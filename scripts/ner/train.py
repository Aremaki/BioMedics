import itertools
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Sized
from itertools import chain, repeat
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

import torch
from accelerate import Accelerator
from confit import Cli, validate_arguments
from confit.utils.random import set_seed
from edsnlp.core.pipeline import Pipeline
from edsnlp.core.registries import registry
from edsnlp.optimization import LinearSchedule, ScheduledOptimizer
from edsnlp.pipes.trainable.embeddings.transformer.transformer import Transformer
from edsnlp.utils.collections import batchify
from edsnlp.utils.typing import AsList
from rich_logger import RichTablePrinter
from spacy.tokens import Doc
from torch.utils.data import DataLoader
from tqdm import trange

from biomedics.ner.reader import EdsMedicReader
from biomedics.ner.scorer import EdsMedicScorer

BASE_DIR = Path.cwd()

app = Cli(pretty_exceptions_show_locals=False)

LOGGER_FIELDS = {
    "step": {},
    "(.*)loss": {
        "goal": "lower_is_better",
        "format": "{:.2e}",
        "goal_wait": 2,
    },
    "(.*)exact_ner/micro/(f)$": {
        "goal": "higher_is_better",
        "format": "{:.2%}",
        "goal_wait": 1,
        "name": r"\1ner_\2",
    },
    "(.*)qualifier/micro/(f)$": {
        "goal": "higher_is_better",
        "format": "{:.2%}",
        "goal_wait": 1,
        "name": r"\1qual_\2",
    },
    "lr": {"format": "{:.2e}"},
    "speed/(.*)": {"format": "{:.2f}", r"name": r"\1"},
    "labels": {"format": "{:.2f}"},
}


def flatten_dict(d, path=""):
    if not isinstance(d, dict):
        return {path: d}

    return {
        k: v
        for key, val in d.items()
        for k, v in flatten_dict(val, f"{path}/{key}" if path else key).items()
    }


class BatchSizeArg:
    """
    Batch size argument validator / caster for confit/pydantic

    Examples
    --------

    ```{ .python .no-check }
    def fn(batch_size: BatchSizeArg):
        return batch_size


    print(fn("10 samples"))
    # Out: (10, "samples")

    print(fn("10 words"))
    # Out: (10, "words")

    print(fn(10))
    # Out: (10, "samples")
    ```
    """

    @classmethod
    def validate(cls, value, config=None):
        value = str(value)
        parts = value.split()
        num = int(parts[0])
        unit = parts[1] if len(parts) == 2 else "samples"
        if (
            len(parts) == 2
            and str(num) == parts[0]
            and unit in ("words", "samples", "spans")
        ):
            return num, unit
        raise Exception(
            f"Invalid batch size: {value}, must be <int> samples|words|spans"
        )

    @classmethod
    def __get_validators__(cls):
        yield cls.validate


if TYPE_CHECKING:
    BatchSizeArg = Tuple[int, str]  # noqa: F811


class LengthSortedBatchSampler:
    """
    Batch sampler that sorts the dataset by length and then batches
    sequences of similar length together. This is useful for transformer
    models that can then be padded more efficiently.

    Parameters
    ----------
    dataset: Iterable
        The dataset to sample from (can be a generator or a fixed size collection)
    batch_size: int
        The batch size
    batch_unit: str
        The unit of the batch size, either "words" or "samples"
    noise: int
        The amount of noise to add to the sequence length before sorting
        (uniformly sampled in [-noise, noise])
    drop_last: bool
        Whether to drop the last batch if it is smaller than the batch size
    buffer_size: Optional[int]
        The size of the buffer to use to shuffle the batches. If None, the buffer
        will be approximately the size of the dataset.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        batch_unit: str,
        noise=1,
        drop_last=True,
        buffer_size: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_unit = batch_unit
        self.noise = noise
        self.drop_last = drop_last
        self.buffer_size = buffer_size

    def __iter__(self):
        # Shuffle the dataset
        if self.batch_unit == "words":

            def sample_len(idx, noise=True):
                count = sum(
                    len(x)
                    for x in next(
                        v
                        for k, v in self.dataset[idx].items()
                        if k.endswith("word_lengths")
                    )
                )
                if not noise:
                    return count
                return count + random.randint(-self.noise, self.noise)

        elif self.batch_unit == "spans":

            def sample_len(idx, noise=True):
                return len(
                    next(
                        v for k, v in self.dataset[idx].items() if k.endswith("begins")
                    )
                )

        else:
            sample_len = lambda idx, noise=True: 1  # noqa: E731

        def make_batches():
            total = 0
            batch = []
            for seq_size, idx in sorted_sequences:
                if total and total + seq_size > self.batch_size:
                    yield batch
                    total = 0
                    batch = []
                total += seq_size
                batch.append(idx)

        # Shuffle the batches in buffer that contain approximately
        # the full dataset to add more randomness
        if isinstance(self.dataset, Sized):
            total_count = sum(sample_len(i, False) for i in range(len(self.dataset)))

        assert (
            isinstance(self.dataset, Sized) or self.buffer_size is not None
        ), "Dataset must have a length or buffer_size must be specified"
        buffer_size = self.buffer_size or math.ceil(total_count / self.batch_size)

        # Sort sequences by length +- some noise
        sorted_sequences = chain.from_iterable(
            sorted((sample_len(i), i) for i in range(len(self.dataset)))
            for _ in repeat(None)
        )

        # Batch sorted sequences
        batches = make_batches()
        buffers = batchify(batches, buffer_size)
        for buffer in buffers:
            random.shuffle(buffer)
            yield from buffer


class SubBatchCollater:
    """
    Collater that splits batches into sub-batches of a maximum size

    Parameters
    ----------
    nlp: Pipeline
        The pipeline object
    embedding: Transformer
        The transformer embedding pipe
    grad_accumulation_max_tokens: int
        The maximum number of tokens (word pieces) to accumulate in a single batch
    """

    def __init__(self, nlp, embedding, grad_accumulation_max_tokens):
        self.nlp = nlp
        self.embedding: Transformer = embedding
        self.grad_accumulation_max_tokens = grad_accumulation_max_tokens

    def __call__(self, seq):
        total = 0
        mini_batches = [[]]
        for sample_features in seq:
            num_tokens = sum(
                math.ceil(len(p) / self.embedding.stride) * self.embedding.window
                for key in sample_features
                if key.endswith("/input_ids")
                for p in sample_features[key]
            )
            if total + num_tokens > self.grad_accumulation_max_tokens:
                print(
                    f"Mini batch size was becoming too large: {total + num_tokens} > "
                    f"{self.grad_accumulation_max_tokens} so it was split"
                )
                total = 0
                mini_batches.append([])
            total += num_tokens
            mini_batches[-1].append(sample_features)
        return [self.nlp.collate(b) for b in mini_batches if len(b)]


def connected_pipes(pipeline: Iterable[Tuple[str, Any]]) -> List[List[str]]:
    pipe_to_params = {}
    for name, pipe in pipeline:
        pipe_to_params[name] = {id(p) for p in pipe.parameters()}
    remaining_pipes = list(pipe_to_params)
    results = []
    while len(remaining_pipes):
        current = [remaining_pipes.pop(0)]
        i = 0
        while i < len(current):
            a = current[i]
            i += 1
            for j, b in enumerate(list(remaining_pipes)):
                if a is not b and pipe_to_params[a] & pipe_to_params[b]:
                    current.append(b)
                    remaining_pipes[j] = None
            remaining_pipes = [p for p in remaining_pipes if p is not None]

        results.append(current)
    return results


@validate_arguments
class TrainingDataLoaderFactory:
    """
    Parameters
    ----------
    data: AsList[EdsMedicReader]
        The training data, can be a EdsMedicReader or a list of EdsMedicReaders
    seed: int
        The seed to use for random number generators when shuffling the data
    batch_size: BatchSizeArg
        The batch size to use for training, support the following units:

        - <int> samples: the number of samples per batch
        - <int> words: the number of words per batch
    grad_accumulation_max_tokens: int
        The maximum number of tokens to accumulate in a single batch

    """

    def __init__(
        self,
        data: AsList[EdsMedicReader],
        batch_size: BatchSizeArg,
        grad_accumulation_max_tokens: int,
        pipe_names: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.data = data
        self.seed = seed
        self.batch_size = batch_size
        self.grad_accumulation_max_tokens = grad_accumulation_max_tokens
        self.pipe_names = pipe_names

    def __call__(self, nlp):
        with nlp.select_pipes(enable=self.pipe_names), set_seed(self.seed):
            trf_pipe = next(
                module
                for name, pipe in nlp.torch_components()
                for module_name, module in pipe.named_component_modules()
                if isinstance(module, Transformer)
            )
            train_docs = [d for td in self.data for d in td(nlp)]
            nlp.post_init(train_docs)
            preprocessed = list(nlp.preprocess_many(train_docs, supervision=True))
            print(
                "Training samples count for "
                f"{', '.join(self.pipe_names)}: {len(preprocessed)}"
            )
            return DataLoader(
                preprocessed,
                batch_sampler=LengthSortedBatchSampler(
                    preprocessed,
                    batch_size=self.batch_size[0],
                    batch_unit=self.batch_size[1],
                ),
                collate_fn=SubBatchCollater(
                    nlp,
                    trf_pipe,
                    grad_accumulation_max_tokens=self.grad_accumulation_max_tokens,
                ),
            )


@app.command(name="train", registry=registry)
def train(
    *,
    nlp: Pipeline,
    train_dataloader: AsList[TrainingDataLoaderFactory],
    val_data: AsList[EdsMedicReader],
    test_data: AsList[EdsMedicReader],
    seed: int = 42,
    max_steps: int = 1000,
    transformer_lr: float = 5e-5,
    task_lr: float = 3e-4,
    validation_interval: int = 10,
    max_grad_norm: float = 5.0,
    warmup_rate: float = 0.1,
    loss_scales: Dict[str, float] = {},
    scorer: EdsMedicScorer,
    output_dir: Optional[Path] = None,
    cpu: bool = False,
):
    """
    Train a model on a dataset.

    Parameters
    ----------
    nlp: Pipeline
        The edsnlp pipeline object
    train_dataloader: AsList[TrainingDataLoaderFactory]
        The training data loader
    val_data: AsList[EdsMedicReader]
        The validation data
    test_data: AsList[EdsMedicReader]
        The test data
    seed: int
        The seed to use for random number generators when initializing the model
    max_steps: int
        The maximum number of training steps
    transformer_lr: float
        The learning rate for the transformer embedding
    task_lr: float
        The learning rate for the task (NER) head
    validation_interval: int
        The number of steps between each validation
    max_grad_norm: float
        The maximum gradient norm to use for gradient clipping
    scorer: EdsMedicScorer
        The scorer object to use for validation
    output_dir: Optional[Path]
        The output directory to save the model and training metrics.
    warmup_rate: float
        The warmup rate to use for the learning rate schedules
    loss_scales: Dict[str, float]
        The loss scales to use for each component
    cpu: bool
        Whether to force the training to run on CPU (useful for M1 chips for which
        all the ops of transformers are not yet supported)

    Returns
    -------
    Pipeline
        The model (trained in place). The artifacts are saved in `artifacts/model-last`
        and `artifacts/train_metrics.json`.
    """

    # ------------------------------
    output_dir = Path(output_dir or BASE_DIR / "models" / "ner" / "temp")
    model_path = output_dir / "model-last"
    train_metrics_path = output_dir / "train_metrics.json"
    os.makedirs(output_dir, exist_ok=True)
    val_docs: List[Doc] = [d for vd in val_data for d in vd(nlp)]

    trainable_pipe_names = {name for name, pipe in nlp.torch_components()}
    print("Trainable components: " + ", ".join(trainable_pipe_names))
    phases = connected_pipes(nlp.torch_components())
    print(
        "Training phases:"
        + "".join(f"\n - {i+1}: {', '.join(n)}" for i, n in enumerate(phases))
    )
    for phase_i, pipe_names in enumerate(phases):
        logger = RichTablePrinter(LOGGER_FIELDS, auto_refresh=False)
        with logger, nlp.select_pipes(disable=trainable_pipe_names - set(pipe_names)):
            print(f"Phase {phase_i+1}: training {', '.join(pipe_names)}")
            set_seed(seed)

            trained_pipes = [nlp.get_pipe(name) for name in pipe_names]
            trf_pipe = next(
                (
                    module
                    for pipe in trained_pipes
                    for name, module in pipe.named_component_modules()
                    if isinstance(module, Transformer)
                ),
                None,
            )

            # Preprocessing training data
            print("Preprocessing data")
            dataloaders = [
                dl(nlp)
                for dl in train_dataloader
                if not dl.pipe_names or set(dl.pipe_names) & set(pipe_names)
            ]

            # Optimizer
            params = set(p for pipe in trained_pipes for p in pipe.parameters())
            trf_params = params & set(trf_pipe.parameters() if trf_pipe else ())
            optim = ScheduledOptimizer(
                torch.optim.AdamW(
                    [
                        {
                            "params": list(params - trf_params),
                            "lr": task_lr,
                            "schedules": LinearSchedule(
                                total_steps=max_steps,
                                warmup_rate=warmup_rate,
                                start_value=task_lr,
                            ),
                        }
                    ]
                    + [
                        {
                            "params": list(trf_params),
                            "lr": transformer_lr,
                            "schedules": LinearSchedule(
                                total_steps=max_steps,
                                warmup_rate=warmup_rate,
                                start_value=0,
                            ),
                        },
                    ][: 1 if transformer_lr else 0]
                )
            )
            grad_params = {p for group in optim.param_groups for p in group["params"]}
            print(
                "Optimizing groups:"
                + "".join(
                    f"\n - {len(group['params'])} weight tensors "
                    f"({sum(p.numel() for p in group['params']):,} parameters)"
                    for group in optim.param_groups
                )
            )
            print(f"Not optimizing {len(params - grad_params):} weight tensors")
            for param in params - grad_params:
                param.requires_grad_(False)

            accelerator = Accelerator(cpu=cpu)
            print("Device:", accelerator.device)
            prep = accelerator.prepare(optim, *dataloaders, *trained_pipes)
            optim = prep[0]
            dataloaders = prep[1 : 1 + len(dataloaders)]
            trained_pipes = prep[1 + len(dataloaders) :]

            cumulated_data = defaultdict(lambda: 0.0, count=0)

            iterators = [
                itertools.chain.from_iterable(itertools.repeat(dl))
                for dl in dataloaders
            ]
            all_metrics = []
            nlp.train(True)
            set_seed(seed)

            # Training loop
            for step in trange(
                max_steps + 1,
                desc="Training model",
                leave=True,
                mininterval=5.0,
            ):
                if (step % validation_interval) == 0:
                    scores = scorer(nlp, val_docs)
                    all_metrics.append(
                        {
                            "step": step,
                            "lr": optim.param_groups[0]["lr"],
                            **cumulated_data,
                            **scores,
                        }
                    )
                    cumulated_data.clear()
                    nlp.to_disk(model_path)
                    train_metrics_path.write_text(json.dumps(all_metrics, indent=2))
                    logger.log_metrics(flatten_dict(all_metrics[-1]))

                if step == max_steps:
                    break

                optim.zero_grad()
                for mini_batch in (b for it in iterators for b in next(it)):
                    loss = torch.zeros((), device=accelerator.device)
                    with nlp.cache():
                        for name, pipe in zip(pipe_names, trained_pipes):
                            if name not in mini_batch:
                                continue
                            res = dict(pipe.module_forward(mini_batch[name]))
                            if "loss" in res:
                                res["loss"] = res["loss"] * loss_scales.get(name, 1)
                                loss += res["loss"]
                                res[f"{name}_loss"] = res["loss"]
                            for key, value in res.items():
                                if key.endswith("loss"):
                                    cumulated_data[key] += float(value)
                            if torch.isnan(loss):
                                raise ValueError(f"NaN loss at component {name}")

                    accelerator.backward(loss)
                    del loss, res, key, value, mini_batch, name, pipe

                torch.nn.utils.clip_grad_norm_(grad_params, max_grad_norm)
                optim.step()

    return nlp


if __name__ == "__main__":
    app()
