import json
from pathlib import Path
from typing import List

import edsnlp
import pandas as pd
import spacy
import torch
from confit import Cli, Config
from confit.utils.random import set_seed
from edsnlp.core.registries import registry

from biomedics.ner.reader import EdsMedicReader
from biomedics.ner.scorer import EdsMedicScorer

app = Cli(pretty_exceptions_show_locals=False)

BASE_DIR = Path(__file__).parent.parent


def flatten_dict(d, depth=-1, path="", current_depth=0):
    if not isinstance(d, dict) or current_depth == depth:
        return {path: d}
    return {
        k: v
        for key, val in d.items()
        for k, v in flatten_dict(
            val, depth, f"{path}/{key}" if path else key, current_depth + 1
        ).items()
    }


@app.command(name="evaluate", registry=registry)
def evaluate(
    *,
    data: EdsMedicReader,
    model_path: Path = BASE_DIR / "artifacts/model-last",
    scorer: EdsMedicScorer,
    data_seed: int = 42,
    quantize: bool = False,
    output: Path = None,
):
    test_metrics_path = model_path.parent / "test_metrics.json"
    per_doc_path = model_path.parent / "test_metrics_per_doc.jsonl"
    overrides = Config()
    if quantize:
        overrides = overrides.merge(
            {
                "components": {
                    "ner": {
                        "embedding": {
                            "embedding": {
                                "quantization": {
                                    "load_in_4bit": True,
                                    "bnb_4bit_compute_dtype": "float16",
                                },
                                # "torch_dtype": torch.float16,
                            }
                        }
                    },
                    "qualifier": {
                        "embedding": {
                            "embedding": {
                                "embedding": {
                                    "quantization": {
                                        "load_in_4bit": True,
                                        "bnb_4bit_compute_dtype": "float16",
                                    },
                                    # "torch_dtype": torch.float16,
                                }
                            }
                        }
                    },
                }
            }
        )
    nlp = edsnlp.load(model_path, overrides=overrides).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    with set_seed(data_seed):
        val_docs: List[spacy.tokens.Doc] = list(data(nlp))
    scores, per_doc = scorer(nlp, val_docs, per_doc=True, output=output)
    print(
        pd.DataFrame.from_dict(
            {
                f"{prefix}{label}": value
                for prefix, group in (
                    ("ner/", scores.get("exact_ner")),
                    ("qlf/", scores.get("qualifier")),
                    ("", {"speed": scores["speed"]}),
                )
                if group is not None
                for label, value in group.items()
            },
            orient="index",
        ).fillna("")
    )
    test_metrics_path.write_text(json.dumps(scores, indent=2))
    with open(per_doc_path, "w") as f:
        for doc_scores in per_doc:
            f.write(json.dumps(doc_scores, separators=(",", ":")))
            f.write("\n")


if __name__ == "__main__":
    app()
