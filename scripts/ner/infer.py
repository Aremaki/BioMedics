import os
import time
from pathlib import Path
from typing import List

import edsnlp
import torch
from confit import Cli, Config
from edsnlp.core.registries import registry
from tqdm import tqdm

from biomedics.ner.brat import BratConnector

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="infer", registry=registry)
def infer(
    *,
    input_folders: List[Path],
    output_folders: List[Path],
    model_path: Path,
    quantize: bool = False,
):
    total_docs = 0
    tic = time.time()
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

    for input_folder, output_folder in zip(input_folders, output_folders):
        assert os.path.isdir(input_folder)
        print(f"Input format is BRAT in {input_folder}")
        input_brat = BratConnector(input_folder)
        input_docs = list(input_brat.brat2docs(nlp))

        total_docs += len(input_docs)
        print("Number of docs:", len(input_docs))

        for doc in input_docs:
            doc.ents = []
            doc.spans.clear()

        predicted = []

        nlp.batch_size = 1

        for doc in tqdm(nlp.pipe(input_docs), total=len(input_docs)):
            doc.user_data = {
                k: v
                for k, v in doc.user_data.items()
                if "note_id" in k
                or "context" in k
                or "split" in k
                or "Action" in k
                # or "Allergie" in k
                or "Certainty" in k
                or "Temporality" in k
                # or "Family" in k
                or "Negation" in k
            }
            predicted.append(doc)

        output_brat = BratConnector(
            output_folder,
            attributes=[
                "Negation",
                # "Family",
                "Temporality",
                "Certainty",
                "Action",
                # "Allergie",
            ],
        )
        output_brat.docs2brat(predicted)
        print(
            (
                f"NER Prediction is saved in BRAT format in the following folder: "
                f"{output_folder}"
            )
        )
    tac = time.time()
    print(f"Processed {total_docs} docs in {tac - tic} secondes")


if __name__ == "__main__":
    app()
