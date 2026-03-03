import os
import time
from pathlib import Path

import edsnlp
import torch
from confit import Cli, Config
from edsnlp.core.registries import registry
from tqdm import tqdm

from biomedics.ner.brat import BratConnector

app = Cli(pretty_exceptions_show_locals=False)


def ensure_empty_ann_files(folder: Path) -> int:
    created = 0
    for txt_file in folder.glob("*.txt"):
        ann_file = txt_file.with_suffix(".ann")
        if not ann_file.exists():
            ann_file.touch()
            created += 1
    return created


@app.command(name="infer", registry=registry)
def infer(
    *,
    input_folder: Path,
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
    base_input_folder = Path(input_folder)
    if not os.path.isdir(base_input_folder):
        raise ValueError(
            f"Input folder does not exist or is not a directory: {base_input_folder}"
        )

    base_output_folder = base_input_folder.parent / "pred_NER"
    base_output_folder.mkdir(parents=True, exist_ok=True)

    base_txt_files = list(base_input_folder.glob("*.txt"))
    subfolders = [
        path
        for path in base_input_folder.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    has_base_txt = len(base_txt_files) > 0
    has_subfolders = len(subfolders) > 0

    if has_base_txt and has_subfolders:
        raise ValueError(
            "Invalid input structure for "
            f"{base_input_folder}: expected either .txt files only in the base "
            "folder (no subfolders), or subfolders only (no base .txt files)."
        )

    if not has_base_txt and not has_subfolders:
        raise ValueError(f"{base_input_folder} is empty.")

    if has_base_txt:
        print(f"Processing base folder: {base_input_folder}")
        subfolders = [base_input_folder]  # Process base folder only, no subfolders

    for current_input_folder in subfolders:
        print(f"Processing folder: {current_input_folder}")
        try:
            relative_folder = current_input_folder.relative_to(base_input_folder)
            output_folder = base_output_folder / relative_folder

            txt_files = list(current_input_folder.glob("*.txt"))
            if not txt_files:
                print(f"Skipping {current_input_folder}: no .txt files found")
                continue

            output_folder.mkdir(parents=True, exist_ok=True)

            created_ann_count = ensure_empty_ann_files(current_input_folder)
            if created_ann_count > 0:
                print(
                    f"Created {created_ann_count} empty .ann file(s) in {current_input_folder}"
                )

            print(f"Input format is BRAT in {current_input_folder}")
            input_brat = BratConnector(current_input_folder)
            input_docs = list(input_brat.brat2docs(nlp))  # type: ignore

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
                    or "Allergie" in k
                    or "Certainty" in k
                    or "Temporality" in k
                    or "Family" in k
                    or "Negation" in k
                    or "RefTemp" in k
                    or "AttDate" in k
                }
                predicted.append(doc)

            edsnlp.data.write_standoff(  # type: ignore
                predicted,
                output_folder,
                overwrite=True,
                span_getter=["*"],
                span_attributes=[
                    "Negation",
                    "Family",
                    "Temporality",
                    "Certainty",
                    "Action",
                    "Allergie",
                    "RefTemp",
                    "AttDate",
                ],
            )
            print(
                f"NER Prediction is saved in BRAT format in the following folder: {output_folder}"
            )
        except Exception as e:
            print(f"NER SKIPPED for {current_input_folder}, error: {e}")
    tac = time.time()
    print(f"Processed {total_docs} docs in {tac - tic} secondes")


if __name__ == "__main__":
    app()
