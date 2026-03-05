import os

from biomedics.utils.extract_pandas_from_brat import discover_brat_dirs

os.environ["OMP_NUM_THREADS"] = "16"

from pathlib import Path
from typing import List

import pandas as pd
from confit import Cli
from loguru import logger

from biomedics.normalization.fuzzy.main import FuzzyNormaliser

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="fuzzy_matching")
def normalize_med_cli(
    *,
    drug_dict_path: Path,
    input_folder: Path,
    label_to_normalize: str,
    qualifiers: List[str],
    method: str,
    threshold: float,
    batch_size: int,
):
    drug_dict = pd.read_pickle(drug_dict_path)

    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        raise ValueError(
            f"Input folder does not exist or is not a directory: {input_folder}"
        )

    base_ner_dir = input_folder.parent / "pred_NER"
    output_dir = input_folder.parent / "pred_NORM"

    # Count the number of .ann files in the brat_dir
    brat_dirs = discover_brat_dirs(base_ner_dir)
    total_ann_files = sum(len(list(brat_dir.glob("*.ann"))) for brat_dir in brat_dirs)
    logger.info(f"Found {total_ann_files} .ann files in {base_ner_dir}")
    # Split into batch
    if total_ann_files > batch_size:
        logger.info(f"Splitting {total_ann_files} .ann files into batches of {batch_size}")
        batch_brats = []
        batch_num = 1
        ann_counts = 0
        for brat_dir in brat_dirs:
            ann_count = len(list((brat_dir).glob("*.ann")))
            ann_counts += ann_count
            if ann_counts > batch_size:
                logger.info(f"Processing batch {batch_num} of {ann_counts - ann_count} .ann files")
                batch_output_dir = output_dir / f"batch_{batch_num}"
                batch_output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    normaliser = FuzzyNormaliser(
                        batch_brats,
                        drug_dict,
                        label_to_normalize,
                        qualifiers,
                        method=method,
                        atc_len=7,
                    )
                    df = normaliser.normalize(threshold=threshold)  # type: ignore
                    df.to_pickle(batch_output_dir / "pred_med_norm.pkl")
                except Exception as e:
                    logger.exception(
                        f"Fuzzy Inference failed for batch {batch_num} of {len(batch_brats)} brats, error: {e}"
                    )
                batch_brats = []
                ann_counts = ann_count
                batch_num += 1
            batch_brats.append(brat_dir)
        if batch_brats:
            logger.info(f"Processing final batch {batch_num} of {ann_counts} .ann files")
            batch_output_dir = output_dir / f"batch_{batch_num}"
            batch_output_dir.mkdir(parents=True, exist_ok=True)
            try:
                normaliser = FuzzyNormaliser(
                    batch_brats,
                    drug_dict,
                    label_to_normalize,
                    qualifiers,
                    method=method,
                    atc_len=7,
                )
                df = normaliser.normalize(threshold=threshold)  # type: ignore
                df.to_pickle(batch_output_dir / "pred_med_norm.pkl")
            except Exception as e:
                logger.exception(
                    f"Fuzzy Inference failed for final batch {batch_num} of {len(batch_brats)} brats, error: {e}"
                )
    else:
        try:
            normaliser = FuzzyNormaliser(
                str(brat_dir),
                drug_dict,
                label_to_normalize,
                qualifiers,
                method=method,
                atc_len=7,
            )
            df = normaliser.normalize(threshold=threshold)  # type: ignore
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_pickle(output_dir / "pred_med_norm.pkl")
        except Exception as e:
            logger.exception(f"Drug Norm SKIPPED, error: {e}")


if __name__ == "__main__":
    app()
