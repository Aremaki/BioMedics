import os

os.environ["OMP_NUM_THREADS"] = "16"

from pathlib import Path

import pandas as pd
import typer

from biomedics.normalization.fuzzy.main import FuzzyNormaliser


def normalize_med_cli(
    drug_dict_path: Path,
    input_dir: Path,
    output_dir: Path,
    label_to_normalize: str,
    with_qualifiers: bool,
    method: str,
    threshold: float,
):
    drug_dict = pd.read_pickle(drug_dict_path)
    normaliser = FuzzyNormaliser(
        str(input_dir),
        drug_dict,
        label_to_normalize,
        with_qualifiers,
        method=method,
        atc_len=7,
    )
    df = normaliser.normalize(threshold=threshold)
    if not os.path.exists(output_dir.parent):
        os.makedirs(output_dir.parent)
    df.to_pickle(output_dir)


if __name__ == "__main__":
    typer.run(normalize_med_cli)
