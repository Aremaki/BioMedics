import os

os.environ["OMP_NUM_THREADS"] = "16"

from pathlib import Path
from typing import List

import pandas as pd
from confit import Cli

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
):
    drug_dict = pd.read_pickle(drug_dict_path)

    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        raise ValueError(
            f"Input folder does not exist or is not a directory: {input_folder}"
        )

    brat_dir = input_folder.parent / "pred_NER"
    output_dir = input_folder.parent / "pred_NORM"
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
        print(f"Drug Norm SKIPPED for {brat_dir}, error: {e}")


if __name__ == "__main__":
    app()
