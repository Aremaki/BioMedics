import os

os.environ["OMP_NUM_THREADS"] = "16"

from pathlib import Path
from typing import List

import pandas as pd
from confit import Cli

from biomedics.normalization.fuzzy.main import FuzzyNormaliser

app = Cli(pretty_exceptions_show_locals=False)


def _discover_brat_dirs(base_dir: Path) -> List[Path]:
    if not base_dir.is_dir():
        return []

    discovered: List[Path] = []

    if list(base_dir.glob("*.txt")):
        discovered.append(base_dir)
        return discovered

    for candidate in base_dir.iterdir():
        if candidate.is_dir() and list(candidate.glob("*.txt")):
            discovered.append(candidate)

    return sorted(discovered)


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

    base_brat_dir = input_folder.parent / "pred_NER"
    base_output_dir = input_folder.parent / "pred_NORM"

    if not base_brat_dir.is_dir():
        raise ValueError(
            f"Expected BRAT prediction folder does not exist: {base_brat_dir}"
        )

    brat_dirs_to_process = _discover_brat_dirs(base_brat_dir)
    if not brat_dirs_to_process:
        raise ValueError(
            f"No BRAT directories with .txt files found in {base_brat_dir}"
        )

    for brat_dir in brat_dirs_to_process:
        relative_dir = brat_dir.relative_to(base_brat_dir)
        output_folder = base_output_dir / relative_dir

        normaliser = FuzzyNormaliser(
            str(brat_dir),
            drug_dict,
            label_to_normalize,
            qualifiers,
            method=method,
            atc_len=7,
        )
        df = normaliser.normalize(threshold=threshold)  # type: ignore
        output_folder.mkdir(parents=True, exist_ok=True)
        df.to_pickle(output_folder / "pred_med_norm.pkl")


if __name__ == "__main__":
    app()
