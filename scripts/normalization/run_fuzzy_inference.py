import os

os.environ["OMP_NUM_THREADS"] = "16"

from pathlib import Path
from typing import List, Optional

import pandas as pd
from confit import Cli

from biomedics.normalization.fuzzy.main import FuzzyNormaliser

app = Cli()


@app.command(name="fuzzy_matching")
def normalize_med_cli(
    *,
    drug_dict_path: Path,
    brat_dirs: List[Path],
    label_to_normalize: str,
    qualifiers: List[str],
    method: str,
    threshold: float,
    output_dirs: Optional[List[Path]] = None,
):
    drug_dict = pd.read_pickle(drug_dict_path)
    if not output_dirs:
        output_dirs = [brat_dir.parent for brat_dir in brat_dirs]
    for brat_dir, output_dir in zip(brat_dirs, output_dirs):
        normaliser = FuzzyNormaliser(
            str(brat_dir),
            drug_dict,
            label_to_normalize,
            qualifiers,
            method=method,
            atc_len=7,
        )
        df = normaliser.normalize(threshold=threshold)  # type: ignore
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_pickle(f"{output_dir}/pred_med_fuzzy_{method}.pkl")


if __name__ == "__main__":
    app()
