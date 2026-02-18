import sys
from pathlib import Path
from typing import List

from edstoolbox import SparkApp  # type: ignore
from loguru import logger

from biomedics.extract_measurement.main import bio_post_processing

# Initialize app
app = SparkApp("bio_post_processing")


def _discover_brat_dirs(base_dir: Path) -> List[Path]:
    if not base_dir.is_dir():
        return []

    discovered: List[Path] = []

    if list(base_dir.glob("*.txt")):
        discovered.append(base_dir)
        return discovered

    for candidate in base_dir.iterdir():
        if list(candidate.glob("*.txt")):
            discovered.append(candidate)

    return sorted(discovered)


@app.submit
def run(spark, sql, config):
    if config["debug"]["debug"]:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    script_config = config["script"]
    input_folder = Path(script_config["input_folder"])

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
        output_dir = base_output_dir / relative_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        bio_post_processing(spark, script_config, brat_dir, output_dir)


if __name__ == "__main__":
    app.run()
