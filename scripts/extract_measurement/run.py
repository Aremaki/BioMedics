import sys
from pathlib import Path

from edstoolbox import SparkApp  # type: ignore
from loguru import logger

from biomedics.extract_measurement.main import bio_post_processing

# Initialize app
app = SparkApp("bio_post_processing")


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

    brat_dir = input_folder.parent / "pred_NER"
    output_dir = input_folder.parent / "pred_NORM"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        bio_post_processing(spark, script_config, brat_dir, output_dir)
    except Exception as e:
        logger.exception(f"Extract Measurement failed for {brat_dir}, error: {e}")


if __name__ == "__main__":
    app.run()
