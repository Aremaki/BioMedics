import sys
from pathlib import Path

from edstoolbox import SparkApp  # type: ignore
from loguru import logger

from biomedics.extract_measurement.main import bio_post_processing
from biomedics.utils.extract_pandas_from_brat import discover_brat_dirs

# Initialize app
app = SparkApp("bio_post_processing")


@app.submit
def run(spark, sql, config):
    if config["debug"]["debug"]:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    script_config = config["script"]
    input_folder = Path(script_config["input_folder"])
    batch_size = script_config.get("batch_size", 80_000)

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
            ann_counts += len(list((brat_dir).glob("*.ann")))
            if ann_counts > batch_size:
                logger.info(f"Processing batch {batch_num} of {ann_counts} .ann files")
                batch_output_dir = output_dir / f"batch_{batch_num}"
                batch_output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    bio_post_processing(spark, script_config, batch_brats, batch_output_dir)
                except Exception as e:
                    logger.exception(
                        f"Extract Measurement failed for batch {batch_num} of {len(batch_brats)} brats, error: {e}"
                    )
                batch_brats = []
                ann_counts = 0
                batch_num += 1
            batch_brats.append(brat_dir)
        if batch_brats:
            logger.info(f"Processing batch {batch_num} of {ann_counts} .ann files")
            batch_output_dir = output_dir / f"batch_{batch_num}"
            batch_output_dir.mkdir(parents=True, exist_ok=True)
            try:
                bio_post_processing(spark, script_config, batch_brats, batch_output_dir)
            except Exception as e:
                logger.exception(
                    f"Extract Measurement failed for final batch {batch_num} of {len(batch_brats)} brats, error: {e}"
                )
    else:
        logger.info(f"Processing all {total_ann_files} .ann files in one batch")
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            bio_post_processing(spark, script_config, brat_dirs, output_dir)
        except Exception as e:
            logger.exception(
                f"Extract Measurement failed for all {total_ann_files} .ann files, error: {e}"
            )

if __name__ == "__main__":
    app.run()
