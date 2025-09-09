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
    brat_dirs = script_config["brat_dirs"]
    output_dirs = script_config.get("output_dirs")
    if not output_dirs:
        output_dirs = [Path(brat_dir).parent for brat_dir in brat_dirs]
    for brat_dir, output_dir in zip(brat_dirs, output_dirs):
        bio_post_processing(spark, script_config, brat_dir, output_dir)


if __name__ == "__main__":
    app.run()
