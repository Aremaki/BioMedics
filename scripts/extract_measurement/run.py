import sys

from edstoolbox import SparkApp
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
    output_dirs = script_config["output_dirs"]
    for brat_dir, output_dir in zip(brat_dirs, output_dirs):
        bio_post_processing(spark, script_config, brat_dir, output_dir)


if __name__ == "__main__":
    app.run()
