from loguru import logger
from edstoolbox import SparkApp
import sys
from biomedics.create_dataset.main import create_dataset

# Initialize app
app = SparkApp("create_dataset")


@app.submit
def run(spark, sql, config):
    if config["debug"]["debug"]:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    script_config = config["script"]
    sql("USE cse_200093_20210402")
    create_dataset(sql, spark, script_config)


if __name__ == "__main__":
    app.run()