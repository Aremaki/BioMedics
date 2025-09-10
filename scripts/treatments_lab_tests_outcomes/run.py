import os

from edstoolbox import SparkApp  # type: ignore

from biomedics.treatments_lab_tests_outcomes.main import (
    compute_save_treatments_lab_tests_outcomes,
)

os.environ["OMP_NUM_THREADS"] = "16"

# Initialize app
app = SparkApp("treatments_lab_tests_outcomes")


@app.submit
def run(spark, sql, config):
    compute_save_treatments_lab_tests_outcomes(sql, config)


if __name__ == "__main__":
    app.run()
