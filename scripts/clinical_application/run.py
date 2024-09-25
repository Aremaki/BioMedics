import sys

import typer
from confection import Config
from loguru import logger

from biomedics import BASE_DIR
from biomedics.clinical_application.main import (
    filter_bio_nlp,
    filter_bio_structured,
    filter_med_nlp,
    filter_med_structured,
)


def main(config_name: str = "config.cfg"):
    # Load config
    config_path = BASE_DIR / "configs" / "clinical_application" / config_name
    config = Config().from_disk(config_path, interpolate=True)
    if config["debug"]["debug"]:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    filter_bio_structured(config["ANABIO_codes"])
    filter_med_structured(config["ATC_codes"])
    filter_bio_nlp(config["CUI_codes"])
    filter_med_nlp(config["ATC_codes"])


if __name__ == "__main__":
    typer.run(main)
