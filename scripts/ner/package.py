from pathlib import Path
from typing import Optional

import edsnlp
from confit import Cli
from tomlkit import parse

from biomedics import BASE_DIR

app = Cli(pretty_exceptions_show_locals=False)

DEFAULT_MODEL = BASE_DIR / "models" / "ner" / "expe_ner_final" / "model-last"


@app.command("package")
def package(
    *,
    model: Optional[Path] = DEFAULT_MODEL,
    name: Optional[str] = None,
    **kwargs,
):
    with open(BASE_DIR / "pyproject.toml") as f:
        doc = parse(f.read())

    try:
        pyproject_model_name = str(doc["tool"]["edsnlp"]["model_name"])
    except KeyError:  # pragma: no cover
        pyproject_model_name = None

    if (pyproject_model_name is None) == (name is None):
        raise ValueError(
            "Please specify pass the --name arg to the script or fill the model_name "
            "in the pyproject.toml (see below):\n"
            "\n"
            "[tool.edsnlp]\n"
            'model_name = "..."\n'
        )
    nlp = edsnlp.load(model)
    nlp.package(name or pyproject_model_name, **kwargs)


if __name__ == "__main__":
    app()
