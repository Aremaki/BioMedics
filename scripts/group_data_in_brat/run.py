import os

os.environ["OMP_NUM_THREADS"] = "16"

import shutil
from pathlib import Path
from typing import List, Optional

import edsnlp
import numpy as np
import pandas as pd
from confit import Cli
from spacy.tokens import Span

from biomedics.ner.brat import BratConnector

app = Cli()


@app.command(name="group_brat")
def group_brat(
    *,
    input_dirs: List[Path],
    conf_path: Path,
    output_dirs: Optional[List[Path]] = None,
):
    np.random.seed(42)

    if not output_dirs:
        output_dirs = [
            Path(f"export/home/brat_data/test_demo/test_{i+1}")
            for i in range(len(input_dirs))
        ]
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        res_bio_df = pd.read_pickle(Path(input_dir).parent / "pred_bio_coder_all.pkl")
        res_drug_df = pd.read_pickle(
            Path(input_dir).parent / "pred_med_fuzzy_jaro_winkler.pkl"
        )
        res_diso_df = pd.read_pickle(
            Path(input_dir).parent / "pred_with_classified_diso.pkl"
        )

        # Process data
        res_diso_df["annotation"] = "Body system : " + res_diso_df["labels"].astype(str)
        res_diso_df["label"] = "DISO"
        res_diso_df = res_diso_df[
            [
                "term",
                "source",
                "span_converted",
                "label",
                "annotation",
            ]
        ]

        res_drug_df["annotation"] = (
            "Match synonyme: "
            + res_drug_df["norm_term"].astype(str)
            + " |ATC codes: "
            + res_drug_df["label"].astype(str)
        )
        res_drug_df["label"] = "Chemical_and_drugs"
        res_drug_df = res_drug_df[
            [
                "term",
                "source",
                "span_converted",
                "label",
                "annotation",
            ]
        ]

        res_bio_comp = res_bio_df.copy()
        res_bio_comp["annotation"] = (
            "Value: "
            + res_bio_comp["value_cleaned"].astype(str)
            + " |Unit: "
            + res_bio_comp["unit"].astype(str)
            + " |Range value: "
            + res_bio_comp["range_value"].astype(str)
            + " |Comment: "
            + res_bio_comp["non_digit_value"].astype(str)
        )
        res_bio_comp["label"] = "BIO_comp"
        res_bio_comp["term"] = res_bio_comp["term_biocomp"]
        res_bio_comp["span_converted"] = res_bio_comp.apply(
            lambda row: [int(row.span_start), int(row.span_end)], axis=1
        )
        res_bio_comp = res_bio_comp[
            ["term", "source", "span_converted", "label", "annotation"]
        ]

        res_bio_df = res_bio_df[~res_bio_df.span_start_bio.isna()]
        res_bio_df["annotation"] = (
            "Match synonyme: "
            + res_bio_df["norm_term"].astype(str)
            + " |CUI code: "
            + res_bio_df["label"].astype(str)
        )
        res_bio_df["label"] = "BIO"
        res_bio_df["term"] = res_bio_df["term_bio"]
        res_bio_df["span_converted"] = res_bio_df.apply(
            lambda row: [int(row.span_start_bio), int(row.span_end_bio)], axis=1
        )
        res_bio_df = res_bio_df[
            ["term", "source", "span_converted", "label", "annotation"]
        ]
        res_df = pd.concat([res_bio_df, res_bio_comp])
        res_df = pd.concat([res_df, res_drug_df, res_diso_df])

        # Load NER data
        doc_list = BratConnector(Path(input_dir)).brat2docs(edsnlp.blank("eds"))  # type: ignore
        docs = edsnlp.data.from_iterable(doc_list)  # type: ignore

        # Add Annotations
        if not Span.has_extension("note"):
            Span.set_extension("note", default=None)
        for doc in doc_list:
            source = doc._.note_id + ".ann"
            res_norm = res_df[res_df.source == source]
            for row in res_norm.itertuples():
                for ent in doc.spans[row.label]:
                    if [ent.start_char, ent.end_char] == row.span_converted:
                        ent._.note = row.annotation
                        break

        # Add Relation
        scheme = {
            "source": [{"label": "Chemical_and_drugs", "attr": None}],
            "target": [
                {"label": "dosage", "attr": None},
                {"label": "strength", "attr": None},
                {"label": "form", "attr": None},
                {"label": "Frequency", "attr": None},
            ],
            "type": "Depend",
            "inv_type": "inv_Depend",
        }

        nlp = edsnlp.blank("eds")

        # Extraction of entities
        nlp.add_pipe("eds.sentences")
        nlp.add_pipe(
            "eds.relations",
            config={
                "scheme": scheme,
                "use_sentences": True,
                "clean_rel": True,
                "proximity_method": "right",
                "max_dist": 40,
            },
        )
        docs = docs.map_pipeline(nlp)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        all_docs = list(docs)
        if len(all_docs) > 100:
            for i in range(0, len(all_docs), 100):
                sub_docs = all_docs[i : i + 100]
                sub_output_dir = f"{output_dir}/sub_folder_{i}"
                if os.path.exists(sub_output_dir):
                    shutil.rmtree(sub_output_dir)
                os.makedirs(sub_output_dir)
                shutil.copy(
                    f"{conf_path}/annotation.conf",
                    f"{sub_output_dir}/annotation.conf",
                )
                shutil.copy(
                    f"{conf_path}/kb_shortcuts.conf",
                    f"{sub_output_dir}/kb_shortcuts.conf",
                )
                shutil.copy(
                    f"{conf_path}/visual.conf",
                    f"{sub_output_dir}/visual.conf",
                )
                edsnlp.data.write_standoff(  # type: ignore
                    sub_docs,
                    sub_output_dir,
                    overwrite=True,
                    span_getter=["*"],
                    span_attributes=[
                        "Negation",
                        "Family",
                        "Temporality",
                        "Certainty",
                        "Action",
                        "Allergie",
                        "RefTemp",
                        "AttDate",
                    ],
                )
        else:
            shutil.copy(
                f"{conf_path}/annotation.conf",
                f"{output_dir}/annotation.conf",
            )
            shutil.copy(
                f"{conf_path}/kb_shortcuts.conf",
                f"{output_dir}/kb_shortcuts.conf",
            )
            shutil.copy(
                f"{conf_path}/visual.conf",
                f"{output_dir}/visual.conf",
            )
            edsnlp.data.write_standoff(  # type: ignore
                all_docs,
                output_dir,
                overwrite=True,
                span_getter=["*"],
                span_attributes=[
                    "Negation",
                    "Family",
                    "Temporality",
                    "Certainty",
                    "Action",
                    "Allergie",
                    "RefTemp",
                    "AttDate",
                ],
            )


if __name__ == "__main__":
    app()
