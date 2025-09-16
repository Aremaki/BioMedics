import os
import pickle
import shutil

import edsnlp
import pandas as pd
from edstoolbox import SparkApp  # type: ignore
from spacy.tokens import Span
from tqdm import tqdm

from biomedics.ner.brat import BratConnector

# Initialize Spark application
app = SparkApp("BioMedics_merge_all")


@app.submit
def main(spark, sql, config):
    # Load script configuration
    script_config = config["script"]

    # Load normalized (NORM) data
    qualifiers_columns_name = script_config["qualifiers_columns_name"]
    res_bio_df = pd.read_pickle(script_config["result_norm_bio"])
    res_drug_df = pd.read_pickle(script_config["result_norm_drug"])

    # Process drug data
    res_drug_df["annotation"] = (
        "Match synonyme: "
        + res_drug_df["norm_term"].astype(str)
        + " |ATC codes: "
        + res_drug_df["label"].astype(str)
    )
    res_drug_df["label"] = "Chemical_and_drugs"
    res_drug_df = res_drug_df[
        ["term", "source", "span_converted", "label", "annotation"]
        + qualifiers_columns_name
    ]

    # Process biological component data
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

    # Process biological entity data
    res_bio_df = res_bio_df.dropna(subset=["span_start_bio"])
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
    res_bio_df = res_bio_df[["term", "source", "span_converted", "label", "annotation"]]

    # Merge all data
    res_df = pd.concat([res_bio_df, res_bio_comp])
    for qualifier in qualifiers_columns_name:
        res_df[qualifier] = None
    res_df = pd.concat([res_df, res_drug_df])

    # Load Named Entity Recognition (NER) data
    for i in [1, 2]:
        print(f"Processing PART {i}")
        docs = BratConnector(script_config[f"result_ner_part_{i}"]).brat2docs(
            edsnlp.blank("eds")  # type: ignore
        )
        docs = edsnlp.data.from_iterable(docs)  # type: ignore

        # Define relation extraction schema
        print(f"Relation extraction PART {i}")
        relation_schema = {
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

        # Process relations using edsnlp
        nlp = edsnlp.blank("eds")
        nlp.add_pipe("eds.sentences")
        nlp.add_pipe(
            "eds.relations",
            config={
                "scheme": relation_schema,
                "use_sentences": True,
                "clean_rel": True,
                "proximity_method": "right",
                "max_dist": 40,
            },
        )
        docs = docs.map_pipeline(nlp)

        # Merge NER and NORM results
        docs_predicted = []
        if not Span.has_extension("note"):
            Span.set_extension("note", default=None)

        for doc in tqdm(docs, desc="Merging NER and NORM data"):
            source = doc._.note_id + ".ann"
            res_norm = res_df[res_df.source == source]
            for row in res_norm.itertuples():
                for ent in doc.spans[row.label]:
                    if [ent.start_char, ent.end_char] == row.span_converted:
                        ent._.note = row.annotation
                        break
            doc.user_data = {
                k: v
                for k, v in doc.user_data.items()
                if "note_id" in k
                or "context" in k
                or "split" in k
                or "Action" in k
                or "Allergie" in k
                or "Certainty" in k
                or "Temporality" in k
                or "Family" in k
                or "Negation" in k
                or "RefTemp" in k
                or "AttDate" in k
                or "note" in k
                or "rel" in k
            }
            docs_predicted.append(doc)

        # Save processed documents as Spacy docs
        print(f"Save Spacy PART {i}")
        with open(script_config[f"spacy_docs_part_{i}"], "wb") as handle:
            pickle.dump(docs_predicted, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save processed documents as BRAT docs
        print(f"Save BRAT PART {i}")
        brat_dir = script_config[f"brat_docs_part_{i}"]
        if os.path.exists(brat_dir):
            shutil.rmtree(brat_dir)
        os.makedirs(brat_dir)

        brat_config_dir = (
            "/export/home/cse200093/brat_data/guillaume/final_pre_annotation_v1"
        )
        for config_file in ["annotation.conf", "kb_shortcuts.conf", "visual.conf"]:
            shutil.copy(
                os.path.join(brat_config_dir, config_file),
                os.path.join(brat_dir, config_file),
            )

        edsnlp.data.write_standoff(  # type: ignore
            docs_predicted,
            brat_dir,
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

    # Convert a sample in BRAT format for visualization
    print("Save sample BRAT")
    sample_brat_dir = script_config["sample_brat"]
    if os.path.exists(sample_brat_dir):
        shutil.rmtree(sample_brat_dir)
    os.makedirs(sample_brat_dir)

    brat_config_dir = (
        "/export/home/cse200093/brat_data/guillaume/final_pre_annotation_v1"
    )
    for config_file in ["annotation.conf", "kb_shortcuts.conf", "visual.conf"]:
        shutil.copy(
            os.path.join(brat_config_dir, config_file),
            os.path.join(sample_brat_dir, config_file),
        )

    edsnlp.data.write_standoff(  # type: ignore
        list(docs)[:100],  # type: ignore
        sample_brat_dir,
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
    app.run()
