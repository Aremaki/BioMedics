import os

os.environ["OMP_NUM_THREADS"] = "16"

import shutil
from pathlib import Path

import edsnlp
import numpy as np
import pandas as pd
from confit import Cli
from loguru import logger
from spacy.tokens import Span

from biomedics.ner.brat import BratConnector
from biomedics.utils.extract_pandas_from_brat import discover_brat_dirs

app = Cli(pretty_exceptions_show_locals=False)

def process_norm_dir(norm_dir):
    drug_norm_path = Path(norm_dir) / "pred_med_norm.pkl"
    bio_norm_path = Path(norm_dir) / "pred_bio_norm.pkl"
    classify_diso_path = Path(norm_dir) / "pred_with_classified_diso.pkl"
    res_dfs = []
    if os.path.exists(classify_diso_path):
        res_diso_df = pd.read_pickle(
            Path(norm_dir) / "pred_with_classified_diso.pkl"
        )

        # Process data
        res_diso_df["annotation"] = "Body system : " + res_diso_df["labels"].astype(
            str
        )
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
        res_dfs.append(res_diso_df)

    if os.path.exists(drug_norm_path):
        res_drug_df = pd.read_pickle(drug_norm_path)
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
        res_dfs.append(res_drug_df)
    if os.path.exists(bio_norm_path):
        res_bio_df = pd.read_pickle(bio_norm_path)
        res_bio_comp = res_bio_df.copy()
        if "value_cleaned" in res_bio_comp.columns:
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
        else:
            res_bio_comp["annotation"] = "No measurement information extracted"
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
        res_dfs.append(res_bio_df)
        res_dfs.append(res_bio_comp)

    if res_dfs:
        res_df = pd.concat(res_dfs)
    else:
        res_df = None
    return res_df

def process_ner_dir(ner_dirs, base_ner_dir, res_df, conf_path, output_folder):
    for ner_dir in ner_dirs:
        # Load NER data
        doc_list = BratConnector(Path(ner_dir)).brat2docs(edsnlp.blank("eds"))  # type: ignore
        docs = edsnlp.data.from_iterable(doc_list)  # type: ignore

        # Add Annotations
        if not Span.has_extension("note"):
            Span.set_extension("note", default=None)
        if res_df is not None:
            for doc in doc_list:
                source = doc._.note_id + ".ann"
                res_norm = res_df[res_df.source == source]
                for row in res_norm.itertuples():  # type: ignore
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

        # Save in BRAT format
        relative_dir = ner_dir.relative_to(base_ner_dir)
        output_dir = output_folder / relative_dir
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

@app.command(name="group_brat")
def group_brat(
    *,
    input_folder: Path,
    conf_path: Path,
    output_folder: Path,
    batch_size: int,
):
    np.random.seed(42)

    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        raise ValueError(
            f"Input folder does not exist or is not a directory: {input_folder}"
        )

    base_ner_dir = input_folder.parent / "pred_NER"
    norm_folder = input_folder.parent / "pred_NORM"
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
            ann_count = len(list((brat_dir).glob("*.ann")))
            ann_counts += ann_count
            if ann_counts > batch_size:
                logger.info(f"Processing batch {batch_num} of {ann_counts - ann_count} .ann files")
                norm_dir = norm_folder / f"batch_{batch_num}"
                norm_dir.mkdir(parents=True, exist_ok=True)
                try:
                    res_df = process_norm_dir(norm_dir)
                    process_ner_dir(batch_brats, base_ner_dir, res_df, conf_path, output_folder)
                except Exception as e:
                    logger.exception(
                        f"Processing failed for batch {batch_num} of {len(batch_brats)} brats, error: {e}"
                    )
                batch_brats = []
                ann_counts = ann_count
                batch_num += 1
            batch_brats.append(brat_dir)
        if batch_brats:
            logger.info(f"Processing final batch {batch_num} of {ann_counts} .ann files")
            norm_dir = norm_folder / f"batch_{batch_num}"
            norm_dir.mkdir(parents=True, exist_ok=True)
            try:
                res_df = process_norm_dir(norm_dir)
                process_ner_dir(batch_brats, base_ner_dir, res_df, conf_path, output_folder)
            except Exception as e:
                logger.exception(
                    f"Processing failed for final batch {batch_num} of {len(batch_brats)} brats, error: {e}"
                )
    else:
        logger.info(f"Processing all {total_ann_files} .ann files in one batch")
        norm_folder.mkdir(parents=True, exist_ok=True)
        try:
            res_df = process_norm_dir(norm_folder)
            process_ner_dir(brat_dirs, base_ner_dir, res_df, conf_path, output_folder)
        except Exception as e:
            logger.exception(
                f"Processing failed for all {total_ann_files} .ann files, error: {e}"
            )

if __name__ == "__main__":
    app()
