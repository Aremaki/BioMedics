import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

os.environ["OMP_NUM_THREADS"] = "16"


def compute_save_treatments_lab_tests_outcomes(sql, config):
    if config["debug"]["debug"]:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    script_config = config["treatments_lab_tests_outcomes"]
    input_dirs = script_config["input_dirs"]
    input_dirs = [Path(input_dir).parent for input_dir in input_dirs]
    output_dirs = script_config.get("output_dirs")
    if not output_dirs:
        output_dirs = input_dirs.copy()
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        # Drugs
        patient_drugs = pd.read_pickle(f"{input_dir}/pred_med_fuzzy_jaro_winkler.pkl")
        patient_drugs = patient_drugs[
            ~(patient_drugs.Negation == "Neg") & (patient_drugs.Certainty == "Certain")
        ][["source", "term", "label", "norm_term"]].explode("label")
        label_name = (
            patient_drugs.explode("norm_term")
            .groupby(["label", "norm_term"], as_index=False)
            .agg({"term": "count"})
        )

        # Laboratory tests
        patient_nlp_bio = pd.read_pickle(f"{input_dir}/pred_bio_coder_all.pkl")
        regex_pos = r"([¦|]?positifs?|[¦|]pos?i?t?\b|[¦|]?positiv?e?s?|\bpos\b|[^a-zA-Z0-9]+(?:\+|p)[^a-zA-Z0-9]*$|^\+|presente?s?|presences?)"
        regex_neg = r"([¦|]?negatifs?|[¦|]neg?a?\b|[¦|]?negati?v?e?s?|\bneg\b|[^a-zA-Z0-9]+(?:\-|n)[^a-zA-Z0-9]*$|^\-|^pas\sd[e']|absente?s?|absences?|indetectables?)"
        patient_nlp_bio["lower_bound"] = (
            patient_nlp_bio["range_value"].str.split(r"[\-–]").str.get(0)
        )
        patient_nlp_bio["lower_bound"] = patient_nlp_bio["lower_bound"].where(
            patient_nlp_bio["range_value"].str.split(r"[\-–]").str.len() == 2,
            None,
        )
        patient_nlp_bio["lower_bound"] = (
            patient_nlp_bio["lower_bound"]
            .mask(
                ~(patient_nlp_bio["range_value"].str.contains(">").isna()),
                patient_nlp_bio.range_value.str.extract(r"(\d+[,\.]?\d*)")[0],
            )
            .str.replace(",", ".")
            .astype(float)
        )
        patient_nlp_bio["upper_bound"] = (
            patient_nlp_bio["range_value"].str.split(r"[\-–<>]").str.get(-1)
        )
        patient_nlp_bio["upper_bound"] = patient_nlp_bio["upper_bound"].where(
            patient_nlp_bio["range_value"].str.split(r"[\-–]").str.len() == 2,
            None,
        )
        patient_nlp_bio["upper_bound"] = (
            patient_nlp_bio["upper_bound"]
            .mask(
                ~(patient_nlp_bio["range_value"].str.contains("<").isna()),
                patient_nlp_bio.range_value.str.extract(r"(\d+[,\.]?\d*)")[0],
            )
            .str.replace(",", ".")
            .astype(float)
        )
        patient_nlp_bio["value_as_number"] = (
            patient_nlp_bio.value_cleaned.str.extract(r"(\d+[,\.]?\d*)")[0]
            .str.replace(",", ".")
            .astype(float)
        )
        patient_nlp_bio["positive_value"] = (
            patient_nlp_bio["value_as_number"] < patient_nlp_bio["lower_bound"]
        ) | (patient_nlp_bio["value_as_number"] > patient_nlp_bio["upper_bound"])
        patient_nlp_bio["positive_value"] = patient_nlp_bio["positive_value"].mask(
            (patient_nlp_bio["value_as_number"].isna())
            | (patient_nlp_bio["range_value"].isna()),
            None,
        )
        patient_nlp_bio["positive_text"] = patient_nlp_bio.non_digit_value.str.match(
            regex_pos
        )
        patient_nlp_bio["negative_text"] = patient_nlp_bio.non_digit_value.str.match(
            regex_neg
        )
        patient_nlp_bio["positive_text"] = patient_nlp_bio.positive_text.where(
            patient_nlp_bio["positive_text"], None
        )
        patient_nlp_bio["negative_text"] = patient_nlp_bio.negative_text.where(
            patient_nlp_bio["negative_text"], None
        )
        patient_nlp_bio = patient_nlp_bio[
            ~patient_nlp_bio.positive_text.isna()
            | ~patient_nlp_bio.negative_text.isna()
            | patient_nlp_bio.positive_value.isna()
            | ~patient_nlp_bio.value_as_number.isna()
        ][
            [
                "source",
                "term_bio",
                "term_biocomp",
                "unit",
                "label",
                "norm_term",
                "positive_text",
                "negative_text",
                "positive_value",
                "value_as_number",
                "lower_bound",
                "upper_bound",
            ]
        ].explode("label")

        label_name = (
            patient_drugs.explode("norm_term")
            .groupby(["label", "norm_term"], as_index=False)
            .agg({"term": "count"})
        )

        label_name = label_name.loc[label_name.groupby("label")["term"].idxmax()][
            ["label", "norm_term"]
        ]
        label_name.columns = ["label", "label_name"]

        patient_drugs = (
            patient_drugs.drop(columns="norm_term")
            .merge(label_name, on="label")
            .drop_duplicates(subset=["source", "label_name"])
        )
        doc_ids = list(set(patient_drugs["source"].str.split(".").str.get(0).to_list()))
        sql("USE cse_200093_20210402")
        outcome_df = sql(
            """SELECT doc.instance_num, cim10.concept_cd AS icd10_codes, concept.name_char AS icd10_names, doc.encounter_num, doc.patient_num, visit.age_visit_in_years_num, visit.start_date, visit.mode_sortie, visit.length_of_stay, patient.death_date FROM i2b2_observation_doc AS doc JOIN i2b2_visit AS visit ON doc.encounter_num = visit.encounter_num JOIN i2b2_patient AS patient ON doc.patient_num = patient.patient_num JOIN i2b2_observation_cim10 AS cim10 ON doc.encounter_num = cim10.encounter_num JOIN i2b2_concept AS concept ON cim10.concept_cd = concept.concept_cd
                          WHERE (doc.concept_cd == 'CR:CRH-HOSPI' OR doc.concept_cd == 'CR:CRH-S')
                          """
        )
        outcome_df = outcome_df.filter(outcome_df.instance_num.isin(doc_ids)).toPandas()
        outcome_df["source"] = outcome_df["instance_num"] + ".ann"
        outcome_df["icd10_codes"] = (
            outcome_df["icd10_codes"] + "|" + outcome_df["icd10_names"]
        )
        outcome_df = outcome_df[
            [
                "source",
                "patient_num",
                "length_of_stay",
                "mode_sortie",
                "start_date",
                "death_date",
                "icd10_codes",
            ]
        ]
        # aggregate icd10 codes
        outcome_df = outcome_df.groupby(
            [
                "source",
                "patient_num",
                "length_of_stay",
                "mode_sortie",
                "start_date",
                "death_date",
            ],
            as_index=False,
        ).agg({"icd10_codes": lambda x: list(set(x))})

        outcome_df["Death_hospit"] = (outcome_df["mode_sortie"] == "6-DC").astype(int)
        outcome_df["0 - Décès à 30 jours"] = (
            outcome_df["death_date"] - outcome_df["start_date"]
            < pd.Timedelta(30, "day")
        ).astype(int)
        outcome_df["1 - Décès à 90 jours"] = (
            outcome_df["death_date"] - outcome_df["start_date"]
            < pd.Timedelta(90, "day")
        ).astype(int)
        outcome_df["2 - Décès à 180 jours"] = (
            outcome_df["death_date"] - outcome_df["start_date"]
            < pd.Timedelta(180, "day")
        ).astype(int)
        outcome_df["label_stay"] = ""

        patient_drugs.to_pickle(f"{output_dir}/treatments.pkl")
        outcome_df.to_pickle(f"{output_dir}/outcomes.pkl")
        patient_nlp_bio.to_pickle(f"{output_dir}/lab_tests.pkl")
