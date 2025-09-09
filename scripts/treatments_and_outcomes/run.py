import os

os.environ["OMP_NUM_THREADS"] = "16"
import sys
from pathlib import Path

import pandas as pd
from edstoolbox import SparkApp  # type: ignore
from loguru import logger

# Initialize app
app = SparkApp("treatments_and_outcomes")


@app.submit
def run(spark, sql, config):
    if config["debug"]["debug"]:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    script_config = config["treatments_and_outcomes"]
    input_dirs = script_config["input_dirs"]
    input_dirs = [Path(input_dir).parent for input_dir in input_dirs]
    output_dirs = script_config.get("output_dirs")
    if not output_dirs:
        output_dirs = input_dirs.copy()
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        patient_drugs = pd.read_pickle(f"{input_dir}/pred_med_fuzzy_jaro_winkler.pkl")
        patient_drugs = patient_drugs[
            ~(patient_drugs.Negation == "Neg") & (patient_drugs.Certainty == "Certain")
        ][["source", "term", "label", "norm_term"]].explode("label")
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
            """SELECT doc.instance_num, doc.encounter_num, doc.patient_num, visit.age_visit_in_years_num, visit.start_date, visit.mode_sortie, visit.length_of_stay, patient.death_date FROM i2b2_observation_doc AS doc JOIN i2b2_visit AS visit ON doc.encounter_num = visit.encounter_num JOIN i2b2_patient AS patient ON doc.patient_num = patient.patient_num
                          WHERE (doc.concept_cd == 'CR:CRH-HOSPI' OR doc.concept_cd == 'CR:CRH-S')
                          """
        )
        outcome_df = outcome_df.filter(outcome_df.instance_num.isin(doc_ids)).toPandas()
        outcome_df["source"] = outcome_df["instance_num"] + ".ann"
        outcome_df = outcome_df[
            [
                "source",
                "patient_num",
                "length_of_stay",
                "mode_sortie",
                "start_date",
                "death_date",
            ]
        ]
        patient_drugs = patient_drugs.merge(outcome_df, on="source")
        patient_drugs["Death_hposit"] = (patient_drugs["mode_sortie"] == "6-DC").astype(
            int
        )
        patient_drugs["0 - Décès à 30 jours"] = (
            patient_drugs["death_date"] - patient_drugs["start_date"]
            < pd.Timedelta(30, "day")
        ).astype(int)
        patient_drugs["1 - Décès à 90 jours"] = (
            patient_drugs["death_date"] - patient_drugs["start_date"]
            < pd.Timedelta(90, "day")
        ).astype(int)
        patient_drugs["2 - Décès à 180 jours"] = (
            patient_drugs["death_date"] - patient_drugs["start_date"]
            < pd.Timedelta(180, "day")
        ).astype(int)
        patient_drugs["label_stay"] = ""
        patient_drugs.to_pickle(f"{output_dir}/treatments_outcomes.pkl")


if __name__ == "__main__":
    app.run()
