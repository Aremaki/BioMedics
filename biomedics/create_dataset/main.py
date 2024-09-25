import os

import pandas as pd
import pyspark.sql.functions as F
from IPython.display import display
from loguru import logger
from rich import print

from biomedics import BASE_DIR


# Save txt function
def save_to_txt(path, txt):
    with open(path, "w") as f:
        print(txt, file=f)


def _get_term_from_c_name(c_name):
    """Extract test name from complete name"""
    return c_name[c_name.index(":") + 1 :].split("_")[0].strip()


def get_docs_df(sql, cim10_list, min_len=1000):
    """Get the EHRs with at least one ICD10 code mentionned and one `CRH-HOSPI`
    or `CRH-S` recorded with at least 1000 characters"""
    docs = sql(
        """
        SELECT doc.instance_num, doc.observation_blob, doc.encounter_num,
               doc.patient_num, doc.start_date AS note_date, cim10.concept_cd,
               visit.age_visit_in_years_num, visit.start_date,
        FROM i2b2_observation_doc AS doc
        JOIN i2b2_observation_cim10 AS cim10
        ON doc.encounter_num = cim10.encounter_num
        JOIN i2b2_visit AS visit
        ON doc.encounter_num = visit.encounter_num
        WHERE (doc.concept_cd == 'CR:CRH-HOSPI' OR doc.concept_cd == 'CR:CRH-S')
        """
    )
    # Filter on cim10_list and export to Pandas
    docs_df = docs.filter(docs.concept_cd.isin(cim10_list)).toPandas().dropna()
    # Keep documents with some information at least
    docs_df = docs_df.loc[docs_df["observation_blob"].apply(len) > min_len].reset_index(
        drop=True
    )
    docs_df = (
        docs_df.groupby("observation_blob")
        .agg(
            {
                "instance_num": set,
                "encounter_num": "first",
                "patient_num": "first",
                "age_visit_in_years_num": "first",
                "start_date": "first",
                "note_date": "first",
                "observation_blob": "first",
            }
        )
        .reset_index(drop=True)
    )
    docs_df["instance_num"] = docs_df["instance_num"].apply(
        lambda instance_num: "_".join(list(instance_num))
    )
    return docs_df


def get_bio_df(sql, spark, docs_df):
    """Get the EHRs with at least one ICD10 code mentionned, one `CRH-HOSPI` or `CRH-S`
    recorded with at least 1000 characters and one laboratory test recorded"""
    bio = sql(
        """SELECT
        bio.instance_num AS bio_id,
        bio.concept_cd,
        bio.units_cd,
        bio.nval_num,
        bio.tval_char,
        bio.quantity_num,
        bio.confidence_num,
        bio.encounter_num, bio.patient_num, bio.start_date, concept.name_char
        FROM i2b2_observation_lab AS bio
        JOIN i2b2_concept AS concept
        ON bio.concept_cd = concept.concept_cd"""
    )
    bio = bio.select(
        *[
            F.col(c).cast("string").alias(c) if t == "timestamp" else F.col(c)
            for c, t in bio.dtypes
        ]
    )

    bio_dfs = {}
    for disease in docs_df.disease.unique():
        unique_visit = docs_df[docs_df.disease == disease][
            ["encounter_num"]
        ].drop_duplicates()
        unique_visit = spark.createDataFrame(unique_visit).hint("broadcast")
        bio_df = bio.join(unique_visit, on="encounter_num").toPandas()
        bio_df["disease"] = disease
        bio_df["terms_linked_to_measurement"] = bio_df["name_char"].apply(
            _get_term_from_c_name
        )
        bio_df.loc[bio_df["units_cd"].isna(), "units_cd"] = "nounit"
        bio_df = bio_df[~((bio_df.nval_num.isna()) & (bio_df.tval_char.isna()))]
        display(bio_df)
        bio_dfs[disease] = bio_df

    return bio_dfs


def get_med_df(sql, spark, docs_df):
    """
    Get the EHRs with at least one ICD10 code mentioned, one `CRH-HOSPI` or
    `CRH-S` recorded with at least 1000 characters and one drug treatment recorded
    """
    med = sql(
        """
        SELECT med.instance_num AS med_id, med.concept_cd, med.value_flag_cd,
        med.encounter_num, med.patient_num, med.start_date, concept.name_char
        FROM i2b2_observation_med AS med
        JOIN i2b2_concept AS concept ON med.concept_cd = concept.concept_cd
        """
    )
    med = med.select(
        *[
            F.col(c).cast("string").alias(c) if t == "timestamp" else F.col(c)
            for c, t in med.dtypes
        ]
    )
    med_dfs = {}
    for disease in docs_df.disease.unique():
        unique_visit = docs_df[docs_df.disease == disease][
            ["encounter_num"]
        ].drop_duplicates()
        unique_visit = spark.createDataFrame(unique_visit).hint("broadcast")
        med_df = med.join(unique_visit, on="encounter_num").toPandas()
        med_df["valueflag_cd"] = med_df["valueflag_cd"].mask(
            med_df.concept_cd == "MED:3400892640778", "P01BA02"
        )
        med_df["disease"] = disease
        display(med_df)
        med_dfs[disease] = med_df

    return med_dfs


def create_dataset(sql, spark, config):
    """
    Save the documents, the laboratory tests, and the drug treatments of the
    study cohort in the specified folders
    """
    # Get docs and save it for each disease
    docs_all_diseases = []
    for disease, cim10_list in config["cim10"].items():
        brat_folder_path = BASE_DIR / "data" / "CRH" / "raw" / disease
        if not os.path.exists(brat_folder_path):
            os.mkdir(brat_folder_path)
        docs_df = get_docs_df(
            sql,
            cim10_list=["CIM10:" + cim10 for cim10 in cim10_list],
            min_len=config["min_len"],
        )
        docs_df.apply(
            lambda row: save_to_txt(
                os.path.join(brat_folder_path, row["instance_num"] + ".txt"),
                row["observation_blob"],
            ),
            axis=1,
        )
        for file in os.listdir(brat_folder_path):
            if file.endswith(".txt"):
                ann_file = os.path.join(brat_folder_path, file[:-3] + "ann")
                open(ann_file, mode="a").close()
        logger.info(disease + f" processed {len(docs_df)} docs and saved")
        docs_df["disease"] = disease
        docs_all_diseases.append(docs_df)
    summary_df_docs = pd.concat(docs_all_diseases)
    summary_df_docs.to_pickle(BASE_DIR / "data" / "CRH" / "summary_df_docs.pkl")

    final_result_folder_path = BASE_DIR / "data" / "final_results"
    if not os.path.exists(final_result_folder_path):
        os.mkdir(final_result_folder_path)
    # Get structured laboratory tests and save them
    bio_from_structured_data = get_bio_df(sql, spark, summary_df_docs)
    bio_from_structured_data = pd.concat(list(bio_from_structured_data.values()))
    bio_from_structured_data.to_pickle(
        final_result_folder_path / "bio_from_structured_data.pkl"
    )

    # Get structured drugs and save them
    med_from_structured_data = get_med_df(sql, spark, summary_df_docs)
    med_from_structured_data = pd.concat(list(med_from_structured_data.values()))
    med_from_structured_data.to_pickle(
        final_result_folder_path / "med_from_structured_data.pkl"
    )
