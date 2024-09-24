---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: biomedics_client
    language: python
    name: biomedics_client
---

```python
%reload_ext autoreload
%autoreload 2
%reload_ext jupyter_black
sc.cancelAllJobs()
```

```python
import os

os.environ["OMP_NUM_THREADS"] = "16"
```

```python
import pandas as pd
from os.path import isfile, isdir, join, basename
from os import listdir, mkdir
import spacy
from edsnlp.processing import pipe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn3, venn2
import altair as alt
from functools import reduce
from knowledge import TO_BE_MATCHED

import sys

BRAT_DIR = "/export/home/cse200093/scratch/BioMedics/data/CRH"
RES_DIR = "/export/home/cse200093/scratch/BioMedics/data/bio_results_v3"
RES_DRUG_DIR = "/export/home/cse200093/scratch/BioMedics/data/drug_results"
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### Functions
<!-- #endregion -->

```python
import pyspark.sql.functions as F

### CELLS TO CREATE THE DATASET CONTAINING ALL TXT FILES WE WANT TO STUDY:
### ALL PATIENTS WITH ONE LINE AT LEAST IN:
# - i2b2_observation_cim10 with correct CIM10 according to `TO_BE_MATCHED`
# - i2b2_observation_doc
# - i2b2_observation_lab (OPTIONAL)

# SHOW DATASETS
sql("USE cse_200093_20210402")
sql("SHOW tables").show(10, False)


# Save txt function
def save_to_txt(path, txt):
    with open(path, "w") as f:
        print(txt, file=f)


def get_docs_df(cim10_list, min_len=1000):
    ### If we filter on `i2b2_observation_lab`
    # docs = sql("""SELECT doc.instance_num, doc.observation_blob, cim10.concept_cd FROM i2b2_observation_doc AS doc
    #               JOIN i2b2_observation_cim10 AS cim10 ON doc.encounter_num = cim10.encounter_num
    #               WHERE ((doc.concept_cd == 'CR:CRH-HOSPI' OR doc.concept_cd == 'CR:CRH-S')
    #               AND EXISTS (SELECT lab.encounter_num FROM i2b2_observation_lab AS lab
    #               WHERE lab.encounter_num = doc.encounter_num))""")

    ### If we don't filter on `i2b2_observation_lab`
    docs = sql(
        """SELECT doc.instance_num, doc.observation_blob, doc.encounter_num, doc.patient_num, doc.start_date AS note_date, visit.age_visit_in_years_num, visit.start_date, cim10.concept_cd FROM i2b2_observation_doc AS doc
                  JOIN i2b2_observation_cim10 AS cim10 ON doc.encounter_num = cim10.encounter_num JOIN i2b2_visit AS visit ON doc.encounter_num = visit.encounter_num
                  WHERE (doc.concept_cd == 'CR:CRH-HOSPI' OR doc.concept_cd == 'CR:CRH-S')
                  """
    )
    ### Filter on cim10_list and export to Pandas
    docs_df = docs.filter(docs.concept_cd.isin(cim10_list)).toPandas().dropna()
    ### Keep documents with some information at least
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


def get_bio_df(summary_docs):
    bio = sql(
        """SELECT bio.instance_num AS bio_id, bio.concept_cd, bio.units_cd, bio.nval_num, bio.tval_char, bio.quantity_num, bio.confidence_num, bio.encounter_num, bio.patient_num, bio.start_date, concept.name_char
        FROM i2b2_observation_lab AS bio JOIN i2b2_concept AS concept ON bio.concept_cd = concept.concept_cd"""
    )
    bio = bio.select(
        *[
            F.col(c).cast("string").alias(c) if t == "timestamp" else F.col(c)
            for c, t in bio.dtypes
        ]
    )

    bio_dfs = {}
    for disease in summary_docs.disease.unique():
        unique_visit = summary_docs[summary_docs.disease == disease][
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


# def get_bio_df_full(cim10_list):
#     docs = sql(
#         """SELECT doc.instance_num, doc.observation_blob, doc.encounter_num, doc.patient_num, doc.start_date AS note_date, visit.age_visit_in_years_num, visit.start_date, cim10.concept_cd FROM i2b2_observation_doc AS doc
#                   JOIN i2b2_observation_cim10 AS cim10 ON doc.encounter_num = cim10.encounter_num JOIN i2b2_visit AS visit ON doc.encounter_num = visit.encounter_num
#                   WHERE (doc.concept_cd == 'CR:CRH-HOSPI' OR doc.concept_cd == 'CR:CRH-S')
#                   """
#     )
#     bio = sql(
#         """SELECT bio.instance_num AS bio_id, bio.concept_cd, bio.units_cd, bio.nval_num, bio.tval_char, bio.quantity_num, bio.confidence_num, bio.encounter_num, bio.patient_num, bio.start_date, concept.name_char, cim10.concept_cd
#         FROM i2b2_observation_lab AS bio JOIN i2b2_concept AS concept ON bio.concept_cd = concept.concept_cd JOIN i2b2_observation_cim10 AS cim10 ON bio.encounter_num = cim10.encounter_num"""
#     )
#     bio_dfs = {}
#     for disease in summary_docs.disease.unique():
#         unique_visit = summary_docs[summary_docs.disease == disease][
#             ["encounter_num"]
#         ].drop_duplicates()
#         unique_visit = spark.createDataFrame(unique_visit).hint("broadcast")
#         bio_df = bio.join(unique_visit, on="encounter_num").toPandas()
#         bio_df["disease"] = disease
#         bio_df["terms_linked_to_measurement"] = bio_df["name_char"].apply(
#             _get_term_from_c_name
#         )
#         bio_df.loc[bio_df["units_cd"].isna(), "units_cd"] = "nounit"
#         bio_df = bio_df[~((bio_df.nval_num.isna()) & (bio_df.tval_char.isna()))]
#         display(bio_df)
#         bio_dfs[disease] = bio_df

#     return bio_dfs


def get_med_df(summary_docs):
    med = sql(
        """SELECT med.instance_num AS med_id, med.concept_cd, med.valueflag_cd, med.encounter_num, med.patient_num, med.start_date, concept.name_char
        FROM i2b2_observation_med AS med JOIN i2b2_concept AS concept ON med.concept_cd = concept.concept_cd"""
    )
    med = med.select(
        *[
            F.col(c).cast("string").alias(c) if t == "timestamp" else F.col(c)
            for c, t in med.dtypes
        ]
    )
    med_dfs = {}
    for disease in summary_docs.disease.unique():
        unique_visit = summary_docs[summary_docs.disease == disease][
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


def _get_term_from_c_name(c_name):
    return c_name[c_name.index(":") + 1 :].split("_")[0].strip()
```

### Get Docs and Bio and Med

```python
# Get docs and save It for each disease
docs_all_diseases = []
for disease, disease_data in TO_BE_MATCHED.items():
    path_to_brat = join(BRAT_DIR, "raw", disease)
    if not os.path.exists(path_to_brat):
        mkdir(path_to_brat)
    docs_df = get_docs_df(["CIM10:" + cim10 for cim10 in disease_data["CIM10"]])
    docs_df.apply(
        lambda row: save_to_txt(
            join(path_to_brat, row["instance_num"] + ".txt"), row["observation_blob"]
        ),
        axis=1,
    )
    for file in os.listdir(path_to_brat):
        if file.endswith(".txt"):
            ann_file = os.path.join(path_to_brat, file[:-3] + "ann")
            open(ann_file, mode="a").close()
    print(disease + f" processed {len(docs_df)} docs and saved")
    docs_df["disease"] = disease
    docs_all_diseases.append(docs_df)
summary_df_docs = pd.concat(docs_all_diseases)
bio_from_structured_data = get_bio_df(summary_df_docs)
bio_from_structured_data = pd.concat(list(bio_from_structured_data.values()))
med_from_structured_data = get_med_df(summary_df_docs)
med_from_structured_data = pd.concat(list(med_from_structured_data.values()))
display(summary_df_docs)
display(bio_from_structured_data)
display(med_from_structured_data)
bio_from_structured_data.to_pickle(join(RES_DIR, "bio_from_structured_data.pkl"))
med_from_structured_data.to_pickle(join(RES_DRUG_DIR, "med_from_structured_data.pkl"))
summary_df_docs.to_pickle(join(BRAT_DIR, "summary_df_docs.pkl"))
```

```python
bio_from_structured_data["found"] = bio_from_structured_data["nval_num"].mask(
    bio_from_structured_data["nval_num"].isna(), bio_from_structured_data["tval_char"]
)
bio_from_structured_data["gold"] = (
    bio_from_structured_data["found"].astype(str)
    + " "
    + bio_from_structured_data["units_cd"]
)
bio_from_structured_data = bio_from_structured_data.groupby(
    ["disease", "encounter_num", "patient_num", "terms_linked_to_measurement"],
    as_index=False,
).agg({"name_char": list, "gold": list})
bio_from_structured_data.to_json(join(RES_DIR, "bio_from_structured_data.json"))
```

# Summary description of the data

```python
import altair as alt

summary_df_docs = pd.read_pickle(join(BRAT_DIR, "summary_df_docs.pkl"))
bio_from_structured_data = pd.read_pickle(join(RES_DIR, "bio_from_structured_data.pkl"))
med_from_structured_data = pd.read_pickle(
    join(RES_DRUG_DIR, "med_from_structured_data.pkl")
)
```

## Number of docs/visit/patients

```python
summary_df_docs.groupby("disease").agg(
    {"instance_num": "nunique", "encounter_num": "nunique", "patient_num": "nunique"}
)
```

```python
complete_case_df = summary_df_docs.merge(
    med_from_structured_data[["encounter_num", "patient_num"]].drop_duplicates(),
    on=["encounter_num", "patient_num"],
    how="inner",
)
complete_case_df = complete_case_df.merge(
    bio_from_structured_data[["encounter_num", "patient_num"]].drop_duplicates(),
    on=["encounter_num", "patient_num"],
    how="inner",
)
complete_case_df.groupby("disease").agg(
    {"instance_num": "nunique", "encounter_num": "nunique", "patient_num": "nunique"}
)
```

## Number of Bio/visit/patient

```python
bio_from_structured_data.groupby("disease").agg(
    {"bio_id": "nunique", "encounter_num": "nunique", "patient_num": "nunique"}
)
```

## Number of Med/visit/patient

```python
med_from_structured_data.groupby("disease").agg(
    {"med_id": "nunique", "encounter_num": "nunique", "patient_num": "nunique"}
)
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Age histogram
<!-- #endregion -->

```python
summary_df_docs["round_age"] = (summary_df_docs["age_visit_in_years_num"] * 2).round(
    -1
) / 2
age_summary = summary_df_docs.groupby(
    ["disease", "age_visit_in_years_num"], as_index=False
).agg({"patient_num": "nunique"})
round_age_summary = summary_df_docs.groupby(
    ["disease", "round_age"], as_index=False
).agg({"patient_num": "nunique"})
total_patient = (
    summary_df_docs.groupby("disease", as_index=False)
    .agg({"patient_num": "nunique"})
    .rename(columns={"patient_num": "total_patient"})
)
age_summary = age_summary.merge(total_patient, on="disease")
age_summary["density"] = age_summary["patient_num"] / age_summary["total_patient"]
display(age_summary)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(round_age_summary).mark_bar(size=12, align="left").encode(
    alt.X("round_age:Q").title("Age at stay"),
    alt.Y("patient_num:Q").title("Number of patients"),
    alt.Row("disease:N"),
).resolve_scale(y="independent").properties(height=200)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(round_age_summary).mark_area(interpolate="step-after").encode(
    alt.X("round_age:Q").title("Age at stay"),
    alt.Y("patient_num:Q").title("Number of patients"),
    alt.Row("disease:N"),
).resolve_scale(y="independent").properties(height=200)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(age_summary).mark_area().encode(
    alt.X("age_visit_in_years_num:Q").title("Age at stay"),
    alt.Y("patient_num:Q").title("Number of patients"),
    alt.Row("disease:N"),
).resolve_scale(y="independent").properties(height=200)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(age_summary).mark_area(interpolate="basis").encode(
    alt.X("age_visit_in_years_num:Q").title("Age at stay"),
    alt.Y("density:Q").title("Density"),
    alt.Row("disease:N"),
).properties(height=200)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(age_summary).mark_bar().encode(
    alt.X("age_visit_in_years_num:Q"),
    alt.Y("density:Q"),
    alt.Row("disease:N"),
    color="disease:N",
).properties(height=200)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(age_summary).mark_area(opacity=0.4).encode(
    alt.X("age_visit_in_years_num:Q"), alt.Y("density:Q").stack(None), color="disease:N"
).properties(height=200)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(age_summary).mark_area().encode(
    alt.X("age_visit_in_years_num:Q"),
    alt.Y("density:Q").stack(True),
    color="disease:N",
).properties(height=200)
```

## Stay start histogramm

```python
summary_df_docs["month_date"] = (
    summary_df_docs["start_date"].dt.strftime("%Y-%m").astype("datetime64[ns]")
)
month_date_summary = summary_df_docs.groupby(
    ["disease", "month_date"], as_index=False
).agg({"encounter_num": "nunique"})
total_visit = (
    summary_df_docs.groupby("disease", as_index=False)
    .agg({"encounter_num": "nunique"})
    .rename(columns={"encounter_num": "total_visit"})
)
month_date_summary = month_date_summary.merge(total_visit, on="disease")
month_date_summary["density"] = (
    month_date_summary["encounter_num"] / month_date_summary["total_visit"]
)
display(month_date_summary)
```

```python
alt.data_transformers.disable_max_rows()
alt.Chart(month_date_summary).mark_bar(align="left").encode(
    alt.X("yearquarter(month_date):T")
    .title("Time (Year)")
    .axis(tickCount="year", labelAngle=0, grid=True, format="%Y"),
    alt.Y("sum(encounter_num):Q").title("Number of stays"),
    alt.Row("disease:N"),
).resolve_scale(y="independent").properties(height=200, width=600)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(month_date_summary).mark_area(interpolate="basis").encode(
    alt.X("month_date:T").title("Time (Year)"),
    alt.Y("density:Q").title("Density"),
    alt.Row("disease:N"),
).properties(height=200, width=600)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(month_date_summary).mark_bar().encode(
    alt.X("month_date:T").title("Time (Year)"),
    alt.Y("density:Q").title("Density"),
    alt.Row("disease:N"),
    color="disease:N",
).properties(height=200)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(month_date_summary).mark_area(opacity=0.4).encode(
    alt.X("month_date:T"), alt.Y("density:Q").stack(None), color="disease:N"
).properties(height=200, width=600)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(month_date_summary).mark_area().encode(
    alt.X("month_date:T"),
    alt.Y("density:Q").stack(True),
    color="disease:N",
).properties(height=200)
```
