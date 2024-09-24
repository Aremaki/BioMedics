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

## TODO
- [x] REMGARDER LE SEUIL pour la positivité
- [x] regarder les patients en communs
- [x] regarder Hémoglobine et DFG
- [x] Finish fine tuning of CODER-EDS. Just execute `/export/home/cse200093/Jacques_Bio/normalisation/py_files/train_coder.sh` file up to 1M iterations (To know the number of iteration, just take a look at where the weigths of CODER-EDS are saved, i.e at `/export/home/cse200093/Jacques_Bio/data_bio/coder_output`. The files are saved with the number of iterations in their names.). Evaluate this model then with the files in `/export/home/cse200093/Jacques_Bio/normalisation/notebooks/coder` for example.
- [X] Requêter les médicaments en structuré !
- [X] Finir la normalisation des médicaments NER
- [ ] Cleaner le code et mettre sur GitHub
- [ ] Récupérer les figures
- [ ] Commencer à rédiger


```python
%reload_ext autoreload
%autoreload 2
%reload_ext jupyter_black
# sc.cancelAllJobs()
```

```python
import os
import sys

os.environ["OMP_NUM_THREADS"] = "16"
```

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
from functools import reduce
from biomedics import BASE_DIR
from biomedics.viz.prepare import (
    prepare_structured_bio_df,
    prepare_structured_med_df,
    prepare_nlp_med_df,
    prepare_nlp_bio_df,
)
from biomedics.viz.plot import plot_hist, plot_venn, plot_summary_med, plot_summary_bio
from confection import Config

config_path = BASE_DIR / "configs" / "clinical_application" / "config.cfg"
config = Config().from_disk(config_path, interpolate=True)
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Summary description of the data
<!-- #endregion -->

```python
summary_df_docs = pd.read_pickle(join(BRAT_DIR, "summary_df_docs.pkl"))
bio_from_structured_data = pd.read_pickle(join(RES_DIR, "bio_from_structured_data.pkl"))
med_from_structured_data = pd.read_pickle(
    join(RES_DRUG_DIR, "med_from_structured_data.pkl")
)
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Number of docs/visit/patients
<!-- #endregion -->

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

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Number of Bio/visit/patient
<!-- #endregion -->

```python
bio_from_structured_data.groupby("disease").agg(
    {"bio_id": "nunique", "encounter_num": "nunique", "patient_num": "nunique"}
)
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Number of Med/visit/patient
<!-- #endregion -->

```python
med_from_structured_data.groupby("disease").agg(
    {"med_id": "nunique", "encounter_num": "nunique", "patient_num": "nunique"}
)
```

## Age histogram

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

# Clinical application results

```python
unit_convert = {
    # "Creatinine": {"µmol/l": 1, "µmol_per_l": 1},
    "Hemoglobin": {"g/dl": 1, "mg/dl": 1},
    "CRP": {"mg/l": 1, "mg_per_l": 1, "mg": 1},
    # "INR": {"nounit": 1},
    # "GFR": {"ml/min/1,73m²": 1},
}
possible_values = {
    # "Creatinine": 500,
    "Hemoglobin": 30,
    "CRP": 300,
    # "INR": 15,
    # "GFR": 200,
}
```

## syndrome_des_anti-phospholipides

```python
disease = "syndrome_des_anti-phospholipides"
english_title = "Antiphospholipid syndrome"
cohort_name = "NLP"
nlp_filtered_res, nlp_patient_group = prepare_nlp_bio_df(disease, config)
structured_filtered_res, structured_patient_group = prepare_structured_bio_df(
    disease, config
)
_, nlp_patient_med_group = prepare_nlp_med_df(disease, config)
_, structured_patient_med_group = prepare_structured_med_df(disease, config)
nlp_structured_patient_group = (
    pd.concat([nlp_patient_group, structured_patient_group])
    .groupby("patient_num", as_index=False)
    .max()
)
nlp_structured_patient_med_group = (
    pd.concat([nlp_patient_med_group, structured_patient_med_group])
    .groupby("patient_num", as_index=False)
    .max()
)
if not os.path.exists(BASE_DIR / "figures" / disease):
    os.makedirs(BASE_DIR / "figures" / disease)
```

```python
from functools import reduce

tests_to_keep = [
    "Anti-cardiolipin antibody positive",
    "Anti-B2GP1 antibody positive",
    "Lupus anticoagulant positive",
]
nlp_patient_group["at_least_one"] = reduce(
    lambda x, y: x | y, (nlp_patient_group[col].astype(bool) for col in tests_to_keep)
)
structured_patient_group["at_least_one"] = reduce(
    lambda x, y: x | y,
    (structured_patient_group[col].astype(bool) for col in tests_to_keep),
)
```

```python
tests_to_keep = [
    "Anti-cardiolipin antibody positive",
    "Anti-B2GP1 antibody positive",
    "Lupus anticoagulant positive",
    "at_least_one",
]
bio_summary_nlp = plot_summary_bio(
    nlp_patient_group,
    structured_patient_group,
    tests_to_keep,
    english_title,
    cohort_name,
)
bio_summary_nlp
```

```python
med_summary_nlp = plot_summary_med(
    nlp_patient_med_group,
    structured_patient_med_group,
    english_title,
    cohort_name,
)
med_summary_nlp
```

```python
bio_venn = dict(
    A="Anti-cardiolipin antibody positive",
    B="Anti-B2GP1 antibody positive",
    C="Lupus anticoagulant positive",
)
plot_venn(
    nlp_structured_patient_group,
    bio_venn,
    "NLP cohort",
    method="Total no. of patients with postive tests",
    first=False,
    remove_pos=True,
)

plt.savefig(BASE_DIR / "figures" / disease / "venn_nlp_structured.jpeg")
plt.show()
```

## Lupus

```python
disease = "lupus_erythemateux_dissemine"
english_title = "Lupus"
cohort_name = "NLP"
nlp_filtered_res, nlp_patient_group = prepare_nlp_bio_df(disease, config)
structured_filtered_res, structured_patient_group = prepare_structured_bio_df(
    disease, config
)
_, nlp_patient_med_group = prepare_nlp_med_df(disease, config)
_, structured_patient_med_group = prepare_structured_med_df(disease, config)
nlp_structured_patient_group = (
    pd.concat([nlp_patient_group, structured_patient_group])
    .groupby("patient_num", as_index=False)
    .max()
)
nlp_structured_patient_med_group = (
    pd.concat([nlp_patient_med_group, structured_patient_med_group])
    .groupby("patient_num", as_index=False)
    .max()
)
if not os.path.exists(BASE_DIR / "figures" / disease):
    os.makedirs(BASE_DIR / "figures" / disease)
```

```python
from functools import reduce

tests_to_keep = [
    "Anti-nuclear antibody positive",
    "Anti-DNA antibodies positive",
    "Anti-Sm positive",
]
nlp_patient_group["at_least_one"] = reduce(
    lambda x, y: x | y, (nlp_patient_group[col].astype(bool) for col in tests_to_keep)
)
structured_patient_group["at_least_one"] = reduce(
    lambda x, y: x | y,
    (structured_patient_group[col].astype(bool) for col in tests_to_keep),
)
```

```python
tests_to_keep = [
    "Anti-nuclear antibody positive",
    "Anti-DNA antibodies positive",
    "Anti-Sm positive",
    "at_least_one",
]
bio_summary_nlp = plot_summary_bio(
    nlp_patient_group,
    structured_patient_group,
    tests_to_keep,
    english_title,
    cohort_name,
)
bio_summary_nlp
```

```python
med_summary_nlp = plot_summary_med(
    nlp_patient_med_group,
    structured_patient_med_group,
    english_title,
    cohort_name,
)
med_summary_nlp
```

```python
bio_venn = dict(
    A="Anti-nuclear antibody positive",
    B="Anti-DNA antibodies positive",
    C="Anti-Sm positive",
)
plot_venn(
    nlp_structured_patient_group,
    bio_venn,
    "NLP cohort",
    method="Total no. of patients with postive tests",
    first=False,
    remove_pos=True,
)
plt.savefig(BASE_DIR / "figures" / disease / "venn_nlp_structured.jpeg")
plt.show()
```

## sclerodermie_systemique

```python
disease = "sclerodermie_systemique"
english_title = "Systemic sclerosis"
cohort_name = "NLP"
nlp_filtered_res, nlp_patient_group = prepare_nlp_bio_df(disease, config)
structured_filtered_res, structured_patient_group = prepare_structured_bio_df(
    disease, config
)
_, nlp_patient_med_group = prepare_nlp_med_df(disease, config)
_, structured_patient_med_group = prepare_structured_med_df(disease, config)
nlp_structured_patient_group = (
    pd.concat([nlp_patient_group, structured_patient_group])
    .groupby("patient_num", as_index=False)
    .max()
)
nlp_structured_patient_med_group = (
    pd.concat([nlp_patient_med_group, structured_patient_med_group])
    .groupby("patient_num", as_index=False)
    .max()
)
if not os.path.exists(BASE_DIR / "figures" / disease):
    os.makedirs(BASE_DIR / "figures" / disease)
```

```python
tests_to_keep = [
    "Anti-RNA pol 3 antibody positive",
    "Anti-SCL 70 positive",
    "Anti-centromere antibody positive",
]
bio_summary_nlp = plot_summary_bio(
    nlp_patient_group,
    structured_patient_group,
    tests_to_keep,
    english_title,
    cohort_name,
)
bio_summary_nlp
```

```python
med_summary_nlp = plot_summary_med(
    nlp_patient_med_group,
    structured_patient_med_group,
    english_title,
    cohort_name,
)
med_summary_nlp
```

```python
bio_venn = dict(
    A="Anti-RNA pol 3 antibody positive",
    B="Anti-SCL 70 positive",
    C="Anti-centromere antibody positive",
)
plot_venn(
    nlp_structured_patient_group,
    bio_venn,
    "NLP cohort",
    method="Total no. of patients with postive tests",
    first=False,
    remove_pos=True,
)
plt.savefig(BASE_DIR / "figures" / disease / "venn_nlp_structured.jpeg")
plt.show()
```

## Maladie de Takayasu

```python
disease = "maladie_de_takayasu"
english_title = "Takayasu´s disease"
cohort_name = "NLP"
nlp_filtered_res, _ = prepare_nlp_bio_df(disease, config)
structured_filtered_res, _ = prepare_structured_bio_df(disease, config)
_, nlp_patient_med_group = prepare_nlp_med_df(disease, config)
_, structured_patient_med_group = prepare_structured_med_df(disease, config)

if not os.path.exists(BASE_DIR / "figures" / disease):
    os.makedirs(BASE_DIR / "figures" / disease)
```

```python
med_summary_nlp = plot_summary_med(
    nlp_patient_med_group,
    structured_patient_med_group,
    english_title,
    cohort_name,
)
med_summary_nlp
```

```python
chart = plot_hist(
    unit_convert,
    possible_values,
    nlp_filtered_res,
    structured_filtered_res,
    True,
    False,
    "Takayasu's arteritis",
)
chart.save(BASE_DIR / "figures" / disease / "histogram_NLP_cohort.html")
display(chart)
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Create manual dataset for error analysis on validation cohort
<!-- #endregion -->

```python
from export_pandas_to_brat import export_pandas_to_brat
import pandas as pd
import shutil

ERROR_DIR = "/export/home/cse200093/brat_data/BioMedics/error_analysis_v2"
BRAT_DIR = "/export/home/cse200093/scratch/BioMedics/data/CRH"
qualifiers_columns_name = ["Action", "Certainty", "Negation", "Temporality"]


def retrieve_span_of_terms_linked_to_measurement(term, term_to_norm, term_span):
    if term.find(term_to_norm) < 0:
        return term_span
    else:
        return [
            term_span[0] + term.find(term_to_norm),
            term_span[0] + term.find(term_to_norm) + len(term_to_norm),
        ]


def convert_term(term, term_to_norm):
    if term.find(term_to_norm) < 0:
        return term
    else:
        return term_to_norm
```

```python jupyter={"outputs_hidden": true}
pd.read_json(join(RES_DIR, "maladie_de_takaysu", "pred_bio_coder_all.json"))
```

```python
np.random.seed(42)
for disease, disease_data in TO_BE_MATCHED.items():
    print(disease)
    if not os.path.exists(join(ERROR_DIR, disease)):
        os.makedirs(join(ERROR_DIR, disease))
    res_bio_df = pd.read_json(join(RES_DIR, disease, "pred_bio_coder_all.json"))
    res_drug_df = pd.read_json(
        join(RES_DIR, disease, "pred_med_fuzzy_jaro_winkler.json")
    )
    res_drug_df["annotation"] = res_drug_df["label"].astype(str) + res_drug_df[
        "norm_term"
    ].astype(str)
    res_drug_df["label"] = "Chemical_and_drugs"
    res_drug_df = res_drug_df[
        ["term", "source", "span_converted", "label", "annotation"]
        + qualifiers_columns_name
    ]

    res_bio_comp = res_bio_df.copy()
    res_bio_comp["annotation"] = res_bio_comp["found"].astype(str)
    res_bio_comp["label"] = "BIO_comp"
    res_bio_comp = res_bio_comp[
        ["term", "source", "span_converted", "label", "annotation"]
    ]

    res_bio_df["span_converted"] = res_bio_df.apply(
        lambda row: retrieve_span_of_terms_linked_to_measurement(
            row["term"], row["term_to_norm"], row["span_converted"]
        ),
        axis=1,
    )
    res_bio_df["term"] = res_bio_df.apply(
        lambda row: convert_term(row["term"], row["term_to_norm"]),
        axis=1,
    )
    res_bio_df["annotation"] = res_bio_df["label"].astype(str)
    res_bio_df["label"] = "BIO"
    res_bio_df = res_bio_df[["term", "source", "span_converted", "label", "annotation"]]
    res_df = pd.concat([res_bio_df, res_bio_comp])
    for qualifier in qualifiers_columns_name:
        res_df[qualifier] = None
    res_df = pd.concat([res_df, res_drug_df])
    nlp_filtered_res, nlp_patient_group = prepare_nlp_bio_df(
        disease, complete_case_only=False
    )
    structured_filtered_res, structured_patient_group = prepare_structured_bio_df(
        disease, complete_case_only=False
    )
    all_filtered_res = nlp_patient_group.merge(
        structured_patient_group, on="patient_num", suffixes=("_nlp", "_tabular")
    )
    # Find patients with Tabular data and not NLP data
    for concept in disease_data["ANABIO_codes"].keys():
        if concept not in ["CRP", "Creatinine", "GFR", "Hemoglobin", "INR"]:
            concept_name = concept.lower().replace(" ", "_")
            if not os.path.exists(join(ERROR_DIR, disease, concept_name)):
                os.makedirs(join(ERROR_DIR, disease, concept_name))
            print(concept_name)
            # Find patients with Tabular data and not NLP data
            patients_list = all_filtered_res[
                ~all_filtered_res[f"{concept} positive_nlp"]
                & all_filtered_res[f"{concept} positive_tabular"]
            ].patient_num.unique()
            # Find 10 random stays with Tabular data and not NLP data
            stays_list = structured_filtered_res[
                structured_filtered_res.patient_num.isin(patients_list)
            ]
            stays_list = stays_list[
                stays_list[f"{concept} positive"]
            ].encounter_num.unique()
            stays_list = np.random.choice(stays_list, 10, replace=False)
            # Take the corresponding CRH NLP data
            for stay in stays_list:
                struct_bio = structured_filtered_res[
                    (structured_filtered_res.encounter_num == stay)
                    & (structured_filtered_res[f"{concept} positive"])
                ]
                text_bio_struct = "\n\n\n###### VALEUR RECUPERER DANS LE DOSSIER EN STRUCTURE ######\n\n"
                for row in struct_bio.itertuples():
                    text_bio_struct += f"Nom du test : {row.name_char} | Date : {row.start_date} | Valeur : {row.value} | Unité : {row.unit} | Seuil : {row.confidence_num} | Commentaire : {row.tval_char}\n"
                docs_to_check = nlp_filtered_res[
                    nlp_filtered_res.encounter_num == stay
                ].instance_num.unique()
                if not os.path.exists(
                    join(ERROR_DIR, disease, concept_name, str(stay))
                ):
                    os.makedirs(join(ERROR_DIR, disease, concept_name, str(stay)))
                shutil.copy(
                    join(ERROR_DIR, "annotation.conf"),
                    join(
                        ERROR_DIR, disease, concept_name, str(stay), "annotation.conf"
                    ),
                )
                shutil.copy(
                    join(ERROR_DIR, "kb_shortcuts.conf"),
                    join(
                        ERROR_DIR, disease, concept_name, str(stay), "kb_shortcuts.conf"
                    ),
                )
                shutil.copy(
                    join(ERROR_DIR, "visual.conf"),
                    join(ERROR_DIR, disease, concept_name, str(stay), "visual.conf"),
                )
                for doc in docs_to_check:
                    ann_file = str(doc) + ".ann"
                    txt_file = str(doc) + ".txt"
                    ann_path = join(
                        ERROR_DIR, disease, concept_name, str(stay), ann_file
                    )
                    txt_path = join(BRAT_DIR, "raw", disease, txt_file)
                    export_pandas_to_brat(
                        ann_path,
                        txt_path,
                        res_df.loc[res_df.source == ann_file].reset_index(),
                        "label",
                        "span_converted",
                        "term",
                        "annotation",
                        qualifiers_columns_name,
                    )
                    shutil.copy(
                        join(txt_path),
                        join(ERROR_DIR, disease, concept_name, str(stay), txt_file),
                    )
                    with open(
                        join(ERROR_DIR, disease, concept_name, str(stay), txt_file), "a"
                    ) as note:
                        note.write(text_bio_struct)
```

```python
from extract_pandas_from_brat import extract_pandas
import pandas as pd

ERROR_DIR = "/export/home/cse200093/brat_data/BioMedics/error_analysis"

dfs = []
for disease, disease_data in TO_BE_MATCHED.items():
    print(disease)
    if not disease == "maladie_de_takayasu":
        for concept in disease_data["ANABIO_codes"].keys():
            concept_name = concept.lower().replace(" ", "_")
            for stay in os.listdir(join(ERROR_DIR, disease, concept_name)):
                try:
                    df = extract_pandas(
                        IN_BRAT_DIR=join(ERROR_DIR, disease, concept_name, stay)
                    )
                    df["disease"] = disease
                    df["concept"] = concept_name
                    df["stay"] = stay
                    dfs.append(df)
                except:
                    pass

brat_df = pd.concat(dfs)
comment = brat_df[brat_df.label == "AnnotatorNotes"].copy()
comment.ann_id = comment.span
comment["comment"] = comment.term
comment = comment[["ann_id", "source", "disease", "comment"]]
brat_df_no_comment = brat_df[~(brat_df.label == "AnnotatorNotes")]
brat_df = brat_df_no_comment.merge(
    comment, how="left", on=["ann_id", "source", "disease"]
)
```

```python
neg = brat_df[brat_df.comment.isin(["infra seuil, non significatif", "neg"])].copy()
neg["error_type"] = "Close to threshold"
```

```python
no_mention = brat_df[brat_df.comment == "pas de mention"].copy()
no_mention["error_type"] = "No mention"
```

```python
NER_fail = brat_df[
    (brat_df.label == "BIO_gold")
    & ~(
        brat_df.comment.isin(["infra seuil, non significatif", "neg", "pas de mention"])
    )
]
NER_span = brat_df[~(brat_df.label == "BIO_gold")][
    ["span", "source", "label", "disease"]
]
NER_fail = NER_fail.merge(
    NER_span, on=["span", "source", "disease"], suffixes=("", "_pred"), how="left"
)
NER_fail = NER_fail[NER_fail.label_pred.isna()].drop_duplicates()
NER_fail["error_type"] = "NER fails"
```

```python
Norm_fail = brat_df[
    (brat_df.label == "BIO_gold")
    & ~(
        brat_df.comment.isin(["infra seuil, non significatif", "neg", "pas de mention"])
    )
]
NER_span = brat_df[~(brat_df.label == "BIO_gold")][
    ["span", "source", "label", "comment", "disease"]
]
Norm_fail = Norm_fail.merge(
    NER_span, on=["span", "source"], suffixes=("", "_pred")
).drop_duplicates()
Norm_fail["error_type"] = "Norm fails"
```

```python
result_error = pd.concat(
    [
        Norm_fail[["stay", "source", "concept", "disease", "error_type"]],
        NER_fail[["stay", "source", "concept", "disease", "error_type"]],
        no_mention[["stay", "source", "concept", "disease", "error_type"]],
        neg[["stay", "source", "concept", "disease", "error_type"]],
    ]
)
a = (
    result_error.groupby(["disease", "concept", "error_type"], as_index=False)
    .agg({"stay": "nunique"})
    .rename(columns={"stay": "Number of stays"})
)
b = (
    brat_df.groupby(["disease", "concept"])
    .agg({"stay": "nunique"})
    .rename(columns={"stay": "Total stays"})
)
a.merge(b, on=["disease", "concept"]).groupby(
    ["disease", "concept", "error_type"]
).first()
```

Faire les copier coller
Ajouter le texte du dossier structuré en haut du document à annoter
Objectif montrer l'apport du structuré quand il est là
Et la room too improve
- % de note avec aucune info ?
- % où on trouve et les types d'erreurs trouvées ?

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Create manual dataset for precision evaluation
<!-- #endregion -->

```python
from export_pandas_to_brat import export_pandas_to_brat
import pandas as pd
import shutil

PREC_DIR = "/export/home/cse200093/brat_data/BioMedics/precision_analysis"
BRAT_DIR = "/export/home/cse200093/scratch/BioMedics/data/CRH"
qualifiers_columns_name = ["Action", "Certainty", "Negation", "Temporality"]


def retrieve_span_of_terms_linked_to_measurement(term, term_to_norm, term_span):
    if term.find(term_to_norm) < 0:
        return term_span
    else:
        return [
            term_span[0] + term.find(term_to_norm),
            term_span[0] + term.find(term_to_norm) + len(term_to_norm),
        ]


def convert_term(term, term_to_norm):
    if term.find(term_to_norm) < 0:
        return term
    else:
        return term_to_norm


def slide_spans(term_span, span_bounds):
    return [
        50 + (term_span[0] - span_bounds[0]),
        50 + +(term_span[0] - span_bounds[0]) + term_span[1] - term_span[0],
    ]
```

```python
np.random.seed(42)
for disease, disease_data in TO_BE_MATCHED.items():
    print(disease)
    if not os.path.exists(join(PREC_DIR, disease)):
        os.makedirs(join(PREC_DIR, disease))
    res_bio_df = pd.read_json(join(RES_DIR, disease, "pred_bio_coder_all.json"))
    res_drug_df = pd.read_json(
        join(RES_DIR, disease, "pred_med_fuzzy_jaro_winkler.json")
    )
    res_drug_df["annotation"] = res_drug_df["label"].astype(str) + res_drug_df[
        "norm_term"
    ].astype(str)
    res_drug_df["label"] = "Chemical_and_drugs"
    res_drug_df = res_drug_df[
        ["term", "source", "span_converted", "label", "annotation"]
        + qualifiers_columns_name
    ]

    res_bio_comp = res_bio_df.copy()
    res_bio_comp["annotation"] = res_bio_comp["found"].astype(str)
    res_bio_comp["label"] = "BIO_comp"
    res_bio_comp = res_bio_comp[
        ["term", "source", "span_converted", "label", "annotation"]
    ]

    res_bio_df["span_converted"] = res_bio_df.apply(
        lambda row: retrieve_span_of_terms_linked_to_measurement(
            row["term"], row["term_to_norm"], row["span_converted"]
        ),
        axis=1,
    )
    res_bio_df["term"] = res_bio_df.apply(
        lambda row: convert_term(row["term"], row["term_to_norm"]),
        axis=1,
    )
    res_bio_df["annotation"] = res_bio_df["label"].astype(str)
    res_bio_df["label"] = "BIO"
    res_bio_df = res_bio_df[["term", "source", "span_converted", "label", "annotation"]]
    res_df = pd.concat([res_bio_df, res_bio_comp])
    for qualifier in qualifiers_columns_name:
        res_df[qualifier] = None
    res_df = pd.concat([res_df, res_drug_df])
    nlp_filtered_res, _ = prepare_nlp_bio_df(disease, complete_case_only=False)
    nlp_med_filtered_res, _ = prepare_nlp_med_df(disease, complete_case_only=False)
    # Find BIO
    bio_concepts = list(disease_data["ANABIO_codes"].keys())
    med_concepts = list(disease_data["ATC_codes"].keys())
    for concept in bio_concepts:
        concept_name = concept.lower().replace(" ", "_")
        if os.path.exists(join(PREC_DIR, disease, concept_name)):
            shutil.rmtree(join(PREC_DIR, disease, concept_name))
        os.makedirs(join(PREC_DIR, disease, concept_name))
        shutil.copy(
            join(PREC_DIR, "annotation.conf"),
            join(PREC_DIR, disease, concept_name, "annotation.conf"),
        )
        shutil.copy(
            join(PREC_DIR, "kb_shortcuts.conf"),
            join(PREC_DIR, disease, concept_name, "kb_shortcuts.conf"),
        )
        shutil.copy(
            join(PREC_DIR, "visual.conf"),
            join(PREC_DIR, disease, concept_name, "visual.conf"),
        )
        print(concept_name)
        if concept in bio_concepts:
            if (
                concept
                not in ["CRP", "Creatinine", "GFR", "Hemoglobin", "INR"] + med_concepts
            ):
                concept = f"{concept} positive"
            concept_df = nlp_filtered_res[nlp_filtered_res[concept]].sample(
                n=10, replace=False
            )
        else:
            concept_df = nlp_med_filtered_res[nlp_med_filtered_res[concept]].sample(
                n=10, replace=False
            )
        for pred_row in concept_df.itertuples():
            source = pred_row.source
            span_bounds = pred_row.span_converted
            ann_file = str(source)
            txt_file = str(source)[:-3] + "txt"
            ann_path = join(PREC_DIR, disease, concept_name, ann_file)
            txt_path = join(BRAT_DIR, "raw", disease, txt_file)
            ann_df = (
                res_df.loc[
                    (res_df.source == ann_file)
                    & (
                        (
                            (res_df.span_converted.str.get(0) <= span_bounds[0])
                            & (res_df.span_converted.str.get(1) >= span_bounds[1])
                        )
                        | (
                            (res_df.span_converted.str.get(0) >= span_bounds[0])
                            & (res_df.span_converted.str.get(1) <= span_bounds[1])
                        )
                    )
                ]
                .drop_duplicates(subset="term")
                .reset_index()
            )
            ann_df["span_converted"] = ann_df.apply(
                lambda row: slide_spans(row["span_converted"], span_bounds), axis=1
            )
            snippet = ""
            with open(txt_path, "r") as note:
                ch_count = 0
                for line in note:
                    for ch in line:
                        if ch_count >= (span_bounds[0] - 50) and ch_count <= (
                            span_bounds[1] + 50
                        ):
                            snippet += ch
                        ch_count += 1
            # print(snippet)
            with open(join(PREC_DIR, disease, concept_name, txt_file), "w") as note:
                note.write(snippet)
            export_pandas_to_brat(
                ann_path,
                join(PREC_DIR, disease, concept_name, txt_file),
                ann_df,
                "label",
                "span_converted",
                "term",
                "annotation",
                qualifiers_columns_name,
            )
```

Prendre 10 snippets par elements (Bio + Médicament)
Prcécision par concept + Précision globale  (colle avec nos jeux de données ?)
