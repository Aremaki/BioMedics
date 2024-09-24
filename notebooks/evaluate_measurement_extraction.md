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
```

# Evaluation

```python
import pandas as pd
import re
from os.path import isfile, isdir, join, basename
from os import listdir
import numpy as np
import altair as alt
from functools import reduce

import sys
from biomedics.utils.extract_pandas_from_brat import extract_pandas
from biomedics import BASE_DIR

QUAERO_DIR = BASE_DIR / "data" / "QUAERO" / "corpus"
BRAT_DIR = BASE_DIR / "data" / "annotated_CRH" / "raw" / "cui_annotations"
BRAT_DIR_BIO_VAL = (
    BASE_DIR / "data" / "annotated_CRH" / "raw" / "measurement_annotations"
)
UMLS_BIO_DIR = BASE_DIR / "data" / "umls" / "lab_snomed_ct_2021AB.pkl"
UMLS_MED_DIR = BASE_DIR / "data" / "umls" / "atc_cui_2023AB.pkl"

## BioVal
PRED_BIO_VAL_ONLY = (
    BASE_DIR
    / "data"
    / "annotated_CRH"
    / "post_processed"
    / "expe_measurement"
    / "pred"
    / "Norm_only"
    / "pred_with_measurement.pkl"
)
PRED_BIO_VAL_NER = (
    BASE_DIR
    / "data"
    / "annotated_CRH"
    / "post_processed"
    / "expe_measurement"
    / "pred"
    / "NER_Norm"
    / "pred_with_measurement.pkl"
)

# BIO
PRED_BIO_CODER = (
    BASE_DIR
    / "data"
    / "annotated_CRH"
    / "post_processed"
    / "expe_complete_pipe"
    / "pred"
    / "NER_Norm"
    / "pred_bio_coder_all.pkl"
)
PRED_BIO_FUZZY = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/CRH/pred/total/Normalisation/pred_bio_fuzzy_jw.json"

# MED
PRED_MED_CODER = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/CRH/pred/total/Normalisation/pred_med_coder_all.json"
PRED_MED_FUZZY = (
    BASE_DIR
    / "data"
    / "annotated_CRH"
    / "post_processed"
    / "expe_complete_pipe"
    / "pred"
    / "NER_Norm"
    / "pred_med_fuzzy_jw.pkl"
)

MICRO = True  # Whether to have micro or macro scale when calculating accuracy
```

## Generate GOLD STANDRAD

```python
# Generate GOLD STANDRAD

from biomedics.ner.brat import BratConnector
import edsnlp

doc_list = BratConnector(BRAT_DIR).brat2docs(edsnlp.blank("eds"))
ents_list = []
for doc in doc_list:
    for ent in doc.spans["BIO"]:
        ent_data = [
            ent.text,
            ent.label_,
            doc._.note_id + ".ann",
            [ent.start_char, ent.end_char],
            ent.text.lower().strip(),
        ]
        for qualifier in ["comments"]:
            ent_data.append(getattr(ent._, qualifier))
        ents_list.append(ent_data)
    for ent in doc.spans["Chemical_and_drugs"]:
        ent_data = [
            ent.text,
            ent.label_,
            doc._.note_id + ".ann",
            [ent.start_char, ent.end_char],
            ent.text.lower().strip(),
        ]
        for qualifier in ["comments"]:
            ent_data.append(getattr(ent._, qualifier))
        ents_list.append(ent_data)
df_columns = ["term", "label", "source", "span_converted", "norm_term"] + ["annotation"]
gold_aphp = pd.DataFrame(ents_list, columns=df_columns)
wrong_annotation = {
    "C0005803": "C0337438 ; C0392201",
    "C0017654": "C4544895",
    "C0948769": "C0201485 ; C0201487",
    "C0201487": "C0201485 ; C0201487",
    "C0282056": "C1262035",
    "C1168447": "C0201362",
    "C5190264": "C0201363",
    "C0134596": "C0201532",
    "C4264019": "C1294200",
    "C5201961": "C1266574",
    "NaN": None,
    "HO2": "H02",
    "H02AB07": "H02AB",
    "N03AX16": "N02BF02 ; N03AX16",
    "N03AX12": "N02BF01 ; N03AX12",
    "L01XC02": "L01XC02 ; L01FA01",
    "L01BA01": "L01BA01 ; L04AX03",
    "J07AL0": "J07AL01",
    "P01BA02": "P01BA02 ; M01CA",
    "C3159309": "B01AF01",
    "C0202194": "C0729816 ; C0202194",
    "C1278302": "C1278302 ; C0201476",
    "C0337438": "C0337438 ; C0392201",
    "C1277776": "C1277776 ; C1294978",
    "C1303280 ; C1295005": "C1303280 ; C1295005 ; C1531512",
    "C1295005": "C1303280 ; C1295005 ; C1531512",
    "C0201804": "C0201804 ; C0202150",
    "C0428472": "C0428472 ; C0392885",
    "C0428474": "C0428474 ; C0202117",
    "C0523115": "C0523115 ; C0023508",
    "C0202178": "C0202178 ; C0523826",
    "C1278049": "C1278049 ; C0202035",
    "C0587344": "C0587344 ; C0079611",
    "C0455288": "C0455288 ; C2732404",
    "C0430072": "C0430072 ; C0428539",
    "C0427417": "C0427417 ; C0523631",
}
gold_aphp["annotation"] = gold_aphp.annotation.str.strip()
gold_aphp.annotation = gold_aphp.annotation.str.split(":").str.get(-1)
gold_aphp = gold_aphp.replace({"annotation": wrong_annotation})
gold_aphp = gold_aphp[~gold_aphp.annotation.isna()]
gold_aphp["annotation"] = gold_aphp.annotation.str.strip()
gold_aphp.span_converted = gold_aphp.span_converted.astype(str)
gold_aphp = gold_aphp.drop_duplicates()
gold_aphp.span_converted = gold_aphp.span_converted.apply(lambda x: eval(x))

# Add umls synonyms for better visualisation
umls_bio_df = (
    pd.read_pickle(UMLS_BIO_DIR).groupby("CUI", as_index=False).agg({"STR": set})
)
umls_med_df = pd.read_pickle(UMLS_MED_DIR)[["CUI", "STR"]].drop_duplicates(subset="CUI")
umls_df = pd.concat([umls_bio_df, umls_med_df])
gold_aphp = gold_aphp.merge(
    umls_df, left_on="annotation", right_on="CUI", how="left"
).drop(columns=["CUI"])

gold_aphp.annotation = gold_aphp.annotation.str.split(" ; ")
gold_aphp.annotation = gold_aphp.annotation.apply(set)
gold_aphp
```

```python
# Generate GOLD STANDRAD BIO VAL
nlp = edsnlp.blank("eds")
nlp.add_pipe("eds.sentences")
nlp.add_pipe("eds.tables")
nlp.add_pipe(
    "eds.measurements",
    config=dict(measurements="all", extract_ranges=True, use_tables=True),
)

REGEX_CONVERT_SPANS = re.compile("^(\d+).*\s(\d+)$")
# Terms which will make the measurements pipe match a positive measurement
positive_terms_from_label_key = (
    "positifs",
    "positives",
    "positivites",
    "presence",
    "presences",
)
# We create a list to match abbreviations of the positive words. This list will
# be the final dictionnary used to match the positive measurements.
positive_terms_from_label_key = [
    word[: i + 1]
    for word in positive_terms_from_label_key
    for i in range(min(len(word) - 1, 1), len(word))
]
# Symbols which will make the measurements pipe match a positive measurement
positive_symbols_from_label_key = ["\+", "p"]
positive_regex_from_label_key = (
    r"^[^a-zA-Z0-9]*(?:% s)"
    % "|".join(positive_symbols_from_label_key + positive_terms_from_label_key)
    + r"[^a-zA-Z0-9]*$"
)

# Terms which will make the measurements pipe match a negative measurement
negative_terms_from_label_key = (
    "negatifs",
    "negatives",
    "negativites",
    "absences",
    "absents",
)
# We create a list to match abbreviations of the negative words. This list will
# be the final dictionnary used to match the negative measurements.
negative_terms_from_label_key = [
    word[: i + 1]
    for word in negative_terms_from_label_key
    for i in range(min(len(word) - 1, 1), len(word))
]
# Symbols which will make the measurements pipe match a positive measurement
negative_symbols_from_label_key = ["\-", "n"]
# To match symbols, we create regex
negative_regex_from_label_key = (
    r"^[^a-zA-Z0-9]*(?:% s)"
    % "|".join(negative_symbols_from_label_key + negative_terms_from_label_key)
    + r"[^a-zA-Z0-9]*$"
)

# Terms which will make the measurements pipe match a normal measurement
normal_terms_from_label_key = ("normales", "normaux", "normalisations", "normalites")
# We create a list to match abbreviations of the normal words. This list will
# be the final dictionnary used to match the normal measurements.
normal_terms_from_label_key = [
    word[: i + 1]
    for word in normal_terms_from_label_key
    for i in range(min(len(word) - 1, 1), len(word))
]


def convert_spans(span):
    span_match = REGEX_CONVERT_SPANS.match(span)
    span_start = int(span_match.group(1))
    span_end = int(span_match.group(2))
    return [span_start, span_end]


raw_bio_val = extract_pandas(BRAT_DIR_BIO_VAL)
gold_bio_val = raw_bio_val.merge(
    raw_bio_val[["span", "source", "term"]].rename(
        columns={"span": "ann_id", "term": "annotation"}
    ),
    on=["ann_id", "source"],
    how="left",
)
gold_bio_val = gold_bio_val[gold_bio_val.label == "BIO_Val"]
gold_bio_val.annotation = gold_bio_val.annotation.str.split("_").str.get(-1)
gold_bio_val["span_converted"] = gold_bio_val["span"].apply(convert_spans)
gold_bio_val["annotation"] = gold_bio_val.annotation.str.strip()
gold_bio_val["annotation"] = gold_bio_val.annotation.str.replace(r"G\/+[lL]", "x10*9/l")
gold_bio_val["annotation"] = gold_bio_val.annotation.str.lower()
all_measures = []
for row in gold_bio_val.annotation:
    res = nlp(row)
    if len(res.spans["measurements"]) == 0:
        measures = None
    else:
        measures = "{} {}".format(
            float(res.spans["measurements"][0]._.value.value),
            res.spans["measurements"][0]._.value.unit,
        )
    all_measures.append(measures)
gold_bio_val = gold_bio_val.reset_index(drop=True).assign(
    norm_annotation=pd.Series(all_measures)
)
gold_bio_val["neg"] = gold_bio_val.annotation.str.match(negative_regex_from_label_key)
gold_bio_val["pos"] = gold_bio_val.annotation.str.match(positive_regex_from_label_key)
gold_bio_val["norm"] = gold_bio_val.annotation.isin(normal_terms_from_label_key)
gold_bio_val["norm_annotation"] = gold_bio_val["norm_annotation"].mask(
    gold_bio_val["norm_annotation"].isna() & gold_bio_val["neg"], "0 bool"
)
gold_bio_val["norm_annotation"] = gold_bio_val["norm_annotation"].mask(
    gold_bio_val["norm_annotation"].isna() & gold_bio_val["pos"], "1 bool"
)
gold_bio_val["norm_annotation"] = gold_bio_val["norm_annotation"].mask(
    gold_bio_val["norm_annotation"].isna() & gold_bio_val["norm"], "0.5 bool"
)
gold_bio_val = gold_bio_val[~gold_bio_val.norm_annotation.isna()]
gold_bio_val
```

## Extract Biology VALUE ONLY

```python
import pandas as pd


pred_bio = pd.read_pickle(PRED_BIO_VAL_ONLY)
pred_bio = pred_bio[~pred_bio.term_bio.isna()]
pred_bio["term"] = pred_bio["term_bio"]


def convert_spans(row):
    return [int(row.span_start_bio), int(row.span_end_bio)]


pred_bio["span_converted"] = pred_bio.apply(convert_spans, axis=1)
pred_bio["found"] = pred_bio["value_cleaned"] + " " + pred_bio["unit"]
all_measures = []
for row in pred_bio.found:
    res = nlp(str(row))
    if len(res.spans["measurements"]) == 0:
        measures = None
    else:
        measures = "{} {}".format(
            float(res.spans["measurements"][0]._.value.value),
            res.spans["measurements"][0]._.value.unit,
        )
    all_measures.append(measures)
pred_bio = pred_bio.reset_index(drop=True).assign(found=pd.Series(all_measures))
pred_bio["neg"] = pred_bio.non_digit_value.str.match(negative_regex_from_label_key)
pred_bio["pos"] = pred_bio.non_digit_value.str.match(positive_regex_from_label_key)
pred_bio["norm"] = pred_bio.non_digit_value.isin(normal_terms_from_label_key)
pred_bio["found"] = pred_bio["found"].mask(
    pred_bio["found"].isna() & pred_bio["neg"], "0 bool"
)
pred_bio["found"] = pred_bio["found"].mask(
    pred_bio["found"].isna() & pred_bio["pos"], "1 bool"
)
pred_bio["found"] = pred_bio["found"].mask(
    pred_bio["found"].isna() & pred_bio["norm"], "0.5 bool"
)
pred_bio = pred_bio[~pred_bio.found.isna()]
pred_bio
```

```python
res_df = {
    "source": [],
    "pred_term": [],
    "gold_term": [],
    "gold_annotation": [],
    "pred_value": [],
    "pred_span": [],
    "gold_span": [],
    "TP": [],
    "Detected": [],
    "FP": [],
    "FN": [],
}
for row in pred_bio.itertuples():
    source = row.source
    span = row.span_converted
    pred_term = row.term
    pred_value = row.found
    match = gold_bio_val[
        (gold_bio_val.source == source)
        & (
            (
                (gold_bio_val.span_converted.str.get(0) <= span[0])
                & (gold_bio_val.span_converted.str.get(1) >= span[0])
            )
            | (
                (gold_bio_val.span_converted.str.get(0) <= span[1])
                & (gold_bio_val.span_converted.str.get(1) >= span[1])
            )
            | (
                (gold_bio_val.span_converted.str.get(0) >= span[0])
                & (gold_bio_val.span_converted.str.get(1) <= span[1])
            )
        )
    ]
    if match.empty:
        TP = False
        FP = True
        Detected = False
        FN = False
        gold_term = None
        gold_annotation = None
        gold_span = None
        continue
    else:
        match_annotation = match[match.norm_annotation == pred_value]
        if match_annotation.empty:
            TP = False
            Detected = True
            FP = True
            FN = False
            gold_term = set(match.term)
            gold_annotation = set(match.norm_annotation)
            gold_span = set(match.span)
        else:
            TP = True
            Detected = True
            FP = False
            FN = False
            gold_term = set(match_annotation.term)
            gold_annotation = set(match_annotation.norm_annotation)
            gold_span = set(match_annotation.span)
    res_df["source"].append(source)
    res_df["pred_term"].append(pred_term)
    res_df["gold_term"].append(gold_term)
    res_df["gold_annotation"].append(gold_annotation)
    res_df["pred_value"].append(pred_value)
    res_df["pred_span"].append(span)
    res_df["gold_span"].append(gold_span)
    res_df["Detected"].append(Detected)
    res_df["TP"].append(TP)
    res_df["FP"].append(FP)
    res_df["FN"].append(FN)

for row in gold_bio_val.itertuples():
    source = row.source
    span = row.span_converted
    gold_term = row.term
    gold_annotation = row.norm_annotation
    match = pred_bio[
        (pred_bio.source == source)
        & (
            (
                (pred_bio.span_converted.str.get(0) <= span[0])
                & (pred_bio.span_converted.str.get(1) >= span[0])
            )
            | (
                (pred_bio.span_converted.str.get(0) <= span[1])
                & (pred_bio.span_converted.str.get(1) >= span[1])
            )
            | (
                (pred_bio.span_converted.str.get(0) >= span[0])
                & (pred_bio.span_converted.str.get(1) <= span[1])
            )
        )
    ]
    if match.empty:
        TP = False
        Detected = False
        FP = False
        FN = True
        pred_term = None
        pred_value = None
        pred_span = None
        res_df["source"].append(source)
        res_df["pred_term"].append(pred_term)
        res_df["gold_term"].append(gold_term)
        res_df["gold_annotation"].append(gold_annotation)
        res_df["pred_value"].append(pred_value)
        res_df["pred_span"].append(pred_span)
        res_df["gold_span"].append(span)
        res_df["Detected"].append(Detected)
        res_df["TP"].append(TP)
        res_df["FP"].append(FP)
        res_df["FN"].append(FN)

res_bio_val = pd.DataFrame(res_df)
```

```python
import numpy as np
from tqdm import tqdm

n_draw = 5000
alpha = 0.05
binary_results = {
    "TP": [0],
    "FP": [0],
    "FN": [0],
    "Precision": [],
    "Recall": [],
    "F1": [],
    "N_entity": [],
}

results_per_doc = {}
docs = list(res_bio_val.source.unique())
for doc in docs:
    res_doc = {}

    res_doc["TP"] = res_bio_val[res_bio_val.source == doc]["TP"].sum()
    res_doc["FP"] = res_bio_val[res_bio_val.source == doc]["FP"].sum()
    res_doc["FN"] = res_bio_val[res_bio_val.source == doc]["FN"].sum()
    binary_results["TP"][0] += res_doc["TP"]
    binary_results["FP"][0] += res_doc["FP"]
    binary_results["FN"][0] += res_doc["FN"]
    results_per_doc[doc] = res_doc

precision_init = binary_results["TP"][0] / (
    binary_results["TP"][0] + binary_results["FP"][0]
)
recall_init = binary_results["TP"][0] / (
    binary_results["TP"][0] + binary_results["FN"][0]
)
n_entity_init = binary_results["TP"][0] + binary_results["FN"][0]
binary_results["Precision"].append(precision_init)
binary_results["Recall"].append(recall_init)
binary_results["F1"].append(
    2 * (precision_init * recall_init) / (precision_init + recall_init)
)
binary_results["N_entity"].append(n_entity_init)
for i in tqdm(range(1, n_draw)):
    draw = np.random.choice(
        docs,
        size=len(docs),
        replace=True,
    )
    binary_result = {"TP": 0, "FP": 0, "FN": 0}
    for doc in draw:
        binary_result["TP"] += results_per_doc[doc]["TP"]
        binary_result["FP"] += results_per_doc[doc]["FP"]
        binary_result["FN"] += results_per_doc[doc]["FN"]

    precision = binary_result["TP"] / (binary_result["TP"] + binary_result["FP"])
    recall = binary_result["TP"] / (binary_result["TP"] + binary_result["FN"])
    f1 = 2 * (precision * recall) / (precision + recall)
    n_entity = binary_result["TP"] + binary_result["FN"]

    binary_results["TP"].append(binary_result["TP"])
    binary_results["FP"].append(binary_result["FP"])
    binary_results["FN"].append(binary_result["FN"])
    binary_results["Precision"].append(precision)
    binary_results["Recall"].append(recall)
    binary_results["F1"].append(f1)
    binary_results["N_entity"].append(n_entity)

n_entities = binary_results["N_entity"][0]
precision = round(binary_results["Precision"][0], 2)
recall = round(binary_results["Recall"][0], 2)
f1 = round(binary_results["F1"][0], 2)
n_entities_lower_bound = int(np.quantile(binary_results["N_entity"], (alpha / 2)))
n_entities_upper_bound = int(np.quantile(binary_results["N_entity"], (1 - alpha / 2)))
precision_lower_bound = round(np.quantile(binary_results["Precision"], (alpha / 2)), 2)
precision_upper_bound = round(
    np.quantile(binary_results["Precision"], (1 - alpha / 2)), 2
)
recall_lower_bound = round(np.quantile(binary_results["Recall"], (alpha / 2)), 2)
recall_upper_bound = round(np.quantile(binary_results["Recall"], (1 - alpha / 2)), 2)
f1_lower_bound = round(np.quantile(binary_results["F1"], (alpha / 2)), 2)
f1_upper_bound = round(np.quantile(binary_results["F1"], (1 - alpha / 2)), 2)
precision_result = "{} [{}, {}]".format(
    precision, precision_lower_bound, precision_upper_bound
)
recall_result = "{} [{}, {}]".format(recall, recall_lower_bound, recall_upper_bound)
f1_result = "{} [{}, {}]".format(f1, f1_lower_bound, f1_upper_bound)
n_entities_result = "{} [{}, {}]".format(
    n_entities, n_entities_lower_bound, n_entities_upper_bound
)
```

```python
pd.DataFrame(
    {
        "Label": ["Measurement"],
        "Dataset": ["Discharge summaries"],
        "Numer of entities": [n_entities_result],
        "Precision": [precision_result],
        "Recall": [recall_result],
        "F1-score": [f1_result],
    }
).set_index(["Label", "Dataset"])
```

## NER + Extract Biology VALUE

```python
import pandas as pd


pred_bio = pd.read_pickle(PRED_BIO_VAL_NER)
pred_bio = pred_bio[~pred_bio.term_bio.isna()]
pred_bio["term"] = pred_bio["term_bio"]


def convert_spans(row):
    return [int(row.span_start_bio), int(row.span_end_bio)]


pred_bio["span_converted"] = pred_bio.apply(convert_spans, axis=1)
pred_bio["found"] = pred_bio["value_cleaned"] + " " + pred_bio["unit"]
all_measures = []
for row in pred_bio.found:
    res = nlp(str(row))
    if len(res.spans["measurements"]) == 0:
        measures = None
    else:
        measures = "{} {}".format(
            float(res.spans["measurements"][0]._.value.value),
            res.spans["measurements"][0]._.value.unit,
        )
    all_measures.append(measures)
pred_bio = pred_bio.reset_index(drop=True).assign(found=pd.Series(all_measures))
pred_bio["neg"] = pred_bio.non_digit_value.str.match(negative_regex_from_label_key)
pred_bio["pos"] = pred_bio.non_digit_value.str.match(positive_regex_from_label_key)
pred_bio["norm"] = pred_bio.non_digit_value.isin(normal_terms_from_label_key)
pred_bio["found"] = pred_bio["found"].mask(
    pred_bio["found"].isna() & pred_bio["neg"], "0 bool"
)
pred_bio["found"] = pred_bio["found"].mask(
    pred_bio["found"].isna() & pred_bio["pos"], "1 bool"
)
pred_bio["found"] = pred_bio["found"].mask(
    pred_bio["found"].isna() & pred_bio["norm"], "0.5 bool"
)
pred_bio = pred_bio[~pred_bio.found.isna()]
pred_bio
```

```python
res_df = {
    "source": [],
    "pred_term": [],
    "gold_term": [],
    "gold_annotation": [],
    "pred_value": [],
    "gold_value": [],
    "pred_span": [],
    "gold_span": [],
    "TP": [],
    "Detected": [],
    "FP": [],
    "FN": [],
}
for row in pred_bio.itertuples():
    source = row.source
    span = row.span_converted
    pred_term = row.term
    pred_value = row.found
    match = gold_bio_val[
        (gold_bio_val.source == source)
        & (
            (
                (gold_bio_val.span_converted.str.get(0) <= span[0])
                & (gold_bio_val.span_converted.str.get(1) >= span[0])
            )
            | (
                (gold_bio_val.span_converted.str.get(0) <= span[1])
                & (gold_bio_val.span_converted.str.get(1) >= span[1])
            )
            | (
                (gold_bio_val.span_converted.str.get(0) >= span[0])
                & (gold_bio_val.span_converted.str.get(1) <= span[1])
            )
        )
    ]
    if match.empty:
        TP = False
        FP = True
        Detected = False
        FN = False
        gold_term = None
        gold_annotation = None
        gold_value = None
        gold_span = None
        continue
    else:
        match_annotation = match[match.norm_annotation == pred_value]
        if match_annotation.empty:
            TP = False
            Detected = True
            FP = True
            FN = False
            gold_term = set(match.term)
            gold_annotation = set(match.annotation)
            gold_value = set(match.norm_annotation)
            gold_span = set(match.span)
        else:
            TP = True
            Detected = True
            FP = False
            FN = False
            gold_term = set(match_annotation.term)
            gold_annotation = set(match_annotation.annotation)
            gold_value = set(match_annotation.norm_annotation)
            gold_span = set(match_annotation.span)
    res_df["source"].append(source)
    res_df["pred_term"].append(pred_term)
    res_df["gold_term"].append(gold_term)
    res_df["gold_annotation"].append(gold_annotation)
    res_df["pred_value"].append(pred_value)
    res_df["gold_value"].append(gold_value)
    res_df["pred_span"].append(span)
    res_df["gold_span"].append(gold_span)
    res_df["Detected"].append(Detected)
    res_df["TP"].append(TP)
    res_df["FP"].append(FP)
    res_df["FN"].append(FN)

for row in gold_bio_val.itertuples():
    source = row.source
    span = row.span_converted
    gold_term = row.term
    gold_annotation = row.annotation
    gold_value = row.norm_annotation
    match = pred_bio[
        (pred_bio.source == source)
        & (
            (
                (pred_bio.span_converted.str.get(0) <= span[0])
                & (pred_bio.span_converted.str.get(1) >= span[0])
            )
            | (
                (pred_bio.span_converted.str.get(0) <= span[1])
                & (pred_bio.span_converted.str.get(1) >= span[1])
            )
            | (
                (pred_bio.span_converted.str.get(0) >= span[0])
                & (pred_bio.span_converted.str.get(1) <= span[1])
            )
        )
    ]
    if match.empty:
        TP = False
        Detected = False
        FP = False
        FN = True
        pred_term = None
        pred_value = None
        pred_span = None
        res_df["source"].append(source)
        res_df["pred_term"].append(pred_term)
        res_df["gold_term"].append(gold_term)
        res_df["gold_annotation"].append(gold_annotation)
        res_df["pred_value"].append(pred_value)
        res_df["gold_value"].append(gold_value)
        res_df["pred_span"].append(pred_span)
        res_df["gold_span"].append(span)
        res_df["Detected"].append(Detected)
        res_df["TP"].append(TP)
        res_df["FP"].append(FP)
        res_df["FN"].append(FN)

res_bio_val = pd.DataFrame(res_df)
```

```python
import numpy as np

n_draw = 5000
alpha = 0.05
binary_results = {
    "TP": [0],
    "FP": [0],
    "FN": [0],
    "Precision": [],
    "Recall": [],
    "F1": [],
    "N_entity": [],
}

results_per_doc = {}
docs = list(res_bio_val.source.unique())
for doc in docs:
    res_doc = {}

    res_doc["TP"] = res_bio_val[res_bio_val.source == doc]["TP"].sum()
    res_doc["FP"] = res_bio_val[res_bio_val.source == doc]["FP"].sum()
    res_doc["FN"] = res_bio_val[res_bio_val.source == doc]["FN"].sum()
    binary_results["TP"][0] += res_doc["TP"]
    binary_results["FP"][0] += res_doc["FP"]
    binary_results["FN"][0] += res_doc["FN"]
    results_per_doc[doc] = res_doc

precision_init = binary_results["TP"][0] / (
    binary_results["TP"][0] + binary_results["FP"][0]
)
recall_init = binary_results["TP"][0] / (
    binary_results["TP"][0] + binary_results["FN"][0]
)
n_entity_init = binary_results["TP"][0] + binary_results["FN"][0]
binary_results["Precision"].append(precision_init)
binary_results["Recall"].append(recall_init)
binary_results["F1"].append(
    2 * (precision_init * recall_init) / (precision_init + recall_init)
)
binary_results["N_entity"].append(n_entity_init)
for i in tqdm(range(1, n_draw)):
    draw = np.random.choice(
        docs,
        size=len(docs),
        replace=True,
    )
    binary_result = {"TP": 0, "FP": 0, "FN": 0}
    for doc in draw:
        binary_result["TP"] += results_per_doc[doc]["TP"]
        binary_result["FP"] += results_per_doc[doc]["FP"]
        binary_result["FN"] += results_per_doc[doc]["FN"]

    precision = binary_result["TP"] / (binary_result["TP"] + binary_result["FP"])
    recall = binary_result["TP"] / (binary_result["TP"] + binary_result["FN"])
    f1 = 2 * (precision * recall) / (precision + recall)
    n_entity = binary_result["TP"] + binary_result["FN"]

    binary_results["TP"].append(binary_result["TP"])
    binary_results["FP"].append(binary_result["FP"])
    binary_results["FN"].append(binary_result["FN"])
    binary_results["Precision"].append(precision)
    binary_results["Recall"].append(recall)
    binary_results["F1"].append(f1)
    binary_results["N_entity"].append(n_entity)

n_entities = binary_results["N_entity"][0]
precision = round(binary_results["Precision"][0], 2)
recall = round(binary_results["Recall"][0], 2)
f1 = round(binary_results["F1"][0], 2)
n_entities_lower_bound = int(np.quantile(binary_results["N_entity"], (alpha / 2)))
n_entities_upper_bound = int(np.quantile(binary_results["N_entity"], (1 - alpha / 2)))
precision_lower_bound = round(np.quantile(binary_results["Precision"], (alpha / 2)), 2)
precision_upper_bound = round(
    np.quantile(binary_results["Precision"], (1 - alpha / 2)), 2
)
recall_lower_bound = round(np.quantile(binary_results["Recall"], (alpha / 2)), 2)
recall_upper_bound = round(np.quantile(binary_results["Recall"], (1 - alpha / 2)), 2)
f1_lower_bound = round(np.quantile(binary_results["F1"], (alpha / 2)), 2)
f1_upper_bound = round(np.quantile(binary_results["F1"], (1 - alpha / 2)), 2)
precision_result = "{} [{}, {}]".format(
    precision, precision_lower_bound, precision_upper_bound
)
recall_result = "{} [{}, {}]".format(recall, recall_lower_bound, recall_upper_bound)
f1_result = "{} [{}, {}]".format(f1, f1_lower_bound, f1_upper_bound)
n_entities_result = "{} [{}, {}]".format(
    n_entities, n_entities_lower_bound, n_entities_upper_bound
)
```

```python
pd.DataFrame(
    {
        "Label": ["Measurement"],
        "Dataset": ["Discharge summaries"],
        "Numer of entities": [n_entities_result],
        "Precision": [precision_result],
        "Recall": [recall_result],
        "F1-score": [f1_result],
    }
).set_index(["Label", "Dataset"])
```

## Biology NER + Norm

```python
threshold = 0

pred_bio_coder = pd.read_pickle(PRED_BIO_CODER)
pred_bio_coder = pred_bio_coder[pred_bio_coder.score >= threshold]
pred_bio_coder
```

```python
gold_df = gold_aphp[gold_aphp.label == "BIO"]
gold_df
```

```python
res_df = {
    "source": [],
    "pred_term": [],
    "gold_term": [],
    "pred_cui": [],
    "gold_cui": [],
    "pred_name_cui": [],
    "gold_name_cui": [],
    "pred_span": [],
    "gold_span": [],
    "TP": [],
    "Detected": [],
    "FP": [],
    "FN": [],
    "score": [],
}
pred_df = pred_bio_coder
for row in pred_df.itertuples():
    source = row.source
    span = row.span_converted
    pred_term = row.term
    pred_cui = row.label
    pred_name_cui = row.norm_term
    score = row.score
    match = gold_df[
        (gold_df.source == source)
        & (gold_df.span_converted.str.get(0) == span[0])
        & (gold_df.span_converted.str.get(1) == span[1])
    ]
    if match.empty:
        TP = False
        FP = True
        Detected = False
        FN = False
        gold_term = None
        gold_cui = None
        gold_span = None
        gold_name_cui = None
    else:
        Detected = True
        match_annotation = match[
            match.annotation.apply(lambda x: len(x.intersection(set(pred_cui))) > 0)
        ]
        if match_annotation.empty:
            TP = False
            FP = True
            FN = True
            gold_term = set(match.term)
            gold_cui = reduce(lambda x, y: x | y, match.annotation)
            gold_span = match.span_converted.iloc[0]
            gold_name_cui = reduce(lambda x, y: x | y, match.STR)
        else:
            TP = True
            FP = False
            FN = False
            gold_term = set(match_annotation.term)
            gold_cui = reduce(lambda x, y: x | y, match_annotation.annotation)
            gold_span = match_annotation.span_converted.iloc[0]
            gold_name_cui = reduce(lambda x, y: x | y, match_annotation.STR)
    res_df["source"].append(source)
    res_df["pred_term"].append(pred_term)
    res_df["gold_term"].append(gold_term)
    res_df["pred_cui"].append(pred_cui)
    res_df["gold_cui"].append(gold_cui)
    res_df["pred_name_cui"].append(pred_name_cui)
    res_df["gold_name_cui"].append(gold_name_cui)
    res_df["pred_span"].append(span)
    res_df["gold_span"].append(gold_span)
    res_df["Detected"].append(Detected)
    res_df["TP"].append(TP)
    res_df["FP"].append(FP)
    res_df["FN"].append(FN)
    res_df["score"].append(score)

for row in gold_df.itertuples():
    source = row.source
    span = row.span_converted
    gold_term = row.term
    gold_cui = row.annotation
    gold_name_cui = row.STR
    match = pred_df[
        (pred_df.source == source)
        & (pred_df.span_converted.str.get(0) == span[0])
        & (pred_df.span_converted.str.get(1) == span[1])
    ]
    if match.empty:
        TP = False
        Detected = False
        FP = False
        FN = True
        pred_term = None
        pred_cui = None
        pred_span = None
        pred_name_cui = None
        score = None
        res_df["source"].append(source)
        res_df["pred_term"].append(pred_term)
        res_df["gold_term"].append(gold_term)
        res_df["pred_cui"].append(pred_cui)
        res_df["gold_cui"].append(gold_cui)
        res_df["pred_name_cui"].append(pred_name_cui)
        res_df["gold_name_cui"].append(gold_name_cui)
        res_df["pred_span"].append(pred_span)
        res_df["gold_span"].append(span)
        res_df["Detected"].append(Detected)
        res_df["TP"].append(TP)
        res_df["FP"].append(FP)
        res_df["FN"].append(FN)
        res_df["score"].append(score)

res_bio = pd.DataFrame(res_df)
```

```python
import numpy as np
from tqdm import tqdm

n_draw = 5000
alpha = 0.05
binary_results = {
    "TP": [0],
    "FP": [0],
    "FN": [0],
    "N_entity": [0],
    "Precision": [],
    "Recall": [],
    "F1": [],
}

detected_init = 0
results_per_doc = {}
docs = list(res_bio.source.unique())
for doc in docs:
    res_doc = {}

    res_doc["N_entity"] = len(gold_df[gold_df.source == doc])
    res_doc["TP"] = res_bio[res_bio.source == doc]["TP"].sum()
    res_doc["FP"] = res_bio[res_bio.source == doc]["FP"].sum()
    res_doc["FN"] = res_bio[res_bio.source == doc]["FN"].sum()
    binary_results["N_entity"][0] += res_doc["N_entity"]
    binary_results["TP"][0] += res_doc["TP"]
    binary_results["FP"][0] += res_doc["FP"]
    binary_results["FN"][0] += res_doc["FN"]
    results_per_doc[doc] = res_doc

precision_init = binary_results["TP"][0] / (
    binary_results["TP"][0] + binary_results["FP"][0]
)
recall_init = binary_results["TP"][0] / (
    binary_results["TP"][0] + binary_results["FN"][0]
)
binary_results["Precision"].append(precision_init)
binary_results["Recall"].append(recall_init)
binary_results["F1"].append(
    2 * (precision_init * recall_init) / (precision_init + recall_init)
)
for i in tqdm(range(1, n_draw)):
    draw = np.random.choice(
        docs,
        size=len(docs),
        replace=True,
    )
    binary_result = {"TP": 0, "FP": 0, "FN": 0, "N_entity": 0}
    for doc in draw:
        binary_result["N_entity"] += results_per_doc[doc]["N_entity"]
        binary_result["TP"] += results_per_doc[doc]["TP"]
        binary_result["FP"] += results_per_doc[doc]["FP"]
        binary_result["FN"] += results_per_doc[doc]["FN"]

    precision = binary_result["TP"] / (binary_result["TP"] + binary_result["FP"])
    recall = binary_result["TP"] / (binary_result["TP"] + binary_result["FN"])
    f1 = 2 * (precision * recall) / (precision + recall)

    binary_results["TP"].append(binary_result["TP"])
    binary_results["FP"].append(binary_result["FP"])
    binary_results["FN"].append(binary_result["FN"])
    binary_results["N_entity"].append(binary_result["N_entity"])
    binary_results["Precision"].append(precision)
    binary_results["Recall"].append(recall)
    binary_results["F1"].append(f1)

n_entities = binary_results["N_entity"][0]
precision = round(binary_results["Precision"][0], 2)
recall = round(binary_results["Recall"][0], 2)
f1 = round(binary_results["F1"][0], 2)
n_entities_lower_bound = int(np.quantile(binary_results["N_entity"], (alpha / 2)))
n_entities_upper_bound = int(np.quantile(binary_results["N_entity"], (1 - alpha / 2)))
precision_lower_bound = round(np.quantile(binary_results["Precision"], (alpha / 2)), 2)
precision_upper_bound = round(
    np.quantile(binary_results["Precision"], (1 - alpha / 2)), 2
)
recall_lower_bound = round(np.quantile(binary_results["Recall"], (alpha / 2)), 2)
recall_upper_bound = round(np.quantile(binary_results["Recall"], (1 - alpha / 2)), 2)
f1_lower_bound = round(np.quantile(binary_results["F1"], (alpha / 2)), 2)
f1_upper_bound = round(np.quantile(binary_results["F1"], (1 - alpha / 2)), 2)
precision_result = "{} [{}-{}]".format(
    precision, precision_lower_bound, precision_upper_bound
)
recall_result = "{} [{}-{}]".format(recall, recall_lower_bound, recall_upper_bound)
f1_result = "{} [{}-{}]".format(f1, f1_lower_bound, f1_upper_bound)
n_entities_result = "{} [{}-{}]".format(
    n_entities, n_entities_lower_bound, n_entities_upper_bound
)
```

```python
bio_table = pd.DataFrame(
    {
        "Label": ["Biological test name"],
        "Dataset": ["Discharge summaries"],
        "Numer of entities": [n_entities_result],
        "Precision": [precision_result],
        "Recall": [recall_result],
        "F1-score": [f1_result],
    }
).set_index(["Label", "Dataset"])
bio_table
```

## MED NER + Norm

```python
threshold = 0.9

pred_med_fuzzy = pd.read_pickle(PRED_MED_FUZZY)
pred_med_fuzzy = pred_med_fuzzy[pred_med_fuzzy.score >= threshold]
pred_med_fuzzy
```

```python
gold_df = gold_aphp[gold_aphp.label == "Chemical_and_drugs"]
gold_df
```

```python
def detail_all_atc(atc_list):
    all_atc = []
    for atc in atc_list:
        if isinstance(atc, str):
            atc_short = ""
            for char in atc:
                atc_short += char
                all_atc.append(atc_short)
    return set(all_atc)


res_df = {
    "source": [],
    "pred_term": [],
    "gold_term": [],
    "pred_cui": [],
    "gold_cui": [],
    "pred_name_cui": [],
    "gold_name_cui": [],
    "pred_span": [],
    "gold_span": [],
    "TP": [],
    "Detected": [],
    "FP": [],
    "FN": [],
    "score": [],
}
pred_df = pred_med_fuzzy
for row in pred_df.itertuples():
    source = row.source
    span = row.span_converted
    pred_term = row.term
    pred_cui = row.label
    pred_name_cui = row.norm_term
    score = row.score
    match = gold_df[
        (gold_df.source == source)
        & (gold_df.span_converted.str.get(0) == span[0])
        & (gold_df.span_converted.str.get(1) == span[1])
    ]
    if match.empty:
        TP = False
        FP = True
        Detected = False
        FN = False
        gold_term = None
        gold_cui = None
        gold_span = None
        gold_name_cui = None
    else:
        Detected = True
        match_annotation = match[
            match.annotation.apply(
                lambda x: len(x.intersection(detail_all_atc(pred_cui))) > 0
            )
        ]
        if match_annotation.empty:
            TP = False
            FP = True
            FN = True
            gold_term = set(match.term)
            gold_cui = reduce(lambda x, y: x | y, match.annotation)
            gold_span = match.span_converted.iloc[0]
            gold_name_cui = reduce(lambda x, y: x | y, match.STR)
        else:
            TP = True
            FP = False
            FN = False
            gold_term = set(match_annotation.term)
            gold_cui = reduce(lambda x, y: x | y, match_annotation.annotation)
            gold_span = match_annotation.span_converted.iloc[0]
            gold_name_cui = reduce(lambda x, y: x | y, match_annotation.STR)
    res_df["source"].append(source)
    res_df["pred_term"].append(pred_term)
    res_df["gold_term"].append(gold_term)
    res_df["pred_cui"].append(pred_cui)
    res_df["gold_cui"].append(gold_cui)
    res_df["pred_name_cui"].append(pred_name_cui)
    res_df["gold_name_cui"].append(gold_name_cui)
    res_df["pred_span"].append(span)
    res_df["gold_span"].append(gold_span)
    res_df["Detected"].append(Detected)
    res_df["TP"].append(TP)
    res_df["FP"].append(FP)
    res_df["FN"].append(FN)
    res_df["score"].append(score)

for row in gold_df.itertuples():
    source = row.source
    span = row.span_converted
    gold_term = row.term
    gold_cui = row.annotation
    gold_name_cui = row.STR
    match = pred_df[
        (pred_df.source == source)
        & (pred_df.span_converted.str.get(0) == span[0])
        & (pred_df.span_converted.str.get(1) == span[1])
    ]
    if match.empty:
        TP = False
        Detected = False
        FP = False
        FN = True
        pred_term = None
        pred_cui = None
        pred_span = None
        pred_name_cui = None
        score = None
        res_df["source"].append(source)
        res_df["pred_term"].append(pred_term)
        res_df["gold_term"].append(gold_term)
        res_df["pred_cui"].append(pred_cui)
        res_df["gold_cui"].append(gold_cui)
        res_df["pred_name_cui"].append(pred_name_cui)
        res_df["gold_name_cui"].append(gold_name_cui)
        res_df["pred_span"].append(pred_span)
        res_df["gold_span"].append(span)
        res_df["Detected"].append(Detected)
        res_df["TP"].append(TP)
        res_df["FP"].append(FP)
        res_df["FN"].append(FN)
        res_df["score"].append(score)

res_med = pd.DataFrame(res_df)
```

```python
import numpy as np
from tqdm import tqdm

n_draw = 5000
alpha = 0.05
binary_results = {
    "TP": [0],
    "FP": [0],
    "FN": [0],
    "N_entity": [0],
    "Precision": [],
    "Recall": [],
    "F1": [],
}

detected_init = 0
results_per_doc = {}
docs = list(res_med.source.unique())
for doc in docs:
    res_doc = {}

    res_doc["N_entity"] = len(gold_df[gold_df.source == doc])
    res_doc["TP"] = res_med[res_med.source == doc]["TP"].sum()
    res_doc["FP"] = res_med[res_med.source == doc]["FP"].sum()
    res_doc["FN"] = res_med[res_med.source == doc]["FN"].sum()
    binary_results["N_entity"][0] += res_doc["N_entity"]
    binary_results["TP"][0] += res_doc["TP"]
    binary_results["FP"][0] += res_doc["FP"]
    binary_results["FN"][0] += res_doc["FN"]
    results_per_doc[doc] = res_doc

precision_init = binary_results["TP"][0] / (
    binary_results["TP"][0] + binary_results["FP"][0]
)
recall_init = binary_results["TP"][0] / (
    binary_results["TP"][0] + binary_results["FN"][0]
)
binary_results["Precision"].append(precision_init)
binary_results["Recall"].append(recall_init)
binary_results["F1"].append(
    2 * (precision_init * recall_init) / (precision_init + recall_init)
)
for i in tqdm(range(1, n_draw)):
    draw = np.random.choice(
        docs,
        size=len(docs),
        replace=True,
    )
    binary_result = {"TP": 0, "FP": 0, "FN": 0, "N_entity": 0}
    for doc in draw:
        binary_result["N_entity"] += results_per_doc[doc]["N_entity"]
        binary_result["TP"] += results_per_doc[doc]["TP"]
        binary_result["FP"] += results_per_doc[doc]["FP"]
        binary_result["FN"] += results_per_doc[doc]["FN"]

    precision = binary_result["TP"] / (binary_result["TP"] + binary_result["FP"])
    recall = binary_result["TP"] / (binary_result["TP"] + binary_result["FN"])
    f1 = 2 * (precision * recall) / (precision + recall)

    binary_results["TP"].append(binary_result["TP"])
    binary_results["FP"].append(binary_result["FP"])
    binary_results["FN"].append(binary_result["FN"])
    binary_results["N_entity"].append(binary_result["N_entity"])
    binary_results["Precision"].append(precision)
    binary_results["Recall"].append(recall)
    binary_results["F1"].append(f1)

n_entities = binary_results["N_entity"][0]
precision = round(binary_results["Precision"][0], 2)
recall = round(binary_results["Recall"][0], 2)
f1 = round(binary_results["F1"][0], 2)
n_entities_lower_bound = int(np.quantile(binary_results["N_entity"], (alpha / 2)))
n_entities_upper_bound = int(np.quantile(binary_results["N_entity"], (1 - alpha / 2)))
precision_lower_bound = round(np.quantile(binary_results["Precision"], (alpha / 2)), 2)
precision_upper_bound = round(
    np.quantile(binary_results["Precision"], (1 - alpha / 2)), 2
)
recall_lower_bound = round(np.quantile(binary_results["Recall"], (alpha / 2)), 2)
recall_upper_bound = round(np.quantile(binary_results["Recall"], (1 - alpha / 2)), 2)
f1_lower_bound = round(np.quantile(binary_results["F1"], (alpha / 2)), 2)
f1_upper_bound = round(np.quantile(binary_results["F1"], (1 - alpha / 2)), 2)
precision_result = "{} [{}-{}]".format(
    precision, precision_lower_bound, precision_upper_bound
)
recall_result = "{} [{}-{}]".format(recall, recall_lower_bound, recall_upper_bound)
f1_result = "{} [{}-{}]".format(f1, f1_lower_bound, f1_upper_bound)
n_entities_result = "{} [{}-{}]".format(
    n_entities, n_entities_lower_bound, n_entities_upper_bound
)
```

```python
med_table = pd.DataFrame(
    {
        "Label": ["Drug name"],
        "Dataset": ["Discharge summaries"],
        "Numer of entities": [n_entities_result],
        "Precision": [precision_result],
        "Recall": [recall_result],
        "F1-score": [f1_result],
    }
).set_index(["Label", "Dataset"])
med_table
```

## NER + NORM Final result

```python
pd.concat([bio_table, med_table])
```

```python

```
