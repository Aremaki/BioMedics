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
from ast import literal_eval
from functools import reduce
import sys
from biomedics.utils.extract_pandas_from_brat import extract_pandas
# from biomedics.normalization.med_config import *
from biomedics.normalization.coder_inference.text_preprocessor import TextPreprocessor

REGEX_CONVERT_SPANS = re.compile("^(\d+).*\s(\d+)$")

# Convert span to list with span_start, span_end. It considers the new lines by adding one character.
def convert_spans(span):
    span_match = REGEX_CONVERT_SPANS.match(span)
    span_start = int(span_match.group(1))
    span_end = int(span_match.group(2))
    return [span_start, span_end]


APHP_DIR = "../data/annotated_CRH/cui_annotations"
SAVE_APHP_BIO_DIR = "../data/annotated_CRH/cui_annotations/aphp_bio.json"
SAVE_APHP_MED_DIR = "../data/annotated_CRH/cui_annotations/aphp_med.json"

MEDLINE_DIR = "../data/QUAERO/corpus/all/MEDLINE"
EMEA_DIR = "../data/QUAERO/corpus/all/EMEA"
SAVE_MEDLINE_BIO_DIR = "../data/QUAERO/corpus/all/MEDLINE/medline_bio.json"
SAVE_EMEA_BIO_DIR = "../data/QUAERO/corpus/all/EMEA/emea_bio.json"
SAVE_MEDLINE_MED_DIR = "../data/QUAERO/corpus/all/MEDLINE/medline_med.json"
SAVE_EMEA_MED_DIR = "../data/QUAERO/corpus/all/EMEA/emea_med.json"
UMLS_BIO_DIR = "../data/umls/lab_snomed_ct_2021AB.pkl"
UMLS_ATC_DIR = "../data/umls/atc_cui_2023AB.pkl"

RES_DIR_BIO_MEDLINE_CODER_ALL = (
    "../data/QUAERO/corpus/pred/MEDLINE/norm_bio/norm_coder_all.json"
)
RES_DIR_BIO_MEDLINE_CODER_ENG_PP = (
    "../data/QUAERO/corpus/pred/MEDLINE/norm_bio/norm_coder_eng_pp.json"
)
RES_DIR_BIO_MEDLINE_SAPBERT_ALL = (
    "../data/QUAERO/corpus/pred/MEDLINE/norm_bio/norm_sapbert_all.json"
)
RES_DIR_BIO_MEDLINE_SAPBERT_EDS = (
    "../data/QUAERO/corpus/pred/MEDLINE/norm_bio/norm_sapbert_eds.json"
)
RES_DIR_BIO_MEDLINE_FUZZY_LEV = (
    "../data/QUAERO/corpus/pred/MEDLINE/norm_bio/norm_fuzzy_lev.json"
)
RES_DIR_BIO_MEDLINE_FUZZY_JW = (
    "../data/QUAERO/corpus/pred/MEDLINE/norm_bio/norm_fuzzy_jaro_winkler.json"
)
RES_DIR_BIO_MEDLINE_FUZZY_CODER = (
    "../data/QUAERO/corpus/pred/MEDLINE/norm_bio/norm_fuzzy_coder.json"
)
RES_DIR_BIO_EMEA_CODER_ALL = (
    "../data/QUAERO/corpus/pred/EMEA/norm_bio/norm_coder_all.json"
)
RES_DIR_BIO_EMEA_CODER_ENG_PP = (
    "../data/QUAERO/corpus/pred/EMEA/norm_bio/norm_coder_eng_pp.json"
)
RES_DIR_BIO_EMEA_SAPBERT_ALL = (
    "../data/QUAERO/corpus/pred/EMEA/norm_bio/norm_sapbert_all.json"
)
RES_DIR_BIO_EMEA_SAPBERT_EDS = (
    "../data/QUAERO/corpus/pred/EMEA/norm_bio/norm_sapbert_eds.json"
)
RES_DIR_BIO_EMEA_FUZZY_LEV = (
    "../data/QUAERO/corpus/pred/EMEA/norm_bio/norm_fuzzy_lev.json"
)
RES_DIR_BIO_EMEA_FUZZY_JW = (
    "../data/QUAERO/corpus/pred/EMEA/norm_bio/norm_fuzzy_jaro_winkler.json"
)
RES_DIR_BIO_EMEA_FUZZY_CODER = (
    "../data/QUAERO/corpus/pred/EMEA/norm_bio/norm_fuzzy_coder.json"
)
RES_DIR_BIO_APHP_CODER_ALL = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/pred_bio_aphp/norm_coder_all.json"
RES_DIR_BIO_APHP_CODER_ENG_PP = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/pred_bio_aphp/norm_coder_eng_pp.json"
RES_DIR_BIO_APHP_SAPBERT_ALL = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/pred_bio_aphp/norm_sapbert_all.json"
RES_DIR_BIO_APHP_SAPBERT_EDS = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/pred_bio_aphp/norm_sapbert_eds.json"
RES_DIR_BIO_APHP_CODER_EDS = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/pred_bio_aphp/norm_coder_eds.json"
RES_DIR_BIO_APHP_FUZZY_LEV = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/pred_bio_aphp/norm_fuzzy_lev.json"
RES_DIR_BIO_APHP_FUZZY_JW = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/pred_bio_aphp/norm_fuzzy_jaro_winkler.json"
RES_DIR_BIO_APHP_FUZZY_CODER = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/pred_bio_aphp/norm_fuzzy_coder.json"
RES_DIR_MED_MEDLINE_CODER_ALL = (
    "../data/QUAERO/corpus/pred/MEDLINE/norm_med/norm_coder_all.json"
)
RES_DIR_MED_MEDLINE_CODER_ENG_PP = (
    "../data/QUAERO/corpus/pred/MEDLINE/norm_med/norm_coder_eng_pp.json"
)
RES_DIR_MED_MEDLINE_SAPBERT_ALL = (
    "../data/QUAERO/corpus/pred/MEDLINE/norm_med/norm_sapbert_all.json"
)
RES_DIR_MED_MEDLINE_SAPBERT_EDS = (
    "../data/QUAERO/corpus/pred/MEDLINE/norm_med/norm_sapbert_eds.json"
)
RES_DIR_MED_MEDLINE_CODER_EDS = (
    "../data/QUAERO/corpus/pred/MEDLINE/norm_med/norm_coder_eds.json"
)
RES_DIR_MED_MEDLINE_FUZZY_LEV = (
    "../data/QUAERO/corpus/pred/MEDLINE/norm_med/norm_fuzzy_lev.json"
)
RES_DIR_MED_MEDLINE_FUZZY_JW = (
    "../data/QUAERO/corpus/pred/MEDLINE/norm_med/norm_fuzzy_jaro_winkler.json"
)
RES_DIR_MED_MEDLINE_FUZZY_CODER = (
    "../data/QUAERO/corpus/pred/MEDLINE/norm_med/norm_fuzzy_coder.json"
)
RES_DIR_MED_EMEA_CODER_ALL = (
    "../data/QUAERO/corpus/pred/EMEA/norm_med/norm_coder_all.json"
)
RES_DIR_MED_EMEA_CODER_ENG_PP = (
    "../data/QUAERO/corpus/pred/EMEA/norm_med/norm_coder_eng_pp.json"
)
RES_DIR_MED_EMEA_SAPBERT_ALL = (
    "../data/QUAERO/corpus/pred/EMEA/norm_med/norm_sapbert_all.json"
)
RES_DIR_MED_EMEA_SAPBERT_EDS = (
    "../data/QUAERO/corpus/pred/EMEA/norm_med/norm_sapbert_eds.json"
)
RES_DIR_MED_EMEA_CODER_EDS = (
    "../data/QUAERO/corpus/pred/EMEA/norm_med/norm_coder_eds.json"
)
RES_DIR_MED_EMEA_FUZZY_LEV = (
    "../data/QUAERO/corpus/pred/EMEA/norm_med/norm_fuzzy_lev.json"
)
RES_DIR_MED_EMEA_FUZZY_JW = (
    "../data/QUAERO/corpus/pred/EMEA/norm_med/norm_fuzzy_jaro_winkler.json"
)
RES_DIR_MED_EMEA_FUZZY_CODER = (
    "../data/QUAERO/corpus/pred/EMEA/norm_med/norm_fuzzy_coder.json"
)
RES_DIR_MED_APHP_CODER_ALL = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/pred_med_aphp/norm_coder_all.json"
RES_DIR_MED_APHP_CODER_ENG_PP = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/pred_med_aphp/norm_coder_eng_pp.json"
RES_DIR_MED_APHP_SAPBERT_ALL = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/pred_med_aphp/norm_sapbert_all.json"
RES_DIR_MED_APHP_SAPBERT_EDS = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/pred_med_aphp/norm_sapbert_eds.json"
RES_DIR_MED_APHP_CODER_EDS = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/pred_med_aphp/norm_coder_eds.json"
RES_DIR_MED_APHP_FUZZY_LEV = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/pred_med_aphp/norm_fuzzy_lev.json"
RES_DIR_MED_APHP_FUZZY_JW = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/pred_med_aphp/norm_fuzzy_jaro_winkler.json"
RES_DIR_MED_APHP_FUZZY_CODER = "/export/home/cse200093/scratch/BioMedics/Normalisation/data/pred_med_aphp/norm_fuzzy_coder.json"
```

## Prepare GOLD DF

```python
# Generate GOLD APHP STANDRAD

gold_aphp = extract_pandas(IN_BRAT_DIR=APHP_DIR)

annotations_df = gold_aphp.loc[gold_aphp["label"] == "AnnotatorNotes"]
annotations_df = annotations_df.rename(
    columns={"term": "annotation", "ann_id": "annotation_id", "span": "ann_id"}
)

gold_aphp = gold_aphp.loc[gold_aphp["label"] != "AnnotatorNotes"]
gold_aphp["span_converted"] = gold_aphp["span"].apply(convert_spans)
gold_aphp = pd.merge(
    gold_aphp[["ann_id", "term", "source", "span_converted", "label"]],
    annotations_df[["ann_id", "annotation", "source"]],
    how="inner",
    on=["source", "ann_id"],
).drop(columns="ann_id")
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
gold_aphp.span_converted = gold_aphp.span_converted.astype(str)
gold_aphp = gold_aphp.drop_duplicates()
gold_aphp.span_converted = gold_aphp.span_converted.apply(lambda x: eval(x))
gold_aphp.annotation = gold_aphp.annotation.str.split(" ; ")
gold_aphp.annotation = gold_aphp.annotation.apply(set)
gold_aphp
```

```python
gold_bio = gold_aphp[gold_aphp.label == "BIO"].drop(columns="label")
gold_bio.to_json(SAVE_APHP_BIO_DIR)
gold_bio
```

```python
gold_med = gold_aphp[gold_aphp.label == "Chemical_and_drugs"].drop(columns="label")
gold_med.to_json(SAVE_APHP_MED_DIR)
gold_med
```

```python
# SKIP THIS SECTION IF FILTERED QUAERO DATASET IS ALREADY BUILT

# Extract all BIOs from MEDLINE

# First list all CUIs in UMLS which are Laboratory Procedure
umls_cuis = pd.read_pickle(UMLS_BIO_DIR)["CUI"].unique().tolist()

REGEX_CONVERT_SPANS = re.compile("^(\d+).*\s(\d+)$")


# Convert span to list with span_start, span_end. It considers the new lines by adding one character.
def convert_spans(span):
    span_match = REGEX_CONVERT_SPANS.match(span)
    span_start = int(span_match.group(1))
    span_end = int(span_match.group(2))
    return [span_start, span_end]


train_df = extract_pandas(IN_BRAT_DIR=MEDLINE_DIR)

annotations_df = train_df.loc[train_df["label"] == "AnnotatorNotes"]
annotations_df = annotations_df.rename(
    columns={"term": "annotation", "ann_id": "annotation_id", "span": "ann_id"}
)

train_df = train_df.loc[train_df["label"] != "AnnotatorNotes"]
train_df["span_converted"] = train_df["span"].apply(convert_spans)
res_df = pd.merge(
    train_df[["ann_id", "term", "source", "span_converted"]],
    annotations_df[["ann_id", "annotation", "source"]],
    how="inner",
    on=["source", "ann_id"],
).drop(columns="ann_id")
res_df = res_df.loc[res_df["annotation"].isin(umls_cuis)]

# Rq: we have already checked that one term is always linked to the same annotation, hence "annotation":"first"
res_df.to_json(SAVE_MEDLINE_BIO_DIR)
res_df
```

```python
# SKIP THIS SECTION IF FILTERED QUAERO DATASET IS ALREADY BUILT

# Extract all BIOs from EMEA

# First list all CUIs in UMLS which are Laboratory Procedure
umls_cuis = pd.read_pickle(UMLS_BIO_DIR)["CUI"].unique().tolist()

REGEX_CONVERT_SPANS = re.compile("^(\d+).*\s(\d+)$")


# Convert span to list with span_start, span_end. It considers the new lines by adding one character.
def convert_spans(span):
    span_match = REGEX_CONVERT_SPANS.match(span)
    span_start = int(span_match.group(1))
    span_end = int(span_match.group(2))
    return [span_start, span_end]


train_df = extract_pandas(IN_BRAT_DIR=EMEA_DIR)

annotations_df = train_df.loc[train_df["label"] == "AnnotatorNotes"]
annotations_df = annotations_df.rename(
    columns={"term": "annotation", "ann_id": "annotation_id", "span": "ann_id"}
)

train_df = train_df.loc[train_df["label"] != "AnnotatorNotes"]
train_df["span_converted"] = train_df["span"].apply(convert_spans)
res_df = pd.merge(
    train_df[["ann_id", "term", "source", "span_converted"]],
    annotations_df[["ann_id", "annotation", "source"]],
    how="inner",
    on=["source", "ann_id"],
).drop(columns="ann_id")
res_df = res_df.loc[res_df["annotation"].isin(umls_cuis)]

# Rq: we have already checked that one term is always linked to the same annotation, hence "annotation":"first"

res_df.to_json(SAVE_EMEA_BIO_DIR)
res_df
```

```python
# SKIP THIS SECTION IF FILTERED QUAERO DATASET IS ALREADY BUILT

# Extract all MEDs from MEDLINE

# First list all CUIs in UMLS which are Laboratory Procedure
umls_cuis = pd.read_pickle(UMLS_ATC_DIR)[["CUI", "ATC"]].rename(
    columns={"CUI": "annotation"}
)
REGEX_CONVERT_SPANS = re.compile("^(\d+).*\s(\d+)$")


# Convert span to list with span_start, span_end. It considers the new lines by adding one character.
def convert_spans(span):
    span_match = REGEX_CONVERT_SPANS.match(span)
    span_start = int(span_match.group(1))
    span_end = int(span_match.group(2))
    return [span_start, span_end]


train_df = extract_pandas(IN_BRAT_DIR=MEDLINE_DIR)

annotations_df = train_df.loc[train_df["label"] == "AnnotatorNotes"]
annotations_df = annotations_df.rename(
    columns={"term": "annotation", "ann_id": "annotation_id", "span": "ann_id"}
)
train_df = train_df.loc[train_df["label"] != "AnnotatorNotes"]
train_df["span_converted"] = train_df["span"].apply(convert_spans)
res_df = pd.merge(
    train_df[["ann_id", "term", "source", "span_converted"]],
    annotations_df[["ann_id", "annotation", "source"]],
    how="inner",
    on=["source", "ann_id"],
).drop(columns="ann_id")
res_df = (
    res_df.merge(umls_cuis)
    .drop(columns="annotation")
    .rename(columns={"ATC": "annotation"})
)
res_df.span_converted = res_df.span_converted.astype(str)

# Rq: we have already checked that one term is always linked to the same annotation, hence "annotation":"first"
res_df = res_df.groupby(["term", "source", "span_converted"], as_index=False).agg(
    {"annotation": set}
)
res_df.to_json(SAVE_MEDLINE_MED_DIR)
res_df
```

```python
# SKIP THIS SECTION IF FILTERED QUAERO DATASET IS ALREADY BUILT

# Extract all MEDs from EMEA

# First list all CUIs in UMLS which are Laboratory Procedure
umls_cuis = pd.read_pickle(UMLS_ATC_DIR)[["CUI", "ATC"]].rename(
    columns={"CUI": "annotation"}
)
REGEX_CONVERT_SPANS = re.compile("^(\d+).*\s(\d+)$")


# Convert span to list with span_start, span_end. It considers the new lines by adding one character.
def convert_spans(span):
    span_match = REGEX_CONVERT_SPANS.match(span)
    span_start = int(span_match.group(1))
    span_end = int(span_match.group(2))
    return [span_start, span_end]


train_df = extract_pandas(IN_BRAT_DIR=EMEA_DIR)

annotations_df = train_df.loc[train_df["label"] == "AnnotatorNotes"]
annotations_df = annotations_df.rename(
    columns={"term": "annotation", "ann_id": "annotation_id", "span": "ann_id"}
)
train_df = train_df.loc[train_df["label"] != "AnnotatorNotes"]
train_df["span_converted"] = train_df["span"].apply(convert_spans)
res_df = pd.merge(
    train_df[["ann_id", "term", "source", "span_converted"]],
    annotations_df[["ann_id", "annotation", "source"]],
    how="inner",
    on=["source", "ann_id"],
).drop(columns="ann_id")
res_df = (
    res_df.merge(umls_cuis)
    .drop(columns="annotation")
    .rename(columns={"ATC": "annotation"})
)
res_df.span_converted = res_df.span_converted.astype(str)

# Rq: we have already checked that one term is always linked to the same annotation, hence "annotation":"first"
res_df = res_df.groupby(["term", "source", "span_converted"], as_index=False).agg(
    {"annotation": set}
)
res_df.to_json(SAVE_EMEA_MED_DIR)
res_df
```

# Now please run ``sbatch basch_scripts/Normalisation/evaluate.sh``


## Prepare FUZZY CODER

```python
threshold = 0.85

bio_quaero_fuzzy_lev = pd.read_json(RES_DIR_BIO_MEDLINE_FUZZY_LEV)
bio_quaero_coder_all = pd.read_json(RES_DIR_BIO_MEDLINE_CODER_ALL).rename(
    columns={
        "label": "label_coder",
        "score": "score_coder",
        "norm_term": "norm_term_coder",
    }
)
bio_quaero_coder_all.span_converted = bio_quaero_coder_all.span_converted.astype(str)
bio_quaero_fuzzy_lev.span_converted = bio_quaero_fuzzy_lev.span_converted.astype(str)
bio_quaero_fuzzy_coder = bio_quaero_fuzzy_lev.merge(
    bio_quaero_coder_all[
        ["source", "span_converted", "norm_term_coder", "label_coder", "score_coder"]
    ],
    on=["source", "span_converted"],
)

bio_quaero_fuzzy_coder.label = bio_quaero_fuzzy_coder.label.where(
    bio_quaero_fuzzy_coder.score >= threshold, bio_quaero_fuzzy_coder.label_coder
)
bio_quaero_fuzzy_coder.norm_term = bio_quaero_fuzzy_coder.norm_term.where(
    bio_quaero_fuzzy_coder.score >= threshold, bio_quaero_fuzzy_coder.norm_term_coder
)
bio_quaero_fuzzy_coder.score = bio_quaero_fuzzy_coder.score.where(
    bio_quaero_fuzzy_coder.score >= threshold, bio_quaero_fuzzy_coder.score_coder
)
bio_quaero_fuzzy_coder.span_converted = bio_quaero_fuzzy_coder.span_converted.apply(
    lambda x: eval(x)
)
bio_quaero_fuzzy_coder = bio_quaero_fuzzy_coder.drop(
    columns=["norm_term_coder", "label_coder", "score_coder"]
)
bio_quaero_fuzzy_coder.to_json(RES_DIR_BIO_MEDLINE_FUZZY_CODER)
```

```python
threshold = 0.85

bio_quaero_fuzzy_lev = pd.read_json(RES_DIR_BIO_EMEA_FUZZY_LEV)
bio_quaero_coder_all = pd.read_json(RES_DIR_BIO_EMEA_CODER_ALL).rename(
    columns={
        "label": "label_coder",
        "score": "score_coder",
        "norm_term": "norm_term_coder",
    }
)
bio_quaero_coder_all.span_converted = bio_quaero_coder_all.span_converted.astype(str)
bio_quaero_fuzzy_lev.span_converted = bio_quaero_fuzzy_lev.span_converted.astype(str)
bio_quaero_fuzzy_coder = bio_quaero_fuzzy_lev.merge(
    bio_quaero_coder_all[
        ["source", "span_converted", "norm_term_coder", "label_coder", "score_coder"]
    ],
    on=["source", "span_converted"],
)

bio_quaero_fuzzy_coder.label = bio_quaero_fuzzy_coder.label.where(
    bio_quaero_fuzzy_coder.score >= threshold, bio_quaero_fuzzy_coder.label_coder
)
bio_quaero_fuzzy_coder.norm_term = bio_quaero_fuzzy_coder.norm_term.where(
    bio_quaero_fuzzy_coder.score >= threshold, bio_quaero_fuzzy_coder.norm_term_coder
)
bio_quaero_fuzzy_coder.score = bio_quaero_fuzzy_coder.score.where(
    bio_quaero_fuzzy_coder.score >= threshold, bio_quaero_fuzzy_coder.score_coder
)
bio_quaero_fuzzy_coder.span_converted = bio_quaero_fuzzy_coder.span_converted.apply(
    lambda x: eval(x)
)
bio_quaero_fuzzy_coder = bio_quaero_fuzzy_coder.drop(
    columns=["norm_term_coder", "label_coder", "score_coder"]
)
bio_quaero_fuzzy_coder.to_json(RES_DIR_BIO_EMEA_FUZZY_CODER)
```

```python
threshold = 0.85

bio_aphp_fuzzy_lev = pd.read_json(RES_DIR_BIO_APHP_FUZZY_LEV)
bio_aphp_coder_all = pd.read_json(RES_DIR_BIO_APHP_CODER_ALL).rename(
    columns={
        "label": "label_coder",
        "score": "score_coder",
        "norm_term": "norm_term_coder",
    }
)
bio_aphp_coder_all.span_converted = bio_aphp_coder_all.span_converted.astype(str)
bio_aphp_fuzzy_lev.span_converted = bio_aphp_fuzzy_lev.span_converted.astype(str)
bio_aphp_fuzzy_coder = bio_aphp_fuzzy_lev.merge(
    bio_aphp_coder_all[
        ["source", "span_converted", "norm_term_coder", "label_coder", "score_coder"]
    ],
    on=["source", "span_converted"],
)

bio_aphp_fuzzy_coder.label = bio_aphp_fuzzy_coder.label.where(
    bio_aphp_fuzzy_coder.score >= threshold, bio_aphp_fuzzy_coder.label_coder
)
bio_aphp_fuzzy_coder.norm_term = bio_aphp_fuzzy_coder.norm_term.where(
    bio_aphp_fuzzy_coder.score >= threshold, bio_aphp_fuzzy_coder.norm_term_coder
)
bio_aphp_fuzzy_coder.score = bio_aphp_fuzzy_coder.score.where(
    bio_aphp_fuzzy_coder.score >= threshold, bio_aphp_fuzzy_coder.score_coder
)
bio_aphp_fuzzy_coder.span_converted = bio_aphp_fuzzy_coder.span_converted.apply(
    lambda x: eval(x)
)
bio_aphp_fuzzy_coder = bio_aphp_fuzzy_coder.drop(
    columns=["norm_term_coder", "label_coder", "score_coder"]
)
bio_aphp_fuzzy_coder.to_json(RES_DIR_BIO_APHP_FUZZY_CODER)
```

```python
threshold = 0.85

med_quaero_fuzzy_lev = pd.read_json(RES_DIR_MED_MEDLINE_FUZZY_LEV)
med_quaero_coder_all = pd.read_json(RES_DIR_MED_MEDLINE_CODER_ALL).rename(
    columns={
        "label": "label_coder",
        "score": "score_coder",
        "norm_term": "norm_term_coder",
    }
)
med_quaero_coder_all.span_converted = med_quaero_coder_all.span_converted.astype(str)
med_quaero_fuzzy_lev.span_converted = med_quaero_fuzzy_lev.span_converted.astype(str)
med_quaero_fuzzy_coder = med_quaero_fuzzy_lev.merge(
    med_quaero_coder_all[
        ["source", "span_converted", "norm_term_coder", "label_coder", "score_coder"]
    ],
    on=["source", "span_converted"],
)

med_quaero_fuzzy_coder.label = med_quaero_fuzzy_coder.label.where(
    med_quaero_fuzzy_coder.score >= threshold, med_quaero_fuzzy_coder.label_coder
)
med_quaero_fuzzy_coder.norm_term = med_quaero_fuzzy_coder.norm_term.where(
    med_quaero_fuzzy_coder.score >= threshold, med_quaero_fuzzy_coder.norm_term_coder
)
med_quaero_fuzzy_coder.score = med_quaero_fuzzy_coder.score.where(
    med_quaero_fuzzy_coder.score >= threshold, med_quaero_fuzzy_coder.score_coder
)
med_quaero_fuzzy_coder.span_converted = med_quaero_fuzzy_coder.span_converted.apply(
    lambda x: eval(x)
)
med_quaero_fuzzy_coder = med_quaero_fuzzy_coder.drop(
    columns=["norm_term_coder", "label_coder", "score_coder"]
)
med_quaero_fuzzy_coder.to_json(RES_DIR_MED_MEDLINE_FUZZY_CODER)
```

```python
threshold = 0.85

med_quaero_fuzzy_lev = pd.read_json(RES_DIR_MED_EMEA_FUZZY_LEV)
med_quaero_coder_all = pd.read_json(RES_DIR_MED_EMEA_CODER_ALL).rename(
    columns={
        "label": "label_coder",
        "score": "score_coder",
        "norm_term": "norm_term_coder",
    }
)
med_quaero_coder_all.span_converted = med_quaero_coder_all.span_converted.astype(str)
med_quaero_fuzzy_lev.span_converted = med_quaero_fuzzy_lev.span_converted.astype(str)
med_quaero_fuzzy_coder = med_quaero_fuzzy_lev.merge(
    med_quaero_coder_all[
        ["source", "span_converted", "norm_term_coder", "label_coder", "score_coder"]
    ],
    on=["source", "span_converted"],
)

med_quaero_fuzzy_coder.label = med_quaero_fuzzy_coder.label.where(
    med_quaero_fuzzy_coder.score >= threshold, med_quaero_fuzzy_coder.label_coder
)
med_quaero_fuzzy_coder.norm_term = med_quaero_fuzzy_coder.norm_term.where(
    med_quaero_fuzzy_coder.score >= threshold, med_quaero_fuzzy_coder.norm_term_coder
)
med_quaero_fuzzy_coder.score = med_quaero_fuzzy_coder.score.where(
    med_quaero_fuzzy_coder.score >= threshold, med_quaero_fuzzy_coder.score_coder
)
med_quaero_fuzzy_coder.span_converted = med_quaero_fuzzy_coder.span_converted.apply(
    lambda x: eval(x)
)
med_quaero_fuzzy_coder = med_quaero_fuzzy_coder.drop(
    columns=["norm_term_coder", "label_coder", "score_coder"]
)
med_quaero_fuzzy_coder.to_json(RES_DIR_MED_EMEA_FUZZY_CODER)
```

```python
threshold = 0.85

med_aphp_fuzzy_lev = pd.read_json(RES_DIR_MED_APHP_FUZZY_LEV)
med_aphp_coder_all = pd.read_json(RES_DIR_MED_APHP_CODER_ALL).rename(
    columns={
        "label": "label_coder",
        "score": "score_coder",
        "norm_term": "norm_term_coder",
    }
)
med_aphp_coder_all.span_converted = med_aphp_coder_all.span_converted.astype(str)
med_aphp_fuzzy_lev.span_converted = med_aphp_fuzzy_lev.span_converted.astype(str)
med_aphp_fuzzy_coder = med_aphp_fuzzy_lev.merge(
    med_aphp_coder_all[
        ["source", "span_converted", "norm_term_coder", "label_coder", "score_coder"]
    ],
    on=["source", "span_converted"],
)

med_aphp_fuzzy_coder.label = med_aphp_fuzzy_coder.label.where(
    med_aphp_fuzzy_coder.score >= threshold, med_aphp_fuzzy_coder.label_coder
)
med_aphp_fuzzy_coder.norm_term = med_aphp_fuzzy_coder.norm_term.where(
    med_aphp_fuzzy_coder.score >= threshold, med_aphp_fuzzy_coder.norm_term_coder
)
med_aphp_fuzzy_coder.score = med_aphp_fuzzy_coder.score.where(
    med_aphp_fuzzy_coder.score >= threshold, med_aphp_fuzzy_coder.score_coder
)
med_aphp_fuzzy_coder.span_converted = med_aphp_fuzzy_coder.span_converted.apply(
    lambda x: eval(x)
)
med_aphp_fuzzy_coder = med_aphp_fuzzy_coder.drop(
    columns=["norm_term_coder", "label_coder", "score_coder"]
)
med_aphp_fuzzy_coder.to_json(RES_DIR_MED_APHP_FUZZY_CODER)
```

## Compute RESULTS

```python
# Load res dataset and merge them in a single dataset
results_norm = []

for data_type in ["BIO", "MED"]:
    for dataset in ["MEDLINE", "EMEA", "APHP"]:
        for model in [
            "CODER_ALL",
            "CODER_ENG_PP",
            "SAPBERT_ALL",
            "SAPBERT_EDS",
            "FUZZY_LEV",
            "FUZZY_JW",
            "FUZZY_CODER",
        ]:
            path = eval("RES_DIR_{}_{}_{}".format(data_type, dataset, model))
            df = pd.read_json(path)
            df["Data type"] = (
                "Biological test name" if data_type == "BIO" else "Drug name"
            )
            df["Dataset"] = "Discharge summaries" if dataset == "APHP" else dataset
            df["Model"] = model.replace("_", " ")
            results_norm.append(df)

# Add umls synonyms for better visualisation
umls_bio_df = pd.read_pickle(UMLS_BIO_DIR)
umls_med_df = pd.read_pickle(UMLS_ATC_DIR)[["CUI", "STR"]].drop_duplicates(subset="CUI")
umls_df = pd.concat([umls_bio_df, umls_med_df])
res_df = pd.concat(results_norm)
res_df.annotation = res_df.annotation.astype(str)
res_df = res_df.merge(umls_df, left_on="annotation", right_on="CUI", how="left")
res_df


def eval_no_error(x):
    try:
        result = eval(x)
    except:
        result = x
    return result


res_df.annotation = res_df.annotation.apply(eval_no_error)
res_df = res_df.drop(columns=["CUI"])
res_df["label"] = res_df["label"].apply(set)
res_df["annotation"] = res_df["annotation"].apply(
    lambda x: set(x) if not (type(x) == str) else set([x])
)
res_df
```

```python
# Show res
display(res_df)
display(
    len(
        res_df.loc[
            res_df.apply(
                lambda row: len(row["annotation"].intersection(row["label"])), axis=1
            )
            > 0
        ]
    )
    / len(res_df)
)
```

## Plot Result


- Revoir Précision F1 Recall vs Accuracy ?
- Revoir par rapport à NER + Norm pour comparaison !
- Lancer la pipe sur tous les CR Lupus 
- Reprendre les erreures vues avec KriKri !

```python
import numpy as np
from tqdm import tqdm

# Bootstrappipng function over list of 0 and 1
N_RESAMPLES = 1000
alpha = 0.05
score_thresholds = {"FUZZY JW": 0.85, "CODER ALL": 0.75}


def bootstrap(results_per_doc, n_resamples=N_RESAMPLES):
    binary_results = {
        "TP": [0],
        "FP": [0],
        "FN": [0],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "N_entity": [],
    }

    docs = list(results_per_doc.keys())
    for doc in docs:
        binary_results["TP"][0] += results_per_doc[doc]["TP"]
        binary_results["FP"][0] += results_per_doc[doc]["FP"]
        binary_results["FN"][0] += results_per_doc[doc]["FN"]

    precision_init = (
        binary_results["TP"][0]
        / (binary_results["TP"][0] + binary_results["FP"][0])
        * 100
    )
    recall_init = (
        binary_results["TP"][0]
        / (binary_results["TP"][0] + binary_results["FN"][0])
        * 100
    )
    n_entity_init = binary_results["TP"][0] + binary_results["FN"][0]
    binary_results["Precision"].append(precision_init)
    binary_results["Recall"].append(recall_init)
    binary_results["F1"].append(
        2 * (precision_init * recall_init) / (precision_init + recall_init)
    )
    binary_results["N_entity"].append(n_entity_init)
    for i in tqdm(range(1, n_resamples)):
        draw = np.random.choice(
            docs,
            size=len(docs),
            replace=True,
        )
        binary_result = {"TP": 0, "FP": 0, "FN": 0, "Detected": 0}
        for doc in draw:
            binary_result["TP"] += results_per_doc[doc]["TP"]
            binary_result["FP"] += results_per_doc[doc]["FP"]
            binary_result["FN"] += results_per_doc[doc]["FN"]

        precision = (
            binary_result["TP"] / (binary_result["TP"] + binary_result["FP"]) * 100
        )
        recall = binary_result["TP"] / (binary_result["TP"] + binary_result["FN"]) * 100
        f1 = 2 * (precision * recall) / (precision + recall)
        n_entity = binary_result["TP"] + binary_result["FN"]

        binary_results["TP"].append(binary_result["TP"])
        binary_results["FP"].append(binary_result["FP"])
        binary_results["FN"].append(binary_result["FN"])
        binary_results["Precision"].append(precision)
        binary_results["Recall"].append(recall)
        binary_results["F1"].append(f1)
        binary_results["N_entity"].append(n_entity)
    return binary_results


def detail_all_atc(atc_list):
    all_atc = []
    for atc in atc_list:
        if isinstance(atc, str):
            atc_short = ""
            for char in atc:
                atc_short += char
                all_atc.append(atc_short)
    return set(all_atc)


def compute_accuracy(group):
    results_per_doc = {}
    for source in group.source.unique():
        source_res = group[group.source == source]
        metrics = {"TP": 0, "FN": 0, "FP": 0}
        n_entity = len(source_res)
        if group.Label.iloc[0] == "Drug name":
            tp = (
                source_res.apply(
                    lambda row: len(
                        detail_all_atc(row["label"]).intersection(row["annotation"])
                    ),
                    axis=1,
                )
                > 0
            ).sum()
        else:
            tp = (
                source_res.apply(
                    lambda row: len(row["label"].intersection(row["annotation"])),
                    axis=1,
                )
                > 0
            ).sum()
        fn = n_entity - tp
        fp = fn - len(source_res[source_res.label.str.len() == 0])
        metrics["TP"] += tp
        metrics["FP"] += fp
        metrics["FN"] += fn
        results_per_doc[source] = metrics
    binary_results = bootstrap(results_per_doc)
    n_entities = binary_results["N_entity"][0]
    precision = round(binary_results["Precision"][0], 1)
    recall = round(binary_results["Recall"][0], 1)
    f1 = round(binary_results["F1"][0], 1)
    n_entities_lower_bound = int(np.quantile(binary_results["N_entity"], (alpha / 2)))
    n_entities_upper_bound = int(
        np.quantile(binary_results["N_entity"], (1 - alpha / 2))
    )
    precision_lower_bound = round(
        np.quantile(binary_results["Precision"], (alpha / 2)), 1
    )
    precision_upper_bound = round(
        np.quantile(binary_results["Precision"], (1 - alpha / 2)), 1
    )
    recall_lower_bound = round(np.quantile(binary_results["Recall"], (alpha / 2)), 1)
    recall_upper_bound = round(
        np.quantile(binary_results["Recall"], (1 - alpha / 2)), 1
    )
    f1_lower_bound = round(np.quantile(binary_results["F1"], (alpha / 2)), 1)
    f1_upper_bound = round(np.quantile(binary_results["F1"], (1 - alpha / 2)), 1)
    precision_result = "{} [{}-{}]".format(
        precision, precision_lower_bound, precision_upper_bound
    )
    recall_result = "{} [{}-{}]".format(recall, recall_lower_bound, recall_upper_bound)
    f1_result = "{} [{}-{}]".format(f1, f1_lower_bound, f1_upper_bound)
    n_entities_result = "{} [{}-{}]".format(
        n_entities, n_entities_lower_bound, n_entities_upper_bound
    )
    return pd.DataFrame(
        {
            "Number of entities": [n_entities_result],
            "Precision": [precision_result],
            "Recall": [recall_result],
            "F1-score": [f1_result],
        }
    )


def adapt_prediction_with_score(group):
    model = group.Model.iloc[0]
    if model in score_thresholds.keys():
        score_threshold = score_thresholds[model]
    else:
        score_threshold = 0
    group.label = group.label.where(group.score >= score_threshold, set())
    return group


res_accuracy = (
    res_df.rename(columns={"Data type": "Label"})
    .groupby("Model")
    .apply(adapt_prediction_with_score)
)
res_accuracy = (
    res_accuracy[
        (
            (res_accuracy.Model == "CODER ALL")
            & (res_accuracy.Label == "Biological test name")
        )
        | ((res_accuracy.Model == "FUZZY JW") & (res_accuracy.Label == "Drug name"))
    ]
    .groupby(["Label", "Model", "Dataset"])
    .apply(compute_accuracy)
)


res_accuracy.droplevel(3)
```

```python
import numpy as np
from tqdm import tqdm

# Bootstrappipng function over list of 0 and 1
N_RESAMPLES = 1000
alpha = 0.05
score_thresholds = {"FUZZY JW": 0.85, "CODER ALL": 0.75}


def bootstrap(results_per_doc, n_resamples=N_RESAMPLES):
    binary_results = {
        "TP": [0],
        "FP": [0],
        "FN": [0],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "N_entity": [],
    }

    docs = list(results_per_doc.keys())
    for doc in docs:
        binary_results["TP"][0] += results_per_doc[doc]["TP"]
        binary_results["FP"][0] += results_per_doc[doc]["FP"]
        binary_results["FN"][0] += results_per_doc[doc]["FN"]

    precision_init = (
        binary_results["TP"][0] / (binary_results["TP"][0] + binary_results["FP"][0])
    ) * 100
    recall_init = (
        binary_results["TP"][0] / (binary_results["TP"][0] + binary_results["FN"][0])
    ) * 100
    n_entity_init = binary_results["TP"][0] + binary_results["FN"][0]
    binary_results["Precision"].append(precision_init)
    binary_results["Recall"].append(recall_init)
    binary_results["F1"].append(
        2 * (precision_init * recall_init) / (precision_init + recall_init)
    )
    binary_results["N_entity"].append(n_entity_init)
    for i in tqdm(range(1, n_resamples)):
        draw = np.random.choice(
            docs,
            size=len(docs),
            replace=True,
        )
        binary_result = {"TP": 0, "FP": 0, "FN": 0, "Detected": 0}
        for doc in draw:
            binary_result["TP"] += results_per_doc[doc]["TP"]
            binary_result["FP"] += results_per_doc[doc]["FP"]
            binary_result["FN"] += results_per_doc[doc]["FN"]

        precision = (
            binary_result["TP"] / (binary_result["TP"] + binary_result["FP"]) * 100
        )
        recall = binary_result["TP"] / (binary_result["TP"] + binary_result["FN"]) * 100
        f1 = 2 * (precision * recall) / (precision + recall)
        n_entity = binary_result["TP"] + binary_result["FN"]

        binary_results["TP"].append(binary_result["TP"])
        binary_results["FP"].append(binary_result["FP"])
        binary_results["FN"].append(binary_result["FN"])
        binary_results["Precision"].append(precision)
        binary_results["Recall"].append(recall)
        binary_results["F1"].append(f1)
        binary_results["N_entity"].append(n_entity)
    return binary_results


def detail_all_atc(atc_list):
    all_atc = []
    for atc in atc_list:
        if isinstance(atc, str):
            atc_short = ""
            for char in atc:
                atc_short += char
                all_atc.append(atc_short)
    return set(all_atc)


def compute_accuracy(group):
    results_per_doc = {}
    for source in group.source.unique():
        source_res = group[group.source == source]
        metrics = {"TP": 0, "FN": 0, "FP": 0}
        n_entity = len(source_res)
        if group.Label.iloc[0] == "Drug name":
            tp = (
                source_res.apply(
                    lambda row: len(
                        detail_all_atc(row["label"]).intersection(row["annotation"])
                    ),
                    axis=1,
                )
                > 0
            ).sum()
        else:
            tp = (
                source_res.apply(
                    lambda row: len(row["label"].intersection(row["annotation"])),
                    axis=1,
                )
                > 0
            ).sum()
        fn = n_entity - tp
        fp = fn - len(source_res[source_res.label.str.len() == 0])
        metrics["TP"] += tp
        metrics["FP"] += fp
        metrics["FN"] += fn
        results_per_doc[source] = metrics
    binary_results = bootstrap(results_per_doc)
    n_entities = binary_results["N_entity"][0]
    precision = round(binary_results["Precision"][0], 1)
    recall = round(binary_results["Recall"][0], 1)
    f1 = round(binary_results["F1"][0], 1)
    n_entities_lower_bound = int(np.quantile(binary_results["N_entity"], (alpha / 2)))
    n_entities_upper_bound = int(
        np.quantile(binary_results["N_entity"], (1 - alpha / 2))
    )
    precision_lower_bound = round(
        np.quantile(binary_results["Precision"], (alpha / 2)), 1
    )
    precision_upper_bound = round(
        np.quantile(binary_results["Precision"], (1 - alpha / 2)), 1
    )
    recall_lower_bound = round(np.quantile(binary_results["Recall"], (alpha / 2)), 1)
    recall_upper_bound = round(
        np.quantile(binary_results["Recall"], (1 - alpha / 2)), 1
    )
    f1_lower_bound = round(np.quantile(binary_results["F1"], (alpha / 2)), 1)
    f1_upper_bound = round(np.quantile(binary_results["F1"], (1 - alpha / 2)), 1)
    precision_result = "{} [{}, {}]".format(
        precision, precision_lower_bound, precision_upper_bound
    )
    recall_result = "{} [{}, {}]".format(recall, recall_lower_bound, recall_upper_bound)
    f1_result = "{} [{}, {}]".format(f1, f1_lower_bound, f1_upper_bound)
    n_entities_result = "{} [{}, {}]".format(
        n_entities, n_entities_lower_bound, n_entities_upper_bound
    )
    return pd.DataFrame(
        {
            "Number of entities": [n_entities_result],
            "Precision": [precision_result],
            "Recall": [recall_result],
            "F1-score": [f1_result],
        }
    )


def adapt_prediction_with_score(group):
    model = group.Model.iloc[0]
    if model in score_thresholds.keys():
        score_threshold = score_thresholds[model]
    else:
        score_threshold = 0
    group.label = group.label.where(group.score >= score_threshold, set())
    return group


res_accuracy = (
    res_df.rename(columns={"Data type": "Label"})
    .groupby("Model")
    .apply(adapt_prediction_with_score)
)

res_accuracy = (
    res_df.rename(columns={"Data type": "Label"})
    .groupby("Model")
    .apply(adapt_prediction_with_score)
)

res_accuracy = (
    res_accuracy[
        (
            (res_accuracy.Dataset == "Discharge summaries")
            # & (res_accuracy.Label == "Biological test name")
        )
        & (
            (
                (res_accuracy.Model != "FUZZY CODER")
                # & (res_accuracy.Label == "Biological test name")
            )
            & (
                (res_accuracy.Model != "CODER ENG PP")
                # & (res_accuracy.Label == "Drug name")
            )
            & (
                (res_accuracy.Model != "SAPBERT EDS")
                # & (res_accuracy.Label == "Drug name")
            )
        )
    ]
    .groupby(["Label", "Model", "Dataset"])
    .apply(compute_accuracy)
)


# res_accuracy.droplevel(3)
```

```python
res_accuracy.stack().unstack(level=2).unstack(level=3).droplevel(2)
```

```python
from IPython.display import display, HTML


def highlight_max(column):
    bio_max = (
        column["Biological test name"].str.split().str.get(0).astype(float)
        == column["Biological test name"].str.split().str.get(0).astype(float).max()
    )
    drug_max = (
        column["Drug name"].str.split().str.get(0).astype(float)
        == column["Drug name"].str.split().str.get(0).astype(float).max()
    )
    s_max = []
    for i in range(len(bio_max)):
        if not column.name[1] == "Number of entities":
            s_max.append(bio_max[i])
        else:
            s_max.append(False)
    for i in range(len(drug_max)):
        if not column.name[1] == "Number of entities":
            s_max.append(drug_max[i])
        else:
            s_max.append(False)
    return ["font-weight: bold" if cell else "" for cell in s_max]


all_res_df = res_accuracy.stack().unstack(level=2).unstack(level=3).droplevel(2)


def remove_confidence(cell):
    return cell.split(r"\n")[0]


pretty_df = (
    all_res_df.drop(
        columns=[
            (model, "Number of entities")
            for model in sorted(list(set(all_res_df.columns.get_level_values(0))))[1:]
        ],
        axis=1,
    )
    # .applymap(remove_confidence)
    .style.apply(highlight_max, axis=0).set_properties(**{"text-align": "center"})
)


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


pretty_print(pretty_df)
```

```python
import numpy as np
from tqdm import tqdm

# Bootstrappipng function over list of 0 and 1
N_RESAMPLES = 1000
alpha = 0.05
score_thresholds = {"FUZZY JW": 0.85, "CODER ALL": 0.75}


def bootstrap(results_per_doc, n_resamples=N_RESAMPLES):
    binary_results = {
        "TP": [0],
        "FP": [0],
        "FN": [0],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "N_entity": [],
    }

    docs = list(results_per_doc.keys())
    for doc in docs:
        binary_results["TP"][0] += results_per_doc[doc]["TP"]
        binary_results["FP"][0] += results_per_doc[doc]["FP"]
        binary_results["FN"][0] += results_per_doc[doc]["FN"]

    precision_init = (
        binary_results["TP"][0]
        / (binary_results["TP"][0] + binary_results["FP"][0])
        * 100
    )
    recall_init = (
        binary_results["TP"][0]
        / (binary_results["TP"][0] + binary_results["FN"][0])
        * 100
    )
    n_entity_init = binary_results["TP"][0] + binary_results["FN"][0]
    binary_results["Precision"].append(precision_init)
    binary_results["Recall"].append(recall_init)
    binary_results["F1"].append(
        2 * (precision_init * recall_init) / (precision_init + recall_init)
    )
    binary_results["N_entity"].append(n_entity_init)
    for i in tqdm(range(1, n_resamples)):
        draw = np.random.choice(
            docs,
            size=len(docs),
            replace=True,
        )
        binary_result = {"TP": 0, "FP": 0, "FN": 0, "Detected": 0}
        for doc in draw:
            binary_result["TP"] += results_per_doc[doc]["TP"]
            binary_result["FP"] += results_per_doc[doc]["FP"]
            binary_result["FN"] += results_per_doc[doc]["FN"]

        precision = (
            binary_result["TP"] / (binary_result["TP"] + binary_result["FP"]) * 100
        )
        recall = binary_result["TP"] / (binary_result["TP"] + binary_result["FN"]) * 100
        f1 = 2 * (precision * recall) / (precision + recall)
        n_entity = binary_result["TP"] + binary_result["FN"]

        binary_results["TP"].append(binary_result["TP"])
        binary_results["FP"].append(binary_result["FP"])
        binary_results["FN"].append(binary_result["FN"])
        binary_results["Precision"].append(precision)
        binary_results["Recall"].append(recall)
        binary_results["F1"].append(f1)
        binary_results["N_entity"].append(n_entity)
    return binary_results


def detail_all_atc(atc_list):
    all_atc = []
    for atc in atc_list:
        if isinstance(atc, str):
            atc_short = ""
            for char in atc:
                atc_short += char
                all_atc.append(atc_short)
    return set(all_atc)


def compute_accuracy(group):
    results_per_doc = {}
    for source in group.source.unique():
        source_res = group[group.source == source]
        metrics = {"TP": 0, "FN": 0, "FP": 0}
        n_entity = len(source_res)
        if group.Label.iloc[0] == "Drug name":
            tp = (
                source_res.apply(
                    lambda row: len(
                        detail_all_atc(row["label"]).intersection(row["annotation"])
                    ),
                    axis=1,
                )
                > 0
            ).sum()
        else:
            tp = (
                source_res.apply(
                    lambda row: len(row["label"].intersection(row["annotation"])),
                    axis=1,
                )
                > 0
            ).sum()
        fn = n_entity - tp
        fp = fn - len(source_res[source_res.label.str.len() == 0])
        metrics["TP"] += tp
        metrics["FP"] += fp
        metrics["FN"] += fn
        results_per_doc[source] = metrics
    binary_results = bootstrap(results_per_doc)
    n_entities = binary_results["N_entity"][0]
    precision = round(binary_results["Precision"][0], 2)
    recall = round(binary_results["Recall"][0], 2)
    f1 = round(binary_results["F1"][0], 2)
    n_entities_lower_bound = int(np.quantile(binary_results["N_entity"], (alpha / 2)))
    n_entities_upper_bound = int(
        np.quantile(binary_results["N_entity"], (1 - alpha / 2))
    )
    precision_lower_bound = round(
        np.quantile(binary_results["Precision"], (alpha / 2)), 2
    )
    precision_upper_bound = round(
        np.quantile(binary_results["Precision"], (1 - alpha / 2)), 2
    )
    recall_lower_bound = round(np.quantile(binary_results["Recall"], (alpha / 2)), 2)
    recall_upper_bound = round(
        np.quantile(binary_results["Recall"], (1 - alpha / 2)), 2
    )
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
    return pd.DataFrame(
        {
            "Number of entities": [n_entities_result],
            "Precision": [precision_result],
            "Recall": [recall_result],
            "F1-score": [f1_result],
        }
    )


def adapt_prediction_with_score(group):
    model = group.Model.iloc[0]
    if model in score_thresholds.keys():
        score_threshold = score_thresholds[model]
    else:
        score_threshold = 0
    group.label = group.label.where(group.score >= score_threshold, set())
    return group


res_accuracy = (
    res_df.rename(columns={"Data type": "Label"})
    .groupby("Model")
    .apply(adapt_prediction_with_score)
)
res_accuracy = (
    res_accuracy[
        (
            (res_accuracy.Model == "CODER ALL")
            & (res_accuracy.Label == "Biological test name")
        )
        | ((res_accuracy.Model == "FUZZY JW") & (res_accuracy.Label == "Drug name"))
    ]
    .groupby(["Label", "Model", "Dataset"])
    .apply(compute_accuracy)
)


res_accuracy.droplevel(3)
```

```python
import numpy as np
from tqdm import tqdm

# Bootstrappipng function over list of 0 and 1
N_RESAMPLES = 1000
alpha = 0.05
score_thresholds = {"FUZZY JW": 0.85, "CODER ALL": 0.75}


def bootstrap(results_per_doc, n_resamples=N_RESAMPLES):
    binary_results = {
        "TP": [0],
        "FP": [0],
        "FN": [0],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "N_entity": [],
    }

    docs = list(results_per_doc.keys())
    for doc in docs:
        binary_results["TP"][0] += results_per_doc[doc]["TP"]
        binary_results["FP"][0] += results_per_doc[doc]["FP"]
        binary_results["FN"][0] += results_per_doc[doc]["FN"]

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
    for i in tqdm(range(1, n_resamples)):
        draw = np.random.choice(
            docs,
            size=len(docs),
            replace=True,
        )
        binary_result = {"TP": 0, "FP": 0, "FN": 0, "Detected": 0}
        for doc in draw:
            binary_result["TP"] += results_per_doc[doc]["TP"]
            binary_result["FP"] += results_per_doc[doc]["FP"]
            binary_result["FN"] += results_per_doc[doc]["FN"]

        precision = binary_result["TP"] / (binary_result["TP"] + binary_result["FP"]) * 100
        recall = binary_result["TP"] / (binary_result["TP"] + binary_result["FN"]) * 100
        f1 = 2 * (precision * recall) / (precision + recall)
        n_entity = binary_result["TP"] + binary_result["FN"]

        binary_results["TP"].append(binary_result["TP"])
        binary_results["FP"].append(binary_result["FP"])
        binary_results["FN"].append(binary_result["FN"])
        binary_results["Precision"].append(precision)
        binary_results["Recall"].append(recall)
        binary_results["F1"].append(f1)
        binary_results["N_entity"].append(n_entity)
    return binary_results


def detail_all_atc(atc_list):
    all_atc = []
    for atc in atc_list:
        if isinstance(atc, str):
            atc_short = ""
            for char in atc:
                atc_short += char
                all_atc.append(atc_short)
    return set(all_atc)


def compute_accuracy(group):
    results_per_doc = {}
    for source in group.source.unique():
        source_res = group[group.source == source]
        metrics = {"TP": 0, "FN": 0, "FP": 0}
        n_entity = len(source_res)
        if group.Label.iloc[0] == "Drug name":
            tp = (
                source_res.apply(
                    lambda row: len(
                        detail_all_atc(row["label"]).intersection(row["annotation"])
                    ),
                    axis=1,
                )
                > 0
            ).sum()
        else:
            tp = (
                source_res.apply(
                    lambda row: len(row["label"].intersection(row["annotation"])),
                    axis=1,
                )
                > 0
            ).sum()
        fn = n_entity - tp
        fp = fn - len(source_res[source_res.label.str.len() == 0])
        metrics["TP"] += tp
        metrics["FP"] += fp
        metrics["FN"] += fn
        results_per_doc[source] = metrics
    binary_results = bootstrap(results_per_doc)
    n_entities = binary_results["N_entity"][0]
    precision = round(binary_results["Precision"][0], 2)
    recall = round(binary_results["Recall"][0], 2)
    f1 = round(binary_results["F1"][0], 2)
    n_entities_lower_bound = int(np.quantile(binary_results["N_entity"], (alpha / 2)))
    n_entities_upper_bound = int(
        np.quantile(binary_results["N_entity"], (1 - alpha / 2))
    )
    precision_lower_bound = round(
        np.quantile(binary_results["Precision"], (alpha / 2)), 2
    )
    precision_upper_bound = round(
        np.quantile(binary_results["Precision"], (1 - alpha / 2)), 2
    )
    recall_lower_bound = round(np.quantile(binary_results["Recall"], (alpha / 2)), 2)
    recall_upper_bound = round(
        np.quantile(binary_results["Recall"], (1 - alpha / 2)), 2
    )
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
    return pd.DataFrame(
        {
            "Number of entities": [n_entities_result],
            "Precision": [precision_result],
            "Recall": [recall_result],
            "F1-score": [f1_result],
        }
    )


def adapt_prediction_with_score(group):
    model = group.Model.iloc[0]
    if model in score_thresholds.keys():
        score_threshold = score_thresholds[model]
    else:
        score_threshold = 0
    group.label = group.label.where(group.score >= score_threshold, set())
    return group


res_accuracy = (
    res_df.rename(columns={"Data type": "Label"})
    .groupby("Model")
    .apply(adapt_prediction_with_score)
)
res_accuracy = (
    res_accuracy[
        (
            (res_accuracy.Model == "CODER ALL")
            # & (res_accuracy.Label == "Biological test name")
        )
        | ((res_accuracy.Model == "FUZZY JW") 
           # & (res_accuracy.Label == "Drug name")
          )
    ]
    .groupby(["Label", "Model", "Dataset"])
    .apply(compute_accuracy)
)


res_accuracy.droplevel(3)
```

```python
import numpy as np
from tqdm import tqdm

# Bootstrappipng function over list of 0 and 1
N_RESAMPLES = 1000
alpha = 0.05
score_thresholds = {
    "FUZZY LEV": 0.85,
    "FUZZY JW": 0.85,
    "FUZZY CODER": 0.75,
    "CODER ALL": 0.75,
    "CODER ENG PP": 0,
    "SAPBERT ALL": 0.75,
    "SAPBERT EDS": 0.75,
}


def bootstrap(results_per_doc, n_resamples=N_RESAMPLES):
    binary_results = {
        "TP": [0],
        "FP": [0],
        "FN": [0],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "N_entity": [],
    }

    docs = list(results_per_doc.keys())
    for doc in docs:
        binary_results["TP"][0] += results_per_doc[doc]["TP"]
        binary_results["FP"][0] += results_per_doc[doc]["FP"]
        binary_results["FN"][0] += results_per_doc[doc]["FN"]

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
    for i in tqdm(range(1, n_resamples)):
        draw = np.random.choice(
            docs,
            size=len(docs),
            replace=True,
        )
        binary_result = {"TP": 0, "FP": 0, "FN": 0, "Detected": 0}
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
    return binary_results


def detail_all_atc(atc_list):
    all_atc = []
    for atc in atc_list:
        if isinstance(atc, str):
            atc_short = ""
            for char in atc:
                atc_short += char
                all_atc.append(atc_short)
    return set(all_atc)


def compute_accuracy(group):
    results_per_doc = {}
    for source in group.source.unique():
        source_res = group[group.source == source]
        metrics = {"TP": 0, "FN": 0, "FP": 0}
        n_entity = len(source_res)
        if group.Label.iloc[0] == "Drug name":
            tp = (
                source_res.apply(
                    lambda row: len(
                        detail_all_atc(row["label"]).intersection(row["annotation"])
                    ),
                    axis=1,
                )
                > 0
            ).sum()
        else:
            tp = (
                source_res.apply(
                    lambda row: len(row["label"].intersection(row["annotation"])),
                    axis=1,
                )
                > 0
            ).sum()
        fn = n_entity - tp
        fp = fn - len(source_res[source_res.label.str.len() == 0])
        metrics["TP"] += tp
        metrics["FP"] += fp
        metrics["FN"] += fn
        results_per_doc[source] = metrics
    binary_results = bootstrap(results_per_doc)
    n_entities = binary_results["N_entity"][0]
    precision = round(binary_results["Precision"][0], 2)
    recall = round(binary_results["Recall"][0], 2)
    f1 = round(binary_results["F1"][0], 2)
    n_entities_lower_bound = int(np.quantile(binary_results["N_entity"], (alpha / 2)))
    n_entities_upper_bound = int(
        np.quantile(binary_results["N_entity"], (1 - alpha / 2))
    )
    precision_lower_bound = round(
        np.quantile(binary_results["Precision"], (alpha / 2)), 2
    )
    precision_upper_bound = round(
        np.quantile(binary_results["Precision"], (1 - alpha / 2)), 2
    )
    recall_lower_bound = round(np.quantile(binary_results["Recall"], (alpha / 2)), 2)
    recall_upper_bound = round(
        np.quantile(binary_results["Recall"], (1 - alpha / 2)), 2
    )
    f1_lower_bound = round(np.quantile(binary_results["F1"], (alpha / 2)), 2)
    f1_upper_bound = round(np.quantile(binary_results["F1"], (1 - alpha / 2)), 2)
    precision_result = "{}\\n[{}, {}]".format(
        precision, precision_lower_bound, precision_upper_bound
    )
    recall_result = "{}\\n[{}, {}]".format(
        recall, recall_lower_bound, recall_upper_bound
    )
    f1_result = "{}\\n[{}, {}]".format(f1, f1_lower_bound, f1_upper_bound)
    n_entities_result = "{}\\n[{}, {}]".format(
        n_entities, n_entities_lower_bound, n_entities_upper_bound
    )
    return pd.DataFrame(
        {
            "Number of entities": [n_entities_result],
            "Precision": [precision_result],
            "Recall": [recall_result],
            "F1-score": [f1_result],
        }
    )


def adapt_prediction_with_score(group):
    model = group.Model.iloc[0]
    if model in score_thresholds.keys():
        score_threshold = score_thresholds[model]
    else:
        score_threshold = 0
    group.label = group.label.where(group.score >= score_threshold, set())
    return group


res_accuracy = (
    res_df.rename(columns={"Data type": "Label"})
    .groupby("Model")
    .apply(adapt_prediction_with_score)
)
res_accuracy = res_accuracy.groupby(["Label", "Dataset", "Model"]).apply(
    compute_accuracy
)
```

```python
from IPython.display import display, HTML


def highlight_max(row):
    Precision_max = (
        row[:, "Precision"].str.split(r"\\n").str.get(0).astype(float)
        == row[:, "Precision"].str.split(r"\\n").str.get(0).astype(float).max()
    )
    Recall_max = (
        row[:, "Recall"].str.split(r"\\n").str.get(0).astype(float)
        == row[:, "Recall"].str.split(r"\\n").str.get(0).astype(float).max()
    )
    F1_max = (
        row[:, "F1-score"].str.split(r"\\n").str.get(0).astype(float)
        == row[:, "F1-score"].str.split(r"\\n").str.get(0).astype(float).max()
    )
    s_max = [False]
    for i in range(len(F1_max)):
        s_max.append(Precision_max[i])
        s_max.append(Recall_max[i])
        s_max.append(F1_max[i])
    return ["font-weight: bold" if cell else "" for cell in s_max]


all_res_df = res_accuracy.stack().unstack(level=2).unstack(level=3).droplevel(2)


def remove_confidence(cell):
    return cell.split(r"\n")[0]


pretty_df = (
    all_res_df.drop(
        columns=[
            (model, "Number of entities")
            for model in sorted(list(set(all_res_df.columns.get_level_values(0))))[1:]
        ],
        axis=1,
    )
    # .applymap(remove_confidence)
    .style.apply(highlight_max, axis=1).set_properties(**{"text-align": "center"})
)


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


pretty_print(pretty_df)
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python
def remove_confidence(cell):
    return cell.split(r"\n")[0]


all_res_df.drop(
    columns=[
        (model, "Number of entities")
        for model in sorted(list(set(all_res_df.columns.get_level_values(0))))[1:]
    ],
    axis=1,
).applymap(remove_confidence)
```

```python
from IPython.display import display, HTML


def highlight_max(row):
    Precision_max = (
        row[:, "Precision"].str.split(r"\\n").str.get(0).astype(float)
        == row[:, "Precision"].str.split(r"\\n").str.get(0).astype(float).max()
    )
    Recall_max = (
        row[:, "Recall"].str.split(r"\\n").str.get(0).astype(float)
        == row[:, "Recall"].str.split(r"\\n").str.get(0).astype(float).max()
    )
    F1_max = (
        row[:, "F1-score"].str.split(r"\\n").str.get(0).astype(float)
        == row[:, "F1-score"].str.split(r"\\n").str.get(0).astype(float).max()
    )
    s_max = [False]
    for i in range(len(F1_max)):
        s_max.append(Precision_max[i])
        s_max.append(Recall_max[i])
        s_max.append(F1_max[i])
    return ["font-weight: bold" if cell else "" for cell in s_max]


all_res_df = res_accuracy.stack().unstack(level=2).unstack(level=3).droplevel(2)

pretty_df = (
    all_res_df.drop(
        columns=[
            (model, "Number of entities")
            for model in sorted(list(set(all_res_df.columns.get_level_values(0))))[1:]
        ],
        axis=1,
    )
    .style.apply(highlight_max, axis=1)
    .set_properties(**{"text-align": "center"})
)


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


pretty_print(pretty_df)
```

```python
# Bootstrappipng function over list of 0 and 1
N_RESAMPLES = 1000
alpha = 0.05


def bootstrap(binary_results, n_resamples=N_RESAMPLES):
    accuracy_init = sum(
        [binary_result.count(1) for binary_result in binary_results.values()]
    ) / sum([len(binary_result) for binary_result in binary_results.values()])
    accuracies = [accuracy_init]
    sources = list(binary_results.keys())
    for i in range(n_resamples - 1):
        sample_sources = np.random.choice(
            sources, size=len(sources), replace=True
        ).tolist()
        samples_binary_result = []
        for source in sample_sources:
            samples_binary_result += binary_results[source]
        accuracies.append(samples_binary_result.count(1) / len(samples_binary_result))
    return accuracies


def detail_all_atc(atc_list):
    all_atc = []
    for atc in atc_list:
        if isinstance(atc, str):
            atc_short = ""
            for char in atc:
                atc_short += char
                all_atc.append(atc_short)
    return set(all_atc)


def compute_accuracy(group):
    binary_results = {}
    for source in group.source.unique():
        source_res = group[group.source == source]
        binary_results[source] = [
            1
            for _ in range(
                len(
                    source_res.loc[
                        source_res.apply(
                            lambda row: len(
                                row["label"].intersection(set(row["annotation"]))
                            ),
                            axis=1,
                        )
                        > 0
                    ]
                )
            )
        ] + [
            0
            for _ in range(
                len(
                    source_res.loc[
                        source_res.apply(
                            lambda row: len(
                                row["label"].intersection(set(row["annotation"]))
                            ),
                            axis=1,
                        )
                        == 0
                    ]
                )
            )
        ]
    accuracy_bootstrap = bootstrap(binary_results)
    accuracy = round(accuracy_bootstrap[0], 3)
    lower_bound = round(np.quantile(accuracy_bootstrap, (alpha / 2)), 3)
    upper_bound = round(np.quantile(accuracy_bootstrap, (1 - alpha / 2)), 3)
    result = "{} [{}, {}]".format(accuracy, lower_bound, upper_bound)
    return pd.Series(result)


def compute_accuracies(group):
    binary_results = {}
    for source in group.source.unique():
        source_res = group[group.source == source]
        binary_results[source] = [
            1
            for _ in range(
                len(
                    source_res.loc[
                        source_res.apply(
                            lambda row: len(
                                row["label"].intersection(set(row["annotation"]))
                            ),
                            axis=1,
                        )
                        > 0
                    ]
                )
            )
        ] + [
            0
            for _ in range(
                len(
                    source_res.loc[
                        source_res.apply(
                            lambda row: len(
                                row["label"].intersection(set(row["annotation"]))
                            ),
                            axis=1,
                        )
                        == 0
                    ]
                )
            )
        ]
    accuracy_bootstrap = bootstrap(binary_results)
    return pd.Series(accuracy_bootstrap)


res_accuracies = (
    res_df.groupby(["data_type", "dataset", "model"])
    .apply(compute_accuracies)
    .stack()
    .to_frame()
    .reset_index()
    .drop(columns="level_3")
    .rename(columns={0: "accuracy"})
)

res_accuracy = res_df.groupby(["data_type", "dataset", "model"]).apply(compute_accuracy)
res_accuracy.stack().unstack(level=2).droplevel(2)
```

```python
# Plotting
alt.data_transformers.disable_max_rows()

bio_df = res_accuracies[res_accuracies["Data type"] == "BIO"]
base_chart_bio = alt.Chart(bio_df, title="BIO").encode(
    x=alt.X("accuracy:Q", bin=alt.BinParams(maxbins=50))
)
bar_bio = base_chart_bio.mark_bar().encode(y="count()")
mean_bio = base_chart_bio.mark_rule(color="red").encode(
    x=alt.X("mean(accuracy):Q", title=None),
    size=alt.value(4),
)
text_bio = base_chart_bio.mark_text(size=14, dx=30).encode(
    x=alt.X("mean(accuracy):Q", title=None),
    text=alt.Text("mean(accuracy):Q", format=",.3f"),
    y=alt.value(25),
)
chart_bio = (bar_bio + mean_bio + text_bio).facet(
    row="Dataset",
    column="Model",
)
display(chart_bio)

med_df = res_accuracies[res_accuracies["Data type"] == "MED"]
base_chart_med = alt.Chart(med_df, title="MED").encode(
    x=alt.X("accuracy:Q", bin=alt.BinParams(maxbins=50))
)
bar_med = base_chart_med.mark_bar().encode(y="count()")
mean_med = base_chart_med.mark_rule(color="red").encode(
    x=alt.X("mean(accuracy):Q", title=None),
    size=alt.value(4),
)
text_med = base_chart_med.mark_text(size=14, dx=30).encode(
    x=alt.X("mean(accuracy):Q", title=None),
    text=alt.Text("mean(accuracy):Q", format=",.3f"),
    y=alt.value(25),
)
chart_med = (bar_med + mean_med + text_med).facet(
    row="Dataset",
    column="Model",
)
display(chart_med)
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python
bio_aphp_coder_all.loc[
    bio_aphp_coder_all.apply(
        lambda row: len(set(row["label"]).intersection({row["annotation"]})),
        axis=1,
    )
    == 0
]
```

```python
bio_aphp_fuzzy_coder.loc[
    bio_aphp_fuzzy_coder.apply(
        lambda row: len(set(row["label"]).intersection({row["annotation"]})),
        axis=1,
    )
    == 0
]
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python
alpha = 0.05

# Plotting
bootstrap_df = pd.DataFrame(accuracies_bio)
alt.data_transformers.disable_max_rows()
chart = (
    alt.Chart(bootstrap_df)
    .mark_bar()
    .encode(
        column="dataset",
        row="model",
        x=alt.X("accuracy:Q", bin=alt.BinParams(maxbins=60)),
        y="count()",
    )
)
display(chart)
print(
    "Confidence interval BIO EXACT: {} [{}, {}]".format(
        round(
            bootstrap_df[
                (bootstrap_df.model == "EXACT") & (bootstrap_df.dataset == "TOTAL")
            ].accuracy.iloc[0],
            4,
        ),
        round(
            np.quantile(
                bootstrap_df[
                    (bootstrap_df.model == "EXACT") & (bootstrap_df.dataset == "TOTAL")
                ].accuracy,
                (alpha / 2),
            ),
            4,
        ),
        round(
            np.quantile(
                bootstrap_df[
                    (bootstrap_df.model == "EXACT") & (bootstrap_df.dataset == "TOTAL")
                ].accuracy,
                (1 - alpha / 2),
            ),
            4,
        ),
    )
)
print(
    "Confidence interval BIO LEV: {} [{}, {}]".format(
        round(
            bootstrap_df[
                (bootstrap_df.model == "LEV") & (bootstrap_df.dataset == "TOTAL")
            ].accuracy.iloc[0],
            4,
        ),
        round(
            np.quantile(
                bootstrap_df[
                    (bootstrap_df.model == "LEV") & (bootstrap_df.dataset == "TOTAL")
                ].accuracy,
                (alpha / 2),
            ),
            4,
        ),
        round(
            np.quantile(
                bootstrap_df[
                    (bootstrap_df.model == "LEV") & (bootstrap_df.dataset == "TOTAL")
                ].accuracy,
                (1 - alpha / 2),
            ),
            4,
        ),
    )
)
print(
    "Confidence interval BIO JARO-WINKLER: {} [{}, {}]".format(
        round(
            bootstrap_df[
                (bootstrap_df.model == "JARO-WINKLER")
                & (bootstrap_df.dataset == "TOTAL")
            ].accuracy.iloc[0],
            4,
        ),
        round(
            np.quantile(
                bootstrap_df[
                    (bootstrap_df.model == "JARO-WINKLER")
                    & (bootstrap_df.dataset == "TOTAL")
                ].accuracy,
                (alpha / 2),
            ),
            4,
        ),
        round(
            np.quantile(
                bootstrap_df[
                    (bootstrap_df.model == "JARO-WINKLER")
                    & (bootstrap_df.dataset == "TOTAL")
                ].accuracy,
                (1 - alpha / 2),
            ),
            4,
        ),
    )
)
print(
    "Confidence interval BIO CODER all: {} [{}, {}]".format(
        round(
            bootstrap_df[
                (bootstrap_df.model == "CODER_ALL") & (bootstrap_df.dataset == "TOTAL")
            ].accuracy.iloc[0],
            4,
        ),
        round(
            np.quantile(
                bootstrap_df[
                    (bootstrap_df.model == "CODER_ALL")
                    & (bootstrap_df.dataset == "TOTAL")
                ].accuracy,
                (alpha / 2),
            ),
            4,
        ),
        round(
            np.quantile(
                bootstrap_df[
                    (bootstrap_df.model == "CODER_ALL")
                    & (bootstrap_df.dataset == "TOTAL")
                ].accuracy,
                (1 - alpha / 2),
            ),
            4,
        ),
    )
)
print(
    "Confidence interval BIO CODER EDS: {} [{}, {}]".format(
        round(
            bootstrap_df[
                (bootstrap_df.model == "CODER_EDS") & (bootstrap_df.dataset == "TOTAL")
            ].accuracy.iloc[0],
            4,
        ),
        round(
            np.quantile(
                bootstrap_df[
                    (bootstrap_df.model == "CODER_EDS")
                    & (bootstrap_df.dataset == "TOTAL")
                ].accuracy,
                (alpha / 2),
            ),
            4,
        ),
        round(
            np.quantile(
                bootstrap_df[
                    (bootstrap_df.model == "CODER_EDS")
                    & (bootstrap_df.dataset == "TOTAL")
                ].accuracy,
                (1 - alpha / 2),
            ),
            4,
        ),
    )
)
```

```python
res_df.to_json(RESULTS_SAVE_DIR)
```
