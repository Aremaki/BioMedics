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

## MED 2023

```python
import zipfile
from tqdm import tqdm
import pandas as pd

atc_cuis = {"CUI": [], "ATC": []}
path = "umls-2023AB-full.zip"
with zipfile.ZipFile(path) as zip_file:
    with zip_file.open("2023AB/META/MRCONSO.RRF", mode="r") as file:
        lines = file.readlines()
        print(f"ALL UMLS 2023AB: {len(lines)} concepts")
        for line in tqdm(lines):
            line = str(line)[2:-3].split("|")
            if line[11] == "ATC":
                atc_cuis["CUI"].append(line[0])
                atc_cuis["ATC"].append(line[13])
atc_cuis_df = pd.DataFrame(atc_cuis).drop_duplicates()
print(f"ATC 2023AB: {len(atc_cuis_df)} cui/code combinations")
```

```python
import zipfile
from tqdm import tqdm
import pandas as pd

med_cuis = set(atc_cuis_df["CUI"].unique())
med_syn = dict(CUI=[], STR=[])
path = "umls-2023AB-full.zip"
with zipfile.ZipFile(path) as zip_file:
    with zip_file.open("2023AB/META/MRCONSO.RRF", mode="r") as file:
        lines = file.readlines()
        print(f"ALL UMLS 2023AB: {len(lines)} concepts")
        for line in tqdm(lines):
            line = str(line)[2:-3].split("|")
            if line[1] in ["FRE", "ENG"]:
                if line[0] in med_cuis:
                    med_syn["CUI"].append(line[0])
                    med_syn["STR"].append(line[14])
med_syn_df = pd.DataFrame(med_syn).drop_duplicates()
print(f"ATC 2023AB: {len(med_syn_df)} synonymes")
```

```python
umls_atc = med_syn_df.merge(atc_cuis_df, on="CUI")[["ATC", "STR"]]
umls_atc.to_pickle("atc_str_2023AB.pkl")
```

### Create Full ATC

```python
import pandas as pd

umls_atc = pd.read_pickle("atc_cui_2023AB.pkl").explode("STR")[["ATC", "STR"]]
romedi_atc = pd.DataFrame.from_dict(
    dict(
        pd.read_pickle(
            "../drug_knowledge/final_dict.pkl"
        )
    ),
    orient="index",
)
romedi_atc = (
    pd.DataFrame(romedi_atc.stack().groupby(level=0).agg(set), columns=["STR"])
    .explode("STR")
    .reset_index()
    .rename(columns={"index": "ATC"})
)
full_atc_str = pd.concat([romedi_atc, umls_atc]).drop_duplicates()
full_atc_str.to_pickle("full_atc_str_2023AB.pkl")
```

## BIO 2023

```python
import zipfile
from tqdm import tqdm

snomed_cuis = []
path = "umls-2023AB-full.zip"
with zipfile.ZipFile(path) as zip_file:
    with zip_file.open("2023AB/META/MRCONSO.RRF", mode="r") as file:
        lines = file.readlines()
        print(f"ALL UMLS 2023AB: {len(lines)} concepts")
        for line in tqdm(lines):
            line = str(line)[2:-3].split("|")
            if line[11] == "SNOMEDCT_US":
                snomed_cuis.append(line[0])
snomed_cuis = set(snomed_cuis)
print(f"SNOMED CT US 2023AB: {len(snomed_cuis)} concepts")
```

```python
from tqdm import tqdm
import zipfile
import pandas as pd

bio_cuis = []
path = "umls-2023AB-full.zip"
with zipfile.ZipFile(path) as zip_file:
    with zip_file.open("2023AB/META/MRSTY.RRF", mode="r") as file:
        lines = file.readlines()
        for line in tqdm(lines):
            line = str(line)[2:-3].split("|")
            if line[1] == "T059":
                bio_cuis.append(line[0])
bio_cuis = set(bio_cuis)
print(f"LABORATORY PROCEDURE 2023AB: {len(bio_cuis)} concepts")
```

```python
bio_snomed_cuis = snomed_cuis.intersection(bio_cuis)
print(f"LAB SNOMED 2023AB: {len(bio_snomed_cuis)} concepts")
```

```python
import zipfile
from tqdm import tqdm
import pandas as pd

snomed_syn = dict(CUI=[], STR=[])
path = "umls-2023AB-full.zip"
with zipfile.ZipFile(path) as zip_file:
    with zip_file.open("2023AB/META/MRCONSO.RRF", mode="r") as file:
        lines = file.readlines()
        print(f"ALL UMLS 2023AB: {len(lines)} concepts")
        for line in tqdm(lines):
            line = str(line)[2:-3].split("|")
            if line[1] in ["FRE", "ENG"]:
                if line[0] in bio_snomed_cuis:
                    snomed_syn["CUI"].append(line[0])
                    snomed_syn["STR"].append(line[14])
snomed_syn_df = pd.DataFrame(snomed_syn).drop_duplicates()
print(f"SNOMED CT US 2023AB: {len(snomed_syn_df)} synonymes")
```

```python
snomed_syn_df.to_pickle("lab_snomed_ct_2023AB.pkl")
```

## BIO 2021

```python
import zipfile
from tqdm import tqdm

snomed_cuis = []
path = "umls-2021AB-full.zip"
with zipfile.ZipFile(path) as zip_file:
    with zip_file.open("2021AB/META/MRCONSO.RRF", mode="r") as file:
        lines = file.readlines()
        print(f"ALL UMLS 2021AB: {len(lines)} concepts")
        for line in tqdm(lines):
            line = str(line)[2:-3].split("|")
            if line[11] == "SNOMEDCT_US":
                snomed_cuis.append(line[0])
snomed_cuis = set(snomed_cuis)
print(f"SNOMED CT US 2021AB: {len(snomed_cuis)} concepts")
```

```python
from tqdm import tqdm
import zipfile
import pandas as pd

bio_cuis = []
path = "umls-2021AB-full.zip"
with zipfile.ZipFile(path) as zip_file:
    with zip_file.open("2021AB/META/MRSTY.RRF", mode="r") as file:
        lines = file.readlines()
        for line in tqdm(lines):
            line = str(line)[2:-3].split("|")
            if line[1] == "T059":
                bio_cuis.append(line[0])
bio_cuis = set(bio_cuis)
print(f"LABORATORY PROCEDURE 2021AB: {len(bio_cuis)} concepts")
```

```python
bio_snomed_cuis = snomed_cuis.intersection(bio_cuis)
print(f"LAB SNOMED 2021AB: {len(bio_snomed_cuis)} concepts")
```

```python
import zipfile
from tqdm import tqdm
import pandas as pd

snomed_syn = dict(CUI=[], STR=[])
path = "umls-2021AB-full.zip"
with zipfile.ZipFile(path) as zip_file:
    with zip_file.open("2021AB/META/MRCONSO.RRF", mode="r") as file:
        lines = file.readlines()
        print(f"ALL UMLS 2021AB: {len(lines)} concepts")
        for line in tqdm(lines):
            line = str(line)[2:-3].split("|")
            if line[1] in ["FRE", "ENG"]:
                if line[0] in bio_snomed_cuis:
                    snomed_syn["CUI"].append(line[0])
                    snomed_syn["STR"].append(
                        str(
                            line[14]
                            .encode()
                            .decode("unicode-escape")
                            .encode("latin1")
                            .decode("utf8")
                        )
                    )
snomed_syn_df = pd.DataFrame(snomed_syn).drop_duplicates()
print(f"SNOMED CT US 2021AB: {len(snomed_syn_df)} synonymes")
```

```python
snomed_syn_df.to_pickle("lab_snomed_ct_2021AB.pkl")
```

## UMLS FR 2021

```python
import zipfile
from tqdm import tqdm
import pandas as pd

umls_fr_syn = dict(CUI=[], STR=[])
path = "umls-2021AB-full.zip"
with zipfile.ZipFile(path) as zip_file:
    with zip_file.open("2021AB/META/MRCONSO.RRF", mode="r") as file:
        lines = file.readlines()
        print(f"ALL UMLS 2021AB: {len(lines)} concepts")
        for line in tqdm(lines):
            line = str(line)[2:-3].split("|")
            if line[1] in ["FRE"]:
                umls_fr_syn["CUI"].append(line[0])
                umls_fr_syn["STR"].append(
                    str(
                        line[14]
                        .encode()
                        .decode("unicode-escape")
                        .encode("latin1")
                        .decode("utf8")
                    )
                )
umls_fr_syn = pd.DataFrame(umls_fr_syn).drop_duplicates()
print(f"UMLS FR 2021AB: {len(umls_fr_syn)} synonymes")
```

```python
umls_fr_syn.to_json("umls_fr_2021AB.json")
```

```python
umls_fr_syn.groupby(["CUI"], as_index=False).agg({"STR": set}).to_pickle(
    "umls_fr_2021AB.pkl"
)
```

```python
umls_fr_syn[umls_fr_syn["CUI"] == "C0518015"].STR.unique()
```

```python
umls_fr_syn[umls_fr_syn["CUI"] == "C0518015"].STR.unique()
```

```python
umls_fr_syn[umls_fr_syn.STR.str.contains("glycémie")]
```

```python

```
