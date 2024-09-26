# BioMedics

<div align="center">
    <img src="figures/logo.svg" alt="BioMedics">
<p align="center">
<a href="https://doi.org/10.5281/zenodo.13838918"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.13838918.svg" alt="DOI"></a>
<a href="https://python-poetry.org/" target="_blank">
    <img src="https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json"
    alt="Poetry">
</a>
<a href="https://www.python.org/" target="_blank">
    <img src="https://img.shields.io/badge/python-%3E%3D%203.7.1%20%7C%20%3C%203.8-brightgreen" alt="Supported Python versions">
</a>
<a href="https://spark.apache.org/docs/2.4.8/" target="_blank">
    <img src="https://img.shields.io/badge/spark-2.4-brightgreen" alt="Supported Java version">
</a>
<a href="https://github.com/astral-sh/ruff" target="_blank">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json"
    alt="Ruff">
</a>
</p>
</div>

## Study

This repositoy contains the computer code that has been executed to generate the results of the article:
```
@unpublished{biomedics,
author = {Adam Remaki and Jacques Ung and Pierre Pages and Perceval Wajsburt and Guillaume Faure and Thomas Petit-Jean and Xavier Tannier and Christel Gérardin},
title = {Improving phenotyping of patients with immune-mediated inflammatory diseases through automated processing of discharge summaries: a multicenter cohort study},
note = {Manuscript submitted for publication},
year = {2024}
}
```

## Data and model availability

The data and trained models are available via a secure Jupyter interface in a kubernetes cluster. Access to the clinical data warehouse's raw data can be granted following the process described on its website: [www.eds.aphp.fr](https://eds.aphp.fr/). A prior validation of the access by the local institutional review board is required (**IRB number: CSE200093**). In the case of non-AP-HP researchers, the signature of a collaboration contract is mandatory.

If you are AP-HP researchers, you can use your own data in BRAT format.

## Overall pipelines

BioMedics stands on the shoulders of the library [EDS-NLP](https://github.com/aphp/edsnlp) (a collaborative NLP framework that aims primarily at extracting information from French clinical notes).
BioMedics aims specifically at extracting laboratory test and drug information from clinical note. It consists of two pipelines:

<img src="figures/overall_pipelines.svg" alt="overall_pipelines">

## How to run the code on AP-HP's data platform

### 1. Setup

- In order to process large-scale data, the study uses [Spark 2.4](https://spark.apache.org/docs/2.4.8/index.html) (an open-source engine for large-scale data processing) which requires to:

  - Install a version of Python $\geq 3.7.1$ and $< 3.8$.
  - Install Java 8 (you can install [OpenJDK 8](https://openjdk.org/projects/jdk8/), an open-source reference implementation of Java 8)

- Clone the repository:

```shell
git clone https://github.com/Aremaki/BioMedics.git
```

- Create a virtual environment with the suitable Python version (**>= 3.7.1 and < 3.8**):

```shell
cd biomedics
python -m venv .venv
source .venv/bin/activate
```

- Install [Poetry](https://python-poetry.org/) (a tool for dependency management and packaging in Python) with the following command line:
    - Linux, macOS, Windows (WSL):

    ```shell
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    - Windows (Powershell):

    ```shell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    ```

    For more details, check the [installation guide](https://python-poetry.org/docs/#installation)

- Install dependencies:

```shell
poetry install
```

### 2. Extract cohort:

This step will extract the study cohort with spark query and will save the discharge summaries in BRAT format in the `data/CRH/raw` folder. If you use another cohort you can skip this step and place your data in the same folder.

- Install EDS-Toolbox (a python library that provides an efficient way of submitting PySpark scripts on AP-HP's data platform. As it is AP-HP specific, it is not available on PyPI):

```shell
pip install edstoolbox
```

- Run the script with EDS-Toolbox:

```shell
cd scripts/create_dataset
bash run.sh
```

### 3. NER + Qualification

The model used for NER and qualification comes from [EDS-Medic](https://gitlab.eds.aphp.fr/entrep-t-de-donn-es-de-sant/eds-tools/datasciencetools/eds-medic) which have been developed by the data science team of AP-HP (Assistance Publique – Hôpitaux de Paris).

<img src="figures/ner_qualif_model.svg" alt="ner_qualif_model">

> Requirements:
- The training of this model requires annotated discharge summaries in BRAT format.
- The training also requires a word embedding model, we recommend to use [EDS-CamemBERT-fine-tuned](https://arxiv.org/abs/2207.12940) available in the AP-HP's model catalogue or [CamemBERT-bio](https://huggingface.co/almanach/camembert-bio-base) available on Hugging Face. Place it in the `models/word_embedding` folder.
- Modify or create your own configuration in `configs/ner/<your_config>.cfg`: You can set the path of your training and testing data folder, set the hyperparameters of the model, set the training parameters...

> Run:
Training, evaluation and inference are gathered into one sbatch but you can comment the part you would like to skip in `run.sh`:

```shell
cd scripts/ner
sbtach run.sh
```

### 4. Extract measurement

It aims at extracting values and units from complete laboratory test in the text. It is based on regular expression processed at very high speed with spark engine. The algorithm has been implemented by the data science team of Pitié-Salpêtrière Hospital.

<img src="figures/extract_measurement.svg" alt="extract_measurement">

```shell
cd scripts/extract_measurement
sbtach run.sh
```

### 5. Normalization

#### Laboratory tests

The normalization for laboratory test is based on CODER, a BERT-based model finetuned on the UMLS:

<img src="figures/normalization_lab.svg" alt="normalization_lab">

> Requirements:
- Download a word embedding model, we recommend to use [CODER-all](https://huggingface.co/GanjinZero/coder_all) available on Hugging Face. Place it in the `models/word_embedding` folder.
- Download the [full release](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html) of UMLS and follow `data/umls/manage_umls.ipynb` notebook.

#### Drugs

The normalization for drug names is a fuzzy matching on a knowledge dictionnary. This dictionary is an aggregation of two open source dictionaries of drug names with their corresponding ATC codes: the UMLS restricted to the ATC vocabulary and the Unique Drug Interoperability Repository (RUIM) created by the French National Agency for Medicines and Health Products Safety (ANSM):

<img src="figures/normalization_drug.svg" alt="normalization_drug">

> Requirement:
- Run `python data/drug_knowledge/dic_generation.py` in order to create the drug knowledge dictionary.

> Run:
Both normalizations are gathered into one sbatch but you can comment the part you would like to skip in `run.sh`:

```shell
cd scripts/normalization/
sbtach run.sh
```
### 6. Filter results for clinical application

This step filter results according to the laboratory tests and drug treatments studied. If you are studying other laboratory tests or drugs, you have to modify this step.

```shell
cd scripts/clinical_application
sbtach run.sh
```

### 7. Generate figures

Generate result figures of the paper from notebooks:

  - Create a Spark-enabled kernel with your environnement:

    ```shell
    eds-toolbox kernel --spark --hdfs
    ```

   - Convert markdown into jupyter notebook:

      ```shell
      cd notebooks
      jupytext --set-formats md,ipynb 'clinical_results.md'
      ```

   - Open *.ipynb* and start the kernel you've just created.
     - Run the cells to obtain the table results.

## Project structure

- `conf`: Configuration files.
- `data`: Saved processed data and knowledge dictionnaries.
- `models`: Trained models.
- `figures`: Saved results.
- `notebooks`: Notebooks that generate figures.
- `biomedics`: Source code.
- `scripts`: Scripts to process data.

## Acknowledgement

We would like to thank [AI4IDF](https://ai4idf.fr/) for funding this project, the data science teams of [Assistance Publique – Hôpitaux de Paris](https://www.aphp.fr/) for contributing to the project.
