# End-to-End Pipeline Usage Guide

This guide will help you configure and run the end2end pipeline for your specific study.

## Step 1: Clone repository

- Clone the repository:
```shell
git clone https://github.com/Aremaki/BioMedics.git
cd BioMedics
```

## Step 2: Python environment

In order to process large-scale data, the study uses [Spark 2.4](https://spark.apache.org/docs/2.4.8/index.html) (an open-source engine for large-scale data processing) which requires to:

- Install a version of Python $\geq 3.7.1$ and $< 3.8$. For instance you can use conda:
   ```shell
   conda create -n py37 python=3.7.16
   conda activate py37
   ```

- Create a virtual environment with the suitable Python version (**>= 3.7.1 and < 3.8**) in the **root of the project** (BioMedics):
   ```shell
   python -m venv .venv
   conda deactivate
   source .venv/bin/activate
   ```

- Install [Poetry](https://python-poetry.org/) (a tool for dependency management and packaging in Python) with the following command line:
   ```shell
   pip install poetry==1.5.1
   ```

- Install dependencies:
   ```shell
   pip install pypandoc==1.7.5
   pip install pyspark==2.4.8
   pip install "edsnlp[ml] @ git+https://github.com/Aremaki/edsnlp.git@rule_based_relation"
   poetry install
   pip uninstall pypandoc
   ```

- Install EDS-Toolbox (a python library that provides an efficient way of submitting PySpark scripts on AP-HP's data platform. As it is AP-HP specific, it is not available on PyPI):

```shell
pip install edstoolbox==0.7.0
```

## Step 3: Download models and data

### NER + QUALIF model
You have two options:
- **Train your own**: Train an NER + QUALIF model using your annotated dataset, following the instructions in the repository’s `README.md` under the **“NER + QUALIF”** section.

- **Use an off-the-shelf model**: Request access to the finetuned **`eds-biomedics-v5`** model from the Data Science team

Store the chosen model in the appropriate models folder used by the pipeline (e.g., `models/ner/` or the path referenced in your config).

### Normalization model
- We recommend the SapBERT-all model from Hugging Face: https://huggingface.co/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR. Download and place it under model/word_embedding/SapBERT_all.

```shell
python -c "from huggingface_hub import snapshot_download; \
snapshot_download(
    repo_id='GanjinZero/coder_all',
    local_dir='models/word_embedding/coder_all',
    local_dir_use_symlinks=False
)"
```

### UMLS data
- Download the full UMLS release (requires a UMLS account and license): https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html
- Prepare and process UMLS according to the repository notebook:
   ```
   data/umls/manage_umls.ipynb
   ```
   Follow that notebook to extract and store the UMLS resources the pipeline expects.

## Step 4: Create a Config File

Before running the pipeline, you must create and modify a configuration file.
Navigate to the config directory:

```
biomedics/configs/end2end/
```

Create a new config file or duplicate an existing one. You can use the following template:

```
config_end_to_end_public.cfg
```

In your configuration file, set the following paths:

In the `vars` section:
- **root_dir**: "/export/home/<YOUR_CSE>"
- **base_dir**: Path to BioMedics folder
In the `infer` section:
- **input folders**: Specify the path to the folders containing your CRH files in `.txt` format.
   > **Note:** Each folder must also include an empty `.ann` file for every `.txt` document.
- **output folders**: Path to an empty folder where the results will be stored.
> ⚠️ Ensure there are as many output folders as input folders.
In the `group_brat` section:
- **output_dirs**: Specify the paths to the folders where BRAT will store the annotated results.

**IMPORTANT**: The script might not detect the $vars.base_dire and $vars.root_dir. If you have errors, you can copy paste the path for each value of $vars in the config

## Step 5: Update the Shell Scripts

Next, ensure that the main shell script use the correct configuration file.
Navigate to the shell script:

```
biomedics/scripts/end2end/run_end2end_public.sh
```

Replace the existing config reference with the name of your config file (e.g., `conf_study_cortico_v1`).

## Step 6: Run the Pipeline

Once everything is configured, you can launch the pipeline by running:

```bash
bash run_end2end_public.sh
```

## Step 7: Visualize the Results in BRAT

You can visualize your model predictions using the BRAT annotation tool.

1. Open your browser and go to:
   [https://brat-cseXXXXXX.eds.aphp.fr](https://brat-cse200093.eds.aphp.fr)

2. Navigate to the folder where you saved the model predictions (as defined in `group_brat`).

This will allow you to inspect the annotated outputs directly in your browser.
