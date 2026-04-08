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

- Install EDS-Toolbox (a python library that provides an efficient way of submitting PySpark scripts on AP-HP's data platform. As it is AP-HP specific, it is not available on PyPI):

```shell
pip install git+https://gitlab.eds.aphp.fr/entrep-t-de-donn-es-de-sant/eds-tools/datasciencetools/eds-toolbox.git
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
    repo_id='cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR',
    local_dir='models/word_embedding/sapbert_all',
    local_dir_use_symlinks=False
)"
```

### UMLS data
**Download the UMLS Metathesaurus Full Subset** (requires a UMLS account and license — [request access here](https://uts.nlm.nih.gov/uts/signup-login))

The fastest way to download it directly to your CSE is via `curl`:

```bash
curl "https://uts-ws.nlm.nih.gov/download?url=https://download.nlm.nih.gov/umls/kss/<year><version>/umls-<year><version>-metathesaurus-full.zip&apiKey=<YOUR_API_KEY>" \
  -o <your_path>/umls-<year><version>.zip
```

Replace the placeholders before running:
- `<YOUR_API_KEY>` — found in [your UMLS profile](https://uts.nlm.nih.gov/uts/profile) once your account is approved
- `<year>` and `<version>` — the UMLS release you want (e.g. `2025` and `AB` for the 2025AB release)
- `<your_path>` the path where you want to save the zip file, it can be :`BioMedics/data/umls`

**Prepare and process** UMLS according to the repository notebook:
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

Create a new config file from the following template:

```
config_end_to_end_other_cse.cfg
```

In your configuration file (`vars` section), set:

- **input_folder**: Path to the folder containing the CRH `.txt` files (subfolders are allowed).
- **ner_model_path**: Path to the BioMedics NER model.
- **normalization_model_path**: Path to the Normalization model (sapbert_all).
- **drug_dict_path**: Path to the BioMedics drug dictionary (data/drug_knowledge/final_dict.pkl).
- **lab_test_termino_path**: Only the name of your lab test dictionary downloaded from UMLS (lab_snomed_ct_<year><version>.csv).
- **brat_config_path**: Path to the BRAT config (configs/brat_data)
- **brat_output_folder**: Path to a folder inside `brat_data` where BRAT files will be saved.

## Step 5: Update the Slurm files with your own GPU parameters

Depending on what GPU you have access to, you may need to update the slurm files. Navigate to the slurm scripts:

```
cd scripts/end2end
```

For each file `*.slurm``, update the SLURM parameters if needed.

Example:

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
```
## Step 6: Run the Pipeline

Once everything is configured, you can launch the pipeline by running:

```bash
export config="<Your_config_name>.cfg"
bash run_end2end_other_cse.sh
```

## Step 7: Visualize the Results in BRAT

When you run the algorithm, it will:

1. Save all predictions as tables in a folder called `pred_NORM` at the root of the input folder.
2. Save all predictions in BRAT format in the specified BRAT output folder with NER and NORM.

You can visualize your model predictions using the BRAT annotation tool.

1. Open your browser and go to:
   [https://brat-cseXXXXXX.eds.aphp.fr](https://brat-cse200093.eds.aphp.fr)

2. Navigate to the folder where you saved the model predictions (as defined in `group_brat`).

This will allow you to inspect the annotated outputs directly in your browser.
