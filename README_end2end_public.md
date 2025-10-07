# End-to-End Pipeline Usage Guide

This guide will help you configure and run the end2end pipeline for your specific study.

## Step 1: Download models and data

### NER + QUALIF model
You have two options:
- **Train your own**: Train an NER + QUALIF model using your annotated dataset, following the instructions in the repository’s `README.md` under the **“NER + QUALIF”** section.

- **Use an off-the-shelf model**: Request access to the finetuned **`eds-biomedics-v4`** model from the Data Science team

Store the chosen model in the appropriate models folder used by the pipeline (e.g., `models/ner/` or the path referenced in your config).

### Normalization model
- We recommend the CODER-all model from Hugging Face: https://huggingface.co/GanjinZero/coder_all
- Download and place it under:
   ```
   models/word_embedding/
   ```

### UMLS data
- Download the full UMLS release (requires a UMLS account and license): https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html
- Prepare and process UMLS according to the repository notebook:
   ```
   data/umls/manage_umls.ipynb
   ```
   Follow that notebook to extract and store the UMLS resources the pipeline expects.

## Step 2: Create a Config File

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

## Step 3: Update the Shell Scripts

Next, ensure that the main shell script use the correct configuration file.
Navigate to the shell script:

```
biomedics/scripts/end2end/run_end2end_public.sh
```

Replace the existing config reference with the name of your config file (e.g., `conf_study_cortico_v1`).

## Step 4: Run the Pipeline

Once everything is configured, you can launch the pipeline by running:

```bash
bash run_end2end_public.sh
```

## Step 5: Visualize the Results in BRAT

You can visualize your model predictions using the BRAT annotation tool.

1. Open your browser and go to:
   [https://brat-cse200093.eds.aphp.fr](https://brat-cse200093.eds.aphp.fr)

2. Navigate to the folder where you saved the model predictions (as defined in `group_brat`).

This will allow you to inspect the annotated outputs directly in your browser.
