# End-to-End Pipeline Usage Guide

This guide will help you configure and run the end2end pipeline for your specific study.

## Step 1: Create a Config File

Before running the pipeline, you must create and modify a configuration file.
Navigate to the config directory:

```
biomedics/configs/end2end/
```

Create a new config file or duplicate an existing one. For example:

```
conf_study_cortico_v1.cfg
```

## Step 2: Set Input and Output Folders

In your configuration file, set the following paths:

In the `infer` section:
- **Input folders**: Path to the CRH files in `.txt` format.
- **Output folders**: Path to an empty folder where the results will be stored.
> ⚠️ Ensure there are as many output folders as input folders.

Then in the `group_brat` section, you must also add the `output_dirs` paths, these paths should point to the BRAT data folders, for example:
```
/export/home/cse200093/brat_data/BioMedics/study_cortico_GF/maladie_de_takayasu
```

## Step 3: Update the Shell Scripts

Next, ensure that all shell scripts use the correct configuration file.
Navigate to the shell scripts directory:

```
biomedics/scripts/end2end/
```

Open each `.sh` file and replace the existing config reference with the name of your config file (e.g., `conf_study_cortico_v1`).

## Step 4: Run the Pipeline

Once everything is configured, you can launch the pipeline by running:

```bash
bash run_end2end.sh
```

## Step 5: Visualize the Results in BRAT

You can visualize your model predictions using the BRAT annotation tool.

1. Open your browser and go to:
   [https://brat-cse200093.eds.aphp.fr](https://brat-cse200093.eds.aphp.fr)

2. Navigate to the folder where you saved the model predictions (as defined in `group_brat`).

This will allow you to inspect the annotated outputs directly in your browser.


## Step 6: Optional: Use the Demo Notebook

You can also explore and interact with the results using the demo notebook provided.

Navigate to:

```
Demo.ipynb
```
