# End-to-End Pipeline Usage Guide

This guide will help you configure and run the end2end pipeline for your specific study.

## Step 1: Create a Config File

Before running the pipeline, you must create and modify a configuration file.
Navigate to the config directory:

```
biomedics/configs/end2end/
```

Create a new config file or duplicate an existing one. You can use the following template:

```
config_end_to_end_public.cfg
```

## Step 2: Set Input and Output Folders

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
