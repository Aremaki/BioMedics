# End-to-End Pipeline Usage Guide

This guide will help you configure and run the end2end pipeline for your specific study.

## Step 1: Create a Config File

Before running the pipeline, you must create and modify a configuration file.
Navigate to the config directory:

```
BioMedics/configs/end2end/
```

Create a new config file from the following template:

```
config_end_to_end_cse200093_template.cfg
```

## Step 2: Set Input and Output Folders

In your configuration file (`vars` section), set:

* **Input folder**: Path to the folder containing the CRH `.txt` files (subfolders are allowed).
* **BRAT output folder**: Path to a folder inside `brat_data` where BRAT files will be saved.

> ⚠️ `brat_data` is the only folder accessible by BRAT.


## Step 3: Run the Pipeline

Once everything is configured, you can launch the pipeline by running:

```bash
cd scripts/end2end
export conifg="<Your_config_name>.cfg"
bash run_end2end_cse200093.sh
```

## Step 4: Visualize the Results in BRAT

When you run the algorithm, it will:

1. Save all predictions as tables in a folder called `pred_NORM` at the root of the input folder.
2. Save all predictions in BRAT format in the specified BRAT output folder with NER and NORM.

You can visualize your model predictions using the BRAT annotation tool.

1. Open your browser and go to:
   [https://brat-cse200093.eds.aphp.fr](https://brat-cse200093.eds.aphp.fr)

2. Navigate to the brat output folder.

This will allow you to inspect the annotated outputs directly in your browser.
