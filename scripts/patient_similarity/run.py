import shutil
from pathlib import Path

import typer
from confection import Config
from loguru import logger

from biomedics import BASE_DIR
from biomedics.patient_similarity.main import process_and_sort_CRH_similarity
from biomedics.patient_similarity.utils import parse_clinical_case

app = typer.Typer()


@app.command()
def main(config_name: str = "config_patient_similarity.cfg"):
    """
    Main function to process clinical cases and find similar patients.
    """
    data_path = BASE_DIR / "data" / "annotated_CRH" / "fictive_clinical_cases"
    config_path = BASE_DIR / "configs" / "end2end" / config_name
    config = Config().from_disk(config_path, interpolate=True)
    cohort_dirs = [
        d for d in data_path.iterdir() if d.is_dir() and d.name.split("_")[0].isdigit()
    ]
    disease_index = {
        "0_takayasu_arteritis": "maladie_de_takayasu",
        "1_systemic_sclerosis": "sclerodermie_systemique",
        "2_antiphospholipid_syndrome": "syndrome_des_anti_phospholipides",
        "3_systemic_lupus_erythematosus": "lupus_erythemateux_dissemine",
    }
    for cohort_dir in cohort_dirs:
        cohort_idx = int(cohort_dir.name.split("_")[0])
        logger.info(f"Processing cohort: {cohort_dir.name}")

        for case_file in cohort_dir.glob("*.txt"):
            logger.info(f"Processing file: {case_file.name}")
            _, cim10_codes, specialties = parse_clinical_case(case_file)

            distances_embedding, icd10_match = process_and_sort_CRH_similarity(
                case_file.stem,
                specialties,
                cohort_idx,
                cim10_codes,
                config_name=config_name,
                seed=int(
                    case_file.name.split(".")[0].split("_")[
                        -1
                    ],  # Use case number as seed
                ),
            )

            distances_embedding.to_pickle(
                f"{cohort_dir}/distances_{case_file.stem}.pkl"
            )
            top_similar_notes = (
                distances_embedding["source"].str.split(".").str[0].head(100).tolist()
            )
            note_to_annotate = (
                distances_embedding[distances_embedding.chosen]["source"]
                .str.split(".")
                .str[0]
                .tolist()
            )

            # Create directory for BRAT annotations
            brat_data_path = Path(config["group_brat"]["conf_path"])
            fictive_case_dir = (
                brat_data_path / "fictive_clinical_cases" / cohort_dir.name
            )
            MIE_folder_annotated = (
                brat_data_path / "MIE_annotated" / case_file.name.split(".")[0]
            )
            MIE_folder = brat_data_path / "MIE" / case_file.name.split(".")[0]
            # Delete and recreate the folders if they already exist
            if MIE_folder_annotated.exists():
                shutil.rmtree(MIE_folder_annotated)
            if MIE_folder.exists():
                shutil.rmtree(MIE_folder)
            MIE_folder_annotated.mkdir(parents=True, exist_ok=True)
            MIE_folder.mkdir(parents=True, exist_ok=True)

            # Convert Doc to BRAT format
            shutil.copy(
                brat_data_path / "annotation.conf",
                MIE_folder_annotated / "annotation.conf",
            )
            shutil.copy(
                brat_data_path / "annotation.conf",
                MIE_folder / "annotation.conf",
            )
            shutil.copy(
                brat_data_path / "kb_shortcuts.conf",
                MIE_folder_annotated / "kb_shortcuts.conf",
            )
            shutil.copy(
                brat_data_path / "kb_shortcuts.conf",
                MIE_folder / "kb_shortcuts.conf",
            )
            shutil.copy(
                brat_data_path / "visual.conf",
                MIE_folder_annotated / "visual.conf",
            )
            shutil.copy(
                brat_data_path / "visual.conf",
                MIE_folder / "visual.conf",
            )

            # Copy case file to MIE folder
            shutil.copy(
                fictive_case_dir / f"case_{case_file.stem}.txt",
                MIE_folder / "fictive_case.txt",
            )
            (MIE_folder / "fictive_case.ann").touch()

            # Copy case file to MIE annotated folder
            shutil.copy(
                fictive_case_dir / f"case_{case_file.stem}.txt",
                MIE_folder_annotated / "fictive_case.txt",
            )
            shutil.copy(
                fictive_case_dir / f"case_{case_file.stem}.ann",
                MIE_folder_annotated / "fictive_case.ann",
            )
            # Copy BRAT note from folder in brat_data
            brat_note_path = (
                brat_data_path
                / "study_cortico_GF"
                / f"{disease_index[cohort_dir.name]}"
            )
            for note in note_to_annotate:
                # Search for the note file in all the directries
                for sub_folder in brat_note_path.iterdir():
                    note_path = sub_folder / f"{note}.txt"
                    ann_path = sub_folder / f"{note}.ann"
                    if note_path.exists() and ann_path.exists():
                        shutil.copy(note_path, MIE_folder_annotated / f"{note}.txt")
                        # Add text at the end of the file to indicate the rank, the distances
                        with open(MIE_folder_annotated / f"{note}.txt", "a") as f:
                            f.write(
                                f"\n\n# Similarity rank: {distances_embedding[distances_embedding['source'] == note + '.ann']['rank'].values[0]}\n"
                            )
                            # Add similarity distance for each selected label
                            for specialty in specialties:  # type: ignore
                                if specialty in distances_embedding.columns:
                                    f.write(
                                        f"# Similarity distance ({specialty}): {distances_embedding[distances_embedding['source'] == note + '.ann'][specialty].values[0]}\n"
                                    )
                            f.write(
                                f"# Mean distance: {distances_embedding[distances_embedding['source'] == note + '.ann']['mean'].values[0]}\n"
                            )
                            f.write(
                                f"# Proba: {distances_embedding[distances_embedding['source'] == note + '.ann']['proba'].values[0]}\n"
                            )
                            f.write(
                                f"# Bucket: {distances_embedding[distances_embedding['source'] == note + '.ann']['bucket'].values[0]}\n"
                            )
                        shutil.copy(note_path, MIE_folder / f"{note}.txt")
                        shutil.copy(ann_path, MIE_folder_annotated / f"{note}.ann")
                        # Create empty .ann file in MIE folder
                        (MIE_folder / f"{note}.ann").touch()
                        break

            for rank, note in enumerate(top_similar_notes):
                # copy note from folder with raw CRH to cohort_dir
                raw_note_path = (
                    BASE_DIR
                    / "data"
                    / "study_cortico_GF"
                    / disease_index[cohort_dir.name]
                    / "raw_CRH"
                    / f"{note}.txt"
                )
                if raw_note_path.exists():
                    # Create the folder if it does not exist
                    (
                        cohort_dir / f"top_similar_note_{case_file.name.split('.')[0]}"
                    ).mkdir(parents=True, exist_ok=True)
                    shutil.copy(
                        raw_note_path,
                        f"{cohort_dir}/top_similar_note_{case_file.name.split('.')[0]}/top_{rank+1}.txt",
                    )
            icd10_match.to_pickle(f"{cohort_dir}/icd10_match_{case_file.stem}.pkl")


if __name__ == "__main__":
    app()
