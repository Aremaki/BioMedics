import typer
from loguru import logger

from biomedics import BASE_DIR
from biomedics.patient_similarity.main import process_and_sort_CRH_similarity
from biomedics.patient_similarity.utils import parse_clinical_case

app = typer.Typer()


@app.command()
def main(config_name: str = "config_study_cortico_v1.cfg"):
    """
    Main function to process clinical cases and find similar patients.
    """
    data_path = BASE_DIR / "data" / "annotated_CRH" / "fictive_clinical_cases"
    cohort_dirs = [d for d in data_path.iterdir() if d.is_dir()]

    for cohort_dir in cohort_dirs:
        cohort_idx = int(cohort_dir.name.split("_")[0])
        logger.info(f"Processing cohort: {cohort_dir.name}")

        for case_file in cohort_dir.glob("*.txt"):
            logger.info(f"Processing file: {case_file.name}")
            clinical_text, cim10_codes, specialties = parse_clinical_case(case_file)

            if clinical_text:
                distances_embedding, icd10_match = process_and_sort_CRH_similarity(
                    clinical_text,
                    specialties,
                    cohort_idx,
                    cim10_codes,
                    config_name=config_name,
                )  # type: ignore
                distances_embedding.to_pickle(
                    f"{cohort_dir}/distances_{case_file.stem}.pkl"
                )  # type: ignore
                icd10_match.to_pickle(f"{cohort_dir}/icd10_match_{case_file.stem}.pkl")


if __name__ == "__main__":
    app()
