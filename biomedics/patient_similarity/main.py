import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from confection import Config
from kneed import KneeLocator
from sklearn.feature_extraction import DictVectorizer

from biomedics import BASE_DIR
from biomedics.patient_similarity.utils import (
    compute_distance,
    get_counts_for_source,
    stratified_sample_indices,
)

warnings.filterwarnings("ignore")


def process_and_sort_CRH_similarity(
    case_name: str,
    selected_specialties,
    cohort_idx,
    cim10_codes,
    config_name: str = "config_patient_similarity.cfg",
    seed: int = 42,
):
    """
    Processes a medical text to find similar patients.
    """
    config_path = BASE_DIR / "configs" / "end2end" / config_name
    config = Config().from_disk(config_path, interpolate=True)

    # Classifier
    df_diso_class = pd.read_pickle(
        Path(config["infer"]["output_folders"][cohort_idx]).parent
        / "pred_with_classified_diso.pkl"
    )
    source_patient = get_counts_for_source(df_diso_class, f"{case_name}.ann")
    if not source_patient or not selected_specialties:
        raise ValueError("No valid source patient or specialties found.")

    # Vectorizer for patient distance
    vectorizer = DictVectorizer(sparse=True)

    # Embeddings
    output_folder = Path(config["infer"]["output_folders"][cohort_idx]).parent
    df_embed = pd.read_pickle(f"{output_folder}/pred_diso_embedding.pkl")
    df_embed = df_embed.drop(columns=["scores", "labels"])
    target_patients = pd.read_pickle(f"{output_folder}/pred_with_classified_diso.pkl")
    target_patients.labels = target_patients.labels.str.split(r" \| ")
    target_patients = target_patients.explode("labels")
    outcomes = pd.read_pickle(f"{output_folder}/outcomes.pkl")
    # compute a df for each source the number of icd10_codes starting with the cim_codes input
    cim10_codes = [
        "CIM10:" + code.split(" : ")[0].replace(".", "") for code in cim10_codes
    ]
    outcomes["icd10_codes"] = outcomes["icd10_codes"].where(
        outcomes["icd10_codes"].isna(),
        outcomes["icd10_codes"].astype(str).str.split("|").str[0],
    )
    outcomes["matched_icd10_codes"] = outcomes["icd10_codes"].apply(
        lambda codes: [
            code
            for code in codes
            if any(code.startswith(cim_code) for cim_code in cim10_codes)
        ]
    )
    outcomes["num_icd10_match"] = outcomes["matched_icd10_codes"].apply(len)
    outcomes["target_icd10_codes"] = [cim10_codes] * len(outcomes)
    icd10_match = outcomes[
        [
            "source",
            "matched_icd10_codes",
            "num_icd10_match",
            "icd10_codes",
            "target_icd10_codes",
        ]
    ]

    # Diso Embeddings
    df_diso_class = pd.read_pickle(
        Path(config["infer"]["output_folders"][cohort_idx]).parent
        / "pred_diso_embedding.pkl"
    )

    # Add new terms to df_embed if not already present
    new_embeddings = df_diso_class[
        ~df_diso_class["normalized_term"].isin(df_embed["normalized_term"])
    ]
    df_embed = pd.concat([df_embed, new_embeddings])

    distances_embedding = compute_distance(
        source_patient,
        target_patients,
        df_embed,
        vectorizer,
        selected_specialties,
    )

    # Add a column with rank value
    distances_embedding = distances_embedding.sort_values(by="mean", ascending=True)
    distances_embedding["rank"] = range(1, len(distances_embedding) + 1)

    # Determine the knee point to filter patients
    x = distances_embedding["rank"].values
    y = distances_embedding["mean"].values
    kneedle = KneeLocator(
        x,
        y,
        curve="concave",  # or "concave" depending on your plot
        direction="increasing",  # or "increasing"
        interp_method="polynomial",
    )

    threshold = kneedle.knee if kneedle.knee else len(distances_embedding)
    distances_embedding["threshold"] = threshold
    distances_embedding["similar"] = distances_embedding["rank"] <= threshold

    # Normalize cosine scores into a probability distribution
    distances_embedding["proba_cosine"] = 1 - distances_embedding["mean"]
    proba_total_top = (
        distances_embedding[distances_embedding["rank"] <= threshold][
            "proba_cosine"
        ].sum()
        * 2
    )
    proba_total_bottom = (
        distances_embedding[distances_embedding["rank"] > threshold][
            "proba_cosine"
        ].sum()
        * 2
    )
    distances_embedding["proba_cosine"] = distances_embedding["proba_cosine"].mask(
        distances_embedding["rank"] > threshold,
        distances_embedding["proba_cosine"] / proba_total_bottom,
    )
    distances_embedding["proba_cosine"] = distances_embedding["proba_cosine"].mask(
        distances_embedding["rank"] <= threshold,
        distances_embedding["proba_cosine"] / proba_total_top,
    )

    # Add AP score as another probability distribution
    Z = len(distances_embedding)
    distances_embedding["proba_elbow_mixed"] = (1.0 / (2.0 * Z)) * np.log(
        Z / distances_embedding["rank"]
    )
    proba_total_top = (
        distances_embedding[distances_embedding["rank"] <= threshold][
            "proba_elbow_mixed"
        ].sum()
        * 2
    )
    proba_total_bottom = (
        distances_embedding[distances_embedding["rank"] > threshold][
            "proba_elbow_mixed"
        ].sum()
        * 2
    )
    distances_embedding["proba_elbow_mixed"] = distances_embedding[
        "proba_elbow_mixed"
    ].mask(
        distances_embedding["rank"] > threshold,
        distances_embedding["proba_elbow_mixed"] / proba_total_bottom,
    )
    distances_embedding["proba_elbow_mixed"] = distances_embedding[
        "proba_elbow_mixed"
    ].mask(
        distances_embedding["rank"] <= threshold,
        distances_embedding["proba_elbow_mixed"] / proba_total_top,
    )

    # Add AP score as another probability distribution
    Z = len(distances_embedding)
    distances_embedding["proba_AP"] = (1.0 / (2.0 * Z)) * np.log(
        Z / distances_embedding["rank"]
    )
    distances_embedding["proba_AP"] /= distances_embedding["proba_AP"].sum()

    # Add probability distribution
    distances_embedding["proba"] = 1 - distances_embedding["mean"]
    proba_total_top = distances_embedding[distances_embedding["rank"] <= threshold][
        "proba"
    ].sum()
    distances_embedding["proba"] = distances_embedding["proba"].mask(
        distances_embedding["rank"] > threshold,
        0.0,
    )
    distances_embedding["proba"] = distances_embedding["proba"].mask(
        distances_embedding["rank"] <= threshold,
        distances_embedding["proba"] / proba_total_top,
    )

    distances_embedding = stratified_sample_indices(
        distances_embedding, m=10, seed=seed
    )

    return distances_embedding, icd10_match
