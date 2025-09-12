import warnings
from pathlib import Path

import edsnlp
import pandas as pd
import torch
from confection import Config
from sklearn.feature_extraction import DictVectorizer
from transformers import CamembertForSequenceClassification, CamembertTokenizer

from biomedics import BASE_DIR
from biomedics.normalization.embedding_similarity.get_normalization_with_embedding import (
    EmbeddingNormalizer,
)
from biomedics.normalization.embedding_similarity.text_preprocessor import (
    TextPreprocessor,
)
from biomedics.patient_similarity.utils import (
    add_atc_code,
    add_label_class,
    compute_distance,
    create_source_terms,
)

warnings.filterwarnings("ignore")


def process_and_sort_CRH_similarity(
    medical_text,
    selected_specialties,
    cohort_idx,
    cim10_codes,
    config_name: str = "config_study_cortico_v1.cfg",
):
    """
    Processes a medical text to find similar patients.
    """
    config_path = BASE_DIR / "configs" / "end2end" / config_name
    config = Config().from_disk(config_path, interpolate=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stopwords = config["emdedding_similarity"]["stopwords"]
    text_preprocessor = TextPreprocessor(cased=False, stopwords=stopwords)
    embedding_normalizer = EmbeddingNormalizer(
        model_name_or_path=config["emdedding_similarity"]["model_path"],
        tokenizer_name_or_path=config["emdedding_similarity"]["model_path"],
        device=device,
    )

    # Load NER model
    nlp = edsnlp.load(config["infer"]["model_path"]).to(device)

    # Normalization data
    drug_dict = pd.read_pickle(config["fuzzy_matching"]["drug_dict_path"])
    atc_len = 7
    drug_df = {}
    for atc_code, values in drug_dict.items():
        shortened_code = atc_code[:atc_len]
        if shortened_code in drug_df:
            drug_df[shortened_code] = list(set(drug_df[shortened_code] + values))
        else:
            drug_df[shortened_code] = values
    drug_df = (
        pd.DataFrame.from_dict({"norm_term": drug_df}, "index")
        .T.explode("norm_term")
        .reset_index()
        .rename(columns={"index": "label"})
    )
    drug_df.norm_term = drug_df.norm_term.str.split(",")
    drug_df = drug_df.explode("norm_term").reset_index(drop=True)

    # Classifier
    labels_path = config["classify_diso"]["labels_path"]
    classif_model_path = config["classify_diso"]["classif_model_path"]
    with open(labels_path, "r") as f_out:
        label_names = f_out.readline().strip().split(",")
    num_labels = len(label_names)
    model = CamembertForSequenceClassification.from_pretrained(
        classif_model_path, num_labels=num_labels
    ).to(device)  # type: ignore
    tokenizer = CamembertTokenizer.from_pretrained(classif_model_path)

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

    # Run NLP model
    doc = nlp(medical_text)
    doc = add_atc_code(doc, drug_df, text_preprocessor)
    doc = add_label_class(doc, model, tokenizer, text_preprocessor, label_names, device)

    source_patient = create_source_terms(doc, selected_specialties, text_preprocessor)
    if not source_patient or not selected_specialties:
        raise ValueError("No valid source patient or specialties found.")

    predicted_entities = [
        text_preprocessor(
            text=ent.text, remove_stopwords=True, remove_special_characters=True
        )
        for ent in doc.spans.get("Signe et symptôme", [])
        if set(ent.kb_id_.split(" | ")).intersection(set(selected_specialties))
    ]
    new_terms = list(set(predicted_entities).difference(set(df_embed.normalized_term)))
    if new_terms:
        new_embeddings = embedding_normalizer.get_bert_embed(
            new_terms,
            normalize=True,
            summary_method="CLS",
            tqdm_bar=False,
            batch_size=2,
        )
        new_embeddings = new_embeddings.to(torch.float16).cpu().numpy()
        new_embeddings = pd.DataFrame(new_embeddings)
        new_embeddings["normalized_term"] = new_terms
        df_embed = pd.concat([df_embed, new_embeddings])

    distances_embedding = compute_distance(
        source_patient,
        target_patients,
        df_embed,
        vectorizer,
        selected_specialties,
    )
    # Add a column with rank value
    distances_embedding = distances_embedding.sort_values(
        by="similarity_distance", ascending=True
    )
    distances_embedding["rank"] = range(1, len(distances_embedding) + 1)

    return distances_embedding, icd10_match, doc
