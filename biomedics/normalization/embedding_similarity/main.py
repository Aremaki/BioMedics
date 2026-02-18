import os

os.environ["OMP_NUM_THREADS"] = "16"
import pickle
from pathlib import Path
from typing import List

import pandas as pd

from biomedics import BASE_DIR
from biomedics.normalization.embedding_similarity.get_normalization_with_embedding import (
    EmbeddingNormalizer,
)
from biomedics.normalization.embedding_similarity.text_preprocessor import (
    TextPreprocessor,
)


def get_embedding_similarity(
    df,
    model_path: Path,
    input_dirs: List[Path],
    umls_path: str,
    labels_column_name: str,
    synonyms_column_name: str,
    column_name_to_normalize: str,
    model_device: str,
    summary_method: str,
    batch_size: int,
    tqdm_bar: bool,
    save_umls_embeddings_dir: bool,
    save_umls_des_dir: bool,
    save_umls_labels_dir: bool,
    save_data_embeddings_dir: bool,
    normalize: bool,
    cased: bool,
    stopwords: List[str],
    remove_stopwords_terms: bool,
    remove_special_characters_terms: bool,
    remove_stopwords_umls: bool,
    remove_special_characters_umls: bool,
):
    # This wrapper is needed to preprocess terms
    # and in case the cells contains list of terms instead of one unique term
    df = df.reset_index(drop=True)
    text_preprocessor = TextPreprocessor(cased=cased, stopwords=stopwords)
    embedding_normalizer = EmbeddingNormalizer(
        model_name_or_path=model_path,  # type: ignore
        tokenizer_name_or_path=model_path,  # type: ignore
        device=model_device,
    )

    # Preprocess UMLS
    print("--- Preprocessing UMLS ---")
    umls_df = pd.read_csv(BASE_DIR / "data" / "umls" / umls_path)
    umls_df[synonyms_column_name] = umls_df[synonyms_column_name].apply(
        lambda term: text_preprocessor(
            text=term,
            remove_stopwords=remove_stopwords_umls,
            remove_special_characters=remove_special_characters_umls,
        )
    )
    umls_df = (
        umls_df.loc[
            (~umls_df[synonyms_column_name].str.isnumeric())
            & (umls_df[synonyms_column_name] != "")
        ]
        .groupby([synonyms_column_name])
        .agg({labels_column_name: set, synonyms_column_name: "first"})
        .reset_index(drop=True)
    )
    umls_des_list = umls_df[synonyms_column_name]
    umls_labels_list = umls_df[labels_column_name]
    if save_umls_des_dir:
        with open(save_umls_des_dir, "wb") as f:
            pickle.dump(umls_des_list, f)
    if save_umls_labels_dir:
        with open(save_umls_labels_dir, "wb") as f:
            pickle.dump(umls_labels_list, f)

    # Preprocessing and inference on terms
    print("--- Preprocessing terms ---")
    if type(df[column_name_to_normalize].iloc[0]) is str:
        data_list = (
            df[column_name_to_normalize]
            .apply(
                lambda term: text_preprocessor(
                    text=term,
                    remove_stopwords=remove_stopwords_terms,
                    remove_special_characters=remove_special_characters_terms,
                )
            )
            .tolist()
        )
        print("--- MODEL inference ---")
        res = embedding_normalizer(
            umls_labels_list=umls_labels_list,
            umls_des_list=umls_des_list,
            data_list=data_list,
            save_umls_embeddings_dir=save_umls_embeddings_dir,
            save_data_embeddings_dir=save_data_embeddings_dir,
            normalize=normalize,
            summary_method=summary_method,
            tqdm_bar=tqdm_bar,
            batch_size=batch_size,
        )
        df[["label", "norm_term", "score"]] = pd.DataFrame(zip(*res))
    else:
        exploded_term_df = (
            pd.DataFrame(
                {"id": df.index, column_name_to_normalize: df[column_name_to_normalize]}
            )
            .explode(column_name_to_normalize)
            .reset_index(drop=True)
        )
        data_list = (
            exploded_term_df[column_name_to_normalize]
            .apply(
                lambda term: text_preprocessor(
                    text=term,
                    remove_stopwords=remove_stopwords_terms,
                    remove_special_characters=remove_special_characters_terms,
                )
            )
            .tolist()
        )
        print("--- MODEL inference ---")
        res = embedding_normalizer(
            umls_labels_list=umls_labels_list,
            umls_des_list=umls_des_list,
            data_list=data_list,
            save_umls_embeddings_dir=save_umls_embeddings_dir,
            save_data_embeddings_dir=save_data_embeddings_dir,
            normalize=normalize,
            summary_method=summary_method,
            tqdm_bar=tqdm_bar,
            batch_size=batch_size,
        )
        exploded_term_df[["label", "norm_term", "score"]] = pd.DataFrame(zip(*res))
        df = (
            pd.merge(
                df.drop(columns=[column_name_to_normalize]),
                exploded_term_df,
                left_index=True,
                right_on="id",
            )
            .drop(columns=["id"])
            .reset_index(drop=True)
        )
    return df
