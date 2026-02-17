import os

os.environ["OMP_NUM_THREADS"] = "16"

from pathlib import Path
from typing import List, Optional

import edsnlp
import pandas as pd
from confit import Cli
from edsnlp.connectors import BratConnector

from biomedics.normalization.embedding_similarity.main import get_embedding_similarity

app = Cli()


@app.command(name="emdedding_similarity")
def coder_inference_cli(
    *,
    model_path: Path,
    input_dirs: List[Path],
    umls_path: str,
    labels_column_name: str,
    synonyms_column_name: str,
    label_to_normalize: str,
    qualifiers: List[str],
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
    output_dirs: Optional[List[Path]] = None,
):
    input_dirs = [input_dir.parent for input_dir in input_dirs]
    if not output_dirs:
        output_dirs = input_dirs.copy()
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        if os.path.isfile(f"{input_dir}/pred_with_measurement.pkl"):
            df = pd.read_pickle(f"{input_dir}/pred_with_measurement.pkl")
            if column_name_to_normalize not in df.columns:
                if "terms_linked_to_measurement" in df.columns:
                    df = df.explode("terms_linked_to_measurement")
                    df = df.rename(
                        columns={
                            "terms_linked_to_measurement": column_name_to_normalize
                        }
                    )
                else:
                    df[column_name_to_normalize] = df.term_bio
        else:
            doc_list = BratConnector(input_dir).brat2docs(edsnlp.blank("eds"))
            ents_list = []
            for doc in doc_list:
                if label_to_normalize in doc.spans.keys():
                    for ent in doc.spans[label_to_normalize]:
                        ent_data = [
                            ent.text,
                            doc._.note_id + ".ann",
                            [ent.start_char, ent.end_char],
                            ent.text.lower().strip(),
                        ]
                        for qualifier in qualifiers:
                            ent_data.append(getattr(ent._, qualifier))
                        ents_list.append(ent_data)
            df_columns = [
                "term",
                "source",
                "span_converted",
                column_name_to_normalize,
            ] + qualifiers
            df = pd.DataFrame(ents_list, columns=df_columns)
        df = df[~df[column_name_to_normalize].isna()]
        df = get_embedding_similarity(
            df=df,
            model_path=model_path,
            cased=cased,
            stopwords=stopwords,
            input_dirs=input_dirs,
            umls_path=umls_path,
            labels_column_name=labels_column_name,
            synonyms_column_name=synonyms_column_name,
            column_name_to_normalize=column_name_to_normalize,
            model_device=model_device,
            summary_method=summary_method,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
            save_umls_embeddings_dir=save_umls_embeddings_dir,
            save_umls_des_dir=save_umls_des_dir,
            save_umls_labels_dir=save_umls_labels_dir,
            save_data_embeddings_dir=save_data_embeddings_dir,
            normalize=normalize,
            remove_stopwords_terms=remove_stopwords_terms,
            remove_special_characters_terms=remove_special_characters_terms,
            remove_stopwords_umls=remove_stopwords_umls,
            remove_special_characters_umls=remove_special_characters_umls,
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_pickle(f"{output_dir}/pred_bio_coder_all.pkl")


if __name__ == "__main__":
    app()
