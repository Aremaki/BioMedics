import os

os.environ["OMP_NUM_THREADS"] = "16"

from pathlib import Path
from typing import List

import edsnlp
import pandas as pd
from confit import Cli
from edsnlp.connectors import BratConnector
from spacy.tokens import Span

from biomedics.normalization.embedding_similarity.main import get_embedding_similarity
from biomedics.utils.extract_pandas_from_brat import discover_brat_dirs

app = Cli(pretty_exceptions_show_locals=False)



@app.command(name="emdedding_similarity")
def coder_inference_cli(
    *,
    model_path: Path,
    input_folder: Path,
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
):
    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        raise ValueError(
            f"Input folder does not exist or is not a directory: {input_folder}"
        )

    output_dir = input_folder.parent / "pred_NORM"
    brat_dir = input_folder.parent / "pred_NER"
    measurement_pickle = output_dir / "pred_with_measurement.pkl"
    if measurement_pickle.is_file():
        print(f"Found measurement pickle for {brat_dir}, loading it.")
        df = pd.read_pickle(measurement_pickle)
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
        print(f"No measurement pickle found for {brat_dir}, processing BRAT files.")
        brat_dirs = discover_brat_dirs(brat_dir)
        ents_list = []
        for brat_dir in brat_dirs:
            doc_list = BratConnector(brat_dir).brat2docs(edsnlp.blank("eds"))
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
                            if not Span.has_extension(qualifier):
                                Span.set_extension(qualifier, default=None)
                            ent_data.append(getattr(ent._, qualifier))
                        ents_list.append(ent_data)
        df_columns = [
            "term",
            "source",
            "span_converted",
            column_name_to_normalize,
        ] + qualifiers
        df = pd.DataFrame(ents_list, columns=df_columns)



        try:
            df = df[~df[column_name_to_normalize].isna()]
            df = get_embedding_similarity(
                df=df,
                model_path=model_path,
                cased=cased,
                stopwords=stopwords,
                input_dirs=[brat_dir],
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
            output_dir.mkdir(parents=True, exist_ok=True)
            if df is not None:
                df.to_pickle(output_dir / "pred_bio_norm.pkl")
        except Exception as e:
            print(f"Bio Norm SKIPPED for {brat_dir}, error: {e}")


if __name__ == "__main__":
    app()
