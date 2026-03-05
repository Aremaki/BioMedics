import os

os.environ["OMP_NUM_THREADS"] = "16"

from pathlib import Path
from typing import List

import edsnlp
import pandas as pd
from confit import Cli
from edsnlp.connectors import BratConnector
from loguru import logger
from spacy.tokens import Span

from biomedics.normalization.embedding_similarity.main import get_embedding_similarity
from biomedics.utils.extract_pandas_from_brat import discover_brat_dirs

app = Cli(pretty_exceptions_show_locals=False)

def run_coder_inference(
    output_dir: Path,
    brat_dirs: list,
    model_path: Path,
    umls_path: str,
    label_to_normalize: str,
    qualifiers: List[str],
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
    measurement_pickle = output_dir  / "pred_with_measurement.pkl"
    if measurement_pickle.is_file():
        print("Found measurement pickle, loading it.")
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
        print("No measurement pickle found, processing BRAT files.")
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
    df = df[~df[column_name_to_normalize].isna()]
    try:
        df = get_embedding_similarity(
            df=df,
            model_path=model_path,
            cased=cased,
            stopwords=stopwords,
            input_dirs=brat_dirs,
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
        if df is not None:
            df.to_pickle(output_dir / "pred_bio_norm.pkl")
        else:
            logger.warning(f"Embedding Similarity returned None for {brat_dirs}, skipping saving.")
    except Exception as e:
        print(f"Bio Norm SKIPPED, error: {e}")

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
    file_batch_size: int,
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
    base_ner_dir = input_folder.parent / "pred_NER"
    # Count the number of .ann files in the brat_dir
    brat_dirs = discover_brat_dirs(base_ner_dir)
    total_ann_files = sum(len(list(brat_dir.glob("*.ann"))) for brat_dir in brat_dirs)
    logger.info(f"Found {total_ann_files} .ann files in {base_ner_dir}")
    # Split into batch
    if total_ann_files > file_batch_size:
        logger.info(f"Splitting {total_ann_files} .ann files into batches of {file_batch_size}")
        batch_brats = []
        batch_num = 1
        ann_counts = 0
        for brat_dir in brat_dirs:
            ann_counts += len(list((brat_dir).glob("*.ann")))
            if ann_counts > file_batch_size:
                logger.info(f"Processing batch {batch_num} of {ann_counts} .ann files")
                batch_output_dir = output_dir / f"batch_{batch_num}"
                batch_output_dir.mkdir(parents=True, exist_ok=True)
                run_coder_inference(
                    output_dir=batch_output_dir,
                    brat_dirs=batch_brats,
                    model_path=model_path,
                    umls_path=umls_path,
                    label_to_normalize=label_to_normalize,
                    qualifiers=qualifiers,
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
                    cased=cased,
                    stopwords=stopwords,
                    remove_stopwords_terms=remove_stopwords_terms,
                    remove_special_characters_terms=remove_special_characters_terms,
                    remove_stopwords_umls=remove_stopwords_umls,
                    remove_special_characters_umls=remove_special_characters_umls,
                )
                batch_brats = []
                ann_counts = 0
                batch_num += 1
            batch_brats.append(brat_dir)
        if batch_brats:
            logger.info(f"Processing batch {batch_num} of {ann_counts} .ann files")
            batch_output_dir = output_dir / f"batch_{batch_num}"
            batch_output_dir.mkdir(parents=True, exist_ok=True)
            run_coder_inference(
                output_dir=batch_output_dir,
                brat_dirs=batch_brats,
                model_path=model_path,
                umls_path=umls_path,
                label_to_normalize=label_to_normalize,
                qualifiers=qualifiers,
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
                cased=cased,
                stopwords=stopwords,
                remove_stopwords_terms=remove_stopwords_terms,
                remove_special_characters_terms=remove_special_characters_terms,
                remove_stopwords_umls=remove_stopwords_umls,
                remove_special_characters_umls=remove_special_characters_umls,
            )
    else:
        print("Processing all files in one batch.")
        run_coder_inference(
            output_dir=output_dir,
            brat_dirs=brat_dirs,
            model_path=model_path,
            umls_path=umls_path,
            label_to_normalize=label_to_normalize,
            qualifiers=qualifiers,
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
            cased=cased,
            stopwords=stopwords,
            remove_stopwords_terms=remove_stopwords_terms,
            remove_special_characters_terms=remove_special_characters_terms,
            remove_stopwords_umls=remove_stopwords_umls,
            remove_special_characters_umls=remove_special_characters_umls,
        )

if __name__ == "__main__":
    app()
