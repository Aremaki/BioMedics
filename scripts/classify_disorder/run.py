import os

os.environ["OMP_NUM_THREADS"] = "16"

from pathlib import Path
from typing import List

import edsnlp
import pandas as pd
import torch
from confit import Cli
from loguru import logger
from spacy.tokens import Span
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bar
from transformers import CamembertForSequenceClassification, CamembertTokenizer

from biomedics.ner.brat import BratConnector
from biomedics.normalization.embedding_similarity.get_normalization_with_embedding import (
    EmbeddingNormalizer,
)
from biomedics.normalization.embedding_similarity.text_preprocessor import (
    TextPreprocessor,
)
from biomedics.utils.extract_pandas_from_brat import discover_brat_dirs

app = Cli(pretty_exceptions_show_locals=False)

def run_classify_diso(
    brat_dirs: list,
    model: CamembertForSequenceClassification,
    tokenizer: CamembertTokenizer,
    device: torch.device,
    label_names: str,
    output_dir: Path,
    stopwords: List[str],
    qualifiers: List[str],
    embedding_model_path: str,
):

  # Load Data
    ents_list = []
    for brat_dir in brat_dirs:
        docs = BratConnector(brat_dir).brat2docs(edsnlp.blank("eds"))  # type: ignore
        docs = edsnlp.data.from_iterable(docs)  # type: ignore

        # Filter DISO entities
        terms = []
        for doc in docs:
            diso_ents = doc.spans.get("DISO") if doc.spans.get("DISO") else []
            for ent in diso_ents:
                ent_data = [
                    ent.text,
                    doc._.note_id + ".ann",
                    [ent.start_char, ent.end_char],
                    os.path.basename(os.path.normpath(brat_dir))
                ]
                for qualifier in qualifiers:
                    if not Span.has_extension(qualifier):
                        Span.set_extension(qualifier, default=None)
                    ent_data.append(getattr(ent._, qualifier))
                ents_list.append(ent_data)
                terms.append(ent.text)
    results_columns = ["term", "source", "span_converted", "folder_name"] + qualifiers
    results = pd.DataFrame(ents_list, columns=results_columns)
    text_preprocessor = TextPreprocessor(cased=False, stopwords=stopwords)
    predicted_entities = [
        text_preprocessor(
            text=ent, remove_stopwords=True, remove_special_characters=True
        )
        for ent in terms
    ]

    # Create a DataLoader for batch processing
    dataloader = DataLoader(
        predicted_entities,  # type: ignore
        batch_size=128,
        collate_fn=lambda x: tokenizer(
            x,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=128,
        ),
    )

    all_probs = []

    # Process each batch with a progress bar
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Batches", unit="batch"):
            batch = {
                k: v.to(model.device) for k, v in batch.items()
            }  # Move to GPU if available
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())  # Move back to CPU to save memory

    # Concatenate all probabilities into a single tensor
    all_probs = torch.cat(all_probs, dim=0)

    scores = []
    labels = []
    for prob in all_probs:
        high_confidence_labels = [
            label_names[i].split("_")[-1].capitalize()
            for i, p in enumerate(prob)
            if p.item() > 0.8
        ]
        high_confidence_scores = [str(p.item()) for p in prob if p.item() > 0.8]
        labels.append(" | ".join(high_confidence_labels))
        scores.append(" | ".join(high_confidence_scores))

    results["normalized_term"] = predicted_entities
    results["labels"] = labels
    results["scores"] = scores

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_pickle(output_dir / "pred_with_classified_diso.pkl")

    all_terms = results[["normalized_term", "labels", "scores"]]
    all_terms = all_terms[~all_terms["normalized_term"].duplicated()]
    predicted_entities = all_terms["normalized_term"].tolist()

    embedding_normalizer = EmbeddingNormalizer(
        model_name_or_path=embedding_model_path,
        tokenizer_name_or_path=embedding_model_path,
        device=device,  # type: ignore
    )

    embeddings = embedding_normalizer.get_bert_embed(
        predicted_entities,
        normalize=True,
        summary_method="CLS",
        tqdm_bar=True,
        batch_size=16,
    )

    embeddings = embeddings.to(torch.float16).cpu().numpy()
    result_embedding = pd.DataFrame(embeddings)
    result_embedding["normalized_term"] = predicted_entities
    result_embedding["labels"] = all_terms["labels"]
    result_embedding["scores"] = all_terms["scores"]
    result_embedding.to_pickle(output_dir / "pred_diso_embedding.pkl")

@app.command(name="classify_diso")
def classify_diso_cli(
    *,
    classif_model_path: str,
    labels_path: str,
    embedding_model_path: str,
    input_folder: Path,
    stopwords: List[str],
    qualifiers: List[str],
    batch_size: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load model classifier
    with open(labels_path, "r") as f_out:
        label_names = f_out.readline().strip().split(",")
    print(f"Label names loaded from {labels_path}")

    num_labels = len(label_names)

    # load the model
    print(f"Load model from {classif_model_path}")
    model = CamembertForSequenceClassification.from_pretrained(
        classif_model_path, num_labels=num_labels
    ).to(device)  # type: ignore
    tokenizer = CamembertTokenizer.from_pretrained(classif_model_path)

    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        raise ValueError(
            f"Input folder does not exist or is not a directory: {input_folder}"
        )

    base_ner_dir = input_folder.parent / "pred_NER"
    output_dir = input_folder.parent / "pred_NORM"
    brat_dirs = discover_brat_dirs(base_ner_dir)
    # Count the number of .ann files in the brat_dir
    total_ann_files = sum(len(list((base_ner_dir / d).glob("*.ann")) for d in brat_dirs))
    logger.info(f"Found {total_ann_files} .ann files in {base_ner_dir}")
    # Split into batch
    if total_ann_files > batch_size:
        logger.info(f"Splitting {total_ann_files} .ann files into batches of {batch_size}")
        batch_brats = []
        batch_num = 1
        ann_counts = 0
        for brat_dir in brat_dirs:
            ann_counts += len(list((brat_dir).glob("*.ann")))
            if ann_counts > batch_size:
                logger.info(f"Processing batch {batch_num} of {ann_counts} .ann files")
                output_dir = output_dir / f"batch_{batch_num}"
                output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    run_classify_diso(
                        brat_dirs=batch_brats,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        label_names=label_names,
                        output_dir=output_dir,
                        stopwords=stopwords,
                        qualifiers=qualifiers,
                        embedding_model_path=embedding_model_path,
                    )
                except Exception as e:
                    logger.exception(
                        f"Classify DISO failed for batch {batch_num} of {len(batch_brats)} brats, error: {e}"
                    )
                batch_brats = []
                ann_counts = 0
                batch_num += 1
            batch_brats.append(brat_dir)
        if batch_brats:
            logger.info(f"Processing final batch {batch_num} of {ann_counts} .ann files")
            output_dir = output_dir / f"batch_{batch_num}"
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                run_classify_diso(
                        brat_dirs=batch_brats,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        label_names=label_names,
                        output_dir=output_dir,
                        stopwords=stopwords,
                        qualifiers=qualifiers,
                        embedding_model_path=embedding_model_path,
                    )
            except Exception as e:
                logger.exception(
                    f"Classify DISO failed for final batch {batch_num} of {len(batch_brats)} brats, error: {e}"
                )
    else:
        logger.info(f"Processing all {total_ann_files} .ann files in one batch.")
        try:
            run_classify_diso(
                brat_dirs=brat_dirs,
                model=model,
                tokenizer=tokenizer,
                device=device,
                label_names=label_names,
                output_dir=output_dir,
                stopwords=stopwords,
                qualifiers=qualifiers,
                embedding_model_path=embedding_model_path,
            )
        except Exception as e:
            logger.exception(
                f"Classify DISO failed for all {total_ann_files} .ann files, error: {e}"
            )

if __name__ == "__main__":
    app()
