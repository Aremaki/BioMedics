import os

os.environ["OMP_NUM_THREADS"] = "16"

from pathlib import Path
from typing import List, Optional

import edsnlp
import pandas as pd
import torch
from confit import Cli
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

app = Cli()


@app.command(name="classify_diso")
def classify_diso_cli(
    *,
    classif_model_path: str,
    labels_path: str,
    embedding_model_path: str,
    brat_dirs: List[Path],
    stopwords: List[str],
    qualifiers: List[str],
    output_dirs: Optional[List[Path]] = None,
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

    # Load Data
    if not output_dirs:
        output_dirs = [brat_dir.parent for brat_dir in brat_dirs]
    for brat_dir, output_dir in zip(brat_dirs, output_dirs):
        docs = BratConnector(brat_dir).brat2docs(edsnlp.blank("eds"))  # type: ignore
        docs = edsnlp.data.from_iterable(docs)  # type: ignore

        # Filter DISO entities
        ents_list = []
        terms = []
        for doc in docs:
            diso_ents = doc.spans.get("DISO") if doc.spans.get("DISO") else []
            for ent in diso_ents:
                ent_data = [
                    ent.text,
                    doc._.note_id + ".ann",
                    [ent.start_char, ent.end_char],
                ]
                for qualifier in qualifiers:
                    if not Span.has_extension(qualifier):
                        Span.set_extension(qualifier, default=None)
                    ent_data.append(getattr(ent._, qualifier))
                ents_list.append(ent_data)
                terms.append(ent.text)
        results_columns = ["term", "source", "span_converted"] + qualifiers
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

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        results.to_pickle(f"{output_dir}/pred_with_classified_diso.pkl")

        all_terms = results[["normalized_term", "labels", "scores"]].drop_duplicates(
            subset="normalized_term"
        )
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
        result_embedding.to_pickle(f"{output_dir}/pred_diso_embedding.pkl")


if __name__ == "__main__":
    app()
