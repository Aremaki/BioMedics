import os
import shutil
import time
from pathlib import Path
from typing import Dict

from confit import validate_arguments
from edsnlp import Pipeline
from edsnlp.scorers import Scorer
from edsnlp.utils.bindings import BINDING_SETTERS
from edsnlp.utils.span_getters import get_spans
from tqdm import tqdm

from biomedics.ner.brat import BratConnector


@validate_arguments
class EdsMedicScorer:
    def __init__(
        self,
        ner: Dict[str, Scorer] = {},
        qualifier: Dict[str, Scorer] = {},
    ):
        self.ner_scorers = ner
        self.qlf_scorers = qualifier

    def __call__(self, nlp: Pipeline, docs, per_doc=False, output: Path = None):
        with nlp.train(False):
            scores = {}
            docs = list(docs)

            # Speed
            t0 = time.time()
            list(nlp.pipe(d.text for d in tqdm(docs, desc="Computing model speed")))
            duration = time.time() - t0
            scores["speed"] = dict(
                wps=sum(len(d) for d in docs) / duration,
                dps=len(docs) / duration,
            )

            per_doc_scores = {doc._.note_id: {"note_id": doc._.note_id} for doc in docs}

            # NER
            if nlp.has_pipe("ner") and "ner" not in nlp.disabled:
                clean_ner_docs = [d.copy() for d in tqdm(docs, desc="Copying docs")]
                for d in clean_ner_docs:
                    d.ents = []
                    d.spans.clear()
                with nlp.select_pipes(enable=["ner"]):
                    ner_preds = list(nlp.pipe(tqdm(clean_ner_docs, desc="Predicting")))
                    if output:
                        ner_folder = output / "ner"
                        if os.path.exists(ner_folder):
                            shutil.rmtree(ner_folder)
                        BratConnector(
                            ner_folder,
                        ).docs2brat(ner_preds)
                        print(
                            (
                                "NER Prediction is saved in BRAT format in the "
                                f"following folder: {ner_folder}"
                            )
                        )
                for name, scorer in self.ner_scorers.items():
                    scores[name] = scorer(docs, ner_preds)

                    if per_doc:
                        for doc, pred in zip(docs, ner_preds):
                            note_id = doc._.note_id
                            doc_scores = scorer([doc], [pred])
                            per_doc_scores[note_id][name] = {
                                label: {
                                    key: value
                                    for key, value in label_scores.items()
                                    if key in ("support", "positives", "tp")
                                }
                                for label, label_scores in doc_scores.items()
                            }

            # Qualification
            if nlp.has_pipe("qualifier") and "qualifier" not in nlp.disabled:
                qlf_pipe = nlp.get_pipe("qualifier")
                clean_qlf_docs = [d.copy() for d in tqdm(docs, desc="Copying docs")]
                for doc in clean_qlf_docs:
                    for span in get_spans(doc, qlf_pipe.span_getter):
                        for qlf in qlf_pipe.qualifiers:
                            BINDING_SETTERS[(qlf, None)](span)
                with nlp.select_pipes(enable=["qualifier"]):
                    qlf_preds = list(nlp.pipe(tqdm(clean_qlf_docs, desc="Predicting")))
                    if output:
                        qlf_folder = output / "qlf"
                        if os.path.exists(qlf_folder):
                            shutil.rmtree(qlf_folder)
                        BratConnector(
                            qlf_folder,
                            attributes=[
                                "Negation",
                                # "Family",
                                "Temporality",
                                "Certainty",
                                "Action",
                                # "Allergie",
                            ],
                        ).docs2brat(qlf_preds)
                        print(
                            "Qualification Prediction is saved in BRAT format in the "
                            f"following folder: {qlf_folder}"
                        )
                for name, scorer in self.qlf_scorers.items():
                    scores[name] = scorer(docs, qlf_preds)

                    if per_doc:
                        for doc, pred in zip(docs, qlf_preds):
                            note_id = doc._.note_id
                            doc_scores = scorer([doc], [pred])
                            per_doc_scores[note_id][name] = {
                                label: {
                                    key: value
                                    for key, value in label_scores.items()
                                    if key in ("support", "positives", "tp")
                                }
                                for label, label_scores in doc_scores.items()
                            }

            if per_doc:
                return scores, list(per_doc_scores.values())

            return scores
