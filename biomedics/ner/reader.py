import random
from typing import Any, Generator, Iterable, List, Optional

import edsnlp
import spacy
from confit import validate_arguments
from spacy.tokens import Doc


def subset_doc(doc: Doc, start: int, end: int) -> Doc:
    """
    Subset a doc given a start and end index.

    Parameters
    ----------
    doc: Doc
        The doc to subset
    start: int
        The start index
    end: int
        The end index

    Returns
    -------
    Doc
    """
    # TODO: review user_data copy strategy
    subset = doc[start:end]
    start_char = subset[0].idx if len(subset) else 0
    end_char = subset[-1].idx + len(subset[-1]) if len(subset) else 0
    new_doc = subset.as_doc(copy_user_data=True)
    new_doc.user_data.update(
        {
            (
                prefix,
                ext,
                s if s is None else max(s - start_char, 0),
                e if e is None else min(e - start_char, end_char - start_char),
            ): v
            for (prefix, ext, s, e), v in doc.user_data.items()
            if (s is None or start_char <= s <= end_char)
            or (e is None or start_char <= e <= end_char)
        }
    )

    for name, group in doc.spans.items():
        new_doc.spans[name] = [
            spacy.tokens.Span(
                new_doc,
                max(0, span.start - start),
                min(end, span.end) - start,
                span.label,
            )
            for span in group
            if span.end > start and span.start < end
        ]

    new_doc._.note_id = f"{doc._.note_id}-{start}-{end}"
    return new_doc


@validate_arguments
class EdsMedicReader:
    """
    Reader that reads docs from a file or a generator, and adapts them to the pipeline.

    Parameters
    ----------
    source: Callable[..., Iterable[Doc]]
        The source of documents (e.g. `edsnlp.data.from_json(...)` or something else)
    limit: Optional[int]
        The maximum number of docs to read
    max_length: int
        The maximum length of the resulting docs
    randomize: bool
        Whether to randomize the split
    multi_sentence: bool
        Whether to split sentences across multiple docs
    filter_expr: Optional[str]
        An expression to filter the docs to generate
    """

    def __init__(
        self,
        source: Any,
        limit: Optional[int] = -1,
        max_length: int = 0,
        randomize: bool = False,
        multi_sentence: bool = True,
        filter_expr: Optional[str] = None,
    ):
        self.source = source
        self.limit = limit
        self.max_length = max_length
        self.randomize = randomize
        self.multi_sentence = multi_sentence
        self.filter_expr = filter_expr

    def __call__(self, nlp) -> Generator[Doc, None, None]:
        filter_fn = eval(f"lambda doc:{self.filter_expr}") if self.filter_expr else None

        blank_nlp = edsnlp.Pipeline(nlp.lang, vocab=nlp.vocab, vocab_config=None)
        blank_nlp.add_pipe("eds.normalizer")
        blank_nlp.add_pipe("eds.sentences")

        docs = blank_nlp.pipe(self.source)

        count = 0

        # Load the jsonl data from path
        if self.randomize:
            docs: List[Doc] = list(docs)
            random.shuffle(docs)

        for doc in docs:
            if 0 <= self.limit <= count:
                break
            if not len(doc):
                continue
            count += 1

            for sub_doc in self.split_doc(doc):
                if filter_fn is not None and not filter_fn(sub_doc):
                    continue
                if len(sub_doc.text.strip()):
                    yield sub_doc
            else:
                continue

    def split_doc(
        self,
        doc: Doc,
    ) -> Iterable[Doc]:
        """
        Split a doc into multiple docs of max_length tokens.

        Parameters
        ----------
        doc: Doc
            The doc to split

        Returns
        -------
        Iterable[Doc]
        """
        max_length = self.max_length
        randomize = self.randomize

        if max_length <= 0:
            yield doc
        else:
            start = 0
            end = 0
            for ent in doc.ents:
                for token in ent:
                    token.is_sent_start = False
            for sent in doc.sents if doc.has_annotation("SENT_START") else (doc[:],):
                # If the sentence adds too many tokens
                if sent.end - start > max_length:
                    # But the current buffer too large
                    while sent.end - start > max_length:
                        subset_end = start + int(
                            max_length * (random.random() ** 0.3 if randomize else 1)
                        )
                        yield subset_doc(doc, start, subset_end)
                        start = subset_end
                    yield subset_doc(doc, start, sent.end)
                    start = sent.end

                if not self.multi_sentence:
                    yield subset_doc(doc, start, sent.end)
                    start = sent.end

                # Otherwise, extend the current buffer
                end = sent.end

            yield subset_doc(doc, start, end)
