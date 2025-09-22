import spacy

if not spacy.tokens.Span.has_extension("event_type"): # type: ignore
    spacy.tokens.Span.set_extension("event_type", default=None) # type: ignore
if not spacy.tokens.Span.has_extension("rel"): # type: ignore
    spacy.tokens.Span.set_extension("rel", default=None) # type: ignore

for ext in [
    "assertion",
    "etat",
    "prise",
    "changement",
    "norme",
    "negation",
    "negated",
    "hypothesis",
    "family",
    "counterindication",
]:
    if not spacy.tokens.Span.has_extension(ext): # type: ignore
        spacy.tokens.Span.set_extension(ext, default=None) # type: ignore
