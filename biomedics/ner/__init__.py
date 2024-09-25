import spacy

if not spacy.tokens.Span.has_extension("event_type"):
    spacy.tokens.Span.set_extension("event_type", default=None)


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
    if not spacy.tokens.Span.has_extension(ext):
        spacy.tokens.Span.set_extension(ext, default=None)
