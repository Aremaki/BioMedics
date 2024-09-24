---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: biomedics_client
    language: python
    name: biomedics_client
---

```python
%reload_ext autoreload
%autoreload 2
%reload_ext jupyter_black
```

```python
import pandas as pd
from functools import reduce
from edsnlp.connectors.brat import BratConnector
from biomedics.ner.evaluate import evaluate_test, evaluate
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Expe Data Size
<!-- #endregion -->

```python
import edsnlp

GOLD_PATH = "../data/annotated_CRH/post_processed/expe_data_size/test"

loader = edsnlp.blank("eds")
brat = BratConnector(GOLD_PATH)
gold_docs = brat.brat2docs(loader)

scores = []
for i in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 62]:
    PRED_PATH = f"../data/annotated_CRH/post_processed/expe_data_size/pred_{i}"
    loader = edsnlp.blank("eds")
    brat = BratConnector(PRED_PATH)
    pred_docs = brat.brat2docs(loader)
    score = pd.DataFrame(
        evaluate_test(
            gold_docs,
            pred_docs,
            boostrap_level="doc",
            exact=True,
            n_draw=5000,
            alpha=0.05,
            digits=5,
            labels_to_remove=[
                "route",
                "DISO",
                "Constantes",
                "SECTION_traitement",
                "SECTION_evolution",
                "SECTION_antecedent",
                "SECTION_motif",
                "SECTION_histoire",
                "SECTION_examen_clinique",
                "SECTION_examen_complementaire",
                "SECTION_mode_de_vie",
                "SECTION_traitement_entree",
                "SECTION_antecedent_familiaux",
                "SECTION_traitement_sortie",
                "SECTION_conclusion",
                "BIO_milieu",
            ],
        )
    ).T.sort_index()
    score[["n_docs"]] = i
    scores.append(score)
```

```python jupyter={"outputs_hidden": true}
import altair as alt
from functools import reduce

alt.data_transformers.disable_max_rows()
result = (
    pd.concat(scores)[["n_docs", "Precision", "Recall", "F1"]]
    .dropna()
    .reset_index()
    .rename(columns={"index": "label"})
    .melt(
        id_vars=["n_docs", "label"],
        value_vars=["Precision", "Recall", "F1"],
        var_name="metric",
        value_name="summary",
    )
)
result["mean"] = result["summary"].str.split(r"\\n").str.get(0)
result["lower"] = (
    result["summary"]
    .str.split(r"\\n")
    .str.get(1)
    .str.split("-")
    .str.get(0)
    .str.slice(1)
)
result["upper"] = (
    result["summary"]
    .str.split(r"\\n")
    .str.get(1)
    .str.split("-")
    .str.get(1)
    .str.slice(0, -1)
)

result = result[
    result.label.isin(
        [
            "Overall",
            "DISO",
            "Constantes",
            "BIO_comp",
            "Chemical_and_drugs",
            "dosage",
            "BIO",
            "strength",
            "form",
            "SECTION_antecedent",
            "SECTION_motif",
            "SECTION_histoire",
            "SECTION_examen_clinique",
            "SECTION_examen_complementaire",
            "SECTION_mode_de_vie",
            "SECTION_traitement_entree",
            "SECTION_antecedent_familiaux",
            "SECTION_traitement_sortie",
            "SECTION_conclusion",
        ]
    )
]
label_dropdown = alt.binding_select(options=list(result.label.unique()), name="Label ")
label_selection = alt.selection_point(
    fields=["label"], bind=label_dropdown, value="Overall"
)

metric_dropdown = alt.binding_select(
    options=list(result.metric.unique()), name="Metric "
)
metric_selection = alt.selection_point(
    fields=["metric"], bind=metric_dropdown, value="F1"
)
result["legend_error_band"] = "95% confidence interval"
line = (
    alt.Chart(result)
    .mark_line(point=True)
    .encode(
        x=alt.X("n_docs:O", title="Number of training documents"),
        y=alt.Y(f"mean:Q", title="F1-score").scale(zero=False),
        stroke=alt.Stroke(
            "legend_error_band",
            title="Error band",
            legend=alt.Legend(
                symbolType="square",
                orient="top",
                labelFontSize=12,
                labelFontStyle="bold",
            ),
        ),
    )
)

band = (
    alt.Chart(result)
    .mark_area(opacity=0.5)
    .encode(
        x="n_docs:O",
        y=alt.Y(f"upper:Q").title(""),
        y2=alt.Y2(f"lower:Q").title(""),
    )
)

chart = line + band
chart = (
    chart.add_params(metric_selection)
    .transform_filter(metric_selection)
    .add_params(label_selection)
    .transform_filter(label_selection)
    .properties(width=600)
)

display(chart)
display(result)
# chart.save("metrics_by_n_docs.html")
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Expe Section
<!-- #endregion -->

```python
from os.path import isfile, isdir, join, basename
import edsnlp
from spacy.tokens import Span

nlp = edsnlp.blank("eds")
nlp.add_pipe("eds.normalizer")
nlp.add_pipe("eds.sections")

GOLD_PATH = "../data/annotated_CRH/post_processed/expe_lang_model/test"

loader = edsnlp.blank("eds")
brat = BratConnector(GOLD_PATH)
gold_docs = brat.brat2docs(loader)

GOLD_PATH = (
    "../data/annotated_CRH/post_processed/expe_lang_model/pred_model_eds_finetune"
)

brat = BratConnector(ML_PRED_PATH)
ml_pred_docs = brat.brat2docs(loader)

mapping = {
    "antécédents": "SECTION_antecedent",
    "motif": "SECTION_motif",
    "histoire de la maladie": "SECTION_histoire",
    "examens": "SECTION_examen_clinique",
    "examens complémentaires": "SECTION_examen_complementaire",
    "habitus": "SECTION_mode_de_vie",
    "traitements entrée": "SECTION_traitement_entree",
    "antécédents familiaux": "SECTION_antecedent_familiaux",
    "traitements sortie": "SECTION_traitement_sortie",
    "conclusion": "SECTION_conclusion",
}
rule_pred_docs = []
for doc in gold_docs:
    rule_pred_doc = nlp(doc.text)
    rule_pred_doc._.note_id = doc._.note_id
    del rule_pred_doc.spans["sections"]
    rule_pred_docs.append(rule_pred_doc)
for doc in rule_pred_docs:
    for label in mapping.values():
        doc.spans[label] = []
    old_ents = list(doc.spans["section_titles"])
    for old_ent in old_ents:
        if old_ent.label_ in mapping.keys():
            new_ent = Span(
                doc, old_ent.start, old_ent.end, label=mapping[old_ent.label_]
            )
            doc.spans[mapping[old_ent.label_]].append(new_ent)

scores_rule = (
    pd.DataFrame(
        evaluate_test(
            gold_docs=gold_docs,
            pred_docs=rule_pred_docs,
            boostrap_level="doc",
            exact=True,
            n_draw=5000,
            alpha=0.05,
            digits=2,
            labels_to_remove=[
                "Chemical_and_drugs",
                "route",
                "DISO",
                "Constantes",
                "BIO",
                "BIO_comp",
                "dosage",
                "form",
                "route",
                "strength",
                "BIO_milieu",
            ],
        )
    )
    .T.sort_index()[["N_entity", "Precision", "Recall", "F1"]]
    .drop(
        index=[
            "ents_per_type",
        ]
    )
)
scores_rule.columns = pd.MultiIndex.from_product(
    [["Rule-Based"], ["N_entity", "Precision", "Recall", "F1"]]
)

scores_ml = (
    pd.DataFrame(
        evaluate_test(
            gold_docs=gold_docs,
            pred_docs=ml_pred_docs,
            boostrap_level="doc",
            exact=True,
            n_draw=5000,
            alpha=0.05,
            digits=2,
            labels_to_remove=[
                "Chemical_and_drugs",
                "route",
                "DISO",
                "Constantes",
                "BIO",
                "BIO_comp",
                "dosage",
                "form",
                "route",
                "strength",
                "BIO_milieu",
            ],
        )
    )
    .T.sort_index()[["Precision", "Recall", "F1"]]
    .drop(
        index=[
            "ents_per_type",
        ]
    )
)
scores_ml.columns = pd.MultiIndex.from_product(
    [["ML (NER)"], ["Precision", "Recall", "F1"]]
)
result = scores_rule.merge(scores_ml, left_index=True, right_index=True)
```

```python
import numpy as np
from IPython.display import display, HTML


def highlight_max(row):
    Precision_max = (
        row[:, "Precision"].str.split(r"\\n").str.get(0).astype(float)
        == row[:, "Precision"].str.split(r"\\n").str.get(0).astype(float).max()
    )
    Recall_max = (
        row[:, "Recall"].str.split(r"\\n").str.get(0).astype(float)
        == row[:, "Recall"].str.split(r"\\n").str.get(0).astype(float).max()
    )
    F1_max = (
        row[:, "F1"].str.split(r"\\n").str.get(0).astype(float)
        == row[:, "F1"].str.split(r"\\n").str.get(0).astype(float).max()
    )
    s_max = [False]
    for i in range(len(F1_max)):
        s_max.append(Precision_max[i])
        s_max.append(Recall_max[i])
        s_max.append(F1_max[i])
    return ["font-weight: bold" if cell else "" for cell in s_max]


def remove_confidence(row):
    return row[:, :].str.split(" ").str.get(0)


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


pretty_print(result.apply(remove_confidence, axis=1).style.apply(highlight_max, axis=1))
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Expe lang models
<!-- #endregion -->

```python
import edsnlp

GOLD_PATH = "../data/annotated_CRH/post_processed/expe_lang_model/test"
pred_docs_by_model = {}

loader = edsnlp.blank("eds")
brat = BratConnector(GOLD_PATH)
gold_docs = brat.brat2docs(loader)

CAM_BASE_PRED_PATH = (
    "../data/annotated_CRH/post_processed/expe_lang_model/pred_model_camembert_base"
)

brat = BratConnector(CAM_BASE_PRED_PATH)
cam_base_pred_docs = brat.brat2docs(loader)
pred_docs_by_model["CamemBert-Base"] = cam_base_pred_docs

CAM_BIO_PRED_PATH = (
    "../data/annotated_CRH/post_processed/expe_lang_model/pred_model_camembert_bio"
)

brat = BratConnector(CAM_BIO_PRED_PATH)
cam_bio_pred_docs = brat.brat2docs(loader)
pred_docs_by_model["CamemBert-Bio"] = cam_bio_pred_docs

DR_BERT_PRED_PATH = (
    "../data/annotated_CRH/post_processed/expe_lang_model/pred_model_DrBert"
)

brat = BratConnector(DR_BERT_PRED_PATH)
DrBert_pred_docs = brat.brat2docs(loader)
pred_docs_by_model["DrBert"] = DrBert_pred_docs

EDS_FINE_PRED_PATH = (
    "../data/annotated_CRH/post_processed/expe_lang_model/pred_model_eds_finetune"
)

brat = BratConnector(EDS_FINE_PRED_PATH)
eds_finetune_pred_docs = brat.brat2docs(loader)
pred_docs_by_model["CamemBert-EDS"] = eds_finetune_pred_docs

# EDS_SCRATCH_PRED_PATH = "/export/home/cse200093/scratch/BioMedics/NER_model/data/NLP_diabeto/expe_lang_model/pred_model_eds_scratch"

# brat = BratConnector(EDS_SCRATCH_PRED_PATH)
# eds_scratch_pred_docs = brat.brat2docs(loader)
# pred_docs_by_model["CamemBert-EDS Scratch"] = eds_scratch_pred_docs
```

```python
scores = []
first = True
for model_name, pred_docs in pred_docs_by_model.items():
    if first:
        columns = ["N_entity", "Precision", "Recall", "F1"]
    else:
        columns = ["Precision", "Recall", "F1"]
    score_model = (
        pd.DataFrame(
            evaluate_test(
                gold_docs,
                pred_docs,
                boostrap_level="doc",
                exact=True,
                n_draw=5000,
                alpha=0.05,
                digits=2,
                labels_to_remove=[
                    "route",
                    "DISO",
                    "Constantes",
                    "SECTION_traitement",
                    "SECTION_evolution",
                    "SECTION_antecedent",
                    "SECTION_motif",
                    "SECTION_histoire",
                    "SECTION_examen_clinique",
                    "SECTION_examen_complementaire",
                    "SECTION_mode_de_vie",
                    "SECTION_traitement_entree",
                    "SECTION_antecedent_familiaux",
                    "SECTION_traitement_sortie",
                    "SECTION_conclusion",
                    "BIO_milieu",
                ],
            )
        )
        .T.rename(
            index={
                "BIO": "Laboratory procedure",
                "BIO_comp": "Complete laboratory procedure",
                "Chemical_and_drugs": "Drug name",
                "dosage": "Drug dosage",
                "form": "Drug form",
                "strength": "Drug strength",
            }
        )
        .sort_index()[columns]
        .drop(
            index=[
                "ents_per_type",
            ]
        )
    )

    score_model.columns = pd.MultiIndex.from_product([[model_name], columns])
    scores.append(score_model)
    first = False

result = reduce(
    lambda left, right: pd.merge(left, right, left_index=True, right_index=True), scores
)
result = result.rename(columns={"N_entity": "Number of entities", "F1": "F1-score"})
```

```python
import numpy as np


def highlight_max(row):
    Precision_max = (
        row[:, "Precision"].str.split(r"\\n").str.get(0).astype(float)
        == row[:, "Precision"].str.split(r"\\n").str.get(0).astype(float).max()
    )
    Recall_max = (
        row[:, "Recall"].str.split(r"\\n").str.get(0).astype(float)
        == row[:, "Recall"].str.split(r"\\n").str.get(0).astype(float).max()
    )
    F1_max = (
        row[:, "F1-score"].str.split(r"\\n").str.get(0).astype(float)
        == row[:, "F1-score"].str.split(r"\\n").str.get(0).astype(float).max()
    )
    s_max = [False]
    for i in range(len(F1_max)):
        s_max.append(Precision_max[i])
        s_max.append(Recall_max[i])
        s_max.append(F1_max[i])
    return ["font-weight: bold" if cell else "" for cell in s_max]


def remove_confidence(row):
    return row[:, :].str.split("\n").str.get(0)


df = result.style.apply(highlight_max, axis=1).set_properties(
    **{"text-align": "center"}
)

from IPython.display import display, HTML


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


pretty_print(df)
```

# Expe QUAERO

```python
import edsnlp

GOLD_PATH = "../data/QUAERO/corpus/test/MEDLINE"
PRED_PATH = "../data/QUAERO/corpus/pred/MEDLINE/ner"

loader = edsnlp.blank("eds")
brat = BratConnector(GOLD_PATH)
gold_docs = brat.brat2docs(loader)


brat = BratConnector(PRED_PATH)
pred_docs = brat.brat2docs(loader)
```

```python
import pandas as pd
from IPython.display import display, HTML

scores = []
columns = ["N_entity", "Precision", "Recall", "F1"]
scores = (
    (
        pd.DataFrame(
            evaluate_test(
                gold_docs,
                pred_docs,
                boostrap_level="doc",
                exact=True,
                n_draw=5000,
                alpha=0.05,
                digits=1,
            )
        ).T
    )
    .sort_index()[columns]
    .drop(
        index=[
            "ents_per_type",
        ]
    )
)
scores = scores.rename(columns={"N_entity": "Number of entities", "F1": "F1-score"})
scores.index.name = "Label"
scores = scores.reindex(
    index=[
        "ANAT",
        "CHEM",
        "DEVI",
        "DISO",
        "GEOG",
        "LIVB",
        "OBJC",
        "PHEN",
        "PHYS",
        "PROC",
        "Overall",
    ]
)


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", " ")))


pretty_print(scores)
```

```python
import edsnlp

GOLD_PATH = "../data/QUAERO/corpus/test/EMEA"
PRED_PATH = "../data/QUAERO/corpus/pred/EMEA/ner"

loader = edsnlp.blank("eds")
brat = BratConnector(GOLD_PATH)
gold_docs = brat.brat2docs(loader)


brat = BratConnector(PRED_PATH)
pred_docs = brat.brat2docs(loader)
```

```python
import pandas as pd
from IPython.display import display, HTML

scores = []
columns = ["N_entity", "Precision", "Recall", "F1"]
scores = (
    (
        pd.DataFrame(
            evaluate_test(
                gold_docs,
                pred_docs,
                boostrap_level="doc",
                exact=True,
                n_draw=5000,
                alpha=0.05,
                digits=1,
            )
        ).T
    )
    .sort_index()[columns]
    .drop(
        index=[
            "ents_per_type",
        ]
    )
)
scores = scores.rename(columns={"N_entity": "Number of entities", "F1": "F1-score"})
scores.index.name = "Label"
scores = scores.reindex(
    index=[
        "ANAT",
        "CHEM",
        "DEVI",
        "DISO",
        "GEOG",
        "LIVB",
        "OBJC",
        "PHEN",
        "PHYS",
        "PROC",
        "Overall",
    ]
)


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", " ")))


pretty_print(scores)
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Expe NER rule based
<!-- #endregion -->

```python
import edsnlp

GOLD_PATH = "../data/annotated_CRH/post_processed/expe_rule_based_ner/test"
PRED_PATH = "../data/annotated_CRH/post_processed/expe_rule_based_ner/pred"

loader = edsnlp.blank("eds")
brat = BratConnector(GOLD_PATH)
gold_docs = brat.brat2docs(loader)


brat = BratConnector(PRED_PATH)
pred_docs = brat.brat2docs(loader)
```

```python
import pandas as pd
from IPython.display import display, HTML

scores = []
columns = ["N_entity", "Precision", "Recall", "F1"]
scores = (
    pd.DataFrame(
        evaluate_test(
            gold_docs,
            pred_docs,
            boostrap_level="doc",
            exact=True,
            n_draw=5000,
            alpha=0.05,
            digits=2,
            labels_to_keep=[
                "BIO",
                "Chemical_and_drugs",
            ],
        )
    )
    .T.rename(
        index={
            "BIO": "Laboratory procedure",
            "Chemical_and_drugs": "Drug name",
        }
    )
    .sort_index()[columns]
    .drop(
        index=[
            "ents_per_type",
        ]
    )
)
scores = scores.rename(columns={"N_entity": "Number of entities", "F1": "F1-score"})
scores.index.name = "Label"


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", " ")))


pretty_print(scores)
```

# Expe NER

```python
import edsnlp

GOLD_PATH = "../data/annotated_CRH/post_processed/expe_ner_final/test"
PRED_PATH = "../data/annotated_CRH/post_processed/expe_ner_final/pred/ner"

loader = edsnlp.blank("eds")
brat = BratConnector(GOLD_PATH)
gold_docs = brat.brat2docs(loader)


brat = BratConnector(PRED_PATH)
pred_docs = brat.brat2docs(loader)
```

```python
import pandas as pd
from IPython.display import display, HTML

scores = []
columns = ["N_entity", "Precision", "Recall", "F1"]
scores = (
    pd.DataFrame(
        evaluate_test(
            gold_docs,
            pred_docs,
            boostrap_level="doc",
            exact=True,
            n_draw=5000,
            alpha=0.05,
            digits=1,
            labels_to_keep=[
                "BIO",
                "BIO_comp",
                "dosage",
                "strength",
                "form",
                "Chemical_and_drugs",
            ],
        )
    )
    .T.rename(
        index={
            "BIO": "0-Laboratory test",
            "BIO_comp": "1-Complete laboratory test",
            "Chemical_and_drugs": "2-Drug name",
            "dosage": "3-Drug dosage",
            "form": "4-Drug form",
            "strength": "5-Drug strength",
        }
    )
    .sort_index()[columns]
    .drop(
        index=[
            "ents_per_type",
        ]
    )
)
scores = scores.rename(columns={"N_entity": "Number of entities", "F1": "F1-score"})
scores.index.name = "Label"


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", " ")))


pretty_print(scores)
```

# Expe Qualification

```python
import edsnlp

GOLD_PATH = "../data/annotated_CRH/post_processed/expe_ner_final/test"
PRED_PATH = "../data/annotated_CRH/post_processed/expe_ner_final/pred/qlf"

loader = edsnlp.blank("eds")
brat = BratConnector(GOLD_PATH)
gold_docs = brat.brat2docs(loader)


brat = BratConnector(PRED_PATH)
pred_docs = brat.brat2docs(loader)
```

```python
import pandas as pd
from IPython.display import display, HTML

columns = ["N_entity", "Precision", "Recall", "F1"]

scores = (
    pd.DataFrame(
        evaluate_test(
            gold_docs,
            pred_docs,
            qualification=True,
            qualif_group=True,
            boostrap_level="doc",
            exact=True,
            n_draw=5000,
            alpha=0.05,
            digits=1,
            labels_to_keep=[
                "Action_Decrease",
                "Action_OtherChange",
                "Action_Increase",
                "Action_Start",
                "Action_UniqueDose",
                "Action_Stop",
                "Temporality_Past",
                "Temporality_Future",
                "Certainty_Hypothetical",
                "Certainty_Conditional",
                "Negation_Neg",
            ],
        )
    )
    .T[columns]
    .drop(
        index=[
            "ents_per_type",
        ]
    )
)

scores = scores.rename(columns={"N_entity": "Number of entities", "F1": "F1-score"})
scores.index.name = "Attributes"


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", " ")))


pretty_print(scores)
```

# Expe Complete Pipe

```python
import edsnlp

GOLD_PATH = "../data/annotated_CRH/post_processed/expe_complete_pipe/test"
PRED_PATH = "../data/annotated_CRH/post_processed/expe_complete_pipe/pred/NER"

loader = edsnlp.blank("eds")
brat = BratConnector(GOLD_PATH)
gold_docs = brat.brat2docs(loader)


brat = BratConnector(PRED_PATH)
pred_docs = brat.brat2docs(loader)
```

```python
import pandas as pd
from IPython.display import display, HTML

scores = []
columns = ["N_entity", "Precision", "Recall", "F1"]
scores = (
    pd.DataFrame(
        evaluate_test(
            gold_docs,
            pred_docs,
            boostrap_level="doc",
            exact=True,
            n_draw=5000,
            alpha=0.05,
            digits=2,
            labels_to_remove=[
                "route",
                "DISO",
                "Constantes",
                "SECTION_traitement",
                "SECTION_autre",
                "SECTION_evolution",
                "SECTION_antecedent",
                "SECTION_motif",
                "SECTION_histoire",
                "SECTION_examen_clinique",
                "SECTION_examen_complementaire",
                "SECTION_mode_de_vie",
                "SECTION_traitement_entree",
                "SECTION_antecedent_familiaux",
                "SECTION_traitement_sortie",
                "SECTION_conclusion",
                "BIO_milieu",
                "bars",
                "doctors",
                "footer",
                "web",
            ],
        )
    )
    .T.rename(
        index={
            "BIO": "0-Laboratory test",
            "BIO_comp": "1-Complete laboratory test",
            "Chemical_and_drugs": "2-Drug name",
            "dosage": "3-Drug dosage",
            "form": "4-Drug form",
            "strength": "5-Drug strength",
        }
    )
    .sort_index()[columns]
    .drop(
        index=[
            "ents_per_type",
        ]
    )
)
scores = scores.rename(columns={"N_entity": "Number of entities", "F1": "F1-score"})
scores.index.name = "Label"


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", " ")))


pretty_print(scores)
```
