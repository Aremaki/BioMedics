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
import spacy
import edsnlp
import edsnlp.connectors as c
import pandas
```

```python
import os
import shutil

# Définir le répertoire source et le répertoire de destination
source_directory = "raw/all"
destination_directory = "post_processed/all_temp"

# Créer le répertoire de destination s'il n'existe pas
if os.path.exists(destination_directory):
    shutil.rmtree(destination_directory)
os.makedirs(destination_directory)

n = 0
for filename in os.listdir(source_directory):
    if filename.endswith(".ann"):
        source_file_path = os.path.join(source_directory, filename)
        dest_file_path = os.path.join(destination_directory, filename)

        # Si le fichier .ann est vide, supprimer les fichiers .ann et .txt associés dans le répertoire de destination
        if os.stat(source_file_path).st_size == 0:
            print(source_file_path + " has not been annotated. Removed")
            if os.path.exists(dest_file_path):
                os.remove(dest_file_path)
            txt_file_path = source_file_path[:-3] + "txt"
            dest_txt_file_path = dest_file_path[:-3] + "txt"
            if os.path.exists(dest_txt_file_path):
                os.remove(dest_txt_file_path)
            continue

        with open(source_file_path, "r+") as file:
            lines = file.readlines()

        modified_lines = []
        for line in lines:
            # if line.endswith("842 851\tCD4 > 500\n"):
            #     print(
            #         "remove buggy line: " + repr(line) + " on file: " + source_file_path
            #     )
            #     n += 1
            #     continue
            if (
                line.startswith("A")
                and any(keyword in line for keyword in ["Negation"])
                and not " Neg\n" in line
            ):
                line = line.rstrip() + " Neg\n"
                n += 1
            elif (
                line.startswith("A")
                and any(keyword in line for keyword in ["Allergie"])
                and not " Allergic\n" in line
            ):
                line = line.rstrip() + " Allergic\n"
                n += 1
            elif (
                line.startswith("A")
                and any(keyword in line for keyword in ["Family"])
                and not " Relative\n" in line
            ):
                line = line.rstrip() + " Relative\n"
                n += 1

            modified_lines.append(line)

        # Écrire les lignes modifiées dans le nouveau fichier de destination
        with open(dest_file_path, "w") as dest_file:
            dest_file.writelines(modified_lines)

    elif filename.endswith(".txt"):
        txt_source_file_path = os.path.join(source_directory, filename)
        txt_dest_file_path = os.path.join(destination_directory, filename)
        a = shutil.copy(txt_source_file_path, txt_dest_file_path)

print(n, "Attribute updated")
```

NLP_diabeto/clean/CRH_-3331052020138721210.txt à analyser, il fait buger le modèle !
C'est une annotation qui le fait bugger (le token CD4 semble séparé en 2 est ce la raison ?). J'ai retiré l'annotation mais il faudrait creuser.

```python
import re
import os
import shutil

# Définir les répertoires source et de destination
source_directory = "post_processed/all_temp"
destination_directory = "post_processed/all"

# Créer le répertoire de destination s'il n'existe pas
if os.path.exists(destination_directory):
    shutil.rmtree(destination_directory)
os.makedirs(destination_directory)

pattern = "T(\d+)"
n = 0

for filename in os.listdir(source_directory):
    if filename.endswith(".ann"):
        source_file_path = os.path.join(source_directory, filename)
        dest_file_path = os.path.join(destination_directory, filename)

        # Lire le contenu du fichier
        with open(source_file_path, "r") as f:
            lines = f.readlines()

        # Dictionnaire pour stocker les lignes d'entités par ID
        entities = {}
        for line in lines:
            if line.startswith("T"):
                ent_id = re.findall(pattern, line)[0]
                entities[ent_id] = line

        to_del = []
        for i in range(len(lines)):
            current_line = lines[i]

            # Si la ligne actuelle commence par "A" et contient le mot "Tech"
            if current_line.startswith("A") and "Tech" in current_line:
                ent_id = re.findall(pattern, current_line)[0]
                if ent_id in entities:
                    # Remplacement de "Chemical_and_drugs" dans la ligne de l'entité par le dernier mot de la ligne actuelle
                    entity_line = entities[ent_id]
                    entity_index = lines.index(entity_line)
                    lines[entity_index] = entity_line.replace(
                        "Chemical_and_drugs", current_line.split()[-1]
                    )
                    # Suppression de la ligne actuelle
                    to_del.append(i)
                    n += 1

            # Si la ligne actuelle commence par "A" et contient le mot "AttTemp"
            if current_line.startswith("A") and "AttTemp" in current_line:
                ent_id = re.findall(pattern, current_line)[0]
                if ent_id in entities:
                    # Remplacement de "Temporal" dans la ligne de l'entité par le dernier mot de la ligne actuelle
                    entity_line = entities[ent_id]
                    entity_index = lines.index(entity_line)
                    lines[entity_index] = entity_line.replace(
                        "Temporal", current_line.split()[-1]
                    )
                    # Suppression de la ligne actuelle
                    to_del.append(i)
                    n += 1

        # Suppression des lignes marquées
        for i in sorted(to_del, reverse=True):
            del lines[i]

        # Écriture du nouveau contenu dans le fichier de destination
        with open(dest_file_path, "w") as f:
            f.writelines(lines)

    # Copier les fichiers .txt correspondants dans le dossier de destination
    elif filename.endswith(".txt"):
        txt_source_file_path = os.path.join(source_directory, filename)
        txt_dest_file_path = os.path.join(destination_directory, filename)
        shutil.copy(txt_source_file_path, txt_dest_file_path)

shutil.rmtree(source_directory)
print(n, "Attribute(s) updated")
```

```python
import shutil

if not os.path.exists("post_processed/expe_complete_pipe"):
    os.mkdir("post_processed/expe_complete_pipe")
# INPUT PATH
PATH = "post_processed/all"
n_test = 0
n_train = 0
# OUTPUT PATH
PATH_train = "post_processed/expe_complete_pipe/train/"
PATH_test = "post_processed/expe_complete_pipe/test/"

if os.path.exists(PATH_train):
    shutil.rmtree(PATH_train)
os.mkdir(PATH_train)
if os.path.exists(PATH_test):
    shutil.rmtree(PATH_test)
os.mkdir(PATH_test)

complete_pipe_doc = []
for doc in os.listdir("raw/cui_annotations"):
    if doc.endswith(".ann") or doc.endswith(".txt"):
        complete_pipe_doc.append(doc)
complete_pipe_doc = list(set(complete_pipe_doc))

for doc in os.listdir(PATH):
    if doc.endswith(".ann") or doc.endswith(".txt"):
        if doc in complete_pipe_doc:
            n_test += 1
            shutil.copyfile(PATH + "/" + doc, PATH_test + "/" + doc)
        else:
            n_train += 1
            shutil.copyfile(PATH + "/" + doc, PATH_train + "/" + doc)

print(f"{int(n_test/2)} test docs saved")
print(f"{int(n_train/2)} train docs saved")
```

```python
import shutil

if not os.path.exists("post_processed/expe_measurement"):
    os.mkdir("post_processed/expe_measurement")
# INPUT PATH
PATH = "post_processed/all"
n_test = 0
n_train = 0
# OUTPUT PATH
PATH_train = "post_processed/expe_measurement/train"
PATH_test = "post_processed/expe_measurement/test"

if os.path.exists(PATH_train):
    shutil.rmtree(PATH_train)
os.mkdir(PATH_train)
if os.path.exists(PATH_test):
    shutil.rmtree(PATH_test)
os.mkdir(PATH_test)

bio_val_doc = []
for doc in os.listdir("raw/measurement_annotations"):
    if doc.endswith(".ann") or doc.endswith(".txt"):
        bio_val_doc.append(doc)

bio_val_doc = list(set(bio_val_doc))

for doc in os.listdir(PATH):
    if doc.endswith(".ann") or doc.endswith(".txt"):
        if doc in bio_val_doc:
            n_test += 1
            shutil.copyfile(PATH + "/" + doc, PATH_test + "/" + doc)
        else:
            n_train += 1
            shutil.copyfile(PATH + "/" + doc, PATH_train + "/" + doc)

print(f"{int(n_test/2)} test docs saved")
print(f"{int(n_train/2)} train docs saved")
```

```python
if not os.path.exists("post_processed/expe_ner_final"):
    os.mkdir("post_processed/expe_ner_final")

# INPUT PATH
PATH = "post_processed/all"

# OUTPUT PATH
PATH_train = "post_processed/expe_ner_final/train"
PATH_test = "post_processed/expe_ner_final/test"
if os.path.exists(PATH_train):
    shutil.rmtree(PATH_train)
os.mkdir(PATH_train)
if os.path.exists(PATH_test):
    shutil.rmtree(PATH_test)
os.mkdir(PATH_test)

# Parameters
draw = 5000
attributes = ["Negation", "Temporality", "Certainty", "Action"]
labels = [
    "DISO",
    "Constantes",
    "BIO_comp",
    "Chemical_and_drugs",
    "dosage",
    "BIO",
    "strength",
    "form",
    "route",
    "Date",
    "Duration",
    "Frequency",
]

brat = c.BratConnector(
    PATH,
    attributes=attributes,
)
empty = edsnlp.blank("fr")
df = brat.brat2docs(empty)

import numpy as np
import random
from collections import Counter
from tqdm import tqdm


n = len(df)

# total
counter_tot_attributes = Counter(
    attr
    for doc in df
    for ent in doc.ents
    for attr in attributes
    if getattr(ent._, attr)
)  # Full distribution of the attributes
print(counter_tot_attributes)
counter_tot_labels = Counter(
    label
    for doc in df
    for label in doc.spans.keys()
    for i in range(len(doc.spans[label]))
    if label in labels
)
print(counter_tot_labels)

# Index Sampling and counter calculation
index_list = [random.sample(range(n), int(0.2 * n)) for _ in range(draw)]
test_indices = [indices[:] for indices in index_list]
train_indices = [list(set(range(n)) - set(indices)) for indices in index_list]

counter_attributes = {
    split: {attr: np.zeros(draw, dtype=int) for attr in attributes}
    for split in ["test"]
}
counter_labels = {
    split: {label: np.zeros(draw, dtype=int) for label in labels} for split in ["test"]
}
for d in tqdm(range(draw)):
    docs_split = {
        "test": [df[i] for i in test_indices[d]],
    }
    for split in docs_split.keys():
        for doc in docs_split[split]:
            for ent in doc.ents:
                for attr in attributes:
                    if getattr(ent._, attr):
                        counter_attributes[split][attr][d] += 1
            for label in doc.spans.keys():
                if label in labels:
                    counter_labels[split][label][d] += len(doc.spans[label])


# Normalization
for split, ratio in {"test": 0.2}.items():
    for attr in attributes:
        counter_attributes[split][attr] = np.round(
            np.abs(
                (counter_attributes[split][attr] / counter_tot_attributes[attr]) - ratio
            ),
            3,
        )
    for label in labels:
        counter_labels[split][label] = np.round(
            np.abs((counter_labels[split][label] / counter_tot_labels[label]) - ratio),
            3,
        )

# Best Index
totals = np.zeros(draw, dtype=float)
for split in ["test"]:
    for d in range(draw):
        for attr in attributes:
            totals[d] += counter_attributes[split][attr][d]
        for label in labels:
            totals[d] += counter_labels[split][label][d]
best_index = np.argmin(totals)


# Dataset generation
TEST = [df[i] for i in index_list[best_index][: int(len(index_list[best_index]))]]
TRAIN = [df[i] for i in range(n) if i not in index_list[best_index]]


print("Train size : ", len(TRAIN))
print("Test size : ", len(TEST))

brat = c.BratConnector(
    PATH_train,
    attributes=attributes,
)
brat.docs2brat(TRAIN)

print("Train saved")

brat = c.BratConnector(
    PATH_test,
    attributes=attributes,
)
brat.docs2brat(TEST)

print("Test saved")
```

```python

```
