import re
from os import listdir
from os.path import basename, isdir, isfile, join

import pandas as pd


def extract_pandas(IN_BRAT_DIR, OUT_DF=None, labels=None):
    assert isdir(IN_BRAT_DIR)

    ENTITY_REGEX = re.compile("^(.\\d+)\t([^ ]+) ([^\t]+)\t(.*)$")

    data = []
    patients = []

    # extract all ann_files from IN_BRAT_DIR
    ann_files = [
        f
        for f in listdir(IN_BRAT_DIR)
        if isfile(join(IN_BRAT_DIR, f))
        if f.endswith(".ann")
    ]
    for ann_file in ann_files:
        ann_path = join(IN_BRAT_DIR, ann_file)
        txt_path = ann_path[:-4] + ".txt"

        # sanity check
        assert isfile(ann_path)
        assert isfile(txt_path)

        # Read text file to get patient number :
        with open(txt_path, "r", encoding="utf-8") as f_txt:
            lines_txt = f_txt.readlines()
        patient_num = lines_txt[0][:-1]
        patients.append(patient_num)

        # Read ann file
        with open(ann_path, "r", encoding="utf-8") as f_in:
            lines = f_in.readlines()

        for line in lines:
            entity_match = ENTITY_REGEX.match(line.strip())
            if entity_match is not None:
                ann_id = entity_match.group(1)
                label = entity_match.group(2)
                offsets = entity_match.group(3)
                term = entity_match.group(4)
                if len(offsets.split(";")) > 1:
                    terms = {}
                    start = 0
                    for char_span in offsets.split(";"):
                        start_span = int(char_span.split()[0])
                        end_span = int(char_span.split()[1])
                        end = start + end_span - start_span
                        terms[(start_span, end_span)] = term[start:end]
                        start = end + 1
                    terms = dict(sorted(terms.items()))
                    term = " ".join(terms.values())
                    offsets = " ".join(
                        [str(list(terms.keys())[0][0]), str(list(terms.keys())[-1][-1])]
                    )
                if (labels is None) or (label in labels):
                    data.append([ann_id, term, label, basename(ann_path), offsets])

    columns = ["ann_id", "term", "label", "source", "span"]
    dataset_df = pd.DataFrame(data=list(data), columns=columns)
    if OUT_DF:
        dataset_df.to_csv(OUT_DF)

    return dataset_df
