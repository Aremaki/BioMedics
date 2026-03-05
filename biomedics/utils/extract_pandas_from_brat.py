import os
import re
from os import listdir
from os.path import basename, isfile, join
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm


def discover_brat_dirs(base_dir: Path) -> List[Path]:
    if not base_dir.is_dir():
        raise ValueError(f"No BRAT directory found: {base_dir}")

    discovered: List[Path] = []

    if list(base_dir.glob("*.txt")):
        discovered.append(base_dir)
        return discovered

    for candidate in base_dir.iterdir():
        if candidate.is_dir() and list(candidate.glob("*.txt")):
            discovered.append(candidate)

    if not discovered:
        raise ValueError(f"No BRAT directory found in {base_dir}")
    return sorted(discovered)

def extract_pandas(brat_dirs, OUT_DF=None, labels=None):

    ENTITY_REGEX = re.compile("^(.\\d+)\t([^ ]+) ([^\t]+)\t(.*)$")

    data = []
    patients = []
    brat_dirs_with_ann_files = [
        (
            brat_dir,
            [
                f
                for f in listdir(brat_dir)
                if isfile(join(brat_dir, f))
                if f.endswith(".ann")
            ],
        )
        for brat_dir in brat_dirs
    ]
    total_ann_files = sum(len(ann_files) for _, ann_files in brat_dirs_with_ann_files)

    brat_dirs_progress = tqdm(
        brat_dirs_with_ann_files,
        desc=f"Parsing BRAT directories (ann_files={total_ann_files})",
        unit="dir",
    )
    for brat_dir, ann_files in brat_dirs_progress:
        brat_dirs_progress.set_postfix_str(
            f"{os.path.basename(os.path.normpath(brat_dir))}: {len(ann_files)} ann_files"
        )
        for ann_file in ann_files:
            ann_path = join(brat_dir, ann_file)
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
                        data.append([ann_id, term, label, basename(ann_path), os.path.basename(os.path.normpath(brat_dir)), offsets])

    columns = ["ann_id", "term", "label", "source", "folder_name", "span"]
    dataset_df = pd.DataFrame(data=list(data), columns=columns)
    if OUT_DF:
        dataset_df.to_csv(OUT_DF)

    return dataset_df
