import re


def export_pandas_to_brat(
    ann_path,
    txt_path,
    df_to_convert,
    label_column_name,
    span_column_name,
    annotation_column_name=None,
    qualifiers_columns_name=None,
):
    """
    - ann_path: str path where to write the ann file.
    - txt_path: str path where is stored the txt linked to ann file. Useful to check if there are newlines.
    - df_to_convert: Pandas df containing at least a column of labels, a column of spans and a column of terms.
    - label_column_name: str name of the column in df_to_convert containing the labels. This column should be filled with str only.
    - span_column_name: str name of the column in df_to_convert containing the spans. This column should be filled with lists only,
    first element of each list being the beginning of the span and second element being the end.
    - annotation_column_name: OPTIONAL str name of the column in df_to_convert containing the annotations. This column should be filled with str only.
    If None, no annotation will be saved.
    """

    SEP = "\t"
    ANNOTATION_LABEL = "AnnotatorNotes"
    brat_raw = ""
    n_annotation = 0
    T_count = 1
    A_count = 1

    with open(txt_path, "r") as f:
        txt_raw = f.read()

    df_to_convert = df_to_convert[
        set(
            [
                label_column_name,
                span_column_name,
                annotation_column_name,
            ]
        )
        .union(qualifiers_columns_name)  # type: ignore
        .intersection(df_to_convert.columns)
    ]

    # Iter through df to write each line of ann file
    for _, row in df_to_convert.iterrows():
        term_raw = txt_raw[row[span_column_name][0] : row[span_column_name][1]]
        if "\n" in term_raw:
            span_str = (
                str(row[span_column_name][0])
                + "".join(
                    " "
                    + str(row[span_column_name][0] + newline_index.start())
                    + ";"
                    + str(row[span_column_name][0] + newline_index.start() + 1)
                    for newline_index in re.finditer("\n", term_raw)
                )
                + " "
                + str(row[span_column_name][1])
            )
        else:
            span_str = (
                str(row[span_column_name][0]) + " " + str(row[span_column_name][1])
            )
        brat_raw += (
            "T"
            + str(T_count)
            + SEP
            + row[label_column_name]
            + " "
            + span_str
            + SEP
            + term_raw.replace("\n", " ")
            + "\n"
        )
        if (
            annotation_column_name in row.index
            and row[annotation_column_name] is not None
        ):
            n_annotation += 1
            brat_raw += (
                "#"
                + str(n_annotation)
                + SEP
                + ANNOTATION_LABEL
                + " "
                + "T"
                + str(T_count)
                + SEP
                + row[annotation_column_name]
                + "\n"
            )
        if qualifiers_columns_name:
            for qualifier in qualifiers_columns_name:
                if qualifier in row.index and row[qualifier]:
                    brat_raw += (
                        "A" + str(A_count) + SEP + qualifier + " " + "T" + str(T_count)
                    )
                    if isinstance(row[qualifier], str):
                        brat_raw += " " + row[qualifier]
                    brat_raw += "\n"
                    A_count += 1

        T_count += 1

    brat_raw = brat_raw[:-1]
    with open(ann_path, "w") as f:
        print(brat_raw, file=f)
