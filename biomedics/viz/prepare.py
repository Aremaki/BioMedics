import pandas as pd
from biomedics import BASE_DIR

regex_pos = r"([¦|]?positifs?|[¦|]pos?i?t?\b|[¦|]?positiv?e?s?|\bpos\b|[^a-zA-Z0-9]+(?:\+|p)[^a-zA-Z0-9]*$|^\+|presente?s?|presences?)"
regex_neg = r"([¦|]?negatifs?|[¦|]neg?a?\b|[¦|]?negati?v?e?s?|\bneg\b|[^a-zA-Z0-9]+(?:\-|n)[^a-zA-Z0-9]*$|^\-|^pas\sd[e']|absente?s?|absences?|indetectables?)"


def prepare_structured_bio_df(disease, config, complete_case_only=False):
    RES_DIR = BASE_DIR / "data" / "final_results"
    summary_filtered_res = pd.read_pickle(RES_DIR / disease / "filtered_bio_from_structured_data.pkl")
    summary_df_docs = pd.read_pickle(BASE_DIR / "data" / "CRH" / "summary_df_docs.pkl")
    summary_df_docs = summary_df_docs[summary_df_docs.disease == disease]
    summary_filtered_res = summary_filtered_res.merge(
        summary_df_docs[["encounter_num", "patient_num"]],
        on=["encounter_num", "patient_num"],
        how="right",
    )
    if complete_case_only:
        med_data = pd.read_pickle(RES_DIR / "med_from_structured_data.pkl")
        bio_data = pd.read_pickle(RES_DIR / "bio_from_structured_data.pkl")
        summary_filtered_res = summary_filtered_res.merge(
            med_data[["encounter_num", "patient_num"]].drop_duplicates(),
            on=["encounter_num", "patient_num"],
            how="inner",
        )
        summary_filtered_res = summary_filtered_res.merge(
            bio_data[["encounter_num", "patient_num"]].drop_duplicates(),
            on=["encounter_num", "patient_num"],
            how="inner",
        )
    summary_filtered_res = summary_filtered_res.rename(
        columns={"nval_num": "value", "units_cd": "unit"}
    )
    summary_patient_group = None
    if len(config["CUI_codes"][disease].keys()) > 0:
        for bio in config["CUI_codes"][disease].keys():
            summary_filtered_res[bio] = summary_filtered_res.bio == bio
            summary_filtered_res[f"{bio} positive"] = (
                summary_filtered_res.bio == bio
            ) & (
                (summary_filtered_res.value > summary_filtered_res.confidence_num)
                | (summary_filtered_res.tval_char.str.contains("posi", case=False))
                | (summary_filtered_res.tval_char.str.contains("présence", case=False))
            )
        summary_patient_group = summary_filtered_res.groupby(
            "patient_num", as_index=False
        ).agg(
            {
                **{
                    bio: "sum"
                    for bio in config["CUI_codes"][disease].keys()
                },
                **{
                    f"{bio} positive": "sum"
                    for bio in config["CUI_codes"][disease].keys()
                },
            }
        )
        for bio in config["CUI_codes"][disease].keys():
            summary_patient_group[bio] = summary_patient_group[bio] >= 1
            summary_patient_group[f"{bio} positive"] = (
                summary_patient_group[f"{bio} positive"] >= 1
            )

    return summary_filtered_res, summary_patient_group

def prepare_structured_med_df(disease, config, complete_case_only=False):
    RES_DIR = BASE_DIR / "data" / "final_results"
    summary_filtered_res = pd.read_pickle(RES_DIR / disease / "filtered_med_from_structured_data.pkl")
    summary_df_docs = pd.read_pickle(BASE_DIR / "data" / "CRH" / "summary_df_docs.pkl")
    summary_df_docs = summary_df_docs[summary_df_docs.disease == disease]
    summary_filtered_res = summary_filtered_res.merge(
        summary_df_docs[["encounter_num", "patient_num"]],
        on=["encounter_num", "patient_num"],
        how="right",
    )
    if complete_case_only:
        med_data = pd.read_pickle(RES_DIR / "med_from_structured_data.pkl")
        bio_data = pd.read_pickle(RES_DIR / "bio_from_structured_data.pkl")
        summary_filtered_res = summary_filtered_res.merge(
            med_data[["encounter_num", "patient_num"]].drop_duplicates(),
            on=["encounter_num", "patient_num"],
            how="inner",
        )
        summary_filtered_res = summary_filtered_res.merge(
            bio_data[["encounter_num", "patient_num"]].drop_duplicates(),
            on=["encounter_num", "patient_num"],
            how="inner",
        )
    summary_patient_group = None
    for med in config["ATC_codes"][disease].keys():
        summary_filtered_res[med] = summary_filtered_res.med == med
    summary_patient_group = summary_filtered_res.groupby(
        "patient_num", as_index=False
    ).agg(
        {
            **{med: "sum" for med in config["ATC_codes"][disease].keys()},
        }
    )
    for med in config["ATC_codes"][disease].keys():
        summary_patient_group[med] = summary_patient_group[med] >= 1

    return summary_filtered_res, summary_patient_group

def prepare_nlp_med_df(
    disease, config, threshold=0.9, complete_case_only=False, filter_certainty=True, filter_negation=True
):
    RES_DIR = BASE_DIR / "data" / "final_results"
    res_filtered_df = pd.read_pickle(RES_DIR / disease / "filtered_med_from_nlp.pkl")
    if filter_negation:
        res_filtered_df = res_filtered_df[res_filtered_df.Negation != "Neg"]
    if filter_certainty:
        res_filtered_df = res_filtered_df[res_filtered_df.Certainty == "Certain"]
    if threshold:
        res_filtered_df = res_filtered_df[res_filtered_df.score > threshold]
    summary_df_docs = pd.read_pickle(BASE_DIR / "data" / "CRH" / "summary_df_docs.pkl")
    summary_df_docs = summary_df_docs[summary_df_docs.disease == disease]
    summary_df_docs["instance_num"] = summary_df_docs.instance_num.str.split("_")
    summary_df_docs = summary_df_docs.explode("instance_num")
    res_filtered_df["instance_num"] = (
        res_filtered_df.source.str.split(".").str.get(0).str.split("_").str.get(0)
    )
    res_filtered_df = res_filtered_df.merge(
        summary_df_docs[["instance_num", "encounter_num", "patient_num", "note_date"]],
        on="instance_num",
        how="right",
    ).rename(columns={"note_date": "start_date"})
    if complete_case_only:
        med_data = pd.read_pickle(RES_DIR / "med_from_structured_data.pkl")
        bio_data = pd.read_pickle(RES_DIR / "bio_from_structured_data.pkl")
        res_filtered_df = res_filtered_df.merge(
            med_data[["encounter_num", "patient_num"]].drop_duplicates(),
            on=["encounter_num", "patient_num"],
            how="inner",
        )
        res_filtered_df = res_filtered_df.merge(
            bio_data[["encounter_num", "patient_num"]].drop_duplicates(),
            on=["encounter_num", "patient_num"],
            how="inner",
        )
    patient_group = None
    for med in config["ATC_codes"][disease].keys():
        res_filtered_df[med] = res_filtered_df.med == med
    patient_group = res_filtered_df.groupby("patient_num", as_index=False).agg(
        {
            **{med: "sum" for med in config["ATC_codes"][disease].keys()},
        }
    )
    for med in config["ATC_codes"][disease].keys():
        patient_group[med] = patient_group[med] >= 1

    return res_filtered_df, patient_group

def prepare_nlp_bio_df(disease, config, threshold=0, complete_case_only=False):
    global regex_pos
    global regex_neg
    RES_DIR = BASE_DIR / "data" / "final_results"
    res_filtered_df = pd.read_pickle(RES_DIR / disease / "filtered_bio_from_nlp.pkl")
    if threshold:
        res_filtered_df = res_filtered_df[res_filtered_df.score > threshold]
    summary_df_docs = pd.read_pickle(BASE_DIR / "data" / "CRH" / "summary_df_docs.pkl")
    summary_df_docs = summary_df_docs[summary_df_docs.disease == disease]
    summary_df_docs["instance_num"] = summary_df_docs.instance_num.str.split("_")
    summary_df_docs = summary_df_docs.explode("instance_num")
    res_filtered_df["instance_num"] = (
        res_filtered_df.source.str.split(".").str.get(0).str.split("_").str.get(0)
    )
    res_filtered_df = res_filtered_df.merge(
        summary_df_docs[["instance_num", "encounter_num", "patient_num", "note_date"]],
        on="instance_num",
        how="right",
    ).rename(columns={"note_date": "start_date"})
    if complete_case_only:
        med_data = pd.read_pickle(RES_DIR / "med_from_structured_data.pkl")
        bio_data = pd.read_pickle(RES_DIR / "bio_from_structured_data.pkl")
        res_filtered_df = res_filtered_df.merge(
            med_data[["encounter_num", "patient_num"]].drop_duplicates(),
            on=["encounter_num", "patient_num"],
            how="inner",
        )
        res_filtered_df = res_filtered_df.merge(
            bio_data[["encounter_num", "patient_num"]].drop_duplicates(),
            on=["encounter_num", "patient_num"],
            how="inner",
        )
    res_filtered_df["lower_bound"] = (
        res_filtered_df["range_value"].str.split("[\-–]").str.get(0)
    )
    res_filtered_df["lower_bound"] = res_filtered_df["lower_bound"].where(
        res_filtered_df["range_value"].str.split("[\-–]").str.len() == 2,
        None,
    )
    res_filtered_df["lower_bound"] = (
        res_filtered_df["lower_bound"]
        .mask(
            res_filtered_df["range_value"].str.contains(">") == True,
            res_filtered_df.range_value.str.extract("(\d+[,\.]?\d*)")[0],
        )
        .str.replace(",", ".")
        .astype(float)
    )
    res_filtered_df["upper_bound"] = (
        res_filtered_df["range_value"].str.split("[\-–<>]").str.get(-1)
    )
    res_filtered_df["upper_bound"] = res_filtered_df["upper_bound"].where(
        res_filtered_df["range_value"].str.split("[\-–]").str.len() == 2,
        None,
    )
    res_filtered_df["upper_bound"] = (
        res_filtered_df["upper_bound"]
        .mask(
            res_filtered_df["range_value"].str.contains("<") == True,
            res_filtered_df.range_value.str.extract("(\d+[,\.]?\d*)")[0],
        )
        .str.replace(",", ".")
        .astype(float)
    )
    res_filtered_df["value_as_number"] = (
        res_filtered_df.value_cleaned.str.extract("(\d+[,\.]?\d*)")[0]
        .str.replace(",", ".")
        .astype(float)
    )
    res_filtered_df["positive_value"] = (
        res_filtered_df["value_as_number"] < res_filtered_df["lower_bound"]
    ) | (res_filtered_df["value_as_number"] > res_filtered_df["upper_bound"])
    res_filtered_df["positive_value"] = res_filtered_df["positive_value"].mask(
        (res_filtered_df["value_as_number"].isna())
        | (res_filtered_df["range_value"].isna()),
        None,
    )
    res_filtered_df["positive_text"] = res_filtered_df.non_digit_value.str.match(
        regex_pos
    )
    res_filtered_df["negative_text"] = res_filtered_df.non_digit_value.str.match(
        regex_neg
    )
    res_filtered_df["positive_text"] = res_filtered_df.positive_text.where(
        res_filtered_df["positive_text"], None
    )
    res_filtered_df["negative_text"] = res_filtered_df.negative_text.where(
        res_filtered_df["negative_text"], None
    )
    patient_group = None
    if len(config["CUI_codes"][disease].keys()) > 0:
        for bio in config["CUI_codes"][disease].keys():
            res_filtered_df[bio] = res_filtered_df.bio == bio
            res_filtered_df[f"{bio} positive"] = (res_filtered_df.bio == bio) & (
                (res_filtered_df["positive_text"])
                | (res_filtered_df["positive_value"])
                | (
                    res_filtered_df["negative_text"].isna()
                    & res_filtered_df["positive_text"].isna()
                    & res_filtered_df["positive_value"].isna()
                )
            )
        patient_group = res_filtered_df.groupby("patient_num", as_index=False).agg(
            {
                **{
                    bio: "sum"
                    for bio in config["CUI_codes"][disease].keys()
                },
                **{
                    f"{bio} positive": "sum"
                    for bio in config["CUI_codes"][disease].keys()
                },
            }
        )
        for bio in config["CUI_codes"][disease].keys():
            patient_group[bio] = patient_group[bio] >= 1
            patient_group[f"{bio} positive"] = patient_group[f"{bio} positive"] >= 1

    return res_filtered_df, patient_group