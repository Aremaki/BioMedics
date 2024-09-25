import os

import pandas as pd
from loguru import logger

from biomedics import BASE_DIR


def filter_bio_structured(config_anabio_codes):
    logger.info("Filter laboratory tests from structured data")
    bio_result_folder_path = BASE_DIR / "data" / "final_results"
    bio_from_structured_data = pd.read_pickle(
        bio_result_folder_path / "bio_from_structured_data.pkl"
    )
    logger.debug("raw df: {}", bio_from_structured_data.shape)
    codes_to_keep = {"disease": [], "concept_cd": [], "bio": []}
    for disease, anabio_codes in config_anabio_codes.items():
        for label, code_list in anabio_codes.items():
            for code in code_list:
                codes_to_keep["disease"].append(disease)
                codes_to_keep["concept_cd"].append(f"LAB:{code}")
                codes_to_keep["bio"].append(label)
    filtered_bio = bio_from_structured_data.merge(
        pd.DataFrame(codes_to_keep), on=["disease", "concept_cd"]
    )
    for disease in config_anabio_codes.keys():
        path_to_res = bio_result_folder_path / disease
        if not os.path.exists(path_to_res):
            os.mkdir(path_to_res)
        filtered_bio[filtered_bio.disease == disease].to_pickle(
            path_to_res / "filtered_bio_from_structured_data.pkl"
        )
    filtered_bio.to_pickle(
        bio_result_folder_path / "filtered_bio_from_structured_data.pkl"
    )
    logger.debug("processed df: {}", filtered_bio.shape)


def filter_med_structured(config_atc_codes):
    logger.info("Filter drug treatments from structured data")
    med_result_folder_path = BASE_DIR / "data" / "final_results"
    med_from_structured_data = pd.read_pickle(
        med_result_folder_path / "med_from_structured_data.pkl"
    )
    logger.debug("raw df: {}", med_from_structured_data.shape)
    codes_to_keep = {"disease": [], "valueflag_cd": [], "med": []}
    for disease, atc_codes in config_atc_codes.items():
        for label, code_list in atc_codes.items():
            for code in code_list:
                codes_to_keep["disease"].append(disease)
                codes_to_keep["valueflag_cd"].append(code)
                codes_to_keep["med"].append(label)
    filtered_med = med_from_structured_data.merge(
        pd.DataFrame(codes_to_keep), on=["disease", "valueflag_cd"]
    )
    med_from_structured_data["valueflag_cd"] = med_from_structured_data[
        "valueflag_cd"
    ].str.slice(stop=5)
    filtered_med_short = med_from_structured_data.merge(
        pd.DataFrame(codes_to_keep), on=["disease", "valueflag_cd"]
    )
    filtered_med = pd.concat([filtered_med, filtered_med_short])
    for disease in config_atc_codes.keys():
        path_to_res = med_result_folder_path / disease
        if not os.path.exists(path_to_res):
            os.mkdir(path_to_res)
        filtered_med[filtered_med.disease == disease].to_pickle(
            path_to_res / "filtered_med_from_structured_data.pkl"
        )
    filtered_med.to_pickle(
        med_result_folder_path / "filtered_med_from_structured_data.pkl"
    )
    logger.debug("processed df: {}", filtered_med.shape)


def filter_bio_nlp(config_cui_codes):
    logger.info("Filter laboratory tests from unstructured data")
    bio_result_folder_path = BASE_DIR / "data" / "final_results"
    # List of df by disease for concatenation
    res_part_filtered_list = []
    for disease, cui_codes in config_cui_codes.items():
        ### Load each res dataset to concat them in one unique df
        res_part_df = pd.read_pickle(
            bio_result_folder_path / disease / "pred_bio_coder_all.pkl"
        )
        logger.debug("raw df for {}: {}", disease, res_part_df.shape)
        res_part_df["disease"] = disease

        ### Filter CUIS to keep
        codes_to_keep = {"disease": [], "label": [], "bio": []}
        for label, code_list in cui_codes.items():
            for code in code_list:
                codes_to_keep["disease"].append(disease)
                codes_to_keep["label"].append(code)
                codes_to_keep["bio"].append(label)
        res_part_df = res_part_df.explode("label")
        res_part_filtered = res_part_df.merge(
            pd.DataFrame(codes_to_keep), on=["disease", "label"]
        )
        res_part_filtered = res_part_filtered.groupby(
            list(res_part_filtered.columns.difference(["label"])),
            as_index=False,
            dropna=False,
        ).agg({"label": list})

        ### Save for future concatenation
        path_to_res = bio_result_folder_path / disease
        if not os.path.exists(path_to_res):
            os.mkdir(path_to_res)
        res_part_filtered.to_pickle(path_to_res / "filtered_bio_from_nlp.pkl")
        logger.debug("processed df for {}: {}", disease, res_part_filtered.shape)
        res_part_filtered_list.append(res_part_filtered)
    res_filtered_df = pd.concat(res_part_filtered_list)
    res_filtered_df.to_pickle(bio_result_folder_path / "filtered_bio_from_nlp.pkl")


def filter_med_nlp(config_atc_codes):
    logger.info("Filter drug treatments from unstructured data")
    med_result_folder_path = BASE_DIR / "data" / "final_results"
    # List of df by disease for concatenation
    res_part_filtered_list = []
    for disease, atc_codes in config_atc_codes.items():
        ### Load each res dataset to concat them in one unique df
        res_part_df = pd.read_pickle(
            med_result_folder_path / disease / "pred_med_fuzzy_jaro_winkler.pkl"
        )
        logger.debug("raw df for {}: {}", disease, res_part_df.shape)
        res_part_df["disease"] = disease
        res_part_df["instance_num"] = res_part_df["source"].str.slice(stop=-4)

        ### Filter ATC to keep
        codes_to_keep = {"disease": [], "label": [], "med": []}
        for label, code_list in atc_codes.items():
            for code in code_list:
                codes_to_keep["disease"].append(disease)
                codes_to_keep["label"].append(code)
                codes_to_keep["med"].append(label)
        res_part_df = res_part_df.explode("label")
        res_part_filtered = res_part_df.merge(
            pd.DataFrame(codes_to_keep), on=["disease", "label"]
        )
        res_part_filtered.norm_term = res_part_filtered.norm_term.astype(str)
        res_part_filtered.span_converted = res_part_filtered.span_converted.astype(str)
        res_part_filtered = res_part_filtered.groupby(
            list(res_part_filtered.columns.difference(["label"])),
            as_index=False,
            dropna=False,
        ).agg({"label": list})
        res_part_filtered.norm_term = res_part_filtered.norm_term.apply(
            lambda x: eval(x)
        )
        res_part_filtered.span_converted = res_part_filtered.span_converted.apply(
            lambda x: eval(x)
        )
        ### Save for future concatenation
        path_to_res = med_result_folder_path / disease
        if not os.path.exists(path_to_res):
            os.mkdir(path_to_res)
        res_part_filtered.to_pickle(path_to_res / "filtered_med_from_nlp.pkl")
        logger.debug("processed df for {}: {}", disease, res_part_filtered.shape)
        res_part_filtered_list.append(res_part_filtered)
    res_filtered_df = pd.concat(res_part_filtered_list)
    res_filtered_df.to_pickle(med_result_folder_path / "filtered_med_from_nlp.pkl")
