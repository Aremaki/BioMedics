import os
import re

from loguru import logger

import time
from datetime import date, datetime, timedelta
from typing import Union
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from biomedics.extract_measurement.bio_lexical_variant import lexical_var_non_digit_values
from biomedics.utils.extract_pandas_from_brat import extract_pandas

def _clean_lexical_variant(lex_var):
    """
    Clean lexical variant to remove pollution

    """
    lex_var_str = str(lex_var).lstrip().rstrip()

    clean_lex_var = re.sub(r"\u2026", " ", lex_var_str)  # remove ...
    clean_lex_var = re.sub(r"\\n", " ", clean_lex_var)  # remove "\n"
    clean_lex_var = re.sub(r"\n", " ", clean_lex_var)
    clean_lex_var = re.sub(r"\best\b", " ", clean_lex_var)  # remove "est"

    clean_lex_var = re.sub(r"[\:\\]", " ", clean_lex_var)

    clean_lex_var = re.sub(r"\bcomprise?\b", " ", clean_lex_var)
    clean_lex_var = re.sub(r"\bavec\b", "", clean_lex_var)
    clean_lex_var = re.sub(r"\bpour\b", "", clean_lex_var)
    clean_lex_var = re.sub(r"\bsoit\b", "", clean_lex_var)
    clean_lex_var = re.sub(r"\bdans\b", "", clean_lex_var)
    clean_lex_var = re.sub(r"\bsans\b", "", clean_lex_var)
    clean_lex_var = re.sub(r"\bselon\b", " ", clean_lex_var)

    clean_lex_var = re.sub(r'\s*\/\s*', '/', clean_lex_var)
    clean_lex_var = re.sub(r'rapp?orts?', ' ', clean_lex_var)
    clean_lex_var = re.sub(r'dosages?', ' ', clean_lex_var)
    clean_lex_var = re.sub(r'numerations?', ' ', clean_lex_var)
    clean_lex_var = re.sub(r'\bratio?\b', ' ', clean_lex_var)
    clean_lex_var = re.sub('µ', 'micro', clean_lex_var)

    return clean_lex_var


def _replace_lexical_var(value):
    """
    Remove non digit values in from the lexical variant
    Terms are reported in the dict lexical_var_non_digit_values

    """

    for key, regex in lexical_var_non_digit_values.items():
        val = re.sub(regex, key, str(value))
    return val


def _clean_value(value):
    """
    Clean value with pollution
    """
    pattern = r"[^\d.,/<>]+"
    value_cleaned = re.sub(pattern, "", str(value))

    match = re.findall(r'([<>]?\d+[\.\,]?\d*\/\d+[\.\,]?\d*|[<>]?\d+[\.\,]?\d*)$', value_cleaned)
    match_unclassic_format = re.findall(r'\b(0\d+)\b', value_cleaned)

    if len(match) > 0:
        val = match[0]
        if "/" in val or ">" in val or "<" in val:
            return value_cleaned
        if "," in val:
            val = val.replace(",", ".")
        if val.count(".") > 1:
            val = re.sub(r"\.{1,10}", ".", val)
        return float(val)
    elif len(match_unclassic_format) > 0:
        return np.nan
    elif str(value) == 'nan':
        return np.nan
    else:
        return value_cleaned


def _clean_unit(lex_var):
    """
    Clean unit

    """
    clean_unit = re.sub(r"gramm?es?", "g", str(lex_var))
    clean_unit = re.sub(r"micr?o?", "µ", clean_unit)
    clean_unit = re.sub(r"micor", "µ", clean_unit)
    clean_unit = re.sub(r"moles?", "mol", clean_unit)
    clean_unit = re.sub(r"molaires?", "mol", clean_unit)
    clean_unit = re.sub(r"mill?i", "m", clean_unit)
    clean_unit = re.sub(r"litres?", "l", clean_unit)
    clean_unit = re.sub(r"secondes?", "sec", clean_unit)
    clean_unit = re.sub(r"secs?", "sec", clean_unit)
    clean_unit = re.sub(r"jours?", "j", clean_unit)
    clean_unit = re.sub(r"par", "/", clean_unit)
    clean_unit = re.sub(r"minutes?", "min", clean_unit)
    clean_unit = re.sub(r"heures?", "h", clean_unit)
    clean_unit = re.sub(r"\_", "", clean_unit)
    clean_unit = re.sub(r"\n", "", clean_unit)
    clean_unit = re.sub(r"\ℓ", "l", clean_unit)
    return clean_unit


def _normalise_unit(unit):
    """
    Normalise units

    """
    
    unit_nospace = str(unit).replace(" ", "").lower()

    clean_unit_stripped = unit_nospace.strip("-,.¦| ")

    match_perc = re.findall(r'%', str(clean_unit_stripped))
    match_24 = re.findall(r'\/24\w\b', str(clean_unit_stripped))
    match_x10E = re.findall(r'x?10\*\d+\/[a-z]{1,2}', str(clean_unit_stripped))
    match_dfg_unit = re.findall(
        r'(ml\s?\/\s?mi?n\s?\/\s?1[.,]73\s?m2|ml\s?\/\s?mi?n\s?\/\s?m2)', str(clean_unit_stripped)
    )
    if len(match_perc) > 0:
        return '%'
    elif len(match_x10E) > 0:
        return str(clean_unit_stripped)
    elif len(match_24) > 0:
        return clean_unit_stripped
    elif len(match_dfg_unit) > 0:
        return clean_unit_stripped
    else:
        return clean_unit_stripped


def normalize_expression(expression):
    return ' '.join(sorted(expression.split()))

def _convert_brat_spans(span):
    span_match = re.compile(r"^(\d+).*\s(\d+)$").match(span)
    span_start = int(span_match.group(1))
    span_end = int(span_match.group(2))
    return [span_start, span_end]

def convert_brat_to_spark(spark, brat_dir, labels):
    # Convert span to list with span_start, span_end. It considers the new lines by adding one character.
    df = extract_pandas(IN_BRAT_DIR=brat_dir)
    df = df.loc[
        df["label"].isin(labels)
    ]
    df["span_converted"] = df["span"].apply(_convert_brat_spans)
    df["span_start"] = df["span_converted"].str.get(0)
    df["span_end"] = df["span_converted"].str.get(1)
    df["lexical_variant"] = df["term"].copy()
    df = df[["term", "lexical_variant", "source", "span_start", "span_end", "label"]]
    spark_df = spark.createDataFrame(df)
    return spark_df


def match_bio_to_biocomp(df_bio, df_biocomp):
    """
    Match bio to biocomp entities retrieved from the EDS-Biomedic model and clean the lexical variants for pollution

    """
    df_biocomp = df_biocomp.withColumnRenamed("lexical_variant", "lexical_variant_biocomp")
    df_biocomp = df_biocomp.withColumnRenamed("term", "term_biocomp")

    for column in df_bio.columns:
        df_bio = df_bio.withColumnRenamed(column, column + "_bio")

    df_biocomp_bio = df_biocomp.join(
        df_bio,
        on=(df_biocomp.source == df_bio.source_bio)
        & (df_biocomp.span_start <= df_bio.span_start_bio)
        & (df_biocomp.span_end >= df_bio.span_end_bio),
        how='left',
    )
    df_biocomp_bio = df_biocomp_bio.dropDuplicates()

    pattern_ellipse = r'((\.\.\.|…)\s?([¹²³⁰⁴⁵⁶⁷⁸⁹0-9]))'
    df_biocomp_bio = df_biocomp_bio.withColumn(
        "lexical_variant_biocomp",
        F.regexp_replace("lexical_variant_biocomp", pattern_ellipse, ' '),
    )
    df_biocomp_bio = df_biocomp_bio.withColumn(
        "lexical_variant_bio", F.regexp_replace("lexical_variant_bio", pattern_ellipse, ' ')
    )

    df_biocomp_bio = df_biocomp_bio.withColumn(
        "lexical_variant_biocomp", F.regexp_replace("lexical_variant_biocomp", '…', ' ')
    )
    df_biocomp_bio = df_biocomp_bio.withColumn(
        "lexical_variant_bio", F.regexp_replace("lexical_variant_bio", '…', ' ')
    )
    df_biocomp_bio = df_biocomp_bio.withColumn(
        "lexical_variant_biocomp",
        F.regexp_replace("lexical_variant_biocomp", r"[⁰¹²³⁴⁵⁶⁷⁸⁹]" , ' '),
    )
    df_biocomp_bio = df_biocomp_bio.withColumn(
        "lexical_variant_bio",
        F.regexp_replace("lexical_variant_bio", r"[⁰¹²³⁴⁵⁶⁷⁸⁹]" , ' '),
    )
    return df_biocomp_bio


def match_date_pattern(df):
    """
    Return date if identified in the lexical_variant.
    If a date format is identified it is assigned to a new column `extracted_date` and remove the date in the lexical_variant
    """
    date_pattern = r"(\d{2}/20\d{2}|\d{2}/\d{2}|20\d{2}-\d{2}-\d{2}|20\d{2}-\d{2}|20\d{2})"

    df_with_dates = df.withColumn(
        "extracted_date", F.regexp_extract("lexical_variant_biocomp", date_pattern, 1)
    )

    df_with_dates = df_with_dates.withColumn(
        "lexical_variant_stripped", F.regexp_replace("lexical_variant_stripped", date_pattern, "")
    )

    return df_with_dates


def extract_clean_range_value(df):
    """
    Return range value if identified in the lexical_variant.
    If a range value format is identified it is assigned to a new column `range_value` and removed in the lexical_variant
    """

    pattern_range_value = r"([|¦][<>]\d+[\.,]?\d*$|\(\s?[nN]?\s?:?\s?[<>]\s?\d+[\.\,]?\d*\s?\)?|\(?\s?[nN]?\s?[:=]?\s?\d+[\.,]?\d*\s?[\-–]\s?\d+[\.,]?\d*\s?\)?)"
    df_range_val = df.withColumn(
        "range_value", F.regexp_extract(F.col("lexical_variant_stripped"), pattern_range_value, 1)
    )

    df_range_val = df_range_val.withColumn(
        "lexical_variant_stripped",
        F.regexp_replace("lexical_variant_stripped", pattern_range_value, " "),
    )
    df_range_val = df_range_val.withColumn(
        'range_value', F.regexp_replace("range_value", r"[^\d><\-–\.,]", "")
    )
    df_range_val = df_range_val.withColumn(
        "range_value",
        F.when(F.col("range_value") == "", None).otherwise(F.col("range_value")),
    )
    return df_range_val


def clean_lexical_variant(df):
    """
    Clean lexical variant to prepare for units and values extractions
    """

    df = df.withColumn(
        'lexical_variant_stripped', F.regexp_replace("lexical_variant_stripped", r'⁹', " ")
    )
    df = df.withColumn(
        'lexical_variant_stripped', F.regexp_replace("lexical_variant_stripped", r'm²', "m2")
    )
    df = df.withColumn(
        'lexical_variant_stripped', F.regexp_replace("lexical_variant_stripped", r'm³', "m3")
    )

    df = df.withColumn(
        'lexical_variant_stripped',
        F.translate(F.lower(F.col('lexical_variant_stripped')), '⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789'),
    )
    df = df.withColumn(
        'lexical_variant_stripped', F.regexp_replace("lexical_variant_stripped", r'à', "=")
    )
    df = df.withColumn(
        'lexical_variant_stripped', F.regexp_replace("lexical_variant_stripped", r'(\:)', " ")
    )
    df = df.withColumn(
        'lexical_variant_stripped',
        F.translate(
            F.lower(F.col('lexical_variant_stripped')),
            'ãäöüáàäčçďéêěèïíîĺľňóôŕšťúûůýž',
            'aaouaaaccdeeeeiiillnoorstuuuyz',
        ),
    )

    clean_lexical_variant_udf = F.udf(_clean_lexical_variant, StringType())
    df_biocomp_bio_clean = df.withColumn(
        "lexical_variant_stripped", clean_lexical_variant_udf(df["lexical_variant_stripped"])
    )
    
    return df_biocomp_bio_clean


def extract_clean_non_digit_value(df):
    """
    Return non digit value if identified in the lexical_variant.
    If a non digit value format is identified it is assigned to a new column `non_digit_value` and removed in the lexical_variant
    """

    replace_lexical_var_udf = F.udf(_replace_lexical_var, StringType())

    df = df.withColumn(
        "lexical_variant_stripped", replace_lexical_var_udf(F.col("lexical_variant_stripped"))
    )

    pattern_non_digit_val = '|'.join(lexical_var_non_digit_values.values())

    df_non_digit_val = df.withColumn(
        "non_digit_value",
        F.regexp_extract(F.col("lexical_variant_stripped"), pattern_non_digit_val, 0),
    )
    df_non_digit_val = df_non_digit_val.withColumn(
        'non_digit_value', F.regexp_replace("non_digit_value", r"[^\w+-]", "")
    )
    df_non_digit_val = df_non_digit_val.withColumn(
        "lexical_variant_stripped",
        F.regexp_replace("lexical_variant_stripped", pattern_non_digit_val, " "),
    )
    
    df_non_digit_val = df_non_digit_val.withColumn(
        "non_digit_value",
        F.when(F.col("non_digit_value") == "", None).otherwise(F.col("non_digit_value")),
    )
    return df_non_digit_val


def extract_clean_units(df):
    """
    Return unit if identified in the lexical_variant.
    If a unit format is identified it is assigned to a new column `unit` and removed in the lexical_variant
    """
    clean_units_udf = F.udf(_clean_unit, StringType())
    df = df.withColumn("lexical_variant_stripped", clean_units_udf(df["lexical_variant_stripped"]))

    pattern_10E = r'(?i)(\d+(?:\.\d+)?)(x?10\*\d+\/[A-Za-z]+)'

    df = df.withColumn(
        "lexical_variant_stripped",
        F.regexp_replace("lexical_variant_stripped", pattern_10E, r'$1 $2'),
    )

    pattern_digit_nospace_wt_unit = r'(?i)(\d+)(g?\/mm3|ml\s?\/\s?mi?n\s?\/\s?m2|ml\s?\/\s?mi?n\s?\/\s?1[.,]73\s?m2|ml\s?\/\s?mi?n|µ?m?n?p?g\/24h|µ?u?m?n?p?g\s?\/\s?d?m?l|m?mol/24h|seco?n?d?e?s?|minu?t?e?s?|µ?u?m?n?p?g\s?\/\s?mmol?|µ?u?m?n?p?mol\s?\/\s?[a-z0-9µ]+|µ?u?m?n?p?g\s?\/\s?\w+?|mm\s?h\s?g\.?|µ?u?m?n?p?mol|\%|fl\b|mgy\.cm|gigas?\s*\/?\w+?|µ?u?m?n?p?g\b|\/?mm\b|gl\b|mil?li\s*grammes?|grammes?\s*\/?\s*\w*|micro\s*mol\/?\w*|\/?p?g?ml|m?osmol\/?\w*|molaires?|u\s*\/\s*l|m?ui\s*\/\s*m?l|k?pa\b|[a-zA-Zµ]+\/\w+|µ?u?m?n?m\b|[|¦]µ?u?m?n?s\b|\/[a-zA-Zµ]+\b)'

    df = df.withColumn(
        "lexical_variant_stripped",
        F.regexp_replace("lexical_variant_stripped", pattern_digit_nospace_wt_unit, r'$1 $2'),
    )

    units = r"(?i)(x?10\*\d+\s?\/[a-z]{1,2}|g?\/mm3|ml\s?\/\s?mi?n\s?\/\s?m2|ml\s?\/\s?mi?n\s?\/\s?1[.,]73\s?m2|ml\s?\/\s?mi?n|μ?µ?m?n?p?g\/24h|µ?u?m?n?p?g\s?\/\s?d?m?l|m?mol/24h|seco?n?d?e?s?|\bminu?t?e?s?|µ?u?m?n?p?g\s?\/\s?mmol?|μ?µ?u?m?n?p?mol\s?\/\s?[a-z0-9µμ]+|μ?µ?u?m?n?p?g\s?\/\s?\w+|mm\s?h\s?g\.?|μ?µ?u?m?n?p?mol|\%|\bfl\b|mgy\.cm|gigas?\s*\/?\w+|\bμ?µ?u?m?n?p?g\b|\/?mm\b|gl\b|mil?li\s*grammes?|grammes?\s*\/?\s*\w*|micro\s*mol\/?\w*|\/?p?g?ml|m?osmol\/?\w*|molaires?|u\s*\/\s*l|m?ui\s*\/\s*m?l|\bk?pa\b|[a-zA-Zµμμ]+\/\w+|\bµ?u?m?n?m\b|[|¦]μ?µ?u?m?n?s\b|[\s|¦]\/[a-zA-Zµμ]+\b)"

    df_units = df.withColumn("unit", F.regexp_extract("lexical_variant_stripped", units, 1))

    df_units = df_units.withColumn(
        "lexical_variant_stripped", F.regexp_replace("lexical_variant_stripped", units, " ")
    )

    normalise_units_udf = F.udf(_normalise_unit, StringType())
    df_units = df_units.withColumn("unit", normalise_units_udf(df_units["unit"]))
    
    df_units = df_units.withColumn(
        "unit",
        F.when(F.col("unit") == "", None).otherwise(F.col("unit")),
    )

    return df_units


def extract_clean_values(df):
    """
    Return value if identified in the lexical_variant.
    If a value format is identified it is assigned to a new column `value` and removed in the lexical_variant
    Value is then cleaned and assigned to a new column `value_cleaned`
    """

    pattern_values = r"([<>]?\s?\d+[\.\,]?\s?\d*\/\d+[\.\,]?\s?\d*|[<>]?\s?\d+[\.\,]?\s?\d*|\d)"

    df_values = df.withColumn(
        "value", F.regexp_extract("lexical_variant_stripped", pattern_values, 1)
    )

    df_values = df_values.withColumn(
        "lexical_variant_stripped",
        F.regexp_replace("lexical_variant_stripped", pattern_values, " "),
    )

    clean_values_udf = F.udf(_clean_value, StringType())
    df_values = df_values.withColumn("value_cleaned", clean_values_udf(df_values["value"]))

    df_values = df_values.withColumn(
        "value_cleaned",
        F.when(F.col("value_cleaned") == "", None).otherwise(F.col("value_cleaned")),
    )

    return df_values


def extract_fluids_source(df):
    """
    Return fluid source the bio was measured in if identified in the lexical_variant.
    If a fluid format is identified it is assigned to a new column `fluid_source` and removed in the lexical_variant
    The fluid terms are normalised usign fuzzy matching using the reference dictionnary fluide_source
    """

    pattern_fluids = r"(?i)(sang\w*|urine\w*|serique\w*|plasma\w*|foetal\w*|capill?air\w*|serum|urinaire\w*|\bur\b|selle\w*|vessie|veineux|veineuse|\blcr\b)"

    df_fluids = df.withColumn(
        "fluid_source", F.regexp_extract("lexical_variant", pattern_fluids, 1)
    )

    df_fluids = df_fluids.withColumn(
        "lexical_variant", F.regexp_replace("lexical_variant", pattern_fluids, " ")
    )

    return df_fluids

def extract_clean_subsequent_lex_var(df):
    """
    Return subsequent lexical variant after removing all the previous entities.
    lexical variant is normalised using fuzzy matching and lexical variant dictionnary bios_lexical_variant_fuzzy_dict
    if the similarity score is higher then threshold the lexical variant term associated is returned in the column `lexical_variant_term`, the associated key
    in the column `lexical_variant_key` and the similarity score in the column `max_fuzzy_score`
    If the the max similarity score is not higher then the threshold then it returns the initial term
    """

    pattern_sub_lexical_var = r"(?i)(\b[A-Za-z]+(\.[A-Za-z]+)*\b)"

    df_subs_lex_var = df.withColumn(
        "lexical_variant",
        F.when(F.col("lexical_variant_bio") != "", F.col("lexical_variant_bio")).otherwise(
            F.regexp_extract(F.col("lexical_variant_stripped"), pattern_sub_lexical_var, 1)
        ),
    )

    df_subs_lex_var = df_subs_lex_var.withColumn(
        'lexical_variant',
        F.translate(
            F.lower(F.col('lexical_variant')),
            'ãäöüáàäčçďéêěèïíîĺľňóôŕšťúûůýž',
            'aaouaaaccdeeeeiiillnoorstuuuyz',
        ),
    )

    clean_lexical_variant_udf = F.udf(_clean_lexical_variant, StringType())
    df_subs_lex_var = df_subs_lex_var.withColumn(
        "lexical_variant", clean_lexical_variant_udf(df_subs_lex_var["lexical_variant"])
    )

    df_subs_lex_var = df_subs_lex_var.withColumn(
        'lexical_variant', F.regexp_replace("lexical_variant", "\bsur\b", "/")
    )
    df_subs_lex_var = df_subs_lex_var.withColumn(
        'lexical_variant', F.regexp_replace("lexical_variant", "[<>-]", " ")
    )
    df_subs_lex_var = df_subs_lex_var.withColumn(
        "lexical_variant", F.ltrim(df_subs_lex_var["lexical_variant"])
    )
    df_subs_lex_var = df_subs_lex_var.withColumn(
        "lexical_variant", F.rtrim(df_subs_lex_var["lexical_variant"])
    )

    df_subs_lex_var_after_removal_fluid_source = extract_fluids_source(df_subs_lex_var)

    return df_subs_lex_var_after_removal_fluid_source


def bio_post_processing(spark, script_config, brat_dir, output_dir):
    logger.info('-------------Load entities-------------')
    
    start_t1 = time.time()
    
    label_key = script_config["label_key"]
    labels_to_remove = script_config["labels_to_remove"]
    all_labels = [label_key] + labels_to_remove
    
    df_ents_sparks = convert_brat_to_spark(spark, brat_dir, all_labels)
    df_ents_bio_comp = df_ents_sparks.filter(F.col('label') == label_key)
    df_ents_bio = df_ents_sparks.filter(F.col('label').isin(labels_to_remove))
    end_t1 = time.time()

    logger.info(
        f"Biocomp table len:  {df_ents_bio_comp.count()} entities\
     {df_ents_bio_comp.select('source').distinct().count()} docs\
     \nBio table len: {df_ents_bio.count()}\
     {df_ents_bio.select('source').distinct().count()} docs\
     \nProcessed in {round(end_t1 - start_t1,3)} secs"
    )

    logger.info('-------------Link bio to bio_comp-------------')
    start_t2 = time.time()
    df_biocomp_bio = match_bio_to_biocomp(df_ents_bio, df_ents_bio_comp)
    end_t2 = time.time()

    logger.info(
        f"Biocomp linked with bio table len:  {df_biocomp_bio.count()} entities \nprocessed in {round(end_t2 - start_t2,3)} secs"
    )

    logger.info('-------------Remove bio from bio_comp-------------')
    start_t3 = time.time()
    df_biocomp_bio = df_biocomp_bio.withColumn(
        "lexical_variant_stripped",
        F.expr("trim(replace(lexical_variant_biocomp, lexical_variant_bio, ' '))"),
    )
    df_biocomp_bio = df_biocomp_bio.withColumn(
        "lexical_variant_stripped",
        F.regexp_replace("lexical_variant_stripped", r"G\/+[lL]\b", "x10*9/l"),
    )
    end_t3 = time.time()

    logger.info(f"processed in {round(end_t3 - start_t3,3)} secs ---")

    logger.info('-------------Extract the date from ents containing one -------------')
    start_t4 = time.time()
    df_biocomp_bio = match_date_pattern(df_biocomp_bio)
    end_t4 = time.time()
    logger.info(f"processed in {round(end_t4 - start_t4,3)} secs ---")

    logger.info('-------------Clean and extract range value -------------')
    start_t5 = time.time()
    df_biocomp_bio = extract_clean_range_value(df_biocomp_bio)
    end_t5 = time.time()
    logger.info(f"processed in {round(end_t5 - start_t5,3)} secs")

    logger.info('-------------Clean the entities col with pollution-------------')
    start_t6 = time.time()
    df_biocomp_bio_clean = clean_lexical_variant(df_biocomp_bio)
    end_t6 = time.time()
    logger.info(f"processed in {round(end_t6 - start_t6,3)} secs")

    logger.info('-------------Clean and extract non digit values-------------')
    start_t7 = time.time()
    df_biocomp_bio_clean = extract_clean_non_digit_value(df_biocomp_bio_clean)
    end_t7 = time.time()
    logger.info(f"processed in {round(end_t7 - start_t7,3)} secs")

    logger.info('-------------Clean and extract units-------------')
    start_t8 = time.time()
    df_biocomp_bio_clean = extract_clean_units(df_biocomp_bio_clean)
    end_t8 = time.time()
    logger.info(f"processed in {round(end_t8 - start_t8,3)} secs")

    logger.info('-------------Clean and extract values-------------')
    start_t9 = time.time()
    df_biocomp_bio_clean = extract_clean_values(df_biocomp_bio_clean)
    end_t9 = time.time()
    logger.info(f"processed in {round(end_t9 - start_t9,3)} secs")

    logger.info('-------------Clean, extract subsequent lexical variant-------------')
    start_t11 = time.time()
    df_biocomp_bio_clean = extract_clean_subsequent_lex_var(df_biocomp_bio_clean)
    end_t11 = time.time()
    logger.info(f"processed in {round(end_t11 - start_t11,3)} secs")

    logger.info(
        f"Dataframe shape after processing: {df_biocomp_bio_clean.count()}\
    \nNumber of unique note_id in the initial df : {df_biocomp_bio.select('source').distinct().count()}, after processing: {df_biocomp_bio_clean.select('source').distinct().count()}"
    )
    
    logger.info('---------------Select columns of interest and convert to Pandas---------------')
    df_final = df_biocomp_bio_clean.select(
        'source',
        'span_start',
        'span_start_bio',
        'span_end',
        'span_end_bio',
        'term_biocomp',
        'term_bio',
        'lexical_variant_stripped',
        'range_value',
        'non_digit_value',
        'unit',
        'lexical_variant',
        'value_cleaned',
        'extracted_date',
        'fluid_source',
    )
    logger.info('---------------Convert to Pandas---------------')
    # Try using pyarrow via HDFS to convert object to pandas as it is way faster.
    try:
        parquet_path = "hdfs://bbsedsi/user/cse200093/temp.parquet"
        df_final.write.mode("overwrite").parquet(parquet_path)
        df_final = pq.read_table(parquet_path)
        df_final = df_final.to_pandas()
    except:
        df_final = df_final.toPandas()
        
    logger.info('---------------Save in pickle---------------')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_final.to_pickle(f"{output_dir}/pred_with_measurement.pkl")

    logger.info('---------------saved---------------')
    logger.info(f'saved path: {output_dir}')

    final_time = time.time()
    logger.info(f"Total processing time:  {round(final_time - start_t1,3)} secs")
