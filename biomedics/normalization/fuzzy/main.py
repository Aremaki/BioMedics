import duckdb
import edsnlp
import pandas as pd
from unidecode import unidecode

from biomedics.ner.brat import BratConnector
from biomedics.normalization.fuzzy.exception import exception_list


class FuzzyNormaliser:
    def __init__(
        self,
        df_path,
        drug_dict,
        label_to_normalize,
        with_qualifiers,
        method="lev",
        atc_len=7,
    ):
        if df_path.endswith("json"):
            self.df = pd.read_json(df_path)
            if "term_to_norm" not in self.df.columns:
                self.df["term_to_norm"] = self.df.term.str.lower().str.strip()
        elif df_path.endswith("pkl"):
            self.df = pd.read_pickle(df_path)
            if "term_to_norm" not in self.df.columns:
                self.df["term_to_norm"] = self.df.term.str.lower().str.strip()
        else:
            self.df = self.gold_generation(df_path, label_to_normalize, with_qualifiers)
        self.unashable_cols = []
        for col in self.df.columns:
            if self.df[col].apply(lambda x: isinstance(x, (set, list))).sum() > 0:
                self.unashable_cols.append(col)
                self.df[col] = self.df[col].astype(str)
        self.df["term_to_norm"] = self.df["term_to_norm"].apply(lambda x: unidecode(x))

        if isinstance(drug_dict, dict):
            self.drug_dict = drug_dict
            merged_dict = {}
            for atc_code, values in self.drug_dict.items():
                # Shorten the ATC code
                shortened_code = atc_code[:atc_len]

                # Check if the shortened_code already exists in the merged dictionary
                if shortened_code in merged_dict:
                    # Merge the arrays
                    merged_dict[shortened_code] = list(
                        set(merged_dict[shortened_code] + values)
                    )

                else:
                    # Add a new entry for the shortened ATC code
                    merged_dict[shortened_code] = values

            merged_dict = (
                pd.DataFrame.from_dict({"norm_term": merged_dict}, "index")
                .T.explode("norm_term")
                .reset_index()
                .rename(columns={"index": "label"})
            )
            merged_dict.norm_term = merged_dict.norm_term.str.split(",")
            merged_dict = merged_dict.explode("norm_term").reset_index(drop=True)
            self.drug_dict = merged_dict
        else:
            self.drug_dict = drug_dict
            if "norm_term" not in self.df.columns:
                self.drug_dict.columns = ["label", "norm_term"]
                self.drug_dict = self.drug_dict.explode("norm_term")
            self.drug_dict.norm_term = self.drug_dict.norm_term.str.lower().str.strip()
        self.method = method

    def get_gold(self):
        return self.df

    def get_dict(self):
        return self.drug_dict

    def gold_generation(self, df_path, label_to_normalize, with_qualifiers):
        qualifiers = (
            ["Temporality", "Certainty", "Action", "Negation"]
            if with_qualifiers
            else []
        )
        doc_list = BratConnector(df_path).brat2docs(edsnlp.blank("eds"))
        ents_list = []
        for doc in doc_list:
            if label_to_normalize in doc.spans.keys():
                for ent in doc.spans[label_to_normalize]:
                    if hasattr(ent._, "Tech") and ent._.Tech:
                        continue
                    ent_data = [
                        ent.text,
                        doc._.note_id + ".ann",
                        [ent.start_char, ent.end_char],
                        ent.text.lower().strip(),
                    ]
                    for qualifier in qualifiers:
                        ent_data.append(getattr(ent._, qualifier))
                    ents_list.append(ent_data)
        df_columns = ["term", "source", "span_converted", "term_to_norm"] + qualifiers
        df = pd.DataFrame(ents_list, columns=df_columns)
        return df

    def normalize(self, threshold=10):
        for index, row in self.df.iterrows():
            for k, v in exception_list.items():
                if row["term_to_norm"] in v:
                    self.df.at[index, "term_to_norm"] = k

        if self.method == "exact":
            self.df = self.df.merge(
                self.drug_dict, how="left", left_on="term_to_norm", right_on="norm_term"
            )
        if self.method == "lev":
            df_1 = self.df.copy()
            self.drug_dict.copy()
            merged_df = duckdb.query(
                (
                    "select *, levenshtein(df_1.term_to_norm, df_2.norm_term) score "
                    f"from df_1, df_2 where score < {threshold}"
                )
            ).to_df()
            merged_df["term_to_norm_len"] = merged_df.term_to_norm.str.len()
            merged_df["norm_term_len"] = merged_df.norm_term.str.len()
            merged_df["max_len"] = merged_df[["norm_term_len", "term_to_norm_len"]].max(
                axis=1
            )
            merged_df["score"] = 1 - (merged_df["score"] / merged_df["max_len"])
            merged_df = merged_df.drop(
                columns=["norm_term_len", "term_to_norm_len", "max_len"]
            )
            idx = (
                merged_df.groupby(["source", "span_converted"])["score"].transform(max)
                == merged_df["score"]
            )
            merged_df = merged_df[idx]
            merged_df = df_1.merge(merged_df, on=list(df_1.columns), how="left")
            self.df = merged_df
        if self.method == "jaro_winkler":
            df_1 = self.df.copy()
            self.drug_dict.copy()
            merged_df = duckdb.query(
                "select *, jaro_winkler_similarity(df_1.term_to_norm, df_2.norm_term)"
                f" score from df_1, df_2 where score > {threshold}"
            ).to_df()
            idx = (
                merged_df.groupby(["source", "span_converted"])["score"].transform(max)
                == merged_df["score"]
            )
            merged_df = merged_df[idx]
            merged_df = df_1.merge(merged_df, on=list(df_1.columns), how="left")
            self.df = merged_df
        self.df = self.df.groupby(
            list(self.df.columns.difference({"label", "norm_term"})),
            as_index=False,
            dropna=False,
        ).agg({"label": list, "norm_term": set})
        for col in self.unashable_cols:
            self.df[col] = self.df[col].apply(lambda x: eval(x))
        return self.df
