from functools import reduce

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn2, venn3


def plot_hist(
    unit_convert,
    possible_values,
    nlp_filtered_df,
    tabular_filtered_df,
    title: bool = False,
    smooth: bool = False,
    top_title: str = None,
    methods=["Tabular", "NLP - Tabular", "NLP + Tabular"],
):
    alt.data_transformers.disable_max_rows()
    bio_charts = []
    first = True
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c"]
    n_patients = nlp_filtered_df.patient_num.nunique()
    for (bio, units), color in zip(unit_convert.items(), colors):
        nlp_filtered_bio = nlp_filtered_df[nlp_filtered_df.bio == bio].rename(
            columns={"value_as_number": "value"}
        )
        nlp_filtered_bio["method"] = "NLP"
        tabular_filtered_bio = tabular_filtered_df[tabular_filtered_df.bio == bio][
            ["patient_num", "bio", "unit", "value"]
        ]
        tabular_filtered_bio["method"] = "Tabular"
        filtered_bio = pd.concat([nlp_filtered_bio, tabular_filtered_bio])
        filtered_bio.unit = filtered_bio.unit.str.lower()
        filtered_bio = filtered_bio[filtered_bio.unit.isin(units.keys())]
        if not filtered_bio.empty:
            for unit, rate in units.items():
                filtered_bio["value"] = filtered_bio["value"].mask(
                    filtered_bio["unit"] == unit, filtered_bio["value"] * rate
                )
            final_unit = (
                "ratio" if list(units.keys())[0] == "nounit" else list(units.keys())[0]
            )
            filtered_bio["unit"] = final_unit
            filtered_bio = filtered_bio[
                (filtered_bio.value >= 0) & (filtered_bio.value <= possible_values[bio])
            ]
            filtered_bio = filtered_bio[
                ["patient_num", "bio", "unit", "value", "method"]
            ].drop_duplicates()
            benefit_nlp = filtered_bio[filtered_bio.method == "NLP"].merge(
                filtered_bio[filtered_bio.method == "Tabular"],
                on=["patient_num", "bio", "unit", "value"],
                how="left",
            )
            benefit_nlp = benefit_nlp[benefit_nlp["method_y"].isna()][
                ["patient_num", "bio", "unit", "value"]
            ]
            benefit_nlp["method"] = "NLP - Tabular"
            filtered_bio_nlp_tabular = filtered_bio.copy()
            filtered_bio_nlp_tabular["method"] = "NLP + Tabular"
            filtered_bio_nlp_tabular = filtered_bio_nlp_tabular[
                ["patient_num", "bio", "unit", "value", "method"]
            ].drop_duplicates()
            filtered_bio = pd.concat(
                [filtered_bio, filtered_bio_nlp_tabular, benefit_nlp]
            )
            method_hists = []
            for method in methods:
                method_filtered_bio = filtered_bio[filtered_bio.method == method]
                n_method_tests = len(method_filtered_bio)
                n_method_patients = method_filtered_bio.patient_num.nunique()
                ratio_patients = n_method_patients / n_patients * 100
                text_value = (
                    [f"+{n_method_tests} laboratory tests"]
                    if method == "NLP - Tabular"
                    else [
                        f"{n_method_tests} laboratory tests",
                        f"{n_method_patients} patients ({ratio_patients:.1f} %)",
                    ]
                )
                if method == "NLP - Tabular":
                    title_method = ["Benefits of the", "unstructured data"]
                elif method == "NLP + Tabular":
                    title_method = ["Structured and", "unstructured data"]
                elif method == "Tabular":
                    title_method = "Structured data only"
                elif method == "NLP":
                    title_method = "Unstructured data only"
                base = alt.Chart(
                    method_filtered_bio,
                    title=alt.TitleParams(
                        text=title_method if first else "",
                        orient="left",
                        anchor="middle",
                        fontSize=24,
                    ),
                )
                if smooth:
                    res_density = (
                        base.transform_density(
                            "value",
                            counts=True,
                            extent=[0, possible_values[bio]],
                            as_=["value", "density"],
                        )
                        .mark_area(color=color)
                        .encode(
                            alt.X("value:Q").title(None),
                            alt.Y("density:Q").title("Density"),
                            alt.Tooltip(["value:Q", "density:Q"]),
                        )
                    ).properties(width=400, height=300)
                else:
                    res_density = (
                        base.mark_bar(color=color).encode(
                            x=alt.X(
                                "value:Q",
                                bin=alt.Bin(
                                    extent=[0, possible_values[bio]],
                                    step=possible_values[bio] / 20,
                                ),
                                scale=alt.Scale(zero=False),
                                title=None,
                            ),
                            y=alt.Y(
                                "count():Q",
                                title="Frequency",
                            ),
                            tooltip=alt.Tooltip(["value:Q", "count():Q"]),
                        )
                    ).properties(width=400, height=300)
                ratio_stay = (
                    alt.Chart({"values": [{}]})
                    .mark_text(align="center", baseline="top", fontSize=20)
                    .encode(
                        x=alt.value(300),  # pixels from left
                        y=alt.value(5),  # pixels from top
                        text=alt.value(text_value),
                    )
                )
                res_box_plot = (
                    alt.Chart(method_filtered_bio)
                    .mark_boxplot(color=color, extent="min-max")
                    .encode(
                        alt.X("value:Q")
                        .scale(domain=[0, possible_values[bio]])
                        .title(f"{bio} ({final_unit})")
                        .axis(ticks=False, labels=False)
                    )
                ).properties(width=400, height=20)
                method_hist = alt.vconcat(
                    res_density + ratio_stay, res_box_plot, spacing=0
                ).resolve_scale(x="shared")
                method_hists.append(method_hist)
            bio_chart = reduce(
                lambda bar_chart_1, bar_chart_2: alt.vconcat(bar_chart_1, bar_chart_2)
                .resolve_scale(x="shared")
                .resolve_scale(y="independent"),
                method_hists,
            )
            if title:
                bio_chart = bio_chart.properties(
                    title=alt.TitleParams(
                        text=bio, orient="top", anchor="middle", fontSize=24
                    )
                )
            first = False
            bio_charts.append(bio_chart)
    chart = (
        reduce(
            lambda bar_chart_1, bar_chart_2: (bar_chart_1 | bar_chart_2)
            .resolve_scale(x="independent")
            .resolve_scale(y="independent"),
            bio_charts,
        )
        .properties(title=f"{top_title} ({n_patients} patients)")
        .configure_title(anchor="middle", fontSize=30, orient="top")
        .configure_axis(
            labelFontSize=18,
            titleFontSize=18,
            labelLimit=500,
        )
        .configure_legend(
            labelFontSize=18,
            titleFontSize=18,
            labelLimit=500,
        )
    )
    return chart


def plot_venn(
    patient_group,
    bio_venn,
    english_title,
    method,
    first: bool = True,
    remove_pos: bool = True,
):
    if remove_pos:
        patient_group = patient_group[list(bio_venn.values()) + ["patient_num"]]
        for key, val in bio_venn.items():
            renamed_val = val.split(" positive")[0]
            patient_group = patient_group.rename(columns={val: renamed_val})
            bio_venn[key] = renamed_val
    if len(bio_venn) == 2:
        subsets = (
            ((patient_group[bio_venn["A"]]) & ~(patient_group[bio_venn["B"]])).sum(),
            (~(patient_group[bio_venn["A"]]) & (patient_group[bio_venn["B"]])).sum(),
            ((patient_group[bio_venn["A"]]) & (patient_group[bio_venn["B"]])).sum(),
        )
        venn = venn2(subsets=subsets, set_labels=bio_venn.values())
    elif len(bio_venn) == 3:
        subsets = (
            (
                (patient_group[bio_venn["A"]])
                & ~(patient_group[bio_venn["B"]])
                & ~(patient_group[bio_venn["C"]])
            ).sum(),
            (
                ~(patient_group[bio_venn["A"]])
                & (patient_group[bio_venn["B"]])
                & ~(patient_group[bio_venn["C"]])
            ).sum(),
            (
                (patient_group[bio_venn["A"]])
                & (patient_group[bio_venn["B"]])
                & ~(patient_group[bio_venn["C"]])
            ).sum(),
            (
                ~(patient_group[bio_venn["A"]])
                & ~(patient_group[bio_venn["B"]])
                & (patient_group[bio_venn["C"]])
            ).sum(),
            (
                (patient_group[bio_venn["A"]])
                & ~(patient_group[bio_venn["B"]])
                & (patient_group[bio_venn["C"]])
            ).sum(),
            (
                ~(patient_group[bio_venn["A"]])
                & (patient_group[bio_venn["B"]])
                & (patient_group[bio_venn["C"]])
            ).sum(),
            (
                (patient_group[bio_venn["A"]])
                & (patient_group[bio_venn["B"]])
                & (patient_group[bio_venn["C"]])
            ).sum(),
        )
        venn = venn3(subsets=subsets, set_labels=bio_venn.values())

    total_patients = patient_group.patient_num.nunique()
    if len(bio_venn) == 3:
        total_pos = patient_group[
            patient_group[bio_venn["A"]]
            | patient_group[bio_venn["B"]]
            | patient_group[bio_venn["C"]]
        ].patient_num.nunique()
    elif len(bio_venn) == 2:
        total_pos = patient_group[
            patient_group[bio_venn["A"]] | patient_group[bio_venn["B"]]
        ].patient_num.nunique()
    for idx, subset in enumerate(venn.subset_labels):
        if subset:
            subset.set_text(
                f"{subset.get_text()}\n{int(subset.get_text())/total_patients*100:.1f}%"
            )
    if first:
        plt.title(
            (
                f"{english_title} ({total_patients} patients) \n\n "
                f"{method}: {total_pos} ({total_pos/total_patients * 100:.1f} %)"
            )
        )
    else:
        plt.title(f"{method}: {total_pos} ({total_pos/total_patients * 100:.1f} %)")
    # plt.show()


def plot_summary_med(
    nlp_patient_group,
    structured_patient_group,
    english_title,
    cohort_name,
):
    n_patient = len(structured_patient_group)
    benefit_nlp = pd.DataFrame()
    for drug in nlp_patient_group.drop(columns="patient_num").columns:
        benefit_nlp[drug] = nlp_patient_group[drug] & ~structured_patient_group[drug]
    benefit_nlp_summary = pd.DataFrame(
        benefit_nlp.sum(),
        columns=["Detected"],
    )
    benefit_nlp_summary["Benefits of the untructured data"] = (
        "+ "
        + benefit_nlp_summary["Detected"].astype(str)
        + " ("
        + (benefit_nlp_summary["Detected"] / n_patient * 100)
        .astype(float)
        .round(1)
        .astype(str)
        + " %)"
    )
    benefit_nlp_summary = benefit_nlp_summary.drop(columns=["Detected"])

    nlp_summary = pd.DataFrame(
        nlp_patient_group.drop(columns="patient_num").sum(),
        columns=["Detected"],
    )
    nlp_summary["Untructured data"] = (
        nlp_summary["Detected"].astype(str)
        + " ("
        + (nlp_summary["Detected"] / n_patient * 100).astype(float).round(1).astype(str)
        + " %)"
    )
    nlp_summary = nlp_summary.drop(columns=["Detected"])

    structued_summary = pd.DataFrame(
        structured_patient_group.drop(columns="patient_num").sum(),
        columns=["Detected"],
    )
    structued_summary["Structured data"] = (
        structued_summary["Detected"].astype(str)
        + " ("
        + (structued_summary["Detected"] / n_patient * 100)
        .astype(float)
        .round(1)
        .astype(str)
        + " %)"
    )
    structued_summary = structued_summary.drop(columns=["Detected"])

    nlp_structured_patient_group = (
        pd.concat([nlp_patient_group, structured_patient_group])
        .groupby("patient_num", as_index=False)
        .max()
    )
    nlp_structued_summary = pd.DataFrame(
        nlp_structured_patient_group.drop(columns="patient_num").sum(),
        columns=["Detected"],
    )
    nlp_structued_summary["Both"] = (
        nlp_structued_summary["Detected"].astype(str)
        + " ("
        + (nlp_structued_summary["Detected"] / n_patient * 100)
        .astype(float)
        .round(1)
        .astype(str)
        + " %)"
    )
    nlp_structued_summary = nlp_structued_summary.drop(columns=["Detected"])

    table = pd.concat(
        [structued_summary, nlp_summary, benefit_nlp_summary, nlp_structued_summary],
        axis=1,
    )
    return table.style.set_caption(f"{cohort_name} cohort ({n_patient} patients)")


def plot_summary_bio(
    nlp_patient_group,
    structured_patient_group,
    tests_to_keep,
    english_title,
    cohort_name,
    remove_pos: bool = True,
):
    if remove_pos:
        nlp_patient_group = nlp_patient_group[tests_to_keep + ["patient_num"]]
        structured_patient_group = structured_patient_group[
            tests_to_keep + ["patient_num"]
        ]
        renamed_tests = []
        for test in tests_to_keep:
            renamed_test = test.split(" positive")[0]
            nlp_patient_group = nlp_patient_group.rename(columns={test: renamed_test})
            structured_patient_group = structured_patient_group.rename(
                columns={test: renamed_test}
            )
            renamed_tests.append(renamed_test)
        tests_to_keep = renamed_tests
    n_patient = len(structured_patient_group)

    benefit_nlp = pd.DataFrame({"patient_num": nlp_patient_group["patient_num"]})
    for test in tests_to_keep:
        benefit_nlp[test] = nlp_patient_group[test] & ~structured_patient_group[test]
    benefit_nlp_summary = pd.DataFrame(
        benefit_nlp[tests_to_keep].sum(),
        columns=["Detected"],
    )
    benefit_nlp_summary["Benefits of the untructured data"] = (
        "+ "
        + benefit_nlp_summary["Detected"].astype(str)
        + " ("
        + (benefit_nlp_summary["Detected"] / n_patient * 100)
        .astype(float)
        .round(1)
        .astype(str)
        + " %)"
    )
    benefit_nlp_summary = benefit_nlp_summary.drop(columns=["Detected"])

    nlp_summary = pd.DataFrame(
        nlp_patient_group[tests_to_keep].sum(),
        columns=["Detected"],
    )
    nlp_summary["Untructured data"] = (
        nlp_summary["Detected"].astype(str)
        + " ("
        + (nlp_summary["Detected"] / n_patient * 100).astype(float).round(1).astype(str)
        + " %)"
    )
    nlp_summary = nlp_summary.drop(columns=["Detected"])

    structued_summary = pd.DataFrame(
        structured_patient_group[tests_to_keep].sum(),
        columns=["Detected"],
    )
    structued_summary["Structured data only"] = (
        structued_summary["Detected"].astype(str)
        + " ("
        + (structued_summary["Detected"] / n_patient * 100)
        .astype(float)
        .round(1)
        .astype(str)
        + " %)"
    )
    structued_summary = structued_summary.drop(columns=["Detected"])

    nlp_structured_patient_group = (
        pd.concat([nlp_patient_group, structured_patient_group])
        .groupby("patient_num", as_index=False)
        .max()
    )
    nlp_structued_summary = pd.DataFrame(
        nlp_structured_patient_group[tests_to_keep].sum(),
        columns=["Detected"],
    )
    nlp_structued_summary["Both"] = (
        nlp_structued_summary["Detected"].astype(str)
        + " ("
        + (nlp_structued_summary["Detected"] / n_patient * 100)
        .astype(float)
        .round(1)
        .astype(str)
        + " %)"
    )
    nlp_structued_summary = nlp_structued_summary.drop(columns=["Detected"])

    table = pd.concat(
        [structued_summary, nlp_summary, benefit_nlp_summary, nlp_structued_summary],
        axis=1,
    )
    return table.style.set_caption(f"{cohort_name} cohort ({n_patient} patients)")
