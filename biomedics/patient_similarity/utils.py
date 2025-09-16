import random
import re
import warnings
from collections import Counter
from pathlib import Path

import altair as alt
import duckdb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import torch
from sklearn.manifold import TSNE
from tqdm.notebook import tqdm
from wordcloud import WordCloud

from biomedics.patient_similarity.XT_distance import distance_files_by_lab

warnings.filterwarnings("ignore")

alt.data_transformers.disable_max_rows()
alt.themes.enable("dark")


def add_atc_code(doc, drug_df, text_preprocessor):
    predicted_entities = [
        text_preprocessor(
            text=ent.text, remove_stopwords=True, remove_special_characters=True
        )
        for ent in doc.ents
        if ent.label_ == "Chemical_and_drugs"
    ]
    if predicted_entities:
        df_1 = pd.DataFrame(  # noqa: F841
            {
                "term": [ent.text for ent in doc.spans["Chemical_and_drugs"]],
                "term_to_norm": predicted_entities,
            }
        )
        df_2 = drug_df  # noqa: F841
        threshold = 0.8
        merged_df = duckdb.query(
            f"""select *, jaro_winkler_similarity(df_1.term_to_norm, df_2.norm_term) score from df_1, df_2 where score > {threshold}"""
        ).to_df()
        idx = (
            merged_df.groupby(["term_to_norm"])["score"].transform(max)
            == merged_df["score"]
        )
        merged_df = merged_df[idx]
        ents = []
        ents_drugs = []
        for ent in doc.ents:
            if ent.label_ == "Chemical_and_drugs":
                if not ent._.Negation:
                    ent.label_ = "Médicament"
                    filter_df = merged_df[merged_df.term == ent.text]
                    if not filter_df.empty:
                        ent.kb_id_ = filter_df.label.iloc[0]
                    ents.append(ent)
                    ents_drugs.append(ent)
            else:
                ents.append(ent)
        doc.ents = ents
        doc.spans["Médicament"] = ents_drugs
    return doc


def add_label_class(doc, model, tokenizer, text_preprocessor, label_names, device):
    predicted_entities = [
        text_preprocessor(
            text=ent.text, remove_stopwords=True, remove_special_characters=True
        )
        for ent in doc.ents
        if ent.label_ == "DISO"
    ]
    if predicted_entities:
        inputs = tokenizer(
            predicted_entities,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        # Ensure inputs are on the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        # Move to CPU for safe Python-side iteration and comparisons
        probs = torch.sigmoid(logits).detach().cpu()

        ents = []
        ents_diso = []
        ents_bio = []
        i = 0
        for ent in doc.ents:
            if ent.label_ == "DISO":
                ent.label_ = "Signe et symptôme"
                if not ent._.Negation:
                    high_confidence_labels = [
                        label_names[j].split("_")[-1].capitalize()
                        for j, p in enumerate(probs[i])
                        if p > 0.8
                    ]
                    ent.kb_id_ = " | ".join(high_confidence_labels)
                    ents_diso.append(ent)
                    ents.append(ent)
                i += 1
            elif ent.label_ in ["BIO", "BIO_comp"]:
                ent.label_ = "Biologie"
                ents_bio.append(ent)
                ents.append(ent)
            elif ent.label_ == "Duration":
                ent.label_ = "Durée"
                ents.append(ent)
            elif ent.label_ == "DATE":
                ent.label_ = "Date"
                ents.append(ent)
            else:
                ents.append(ent)
        doc.ents = ents
        doc.spans["Signe et symptôme"] = ents_diso
        doc.spans["Biologie"] = ents_bio
    return doc


def create_source_terms(doc, selected_labels, text_preprocessor):
    source_patient = {}
    for selected_label in selected_labels:
        terms = []
        labels = []
        for ent in doc.ents:
            if ent.label_ == "Signe et symptôme" and ent.kb_id_:
                if selected_label in ent.kb_id_.split(" | "):
                    terms.append(
                        text_preprocessor(
                            text=ent.text,
                            remove_stopwords=True,
                            remove_special_characters=True,
                        )
                    )
                    labels.append(ent.kb_id_)
        if not terms:
            return None
        source_patient[selected_label] = (
            pd.DataFrame({"terms": terms, "labels": labels})
            .groupby(["terms"])
            .size()
            .to_dict()
        )
    return source_patient


def compute_distance(
    source_patient,
    target_patients,
    df_embed,
    vectorizer,
    selected_labels,
):
    selected_patients = target_patients[target_patients.labels.isin(selected_labels)]
    sources = selected_patients["source"].drop_duplicates().tolist()
    selected_patients = selected_patients.groupby(
        ["source", "labels", "normalized_term"]
    ).size()
    distances = {selected_label: [] for selected_label in selected_labels}
    distances["mean"] = []
    distances["source"] = []
    for source in tqdm(
        sources,
        desc="Computing distances",
        leave=True,
        bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}",
    ):
        selected_patient = selected_patients[source]
        if set(selected_patient.keys().get_level_values(0)) >= set(selected_labels):
            distance = 0
            distances["source"].append(source)
            for selected_label in selected_labels:
                vector = vectorizer.fit_transform(
                    [
                        source_patient[selected_label],
                        selected_patient.get(selected_label).to_dict(),
                    ]
                )
                embedding_matrix = (
                    df_embed[
                        df_embed["normalized_term"].isin(
                            vectorizer.get_feature_names_out()
                        )
                    ]
                    .drop(columns="normalized_term")
                    .values.tolist()
                )
                local_dist_matrix = sklearn.metrics.pairwise.cosine_distances(  # type: ignore
                    embedding_matrix
                )
                label_distance = distance_files_by_lab(
                    vector[0],
                    vector[1],
                    distance_matrix=local_dist_matrix,
                    track_time=False,
                    verbose=False,
                )
                distances[selected_label].append(label_distance)
                distance += label_distance  # type: ignore
            distances["mean"].append(distance / len(selected_labels))

    # convert to pandas dataframe
    distances = pd.DataFrame(distances)

    return distances


def plot_output_treatment(
    patient_drugs, most_similar_patients, atc_code_path, ATC_rank=5
):
    patient_drugs = patient_drugs[
        patient_drugs.source.isin(most_similar_patients.keys())
    ]
    atc_code = pd.read_excel(atc_code_path)[["ATC_code", "Libellé français"]].rename(
        columns={"ATC_code": "label", "Libellé français": "label_name"}
    )
    patient_drugs["label"] = patient_drugs.label.str[:ATC_rank]
    atc_code["label_name"] = atc_code["label_name"].str.capitalize()
    patient_drugs = patient_drugs.drop(columns="label_name").merge(atc_code, on="label")
    base = alt.Chart(patient_drugs)

    treatment_selection = alt.selection_point(fields=["label_name"])

    label_color = alt.condition(
        treatment_selection,
        alt.Color(
            "label_name:N",
            legend=None,
            sort="-x",
        ),
        alt.value("lightgray"),
    )

    treatment_chart = (
        base.mark_bar()
        .encode(
            x=alt.X("distinct(source)")
            .title("Nombre de patients traités")
            .axis(labelFontSize=14, titleFontSize=16, orient="top"),  # X-axis on top
            y=alt.Y("label_name:N")
            .sort("-x")
            .axis(
                title="Traitements",
                titleAngle=0,
                titleAlign="right",
                titleY=-2,
                titleX=0,
                labelFontSize=14,
                titleFontSize=16,
            ),
            color=label_color,
            tooltip=[
                alt.Tooltip("label_name", title="Médicament"),
                alt.Tooltip("distinct(source)", title="Nombre de patients"),
            ],  # Tooltip added
        )
        .add_params(treatment_selection)
    )

    length_chart = (
        base.transform_filter(treatment_selection)
        .transform_aggregate(
            unique_length_of_stay="mean(length_of_stay)",
            groupby=["source", "label_stay"],
        )
        .mark_boxplot(extent=50)
        .encode(
            x=alt.X("label_stay")
            .title("Durée d'hospitalisation")
            .axis(labelFontSize=14, titleFontSize=16, orient="top"),
            y=alt.Y("mean(unique_length_of_stay):Q")
            .title("Nombre de jours")
            .scale(zero=False, domainMax=50, clamp=True)
            .axis(labelFontSize=14, titleFontSize=16),
        )
    ).properties(height=300)

    death_chart = (
        alt.Chart(patient_drugs, title="Décès")
        .transform_fold(
            [
                "0 - Décès à 30 jours",
                "1 - Décès à 90 jours",
                "2 - Décès à 180 jours",
            ],
            as_=["death_type", "death_status"],
        )
        .transform_filter(treatment_selection)
        .transform_aggregate(
            total_alive="distinct(source)",
            groupby=["death_type", "death_status"],
        )
        .transform_joinaggregate(
            total_patients="sum(total_alive)",
            groupby=["death_type"],
        )
        .transform_filter("datum.death_status == 0")
        .transform_calculate(total_death="datum.total_patients - datum.total_alive")
        .transform_calculate(perc_death="datum.total_death/datum.total_patients")
        .mark_bar()
        .encode(
            x=alt.X("death_type:N").title("").axis(labelFontSize=14, titleFontSize=16),
            y=alt.Y("perc_death:Q")
            .title("Pourcentage de décès")
            .axis(format=".1%", labelFontSize=14, titleFontSize=16)
            .scale(zero=True),
            tooltip=[
                alt.Tooltip("total_death:Q", title="Nombre de décès"),
                alt.Tooltip("total_patients:Q", title="Nombre de patients"),
                alt.Tooltip("perc_death:Q", format=".1%", title="Pourcentage de décès"),
            ],
        )
    ).properties(height=300)

    output_chart = treatment_chart | (death_chart & length_chart)

    return output_chart.configure_title(fontSize=16)


def plot_similartiy_network(distances, max_distance, sample_size=500):
    # Randomly select keys from the filtered dictionary
    if len(distances) > sample_size:
        sample_keys = random.sample(list(distances.keys()), k=sample_size)
    else:
        sample_keys = distances

    # Build a new dictionary with the sampled keys
    sampled_distances = {pid: distances[pid] for pid in sample_keys}
    map_distances = [0] + list(sampled_distances.values())  # Random distances
    num_nodes = len(map_distances)

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from((0, i) for i in range(1, num_nodes))  # Connect all nodes to 0

    # Generate spherical coordinates for uniform distribution
    phi = np.arccos(1 - 2 * np.random.rand(num_nodes))
    theta = 2 * np.pi * np.random.rand(num_nodes)

    # Convert to Cartesian coordinates
    pos = {
        i: (
            map_distances[i] * np.sin(phi[i]) * np.cos(theta[i]),
            map_distances[i] * np.sin(phi[i]) * np.sin(theta[i]),
            map_distances[i] * np.cos(phi[i]),
        )
        for i in range(num_nodes)
    }

    # Get Patient 1's position (center of sphere)
    p1_x, p1_y, p1_z = pos[0]

    # Extract edge coordinates
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        x0, y0, z0, x1, y1, z1 = *pos[u], *pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    # Define node sizes
    node_sizes = [
        30 if i == 0 else 12 for i in range(num_nodes)
    ]  # Bigger for Patient 1

    # 🎨 **Updated Colors**
    node_colors = [
        "#FF1493"
        if map_distances[i] > max_distance
        else "#00BFFF"  # Magenta for outside, Light Blue for inside
        for i in range(num_nodes)
    ]
    node_colors[0] = "#FFD700"  # Patient 1 in Gold

    # Add hover text for distances
    hover_texts = [
        f"Patient {i}<br> Distance avec P0: {map_distances[i]:.2f}"
        for i in range(num_nodes)
    ]

    # Generate a wireframe sphere (instead of a solid sphere)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)

    x_sphere = p1_x + max_distance * np.outer(np.cos(u), np.sin(v))
    y_sphere = p1_y + max_distance * np.outer(np.sin(u), np.sin(v))
    z_sphere = p1_z + max_distance * np.outer(np.ones_like(u), np.cos(v))

    # Convert sphere into wireframe lines
    sphere_lines = []
    for i in range(len(u)):
        sphere_lines.append(
            go.Scatter3d(
                x=x_sphere[i, :],
                y=y_sphere[i, :],
                z=z_sphere[i, :],
                mode="lines",
                line=dict(color="cyan", width=1.5),
                hoverinfo="skip",
                showlegend=False,
                opacity=0.2,  # Lower opacity to avoid blocking hover
            )
        )
    for j in range(len(v)):
        sphere_lines.append(
            go.Scatter3d(
                x=x_sphere[:, j],
                y=y_sphere[:, j],
                z=z_sphere[:, j],
                mode="lines",
                line=dict(color="cyan", width=1.5),
                hoverinfo="skip",
                showlegend=False,
                opacity=0.2,  # Lower opacity to avoid blocking hover
            )
        )

    # Create figure with dark theme
    fig = go.Figure()

    # 🔵 Add Sphere Wireframe (Non-blocking!)
    for line in sphere_lines:
        fig.add_trace(line)

    # 🔵 Edges
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=edge_x,
    #         y=edge_y,
    #         z=edge_z,
    #         mode="lines",
    #         line=dict(width=1.5, color="#808080"),  # Grayish Blue
    #         hoverinfo="none",
    #         showlegend=False,
    #     )
    # )

    # 🔴 Nodes
    fig.add_trace(
        go.Scatter3d(
            x=[pos[i][0] for i in G.nodes()],
            y=[pos[i][1] for i in G.nodes()],
            z=[pos[i][2] for i in G.nodes()],
            mode="markers+text",
            textposition="top center",
            marker=dict(size=node_sizes, color=node_colors, opacity=1.0),
            hoverinfo="text",
            hovertext=hover_texts,
            showlegend=False,
        )
    )

    # Apply dark mode settings
    fig.update_layout(
        title="",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="white"),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(0, 0, 0, 0)",
        ),
        dragmode="turntable",  # Allows easier rotation to access inner nodes
    )
    return fig


def plot_word_embeddings(sample_embed_with_lab):
    tsne = TSNE(n_components=3, random_state=0)
    projections = tsne.fit_transform(
        sample_embed_with_lab.drop(columns=["normalized_term", "labels"])
    )
    fig = px.scatter_3d(
        pd.DataFrame(projections, columns=["x", "y", "z"]),
        x="x",
        y="y",
        z="z",
        hover_name=sample_embed_with_lab["normalized_term"],
        color=sample_embed_with_lab["labels"],
        labels={"color": "Spécialité"},
    )

    fig.update_traces(marker_size=8)

    fig.update_layout(
        {
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
        },
        font=dict(color="white"),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
    )
    return fig


# Function to generate a word cloud
def _generate_wordcloud(term_list, title):
    # Make a dictionary {text: count}
    count_dict = dict(Counter(term_list))
    # Clean keys: remove newlines and strip spaces
    count_dict = {str(k).replace("\n", " ").strip(): v for k, v in count_dict.items()}

    wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="cool"
    ).generate_from_frequencies(count_dict)
    return wordcloud


def generate_wordcloud(target_patients, selected_specialties, most_similar_patients):
    terms_by_label = (
        target_patients[
            target_patients.source.isin(most_similar_patients.keys())
            & target_patients.labels.isin(selected_specialties)
        ]
        .groupby("labels")
        .agg({"normalized_term": list})
        .to_dict(orient="index")
    )
    # Generate word clouds
    plt.style.use("dark_background")  # Dark theme for better contrast
    for label in terms_by_label.keys():
        _generate_wordcloud(terms_by_label[label]["normalized_term"], label)


def parse_clinical_case(file_path: Path):
    """
    Parses a clinical case file to extract the clinical text, CIM-10 codes, and axes.

    Args:
        file_path (Path): The path to the clinical case file.

    Returns:
        tuple: A tuple containing the clinical text, a list of CIM-10 codes, and a list of axes.
               Returns (None, None, None) if the file cannot be parsed.
    """
    text = file_path.read_text(encoding="utf-8")

    # Split the text to isolate the clinical notes
    parts = re.split(r"-----Codes CIM-10-----", text, flags=re.IGNORECASE)
    if len(parts) < 2:
        return None, None, None

    clinical_text = parts[0].strip()

    # Further split to get CIM-10 codes and Axes
    remaining_text = parts[1]
    sub_parts = re.split(r"---Axes---", remaining_text, flags=re.IGNORECASE)
    if len(sub_parts) < 2:
        return clinical_text, [], []

    cim10_section = sub_parts[0].strip()
    sepcialities_section = sub_parts[1].strip()

    # Extract CIM-10 codes
    cim10_codes = [line.strip() for line in cim10_section.split("\n") if line.strip()]

    # Extract sepcialities
    sepcialities = [
        line.strip() for line in sepcialities_section.split("\n") if line.strip()
    ]

    return clinical_text, cim10_codes, sepcialities


def plot_output_outcomes(
    topk_outcomes,
    topk_drugs,
    topk_lab_tests,
    topk_disorder,
    specialties,
    bio_config,
    treatment_config,
    total_note,
    max_icd10=30,
):
    """
    Interactive Altair plots for outcomes:
    - Length of stay (boxplot, filtered)
    - Death outcomes (bar chart % with selection)
    - Top ICD-10 codes (bar chart, filtered)
    """

    # Disorder word_cloud
    wordclouds = []
    topk_disorder["labels"] = topk_disorder["labels"].str.split(r" \| ")
    disorder_df = topk_disorder.explode("labels")
    disorder_df = (
        disorder_df[disorder_df.labels.isin(specialties)]
        .groupby("labels")
        .agg({"normalized_term": list})
        .to_dict(orient="index")
    )
    # Generate word clouds
    for label in disorder_df.keys():
        wordclouds.append(
            _generate_wordcloud(disorder_df[label]["normalized_term"], label)
        )

    # --- Death dataframe ---
    death_cols = [
        c for c in topk_outcomes.columns if "Décès" in c or c == "Death_hospit"
    ]

    death_df = topk_outcomes[["source"] + death_cols].melt(
        id_vars="source", var_name="death_type", value_name="death_status"
    )
    # Keep numeric prefix for ordering but create clean label for x-axis
    death_df["death_order"] = death_df["death_type"].apply(
        lambda x: int(re.match(r"^(\d+)", x).group(1)) + 1  # type: ignore
        if re.match(r"^(\d+)", x)
        else 0
    )
    death_df["death_label"] = (
        death_df["death_type"]
        .apply(lambda x: re.sub(r"^\d+\s*-\s*", "", x))
        .str.replace("Décès à", "Death within")
        .str.replace("Death_hospit", "Death during hospitalization")
        .str.replace("jours", "days")
    )

    # Aggregate for chart
    death_agg = (
        death_df.groupby(["death_order", "death_label", "death_status"])["source"]
        .nunique()
        .reset_index(name="count")
    )
    total = (
        death_agg.groupby(["death_order", "death_label"])["count"]
        .sum()
        .reset_index(name="total")
    )
    death_agg = death_agg.merge(total, on=["death_order", "death_label"])
    death_agg["total"] = total_note
    death_agg["perc"] = death_agg["count"] / death_agg["total"]
    death_agg = death_agg[death_agg["death_status"] == 1]

    # --- Death outcomes chart ---
    death_chart = (
        alt.Chart(death_agg, title="Death")
        .mark_bar()
        .encode(
            x=alt.X(
                "death_label:N",
                title="",
                sort=alt.EncodingSortField(field="death_order", order="ascending"),
                axis=alt.Axis(labelFontSize=12, titleFontSize=14),
            ),
            y=alt.Y(
                "perc:Q",
                title="Percentage of Deaths",
                axis=alt.Axis(format="%", labelFontSize=12, titleFontSize=14),
            ),
            # color=alt.Color("death_label:N", legend=None),
            tooltip=[
                alt.Tooltip("death_label:N", title="Type"),
                alt.Tooltip("count:Q", title="Number of Deaths"),
                alt.Tooltip("total:Q", title="Total Note"),
                alt.Tooltip("perc:Q", format=".1%", title="Percentage"),
            ],
        )
        .properties(height=300, width=250)
    )

    # --- Length of stay chart (filtered) ---
    length_chart = (
        alt.Chart(topk_outcomes, title="Length of hospitalization")
        .transform_aggregate(
            unique_length_of_stay="mean(length_of_stay)",
            groupby=["source"],
        )
        .mark_boxplot(extent=100)
        .encode(
            y=alt.Y(
                "mean(unique_length_of_stay):Q",
                title="Days",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14),
            ).scale(zero=False, domainMax=100, clamp=True),
            color=alt.value("#1f77b4"),
        )
    ).properties(height=300, width=50)

    # --- ICD-10 codes chart (filtered) ---
    icd10_df = (
        topk_outcomes[["source", "icd10_codes"]]
        .explode("icd10_codes")
        .drop_duplicates()
    )
    icd10_df["icd10_codes"] = icd10_df["icd10_codes"].str.split("|").str.get(1)
    icd10_df = (
        icd10_df.groupby("icd10_codes", as_index=False)["source"]
        .nunique()
        .sort_values("source", ascending=False)
        .head(max_icd10)
    )
    icd10_df["total"] = total_note
    icd10_df["perc"] = icd10_df["source"] / icd10_df["total"]
    icd_chart = (
        alt.Chart(icd10_df, title=f"Top {max_icd10} ICD-10 Codes")
        .mark_bar()
        .encode(
            x=alt.X(
                "perc:Q",
                title="Percentage",
                axis=alt.Axis(format="%", labelFontSize=12, titleFontSize=14),
            ),
            y=alt.Y(
                "icd10_codes:N",
                sort="-x",
                title="",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14),
            ),
            tooltip=[
                alt.Tooltip("icd10_codes:N", title="Code"),
                alt.Tooltip("source:Q", title="Frequency"),
                alt.Tooltip("perc:Q", format=".1%", title="Percentage"),
                alt.Tooltip("total:Q", title="Total Note"),
            ],
        )
        .properties(height=300, width=300)
    )

    # --- Drugs chart (Filtered) ---

    # Step 1: build reverse mapping {code -> label}
    code_to_label = {
        code: label for label, codes in treatment_config.items() for code in codes
    }
    drug_mapping = pd.DataFrame(
        list(code_to_label.items()), columns=["label", "drug_name"]
    )
    filtered_df = topk_drugs.merge(drug_mapping, on="label")
    filtered_df = filtered_df[["source", "drug_name"]].drop_duplicates()
    filtered_df = (
        filtered_df.groupby("drug_name", as_index=False)["source"]
        .nunique()
        .sort_values("source", ascending=False)
    )
    filtered_df["total"] = total_note
    filtered_df["perc"] = filtered_df["source"] / filtered_df["total"]
    filtered_drug_chart = (
        alt.Chart(filtered_df, title="Filtered treatments in text")
        .mark_bar()
        .encode(
            x=alt.X(
                "perc:Q",
                title="Percentage",
                axis=alt.Axis(format="%", labelFontSize=12, titleFontSize=14),
            ),
            y=alt.Y(
                "drug_name:N",
                sort="-x",
                title="",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14),
            ),
            tooltip=[
                alt.Tooltip("drug_name:N", title="Drug"),
                alt.Tooltip("source:Q", title="Frequency"),
                alt.Tooltip("perc:Q", format=".1%", title="Percentage"),
                alt.Tooltip("total:Q", title="Total Note"),
            ],
        )
        .properties(height=300, width=300)
    )

    # --- Drugs chart (TOP) ---
    topk_drugs["label"] = topk_drugs["label"] + " : " + topk_drugs["label_name"]
    drugs_df = topk_drugs[["source", "label"]].drop_duplicates()
    drugs_df = (
        drugs_df.groupby("label", as_index=False)["source"]
        .nunique()
        .sort_values("source", ascending=False)
        .head(max_icd10)
    )
    drugs_df["total"] = total_note
    drugs_df["perc"] = drugs_df["source"] / drugs_df["total"]
    drug_chart = (
        alt.Chart(drugs_df, title=f"Top {max_icd10} treatments in text")
        .mark_bar()
        .encode(
            x=alt.X(
                "perc:Q",
                title="Percentage",
                axis=alt.Axis(format="%", labelFontSize=12, titleFontSize=14),
            ),
            y=alt.Y(
                "label:N",
                sort="-x",
                title="",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Code"),
                alt.Tooltip("source:Q", title="Frequency"),
                alt.Tooltip("perc:Q", format=".1%", title="Percentage"),
                alt.Tooltip("total:Q", title="Total Note"),
            ],
        )
        .properties(height=300, width=300)
    )

    # --- Bio chart (Filtered) ---

    # Step 1: build reverse mapping {code -> label}
    code_to_label = {
        code: label for label, codes in bio_config.items() for code in codes
    }
    bio_mapping = pd.DataFrame(
        list(code_to_label.items()), columns=["label", "bio_name"]
    )
    filtered_df = topk_lab_tests.merge(bio_mapping, on="label")
    filtered_df = filtered_df[
        (filtered_df.positive_value.eq(True)) | (filtered_df.positive_text.eq(True))
    ]
    filtered_df = filtered_df[["source", "bio_name"]].drop_duplicates()
    filtered_df = (
        filtered_df.groupby("bio_name", as_index=False)["source"]
        .nunique()
        .sort_values("source", ascending=False)
    )
    filtered_df["total"] = total_note
    filtered_df["perc"] = filtered_df["source"] / filtered_df["total"]
    filtered_bio_chart = (
        alt.Chart(filtered_df, title="Positive antibody in text")
        .mark_bar()
        .encode(
            x=alt.X(
                "perc:Q",
                title="Percentage",
                axis=alt.Axis(format="%", labelFontSize=12, titleFontSize=14),
            ),
            y=alt.Y(
                "bio_name:N",
                sort="-x",
                title="",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14),
            ),
            tooltip=[
                alt.Tooltip("bio_name:N", title="Laboratory test"),
                alt.Tooltip("source:Q", title="Frequency"),
                alt.Tooltip("perc:Q", format=".1%", title="Percentage"),
                alt.Tooltip("total:Q", title="Total Note"),
            ],
        )
        .properties(height=300, width=300)
    )

    # --- Bio chart (TOP) ---
    topk_lab_tests["label"] = (
        topk_lab_tests["label"] + " : " + topk_lab_tests["norm_term"]
    )
    bio_df = topk_lab_tests[["source", "label"]].drop_duplicates()
    bio_df = (
        bio_df.groupby("label", as_index=False)["source"]
        .nunique()
        .sort_values("source", ascending=False)
        .head(max_icd10)
    )
    bio_df["total"] = total_note
    bio_df["perc"] = bio_df["source"] / bio_df["total"]
    bio_chart = (
        alt.Chart(bio_df, title=f"Top {max_icd10} laboratory tests in text")
        .mark_bar()
        .encode(
            x=alt.X(
                "perc:Q",
                title="Percentage",
                axis=alt.Axis(format="%", labelFontSize=12, titleFontSize=14),
            ),
            y=alt.Y(
                "label:N",
                sort="-x",
                title="",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Code"),
                alt.Tooltip("source:Q", title="Frequency"),
                alt.Tooltip("perc:Q", format=".1%", title="Percentage"),
                alt.Tooltip("total:Q", title="Total Note"),
            ],
        )
        .properties(height=300, width=300)
    )

    return (
        death_chart,
        length_chart,
        icd_chart,
        drug_chart,
        filtered_drug_chart,
        bio_chart,
        filtered_bio_chart,
        wordclouds,
    )


# -------------------------
#  Stevens stratified sampling
# -------------------------
def stratified_sample_indices(
    distances: pd.DataFrame, m: int = 10, seed: int = 42
) -> pd.DataFrame:
    """
    weights: prior over documents (length N), sums to 1.
    m: desired sample size (e.g., 10)
    Returns:
      - selected_indices: list of sampled document indices (length <= m, ideally m unique)
      - bucket_inclusion_probs: array of 'g' for each bucket (sum of bucket weights)
      - buckets: list of (start_idx, end_idx) tuples (ranges in the sorted order)
    Implementation choices follow the paper's description:
      - sort items by weight descending
      - create buckets of size m
      - sample buckets with replacement m times with prob proportional to bucket weight sum
      - from each picked bucket pick one document uniformly at random without replacement inside that bucket
    We return indices w.r.t. the original order (weights input).
    """
    rng = np.random.default_rng(seed)
    N = len(distances)
    # 1) bucketize into groups of size m
    buckets = []
    bucket_probs = []
    bucket_probs_unique = []
    for i, start in enumerate(range(0, N, m)):
        end = min(start + m, N)
        bucket_size = end - start
        buckets.extend([i] * bucket_size)
        bucket_prob = distances["proba"][start:end].mean()
        bucket_probs_unique.append(bucket_prob)
        bucket_probs.extend([bucket_prob] * bucket_size)
    distances["bucket"] = buckets
    distances["bucket_prob"] = bucket_probs
    # check bucket prob sum is one
    assert np.isclose(
        distances.bucket_prob.sum(), 1.0
    ), "Bucket probabilities do not sum to 1 but sum to {}".format(
        distances.bucket_prob.sum()
    )
    # 3) pick buckets with replacement m times
    picks = rng.choice(
        range(distances["bucket"].max() + 1),
        size=m,
        replace=True,
        p=bucket_probs_unique,
    )
    picks_freq = {i: 0 for i in range(distances["bucket"].max() + 1)}
    for p in picks:
        picks_freq[p] += 1

    # 4) from each picked bucket, sample uniformly without replacement
    distances["chosen"] = False
    for bucket, freq in picks_freq.items():
        if freq > 0:
            bucket_distance = distances[distances.bucket == bucket]
            chosen = rng.choice(range(len(bucket_distance)), size=freq, replace=False)
            chosen_sources = bucket_distance.iloc[chosen].source
            distances.loc[distances.source.isin(chosen_sources), "chosen"] = True
    return distances
