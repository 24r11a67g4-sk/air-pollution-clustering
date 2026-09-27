"""Air Pollution Data Analysis Using Clustering.

This Flask application is intentionally self-contained so it can be opened
directly in VS Code and run with `python app.py`.
"""

from __future__ import annotations

import os
import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, jsonify, render_template, request
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "air_pollution.csv"

REQUIRED_COLUMNS = ["City", "PM2.5", "PM10", "NO2", "SO2", "CO", "AQI"]
NUMERIC_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "AQI"]
PLOT_NAMES = [
    "elbow.png",
    "aqi_chart.png",
    "pm25_chart.png",
    "clusters.png",
    "cluster_distribution.png",
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

current_data = pd.DataFrame()
current_filename = DATA_PATH.name


def load_data(file_path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load a CSV file and validate that it has the required data columns."""
    data = pd.read_csv(file_path)
    validate_columns(data)
    return data


def validate_columns(data: pd.DataFrame) -> None:
    """Raise a clear error when an uploaded file is not an air-quality CSV."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s): {missing}")


def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler, int]:
    """Clean values, fill missing data, and standardize pollution features."""
    processed = data[REQUIRED_COLUMNS].copy()
    processed["City"] = processed["City"].fillna("Unknown City").astype(str).str.strip()
    processed.loc[processed["City"] == "", "City"] = "Unknown City"

    # Coerce numeric text to NaN so uploaded CSVs are handled consistently.
    for feature in NUMERIC_FEATURES:
        processed[feature] = pd.to_numeric(processed[feature], errors="coerce")

    missing_values = int(processed[NUMERIC_FEATURES].isna().sum().sum())
    for feature in NUMERIC_FEATURES:
        feature_mean = processed[feature].mean()
        processed[feature] = processed[feature].fillna(0 if pd.isna(feature_mean) else feature_mean)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(processed[NUMERIC_FEATURES])
    return processed, scaled_features, scaler, missing_values


def perform_kmeans(scaled_features: np.ndarray, k: int) -> tuple[np.ndarray, KMeans]:
    """Run real K-Means clustering with the requested reproducible settings."""
    if k not in (2, 3, 4):
        raise ValueError("The number of clusters must be 2, 3, or 4.")
    if len(scaled_features) < k:
        raise ValueError(f"At least {k} records are required for {k} clusters.")

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_features)
    return labels, model


def _classification_labels(cluster_count: int) -> list[str]:
    """Return labels ordered from lowest to highest average pollution."""
    return {
        2: ["Low Pollution", "High Pollution"],
        3: ["Low Pollution", "Moderate Pollution", "High Pollution"],
        4: ["Low Pollution", "Moderate Pollution", "High Pollution", "Severe Pollution"],
    }[cluster_count]


def calculate_cluster_statistics(data: pd.DataFrame, k: int) -> tuple[pd.DataFrame, dict[int, str]]:
    """Summarize each cluster and classify it by its actual average AQI."""
    averages = (
        data.groupby("Cluster")[NUMERIC_FEATURES]
        .mean()
        .reset_index()
        .sort_values("AQI")
        .reset_index(drop=True)
    )
    labels = _classification_labels(k)
    classification_by_cluster: dict[int, str] = {}
    for rank, cluster_number in enumerate(averages["Cluster"].astype(int)):
        classification_by_cluster[cluster_number] = labels[rank]

    stats = (
        data.groupby("Cluster")
        .agg(
            Records=("City", "size"),
            Average_AQI=("AQI", "mean"),
            Average_PM25=("PM2.5", "mean"),
        )
        .reset_index()
    )
    stats["Classification"] = stats["Cluster"].map(classification_by_cluster)
    return stats.sort_values("Cluster").reset_index(drop=True), classification_by_cluster


def create_elbow_plot(scaled_features: np.ndarray) -> str:
    """Create the inertia curve for K values from 1 through 10."""
    max_k = min(10, len(scaled_features))
    k_values = list(range(1, max_k + 1))
    inertia_values = []

    for cluster_count in k_values:
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        model.fit(scaled_features)
        inertia_values.append(model.inertia_)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k_values, inertia_values, marker="o", linewidth=2.5, color="#0f766e")
    ax.set_title("Elbow Method: Choosing K", fontweight="bold")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_xticks(k_values)
    ax.grid(alpha=0.2)

        fig.tight_layout()
    return figure_to_data_uri(fig)

def create_cluster_plot(
    data: pd.DataFrame,
    model: KMeans,
    scaler: StandardScaler,
) -> str:
    """Plot PM2.5 vs AQI and project K-Means centroids back to original units."""
    centers = scaler.inverse_transform(model.cluster_centers_)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = sns.color_palette("viridis", n_colors=model.n_clusters)
    sns.scatterplot(
        data=data,
        x="PM2.5",
        y="AQI",
        hue="Cluster",
        palette=palette,
        s=95,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )
    ax.scatter(
        centers[:, NUMERIC_FEATURES.index("PM2.5")],
        centers[:, NUMERIC_FEATURES.index("AQI")],
        marker="X",
        s=230,
        c="#0f172a",
        edgecolor="white",
        linewidth=1.2,
        label="Centroid",
    )
    ax.set_title("Pollution Clusters: PM2.5 vs AQI", fontweight="bold")
    ax.set_xlabel("PM2.5")
    ax.set_ylabel("AQI")
    ax.legend(title="Cluster", frameon=False)
    ax.grid(alpha=0.16)
    fig.tight_layout()
    return figure_to_data_uri(fig)


def create_aqi_plot(data: pd.DataFrame) -> str:
    """Create a city-level AQI bar chart."""
    chart_data = data.sort_values("AQI", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=chart_data, x="AQI", y="City", hue="City", palette="crest", legend=False, ax=ax)
    ax.set_title("AQI by City", fontweight="bold")
    ax.set_xlabel("AQI")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.16)
    fig.tight_layout()
    return figure_to_data_uri(fig)


def create_pm25_plot(data: pd.DataFrame) -> str:
    """Create a city-level PM2.5 bar chart."""
    chart_data = data.sort_values("PM2.5", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=chart_data,
        x="PM2.5",
        y="City",
        hue="City",
        palette="mako",
        legend=False,
        ax=ax,
    )
    ax.set_title("PM2.5 by City", fontweight="bold")
    ax.set_xlabel("PM2.5")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.16)
    fig.tight_layout()
   return figure_to_data_uri(fig)

def create_cluster_distribution(data: pd.DataFrame) -> str:
    """Create a distribution chart for the number of records per cluster."""
    counts = data["Cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = sns.color_palette("viridis", n_colors=len(counts))
    ax.pie(
        counts.values,
        labels=[f"Cluster {cluster}" for cluster in counts.index],
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"linewidth": 2, "edgecolor": "white"},
    )
    ax.set_title("Cluster Distribution", fontweight="bold")
    fig.tight_layout()
    return figure_to_data_uri(fig)


def _round_value(value: Any) -> float:
    return round(float(value), 2)


def run_analysis(data: pd.DataFrame, k: int) -> dict[str, Any]:
    """Run preprocessing, K-Means, statistics, and all requested plots."""
    validate_columns(data)
    processed, scaled_features, scaler, missing_values = preprocess_data(data)
    labels, model = perform_kmeans(scaled_features, k)
    processed["Cluster"] = labels
    stats, classification_by_cluster = calculate_cluster_statistics(processed, k)

                    "plots": {
            "elbow": elbow_plot,
            "aqi_chart": aqi_plot,
            "pm25_chart": pm25_plot,
            "clusters": cluster_plot,
            "cluster_distribution": distribution_plot,
        },
    records = []
    for row in processed.to_dict(orient="records"):
        records.append(
            {
                "City": row["City"],
                "PM2.5": _round_value(row["PM2.5"]),
                "PM10": _round_value(row["PM10"]),
                "NO2": _round_value(row["NO2"]),
                "SO2": _round_value(row["SO2"]),
                "CO": _round_value(row["CO"]),
                "AQI": _round_value(row["AQI"]),
                "Cluster": int(row["Cluster"]),
                "Classification": classification_by_cluster[int(row["Cluster"])],
            }
        )

    statistics = [
        {
            "cluster": int(row["Cluster"]),
            "records": int(row["Records"]),
            "average_aqi": _round_value(row["Average_AQI"]),
            "average_pm25": _round_value(row["Average_PM25"]),
            "classification": row["Classification"],
        }
        for row in stats.to_dict(orient="records")
    ]
    highest_row = processed.loc[processed["AQI"].idxmax()]
    lowest_row = processed.loc[processed["AQI"].idxmin()]

    return {
        "filename": current_filename,
        "records": records,
        "statistics": statistics,
        "summary": {
            "total_records": int(len(processed)),
            "average_aqi": _round_value(processed["AQI"].mean()),
            "highest_aqi": _round_value(processed["AQI"].max()),
            "lowest_aqi": _round_value(processed["AQI"].min()),
            "number_of_clusters": k,
            "missing_values": missing_values,
        },
        "extremes": {
            "most_polluted_city": str(highest_row["City"]),
            "least_polluted_city": str(lowest_row["City"]),
            "highest_aqi": _round_value(highest_row["AQI"]),
            "lowest_aqi": _round_value(lowest_row["AQI"]),
        },
        "preprocessing": {
            "selected_features": NUMERIC_FEATURES,
            "standardization": "StandardScaler (zero mean, unit variance)",
        },
        "plots": {
    "elbow": elbow_plot,
    "aqi_chart": aqi_plot,
    "pm25_chart": pm25_plot,
    "clusters": cluster_plot,
    "cluster_distribution": distribution_plot,
},
    }


def _initial_data() -> pd.DataFrame:
    return load_data(DATA_PATH)


@app.route("/")
def index():
    """Render the academic dashboard shell."""
    return render_template(
        "index.html",
        required_columns=REQUIRED_COLUMNS,
        default_clusters=3,
    )


@app.route("/cluster", methods=["GET", "POST"])
def cluster():
    """Run clustering for the selected K and optionally replace the dataset."""
    global current_data, current_filename

    try:
        if request.method == "POST" and "file" in request.files:
            uploaded_file = request.files["file"]
            if uploaded_file.filename:
                uploaded_data = pd.read_csv(uploaded_file)
                validate_columns(uploaded_data)
                current_data = uploaded_data
                current_filename = uploaded_file.filename
        elif current_data.empty:
            current_data = _initial_data()
            current_filename = DATA_PATH.name

        requested_k = request.form.get("k") or (request.json or {}).get("k") if request.is_json else request.form.get("k")
        k = int(requested_k or 3)
        return jsonify(run_analysis(current_data, k))
    except (ValueError, TypeError, KeyError, pd.errors.ParserError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:  # Keep the browser response useful for malformed uploads.
        app.logger.exception("Analysis failed")
        return jsonify({"error": f"Analysis failed: {error}"}), 500


if __name__ == "__main__":
    current_data = _initial_data()
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
