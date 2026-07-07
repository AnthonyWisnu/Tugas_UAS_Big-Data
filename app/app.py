import json
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from joblib import load


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "transfermarkt_clean.csv"
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "model" / "players_model.csv"
FEATURE_LIST_PATH = PROJECT_ROOT / "data" / "model" / "feature_list.json"
MODEL_METRICS_PATH = PROJECT_ROOT / "data" / "output" / "model_metrics.csv"
TEST_METRICS_PATH = PROJECT_ROOT / "data" / "output" / "test_metrics.csv"
CLASSIFICATION_REPORT_PATH = PROJECT_ROOT / "data" / "output" / "classification_report.csv"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "data" / "output" / "confusion_matrix.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "data" / "output" / "feature_importance.csv"
OVERSAMPLING_SUMMARY_PATH = PROJECT_ROOT / "data" / "output" / "oversampling_summary.csv"
BEST_MODEL_SUMMARY_PATH = PROJECT_ROOT / "data" / "output" / "best_model_summary.csv"
TRAIN_BEFORE_OVERSAMPLING_PATH = PROJECT_ROOT / "data" / "model" / "train_before_oversampling.csv"
TRAIN_AFTER_OVERSAMPLING_PATH = PROJECT_ROOT / "data" / "model" / "train_after_oversampling.csv"
HIGH_BEFORE_OVERSAMPLING_PATH = PROJECT_ROOT / "data" / "model" / "train_high_before_oversampling.csv"
HIGH_AFTER_OVERSAMPLING_PATH = PROJECT_ROOT / "data" / "model" / "train_high_after_oversampling.csv"
MATCHING_PATH = PROJECT_ROOT / "data" / "interim" / "player_matching_result.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"

LABEL_ORDER = ["Rendah", "Menengah", "Tinggi"]
LABEL_COLORS = {
    "Rendah": "#3B82F6",
    "Menengah": "#10B981",
    "Tinggi": "#F59E0B",
}
VALUE_RANGES = {
    "Rendah": "EUR 5 juta sampai < EUR 15 juta",
    "Menengah": "EUR 15 juta sampai EUR 35 juta",
    "Tinggi": "> EUR 35 juta",
}
OVERSAMPLING_RATIO = 0.85
PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}


st.set_page_config(
    page_title="Market Value Dashboard",
    layout="wide",
)


def supports_argument(function, argument):
    return argument in inspect.signature(function).parameters


PLOTLY_CHART_SUPPORTS_WIDTH = supports_argument(st.plotly_chart, "width")
DATAFRAME_SUPPORTS_WIDTH = supports_argument(st.dataframe, "width")


def render_plotly_chart(fig):
    if PLOTLY_CHART_SUPPORTS_WIDTH:
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    else:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def render_dataframe(data, hide_index=True):
    data = data.copy()
    for column in data.columns:
        if data[column].dtype == "object":
            data[column] = data[column].where(data[column].notna(), "").astype(str)
    if DATAFRAME_SUPPORTS_WIDTH:
        st.dataframe(data, width="stretch", hide_index=hide_index)
    else:
        st.dataframe(data, use_container_width=True, hide_index=hide_index)


def require_artifacts(paths):
    missing = [path.relative_to(PROJECT_ROOT).as_posix() for path in paths if not path.exists() or path.stat().st_size == 0]
    if missing:
        st.error("Artifact belum tersedia. Jalankan 02_preprocessing.ipynb dan 03_training_model.ipynb terlebih dahulu.")
        st.caption("File yang belum tersedia:")
        st.code("\n".join(missing))
        st.stop()


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_json_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_prediction_artifacts():
    return load(BEST_MODEL_PATH), load(LABEL_ENCODER_PATH)


@st.cache_data
def load_all_data():
    clean_df = load_csv(CLEAN_DATA_PATH)
    model_df = load_csv(MODEL_DATA_PATH)
    metrics_df = load_csv(MODEL_METRICS_PATH)
    test_metrics_df = load_csv(TEST_METRICS_PATH)
    report_df = load_csv(CLASSIFICATION_REPORT_PATH)
    confusion_df = load_csv(CONFUSION_MATRIX_PATH)
    feature_df = load_csv(FEATURE_IMPORTANCE_PATH)
    oversampling_df = load_csv(OVERSAMPLING_SUMMARY_PATH)
    best_model_summary_df = load_csv(BEST_MODEL_SUMMARY_PATH)
    matching_df = load_csv(MATCHING_PATH)
    feature_list = load_json_file(FEATURE_LIST_PATH)
    return (
        clean_df,
        model_df,
        metrics_df,
        test_metrics_df,
        report_df,
        confusion_df,
        feature_df,
        oversampling_df,
        best_model_summary_df,
        matching_df,
        feature_list,
    )


@st.cache_data
def load_training_dataset(path):
    return pd.read_csv(path)


def add_split_column(df):
    result = df.copy()
    result["split"] = result["season"].apply(
        lambda season: "Train" if 2017 <= season <= 2021 else ("Validation" if season == 2022 else "Test")
    )
    return result


def apply_filters(df):
    st.sidebar.header("Filter Data")

    season_values = sorted(df["season"].dropna().astype(int).unique().tolist())
    selected_seasons = st.sidebar.multiselect("Season", season_values, default=season_values)

    league_values = sorted(df["league"].dropna().astype(str).unique().tolist())
    selected_leagues = st.sidebar.multiselect("League", league_values, default=league_values)

    label_values = [label for label in LABEL_ORDER if label in set(df["market_value_category"].dropna())]
    selected_labels = st.sidebar.multiselect("Market Value Category", label_values, default=label_values)

    position_values = sorted(df["pos_category"].dropna().astype(str).unique().tolist())
    selected_positions = st.sidebar.multiselect("Position", position_values, default=position_values)

    performance_filter = st.sidebar.selectbox(
        "FBref Stats",
        ["Semua", "Matched only", "Unmatched only"],
        index=0,
    )

    filtered = df[
        df["season"].isin(selected_seasons)
        & df["league"].isin(selected_leagues)
        & df["market_value_category"].isin(selected_labels)
        & df["pos_category"].isin(selected_positions)
    ].copy()

    if "has_performance_stats" in filtered.columns:
        if performance_filter == "Matched only":
            filtered = filtered[filtered["has_performance_stats"] == 1]
        elif performance_filter == "Unmatched only":
            filtered = filtered[filtered["has_performance_stats"] == 0]

    return filtered


def format_percent(value):
    if pd.isna(value):
        return "-"
    return f"{value * 100:.2f}%"


def format_eur_mio(value):
    if pd.isna(value):
        return "-"
    return f"EUR {float(value):,.2f} juta"


def plot_label_distribution(df):
    data = (
        df["market_value_category"]
        .value_counts()
        .reindex(LABEL_ORDER)
        .dropna()
        .rename_axis("category")
        .reset_index(name="records")
    )
    fig = px.bar(
        data,
        x="category",
        y="records",
        color="category",
        color_discrete_map=LABEL_COLORS,
        text="records",
        title="Distribusi Label",
        labels={"category": "Kategori", "records": "Jumlah record"},
    )
    fig.update_layout(showlegend=False, height=380, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_split_distribution(df):
    data = add_split_column(df)
    grouped = data.groupby(["split", "market_value_category"], as_index=False).size()
    split_order = ["Train", "Validation", "Test"]
    fig = px.bar(
        grouped,
        x="split",
        y="size",
        color="market_value_category",
        category_orders={"split": split_order, "market_value_category": LABEL_ORDER},
        color_discrete_map=LABEL_COLORS,
        barmode="group",
        title="Distribusi Label per Split",
        labels={"split": "Split", "size": "Jumlah record", "market_value_category": "Kategori"},
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def build_train_oversampling_distribution(oversampling_df):
    summary = oversampling_df.set_index("category").reindex(LABEL_ORDER).fillna(0).reset_index()
    return pd.concat(
        [
            summary[["category", "before_records"]]
            .rename(columns={"before_records": "records"})
            .assign(status="Sebelum oversampling"),
            summary[["category", "after_records"]]
            .rename(columns={"after_records": "records"})
            .assign(status="Sesudah oversampling"),
        ],
        ignore_index=True,
    )


def plot_train_oversampling_distribution(oversampling_df):
    data = build_train_oversampling_distribution(oversampling_df)
    fig = px.bar(
        data,
        x="category",
        y="records",
        color="category",
        facet_col="status",
        category_orders={"category": LABEL_ORDER, "status": ["Sebelum oversampling", "Sesudah oversampling"]},
        color_discrete_map=LABEL_COLORS,
        text="records",
        title="Distribusi Train Set Sebelum dan Sesudah Oversampling",
        labels={"category": "Kategori", "records": "Jumlah record", "status": ""},
    )
    fig.update_layout(showlegend=False, height=380, margin=dict(l=20, r=20, t=60, b=20))
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    return fig


def plot_oversampling_added_records(oversampling_df):
    data = oversampling_df.copy()
    data["added_records"] = data["added_records"].astype(int)
    fig = px.bar(
        data,
        x="category",
        y="added_records",
        color="category",
        category_orders={"category": LABEL_ORDER},
        color_discrete_map=LABEL_COLORS,
        text="added_records",
        title="Record Tambahan dari Oversampling Train Set",
        labels={"category": "Kategori", "added_records": "Record tambahan"},
    )
    fig.update_layout(showlegend=False, height=360, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_prediction_probabilities(probability_df):
    fig = px.bar(
        probability_df,
        x="category",
        y="probability",
        color="category",
        category_orders={"category": LABEL_ORDER},
        color_discrete_map=LABEL_COLORS,
        text=probability_df["probability"].map(format_percent),
        title="Probabilitas Prediksi per Kategori",
        labels={"category": "Kategori", "probability": "Probabilitas"},
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(showlegend=False, height=360, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def get_model_classes(model):
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return classes
    final_model = model.named_steps.get("model") if hasattr(model, "named_steps") else None
    return getattr(final_model, "classes_", [])


def prepare_training_table(df):
    preferred_columns = [
        "_oversampling_status",
        "_is_oversampled_duplicate",
        "_source_row_id",
        "season",
        "market_value_category",
        "league",
        "pos_category",
        "age",
        "height_m",
        "preferred_foot",
        "league_rank",
        "club_total_mv_mio",
        "prev_season_mv",
        "prev_mv_category",
        "minutes",
        "goals",
        "assists",
        "starts_rate",
        "has_performance_stats",
    ]
    available_columns = [column for column in preferred_columns if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in available_columns]
    return df[available_columns + remaining_columns]


def plot_records_by_season(df):
    data = df.groupby(["season", "market_value_category"], as_index=False).size()
    fig = px.line(
        data,
        x="season",
        y="size",
        color="market_value_category",
        markers=True,
        category_orders={"market_value_category": LABEL_ORDER},
        color_discrete_map=LABEL_COLORS,
        title="Jumlah Record per Season",
        labels={"season": "Season", "size": "Jumlah record", "market_value_category": "Kategori"},
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_records_by_league(df):
    data = df.groupby(["league", "market_value_category"], as_index=False).size()
    fig = px.bar(
        data,
        x="league",
        y="size",
        color="market_value_category",
        category_orders={"market_value_category": LABEL_ORDER},
        color_discrete_map=LABEL_COLORS,
        title="Jumlah Record per League",
        labels={"league": "League", "size": "Jumlah record", "market_value_category": "Kategori"},
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_market_value_box(df):
    fig = px.box(
        df,
        x="market_value_category",
        y="market_value_mio",
        color="market_value_category",
        category_orders={"market_value_category": LABEL_ORDER},
        color_discrete_map=LABEL_COLORS,
        title="Sebaran Market Value per Kategori",
        labels={"market_value_category": "Kategori", "market_value_mio": "Market value EUR juta"},
    )
    fig.update_layout(showlegend=False, height=420, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_position_market_value(df):
    data = (
        df.groupby("pos_category", as_index=False)
        .agg(records=("player_id", "count"), avg_market_value=("market_value_mio", "mean"))
        .sort_values("avg_market_value", ascending=False)
    )
    fig = px.bar(
        data,
        x="pos_category",
        y="avg_market_value",
        text=data["avg_market_value"].round(2),
        title="Rata-Rata Market Value per Posisi",
        labels={"pos_category": "Posisi", "avg_market_value": "Rata-rata EUR juta"},
        color="records",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_matching_summary(matching_df):
    data = matching_df["matched"].map({True: "Matched", False: "Unmatched"}).value_counts().reset_index()
    data.columns = ["status", "records"]
    fig = px.pie(
        data,
        names="status",
        values="records",
        hole=0.45,
        title="Ringkasan Matching FBref",
        color="status",
        color_discrete_map={"Matched": "#10B981", "Unmatched": "#EF4444"},
    )
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_model_comparison(metrics_df, metric_name):
    validation = metrics_df[metrics_df["split"] == "validation"].copy()
    if validation.empty:
        return go.Figure()
    validation["candidate"] = validation["model"] + " | " + validation["scenario"]
    validation = validation.sort_values(metric_name, ascending=False)
    fig = px.bar(
        validation,
        x=metric_name,
        y="candidate",
        orientation="h",
        text=validation[metric_name].map(format_percent),
        title=f"Perbandingan Validation {metric_name.replace('_', ' ').title()}",
        labels={metric_name: metric_name.replace("_", " ").title(), "candidate": "Model dan skenario"},
        color=metric_name,
        color_continuous_scale="Teal",
    )
    fig.update_layout(height=360, yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_confusion_matrix(confusion_df):
    matrix = confusion_df.set_index("actual_label")
    labels = [label.replace("predicted_", "") for label in matrix.columns]
    actual = [label.replace("actual_", "") for label in matrix.index]
    fig = px.imshow(
        matrix.values,
        x=labels,
        y=actual,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix Test Set",
        labels={"x": "Predicted", "y": "Actual", "color": "Records"},
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_feature_importance(feature_df, top_n):
    data = feature_df.head(top_n).sort_values("importance", ascending=True)
    fig = px.bar(
        data,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {top_n} Feature Importance",
        labels={"importance": "Importance", "feature": "Feature"},
        color="importance",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=max(420, top_n * 28), margin=dict(l=20, r=20, t=60, b=20))
    return fig


required_paths = [
    CLEAN_DATA_PATH,
    MODEL_DATA_PATH,
    FEATURE_LIST_PATH,
    MODEL_METRICS_PATH,
    TEST_METRICS_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURE_IMPORTANCE_PATH,
    OVERSAMPLING_SUMMARY_PATH,
    BEST_MODEL_SUMMARY_PATH,
    TRAIN_BEFORE_OVERSAMPLING_PATH,
    TRAIN_AFTER_OVERSAMPLING_PATH,
    HIGH_BEFORE_OVERSAMPLING_PATH,
    HIGH_AFTER_OVERSAMPLING_PATH,
    MATCHING_PATH,
    BEST_MODEL_PATH,
    LABEL_ENCODER_PATH,
]
require_artifacts(required_paths)

(
    clean_df,
    model_df,
    metrics_df,
    test_metrics_df,
    report_df,
    confusion_df,
    feature_df,
    oversampling_df,
    best_model_summary_df,
    matching_df,
    feature_list,
) = load_all_data()
filtered_df = apply_filters(clean_df)

st.title("Football Market Value Dashboard")
st.caption("Big 5 European leagues, seasons 2017 sampai 2024, pemain dengan market value minimal EUR 5 juta.")

if filtered_df.empty:
    st.warning("Tidak ada data yang sesuai dengan filter.")
    st.stop()

latest_test = test_metrics_df.iloc[0] if not test_metrics_df.empty else pd.Series(dtype="object")
matched_rows = int(matching_df["matched"].sum()) if "matched" in matching_df.columns else 0
unmatched_rows = int((~matching_df["matched"]).sum()) if "matched" in matching_df.columns else 0

overview_cols = st.columns(5)
overview_cols[0].metric("Records", f"{len(filtered_df):,}")
overview_cols[1].metric("Players", f"{filtered_df['player_id'].nunique():,}")
overview_cols[2].metric("Clubs", f"{filtered_df['club'].nunique():,}")
overview_cols[3].metric("Test Accuracy", format_percent(latest_test.get("accuracy")))
overview_cols[4].metric("Test Macro F1", format_percent(latest_test.get("macro_f1")))

overview_tab, market_tab, model_tab, prediction_tab, feature_tab, explorer_tab = st.tabs(
    ["Overview", "Market Value", "Model Evaluation", "Prediction", "Feature Importance", "Data Explorer"]
)

with overview_tab:
    st.subheader("Dataset Overview")
    col_left, col_right = st.columns(2)
    with col_left:
        render_plotly_chart(plot_label_distribution(filtered_df))
        render_plotly_chart(plot_records_by_season(filtered_df))
    with col_right:
        render_plotly_chart(plot_split_distribution(filtered_df))
        render_plotly_chart(plot_matching_summary(matching_df))

    match_cols = st.columns(3)
    match_cols[0].metric("FBref Matched", f"{matched_rows:,}")
    match_cols[1].metric("FBref Unmatched", f"{unmatched_rows:,}")
    match_rate = matched_rows / (matched_rows + unmatched_rows) if matched_rows + unmatched_rows else 0
    match_cols[2].metric("FBref Match Rate", format_percent(match_rate))

with market_tab:
    st.subheader("Market Value Analysis")
    col_left, col_right = st.columns(2)
    with col_left:
        render_plotly_chart(plot_records_by_league(filtered_df))
        render_plotly_chart(plot_position_market_value(filtered_df))
    with col_right:
        render_plotly_chart(plot_market_value_box(filtered_df))
        top_players = (
            filtered_df[
                [
                    "player_name",
                    "club",
                    "league",
                    "season",
                    "pos_category",
                    "market_value_mio",
                    "market_value_category",
                ]
            ]
            .sort_values("market_value_mio", ascending=False)
            .head(15)
        )
        render_dataframe(top_players)

with model_tab:
    st.subheader("Model Evaluation")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Test Accuracy", format_percent(latest_test.get("accuracy")))
    metric_cols[1].metric("Test Macro F1", format_percent(latest_test.get("macro_f1")))
    metric_cols[2].metric("Test Weighted F1", format_percent(latest_test.get("weighted_f1")))
    metric_cols[3].metric("Best Model", str(latest_test.get("model", "-")))

    before_total = int(oversampling_df["before_records"].sum())
    after_total = int(oversampling_df["after_records"].sum())
    added_total = int(oversampling_df["added_records"].sum())
    high_added = int(
        oversampling_df.loc[oversampling_df["category"] == "Tinggi", "added_records"].sum()
    )
    sampling_cols = st.columns(4)
    sampling_cols[0].metric("Train Sebelum OS", f"{before_total:,}")
    sampling_cols[1].metric("Train Sesudah OS", f"{after_total:,}")
    sampling_cols[2].metric("Record Tambahan", f"{added_total:,}")
    sampling_cols[3].metric("Tambahan Kelas Tinggi", f"{high_added:,}")

    col_sampling_left, col_sampling_right = st.columns(2)
    with col_sampling_left:
        render_plotly_chart(plot_train_oversampling_distribution(oversampling_df))
    with col_sampling_right:
        render_plotly_chart(plot_oversampling_added_records(oversampling_df))
    st.caption(
        "Grafik ini membaca file hasil oversampling aktual dari proses training. Validation dan test tetap "
        "memakai distribusi asli agar evaluasi model merepresentasikan data nyata."
    )

    col_left, col_right = st.columns(2)
    with col_left:
        render_plotly_chart(plot_model_comparison(metrics_df, "accuracy"))
        render_plotly_chart(plot_confusion_matrix(confusion_df))
    with col_right:
        render_plotly_chart(plot_model_comparison(metrics_df, "macro_f1"))
        render_dataframe(report_df)

with prediction_tab:
    st.subheader("Prediction")
    st.caption(
        "Pilih record pemain dari dataset, lalu model akan memprediksi kategori market value. "
        "Estimasi valuasi ditampilkan sebagai rentang kategori, bukan angka valuasi absolut."
    )

    if len(clean_df) != len(model_df):
        st.error(
            "Jumlah baris clean dataset dan model dataset berbeda, sehingga prediksi per pemain tidak bisa "
            "dipetakan dengan aman. Jalankan ulang preprocessing dan training."
        )
        st.stop()

    missing_prediction_features = [column for column in feature_list if column not in model_df.columns]
    if missing_prediction_features:
        st.error(f"Fitur prediksi tidak tersedia di model dataset: {missing_prediction_features}")
        st.stop()

    best_model, label_encoder = load_prediction_artifacts()
    best_summary = best_model_summary_df.iloc[0] if not best_model_summary_df.empty else pd.Series(dtype="object")

    predictor_pool = clean_df.reset_index(drop=False).rename(columns={"index": "record_id"}).copy()
    pred_col_left, pred_col_mid, pred_col_right = st.columns(3)
    with pred_col_left:
        prediction_seasons = sorted(predictor_pool["season"].dropna().astype(int).unique().tolist())
        selected_prediction_season = st.selectbox("Season prediksi", prediction_seasons, index=len(prediction_seasons) - 1)
    with pred_col_mid:
        season_pool = predictor_pool[predictor_pool["season"].astype(int) == selected_prediction_season]
        prediction_leagues = ["Semua"] + sorted(season_pool["league"].dropna().astype(str).unique().tolist())
        selected_prediction_league = st.selectbox("League prediksi", prediction_leagues)
    with pred_col_right:
        prediction_categories = ["Semua"] + [label for label in LABEL_ORDER if label in set(season_pool["market_value_category"])]
        selected_prediction_category = st.selectbox("Actual category", prediction_categories)

    prediction_pool = season_pool.copy()
    if selected_prediction_league != "Semua":
        prediction_pool = prediction_pool[prediction_pool["league"].astype(str) == selected_prediction_league]
    if selected_prediction_category != "Semua":
        prediction_pool = prediction_pool[prediction_pool["market_value_category"] == selected_prediction_category]

    prediction_search = st.text_input("Cari pemain untuk prediksi")
    if prediction_search.strip():
        needle = prediction_search.strip().lower()
        search_columns = ["player_name", "club", "league", "nationality", "pos_category"]
        mask = pd.Series(False, index=prediction_pool.index)
        for column in search_columns:
            if column in prediction_pool.columns:
                mask = mask | prediction_pool[column].astype(str).str.lower().str.contains(needle, na=False)
        prediction_pool = prediction_pool[mask]

    if prediction_pool.empty:
        st.warning("Tidak ada pemain yang sesuai dengan filter prediksi.")
        st.stop()

    prediction_pool = prediction_pool.sort_values(["market_value_mio", "player_name"], ascending=[False, True])
    option_labels = {
        int(row.record_id): (
            f"{row.player_name} | {row.club} | {int(row.season)} | "
            f"{format_eur_mio(row.market_value_mio)} | {row.market_value_category}"
        )
        for row in prediction_pool.itertuples(index=False)
    }
    selected_record_id = st.selectbox(
        "Pilih pemain",
        list(option_labels.keys()),
        format_func=lambda record_id: option_labels.get(int(record_id), str(record_id)),
    )

    selected_actual = clean_df.iloc[int(selected_record_id)]
    selected_features = model_df.loc[[int(selected_record_id)], feature_list]
    predicted_encoded = int(best_model.predict(selected_features)[0])
    predicted_label = label_encoder.inverse_transform([predicted_encoded])[0]

    probability_df = pd.DataFrame(columns=["category", "probability"])
    confidence = np.nan
    if hasattr(best_model, "predict_proba"):
        probabilities = best_model.predict_proba(selected_features)[0]
        encoded_classes = get_model_classes(best_model)
        class_labels = label_encoder.inverse_transform([int(value) for value in encoded_classes])
        probability_df = pd.DataFrame({
            "category": class_labels,
            "probability": probabilities,
        })
        probability_df = (
            probability_df.set_index("category")
            .reindex(LABEL_ORDER)
            .fillna(0)
            .reset_index()
        )
        confidence = float(
            probability_df.loc[probability_df["category"] == predicted_label, "probability"].iloc[0]
        )

    actual_label = selected_actual.get("market_value_category", "-")
    prediction_match = predicted_label == actual_label
    prediction_cols = st.columns(5)
    prediction_cols[0].metric("Prediksi Kategori", predicted_label)
    prediction_cols[1].metric("Confidence", format_percent(confidence))
    prediction_cols[2].metric("Actual Category", str(actual_label))
    prediction_cols[3].metric("Actual Market Value", format_eur_mio(selected_actual.get("market_value_mio")))
    prediction_cols[4].metric("Hasil", "Sesuai" if prediction_match else "Berbeda")

    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.markdown("**Profil Record**")
        profile_columns = [
            "player_name",
            "club",
            "league",
            "season",
            "pos_category",
            "age",
            "nationality",
            "market_value_mio",
            "market_value_category",
        ]
        profile_data = pd.DataFrame([{
            column: selected_actual.get(column, "-")
            for column in profile_columns
            if column in clean_df.columns
        }])
        render_dataframe(profile_data)
        st.markdown("**Interpretasi Valuasi**")
        st.info(f"Kategori {predicted_label}: {VALUE_RANGES.get(predicted_label, '-')}")

    with detail_right:
        if probability_df.empty:
            st.warning("Model terpilih tidak menyediakan probabilitas kelas.")
        else:
            render_plotly_chart(plot_prediction_probabilities(probability_df))

    st.markdown("**Fitur yang Masuk ke Model**")
    feature_preview = selected_features.transpose().reset_index()
    feature_preview.columns = ["feature", "value"]
    render_dataframe(feature_preview)

    st.caption(
        f"Model aktif: {best_summary.get('model', '-')} | skenario: {best_summary.get('scenario', '-')} | "
        f"target: {best_summary.get('target_column', 'market_value_category')}"
    )

with feature_tab:
    st.subheader("Feature Importance")
    top_n = st.slider("Jumlah feature ditampilkan", min_value=5, max_value=min(30, len(feature_df)), value=min(15, len(feature_df)))
    render_plotly_chart(plot_feature_importance(feature_df, top_n))
    render_dataframe(feature_df.head(top_n))

with explorer_tab:
    st.subheader("Data Explorer")
    search_text = st.text_input("Cari pemain, klub, atau liga")
    table_df = filtered_df.copy()
    if search_text.strip():
        needle = search_text.strip().lower()
        search_columns = ["player_name", "club", "league", "nationality", "pos_category"]
        mask = pd.Series(False, index=table_df.index)
        for column in search_columns:
            if column in table_df.columns:
                mask = mask | table_df[column].astype(str).str.lower().str.contains(needle, na=False)
        table_df = table_df[mask]

    selected_columns = [
        "player_name",
        "club",
        "league",
        "season",
        "pos_category",
        "age",
        "market_value_mio",
        "market_value_category",
        "prev_season_mv",
        "minutes",
        "goals",
        "assists",
        "has_performance_stats",
    ]
    available_columns = [column for column in selected_columns if column in table_df.columns]
    render_dataframe(
        table_df[available_columns].sort_values(["season", "market_value_mio"], ascending=[False, False]),
    )

    st.divider()
    st.subheader("Training Dataset Oversampling")
    st.caption(
        "Bagian ini membaca file train sebelum/sesudah oversampling. Data training bersifat model-ready, "
        "jadi metadata seperti player_name dan club tidak ikut disimpan sebagai fitur training."
    )

    training_options = {
        "Sebelum oversampling - semua kelas": TRAIN_BEFORE_OVERSAMPLING_PATH,
        "Sesudah oversampling - semua kelas": TRAIN_AFTER_OVERSAMPLING_PATH,
        "Sebelum oversampling - kelas Tinggi": HIGH_BEFORE_OVERSAMPLING_PATH,
        "Sesudah oversampling - kelas Tinggi": HIGH_AFTER_OVERSAMPLING_PATH,
    }
    training_choice = st.selectbox("Dataset training", list(training_options.keys()), index=1)
    training_df = load_training_dataset(training_options[training_choice])

    status_values = ["Semua"]
    if "_oversampling_status" in training_df.columns:
        status_values += sorted(training_df["_oversampling_status"].dropna().astype(str).unique().tolist())
    selected_status = st.selectbox("Status oversampling", status_values)
    if selected_status != "Semua" and "_oversampling_status" in training_df.columns:
        training_df = training_df[training_df["_oversampling_status"].astype(str) == selected_status]

    training_search = st.text_input("Cari pada dataset training", key="training_dataset_search")
    if training_search.strip():
        needle = training_search.strip().lower()
        search_columns = [
            "market_value_category",
            "league",
            "pos_category",
            "preferred_foot",
            "prev_mv_category",
            "two_seasons_ago_mv_category",
            "_oversampling_status",
        ]
        mask = pd.Series(False, index=training_df.index)
        for column in search_columns:
            if column in training_df.columns:
                mask = mask | training_df[column].astype(str).str.lower().str.contains(needle, na=False)
        training_df = training_df[mask]

    duplicate_count = (
        int(training_df["_is_oversampled_duplicate"].sum())
        if "_is_oversampled_duplicate" in training_df.columns
        else 0
    )
    training_metric_cols = st.columns(4)
    training_metric_cols[0].metric("Rows Ditampilkan", f"{len(training_df):,}")
    training_metric_cols[1].metric(
        "Kelas Tinggi",
        f"{int((training_df['market_value_category'] == 'Tinggi').sum()):,}"
        if "market_value_category" in training_df.columns
        else "0",
    )
    training_metric_cols[2].metric("Tambahan Oversampling", f"{duplicate_count:,}")
    training_metric_cols[3].metric("Kolom", f"{len(training_df.columns):,}")

    render_dataframe(prepare_training_table(training_df), hide_index=True)

