import inspect
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "transfermarkt_clean.csv"
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "model" / "players_model.csv"
MODEL_METRICS_PATH = PROJECT_ROOT / "data" / "output" / "model_metrics.csv"
TEST_METRICS_PATH = PROJECT_ROOT / "data" / "output" / "test_metrics.csv"
CLASSIFICATION_REPORT_PATH = PROJECT_ROOT / "data" / "output" / "classification_report.csv"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "data" / "output" / "confusion_matrix.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "data" / "output" / "feature_importance.csv"
MATCHING_PATH = PROJECT_ROOT / "data" / "interim" / "player_matching_result.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"

LABEL_ORDER = ["Rendah", "Menengah", "Tinggi"]
LABEL_COLORS = {
    "Rendah": "#3B82F6",
    "Menengah": "#10B981",
    "Tinggi": "#F59E0B",
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
def load_all_data():
    clean_df = load_csv(CLEAN_DATA_PATH)
    model_df = load_csv(MODEL_DATA_PATH)
    metrics_df = load_csv(MODEL_METRICS_PATH)
    test_metrics_df = load_csv(TEST_METRICS_PATH)
    report_df = load_csv(CLASSIFICATION_REPORT_PATH)
    confusion_df = load_csv(CONFUSION_MATRIX_PATH)
    feature_df = load_csv(FEATURE_IMPORTANCE_PATH)
    matching_df = load_csv(MATCHING_PATH)
    return clean_df, model_df, metrics_df, test_metrics_df, report_df, confusion_df, feature_df, matching_df


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


def build_train_oversampling_distribution(df):
    train_df = add_split_column(df)
    train_df = train_df[train_df["split"] == "Train"]
    before = (
        train_df["market_value_category"]
        .value_counts()
        .reindex(LABEL_ORDER)
        .fillna(0)
        .astype(int)
    )

    majority_count = int(before.max()) if not before.empty else 0
    target_count = int(majority_count * OVERSAMPLING_RATIO)
    after = before.apply(lambda count: max(int(count), target_count) if count > 0 else 0)

    return pd.concat(
        [
            before.rename("records").reset_index().assign(status="Sebelum oversampling"),
            after.rename("records").reset_index().assign(status="Sesudah oversampling"),
        ],
        ignore_index=True,
    ).rename(columns={"market_value_category": "category"})


def plot_train_oversampling_distribution(df):
    data = build_train_oversampling_distribution(df)
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
    MODEL_METRICS_PATH,
    TEST_METRICS_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURE_IMPORTANCE_PATH,
    MATCHING_PATH,
    BEST_MODEL_PATH,
    LABEL_ENCODER_PATH,
]
require_artifacts(required_paths)

clean_df, model_df, metrics_df, test_metrics_df, report_df, confusion_df, feature_df, matching_df = load_all_data()
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

tabs = st.tabs(["Overview", "Market Value", "Model Evaluation", "Feature Importance", "Data Explorer"])

with tabs[0]:
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

with tabs[1]:
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

with tabs[2]:
    st.subheader("Model Evaluation")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Test Accuracy", format_percent(latest_test.get("accuracy")))
    metric_cols[1].metric("Test Macro F1", format_percent(latest_test.get("macro_f1")))
    metric_cols[2].metric("Test Weighted F1", format_percent(latest_test.get("weighted_f1")))
    metric_cols[3].metric("Best Model", str(latest_test.get("model", "-")))

    render_plotly_chart(plot_train_oversampling_distribution(model_df))
    st.caption(
        "Oversampling hanya diterapkan pada train set. Validation dan test tetap memakai distribusi asli "
        "agar evaluasi model tetap merepresentasikan data nyata."
    )

    col_left, col_right = st.columns(2)
    with col_left:
        render_plotly_chart(plot_model_comparison(metrics_df, "accuracy"))
        render_plotly_chart(plot_confusion_matrix(confusion_df))
    with col_right:
        render_plotly_chart(plot_model_comparison(metrics_df, "macro_f1"))
        render_dataframe(report_df)

with tabs[3]:
    st.subheader("Feature Importance")
    top_n = st.slider("Jumlah feature ditampilkan", min_value=5, max_value=min(30, len(feature_df)), value=min(15, len(feature_df)))
    render_plotly_chart(plot_feature_importance(feature_df, top_n))
    render_dataframe(feature_df.head(top_n))

with tabs[4]:
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

