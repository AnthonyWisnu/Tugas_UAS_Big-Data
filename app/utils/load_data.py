from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "transfermarkt_dataset_clean.csv"
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "model" / "players_model_with_performance.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "output" / "model_metrics_with_performance.csv"
CLASSIFICATION_REPORT_PATH = PROJECT_ROOT / "data" / "output" / "classification_report_best_model_with_performance.csv"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "data" / "output" / "confusion_matrix_best_model_with_performance.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "data" / "output" / "feature_importance_best_model_with_performance.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_model_with_performance.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder_with_performance.pkl"


def require_file(path):
    path = Path(path)
    if not path.exists():
        st.error(f"File tidak ditemukan: {path}")
        st.stop()
    return path


@st.cache_data(show_spinner=False)
def load_processed_data():
    return pd.read_csv(require_file(PROCESSED_DATA_PATH))


@st.cache_data(show_spinner=False)
def load_model_data():
    return pd.read_csv(require_file(MODEL_DATA_PATH))


@st.cache_data(show_spinner=False)
def load_metrics():
    return pd.read_csv(require_file(METRICS_PATH))


@st.cache_data(show_spinner=False)
def load_classification_report():
    return pd.read_csv(require_file(CLASSIFICATION_REPORT_PATH))


@st.cache_data(show_spinner=False)
def load_confusion_matrix():
    return pd.read_csv(require_file(CONFUSION_MATRIX_PATH))


@st.cache_data(show_spinner=False)
def load_feature_importance():
    return pd.read_csv(require_file(FEATURE_IMPORTANCE_PATH))


@st.cache_resource(show_spinner=False)
def load_best_model():
    try:
        import joblib
    except ModuleNotFoundError:
        st.error("Package joblib belum terinstall. Jalankan: pip install joblib")
        st.stop()
    return joblib.load(require_file(BEST_MODEL_PATH))


@st.cache_resource(show_spinner=False)
def load_label_encoder():
    try:
        import joblib
    except ModuleNotFoundError:
        st.error("Package joblib belum terinstall. Jalankan: pip install joblib")
        st.stop()
    return joblib.load(require_file(LABEL_ENCODER_PATH))


def sorted_options(df, column):
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())


def sidebar_filters(df):
    st.sidebar.header("Filters")
    filtered = df.copy()

    filter_columns = [
        ("season", "Season"),
        ("league", "League"),
        ("club", "Club"),
        ("pos_category", "Position"),
        ("nationality", "Nationality"),
        ("market_value_category", "Market Value Category"),
    ]

    selected = {}
    for column, label in filter_columns:
        if column not in filtered.columns:
            continue
        options = sorted_options(filtered, column)
        values = st.sidebar.multiselect(label, options=options, default=[])
        selected[column] = values
        if values:
            filtered = filtered[filtered[column].astype(str).isin(values)]

    st.sidebar.caption(f"Filtered records: {len(filtered):,}")
    return filtered, selected


def format_number(value):
    if pd.isna(value):
        return "0"
    return f"{int(value):,}"
