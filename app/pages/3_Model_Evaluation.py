import streamlit as st
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.load_data import (
    load_classification_report,
    load_confusion_matrix,
    load_feature_importance,
    load_metrics,
)
from utils.plotting import confusion_matrix_chart, feature_importance_chart


st.set_page_config(page_title="Model Evaluation", layout="wide")
st.title("Model Evaluation")
st.caption("The original 2023-2024 test split is the main final evaluation. Validation is used for model selection.")

metrics_df = load_metrics()
report_df = load_classification_report()
confusion_df = load_confusion_matrix()
feature_df = load_feature_importance()

test_metrics = metrics_df[metrics_df["split"] == "test"].copy()
if test_metrics.empty:
    st.error("Test metrics are missing from model_metrics_improved.csv.")
    st.stop()

best_row = test_metrics.iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Best Model", str(best_row["model"]))
col2.metric("Scenario", str(best_row["scenario"]))
col3.metric("Test Accuracy", f"{best_row['accuracy']:.3f}")
col4.metric("Test Macro F1", f"{best_row['macro_f1']:.3f}")

st.subheader("Best Model Test Metrics")
st.dataframe(test_metrics, use_container_width=True, hide_index=True)

st.subheader("Validation Comparison")
validation_df = metrics_df[metrics_df["split"] == "validation"].sort_values("macro_f1", ascending=False)
st.dataframe(validation_df.head(30), use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.subheader("Classification Report")
    st.dataframe(report_df, use_container_width=True, hide_index=True)
with right:
    st.subheader("Confusion Matrix")
    st.plotly_chart(confusion_matrix_chart(confusion_df), use_container_width=True)

st.subheader("Feature Importance Top 20")
st.plotly_chart(feature_importance_chart(feature_df, n=20), use_container_width=True)
st.dataframe(feature_df.head(20), use_container_width=True, hide_index=True)
