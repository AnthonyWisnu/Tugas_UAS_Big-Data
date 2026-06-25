import streamlit as st
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.load_data import load_processed_data, sidebar_filters
from utils.plotting import (
    label_distribution_chart,
    records_by_league_chart,
    records_by_season_chart,
)


st.set_page_config(page_title="Overview", layout="wide")
st.title("Overview")

df = load_processed_data()
filtered_df, _ = sidebar_filters(df)

if filtered_df.empty:
    st.warning("No records match the current filters.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(filtered_df):,}")
col2.metric("Total Players", f"{filtered_df['player_id'].nunique():,}" if "player_id" in filtered_df.columns else "N/A")
col3.metric("Total Clubs", f"{filtered_df['club'].nunique():,}")
col4.metric("Season Range", f"{int(filtered_df['season'].min())}-{int(filtered_df['season'].max())}")

left, right = st.columns(2)
with left:
    st.plotly_chart(label_distribution_chart(filtered_df), use_container_width=True)
    st.plotly_chart(records_by_season_chart(filtered_df), use_container_width=True)
with right:
    st.plotly_chart(records_by_league_chart(filtered_df), use_container_width=True)
    st.dataframe(filtered_df.head(50), use_container_width=True, hide_index=True)
