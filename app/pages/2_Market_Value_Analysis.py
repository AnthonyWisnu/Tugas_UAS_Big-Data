import streamlit as st
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.load_data import load_processed_data, sidebar_filters
from utils.plotting import (
    average_market_value_by_league,
    average_market_value_by_season,
    market_value_by_position,
    market_value_distribution_by_category,
    top_clubs_by_total_market_value,
    top_players_by_market_value,
)


st.set_page_config(page_title="Market Value Analysis", layout="wide")
st.title("Market Value Analysis")

df = load_processed_data()
filtered_df, _ = sidebar_filters(df)

if filtered_df.empty:
    st.warning("No records match the current filters.")
    st.stop()

left, right = st.columns(2)
with left:
    st.plotly_chart(average_market_value_by_season(filtered_df), use_container_width=True)
    st.plotly_chart(market_value_distribution_by_category(filtered_df), use_container_width=True)
with right:
    st.plotly_chart(average_market_value_by_league(filtered_df), use_container_width=True)
    st.plotly_chart(market_value_by_position(filtered_df), use_container_width=True)

st.subheader("Top 10 Players by Market Value")
st.dataframe(top_players_by_market_value(filtered_df), use_container_width=True, hide_index=True)

st.subheader("Top 10 Clubs by Club Total Market Value")
st.dataframe(top_clubs_by_total_market_value(filtered_df), use_container_width=True, hide_index=True)
