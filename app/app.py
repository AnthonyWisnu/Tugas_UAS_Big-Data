import streamlit as st
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.load_data import load_processed_data, sidebar_filters
from utils.plotting import (
    label_distribution_chart,
    records_by_league_chart,
    records_by_season_chart,
)


st.set_page_config(
    page_title="Football Market Value Dashboard",
    page_icon=None,
    layout="wide",
)


st.title("Football Player Market Value Classification")
st.caption("Big 5 European leagues, seasons 2017-2024, players with market value at least EUR 5 million.")

df = load_processed_data()
filtered_df, _ = sidebar_filters(df)

total_records = len(filtered_df)
total_players = filtered_df["player_id"].nunique() if "player_id" in filtered_df.columns else 0
total_clubs = filtered_df["club"].nunique() if "club" in filtered_df.columns else 0
season_min = int(filtered_df["season"].min()) if not filtered_df.empty else 0
season_max = int(filtered_df["season"].max()) if not filtered_df.empty else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{total_records:,}")
col2.metric("Total Players", f"{total_players:,}")
col3.metric("Total Clubs", f"{total_clubs:,}")
col4.metric("Season Range", f"{season_min}-{season_max}" if total_records else "No data")

if filtered_df.empty:
    st.warning("No records match the current filters.")
    st.stop()

left, right = st.columns(2)
with left:
    st.plotly_chart(label_distribution_chart(filtered_df), use_container_width=True)
    st.plotly_chart(records_by_season_chart(filtered_df), use_container_width=True)
with right:
    st.plotly_chart(records_by_league_chart(filtered_df), use_container_width=True)
    st.dataframe(
        filtered_df[["player_name", "club", "league", "season", "pos_category", "market_value_mio", "market_value_category"]]
        .sort_values("market_value_mio", ascending=False)
        .head(20),
        use_container_width=True,
        hide_index=True,
    )
