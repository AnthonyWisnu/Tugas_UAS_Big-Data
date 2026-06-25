import streamlit as st
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.load_data import (
    load_best_model,
    load_label_encoder,
    load_model_data,
    load_processed_data,
    sorted_options,
)
from utils.prediction import build_prediction_row, predict_category


st.set_page_config(page_title="Player Prediction", layout="wide")
st.title("Player Prediction")
st.caption("Prediction uses the saved best model artifact. The dashboard does not retrain models.")

clean_df = load_processed_data()
model_df = load_model_data()
model = load_best_model()
label_encoder = load_label_encoder()

model_columns = list(model_df.columns)
performance_inputs = [
    "matches_played",
    "starts",
    "minutes",
    "goals",
    "assists",
    "non_penalty_goals",
    "yellow_cards",
    "red_cards",
    "shots_total",
    "shots_on_target",
    "fouls_committed",
    "fouls_drawn",
    "saves",
    "clean_sheets",
    "goals_against",
    "shots_on_target_against",
]
available_performance_inputs = [column for column in performance_inputs if column in model_columns]

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=15.0, max_value=45.0, value=25.0, step=1.0)
        height_m = st.number_input("Height, meter", min_value=1.40, max_value=2.20, value=1.80, step=0.01)
        preferred_foot = st.selectbox("Preferred Foot", sorted_options(model_df, "preferred_foot") or ["right", "left", "both"])
        pos_category = st.selectbox("Position", sorted_options(model_df, "pos_category") or ["Attack", "Midfield", "Defender", "Goalkeeper"])
    with col2:
        nationality = st.selectbox("Nationality", sorted_options(model_df, "nationality") or ["Unknown"])
        league = st.selectbox("League", sorted_options(model_df, "league") or ["Premier League"])
        league_rank_map = (
            clean_df[["league", "league_rank"]]
            .drop_duplicates()
            .set_index("league")["league_rank"]
            .to_dict()
            if {"league", "league_rank"} <= set(clean_df.columns)
            else {}
        )
        league_rank = st.number_input("League Rank", min_value=1, max_value=5, value=int(league_rank_map.get(league, 1)), step=1)
        season = st.number_input("Season", min_value=2017, max_value=2024, value=2024, step=1)
    with col3:
        clubs = sorted(clean_df[clean_df["league"] == league]["club"].dropna().astype(str).unique().tolist())
        club = st.selectbox("Club", clubs or sorted_options(model_df, "club") or ["Unknown"])
        club_total_mv_mio = st.number_input("Club Total Market Value, EUR Mio", min_value=0.0, value=250.0, step=10.0)
        prev_season_mv = st.number_input("Previous Season Market Value, EUR Mio", min_value=0.0, value=10.0, step=1.0)
        two_seasons_ago_mv = st.number_input("Two Seasons Ago Market Value, EUR Mio", min_value=0.0, value=8.0, step=1.0)
        mv_history_count = st.number_input("Market Value History Count", min_value=0, value=1, step=1)

    performance_values = {}
    if available_performance_inputs:
        st.subheader("Performance Inputs")
        perf_cols = st.columns(4)
        for index, column in enumerate(available_performance_inputs):
            with perf_cols[index % 4]:
                performance_values[column] = st.number_input(
                    column.replace("_", " ").title(),
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                )

    submitted = st.form_submit_button("Predict")

if submitted:
    inputs = {
        "age": age,
        "height_m": height_m,
        "preferred_foot": preferred_foot,
        "pos_category": pos_category,
        "nationality": nationality,
        "club": club,
        "league": league,
        "league_rank": league_rank,
        "season": season,
        "club_total_mv_mio": club_total_mv_mio,
        "prev_season_mv": prev_season_mv,
        "two_seasons_ago_mv": two_seasons_ago_mv,
        "mv_history_count": mv_history_count,
    }
    inputs.update(performance_values)
    input_df = build_prediction_row(inputs, clean_df, model_columns)
    prediction, probabilities = predict_category(model, label_encoder, input_df)
    st.subheader("Prediction Result")
    st.metric("Predicted Category", prediction)

    if probabilities is not None:
        st.subheader("Prediction Probabilities")
        st.dataframe(probabilities, use_container_width=True, hide_index=True)

    with st.expander("Model Input Row"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)
