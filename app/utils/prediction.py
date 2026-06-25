import math

import pandas as pd


TARGET_COLUMN = "market_value_category"

PERFORMANCE_INPUT_COLUMNS = [
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
]


def create_market_value_category(value):
    if value < 10:
        return "Rendah"
    if value <= 30:
        return "Menengah"
    return "Tinggi"


def position_flags(pos_category):
    return {
        "is_goalkeeper": int(pos_category == "Goalkeeper"),
        "is_defender": int(pos_category == "Defender"),
        "is_midfielder": int(pos_category == "Midfield"),
        "is_forward": int(pos_category == "Attack"),
    }


def age_group(age):
    if age <= 20:
        return "U20"
    if age <= 24:
        return "21-24"
    if age <= 28:
        return "25-28"
    if age <= 32:
        return "29-32"
    return "33plus"


def estimate_club_rank(clean_df, league, season, club_total_mv_mio):
    subset = clean_df[(clean_df["league"] == league) & (clean_df["season"] == season)]
    if subset.empty or "club_total_mv_mio" not in subset.columns:
        return 1.0, 1.0
    club_values = (
        subset[["club", "club_total_mv_mio"]]
        .drop_duplicates()
        .copy()
    )
    candidate = pd.DataFrame([{"club": "__candidate__", "club_total_mv_mio": club_total_mv_mio}])
    ranked = pd.concat([club_values, candidate], ignore_index=True)
    ranked["rank"] = ranked["club_total_mv_mio"].rank(method="dense", ascending=False)
    ranked["pct"] = ranked["club_total_mv_mio"].rank(method="average", pct=True)
    row = ranked[ranked["club"] == "__candidate__"].iloc[0]
    return float(row["rank"]), float(row["pct"])


def build_prediction_row(inputs, clean_df, model_columns):
    age = float(inputs["age"])
    club_total_mv_mio = float(inputs["club_total_mv_mio"])
    prev_season_mv = float(inputs["prev_season_mv"])
    two_seasons_ago_mv = float(inputs["two_seasons_ago_mv"])
    season = int(inputs["season"])
    league = inputs["league"]
    pos_category = inputs["pos_category"]

    if two_seasons_ago_mv > 0:
        prev_growth_rate = (prev_season_mv - two_seasons_ago_mv) / two_seasons_ago_mv
    else:
        prev_growth_rate = 0.0

    rank, pct = estimate_club_rank(clean_df, league, season, club_total_mv_mio)

    matches_played = float(inputs.get("matches_played", 0))
    starts = float(inputs.get("starts", 0))
    minutes = float(inputs.get("minutes", 0))
    goals = float(inputs.get("goals", 0))
    assists = float(inputs.get("assists", 0))
    non_penalty_goals = float(inputs.get("non_penalty_goals", goals))
    yellow_cards = float(inputs.get("yellow_cards", 0))
    red_cards = float(inputs.get("red_cards", 0))
    shots_total = float(inputs.get("shots_total", 0))
    shots_on_target = float(inputs.get("shots_on_target", 0))
    fouls_committed = float(inputs.get("fouls_committed", 0))
    fouls_drawn = float(inputs.get("fouls_drawn", 0))
    saves = float(inputs.get("saves", 0))
    clean_sheets = float(inputs.get("clean_sheets", 0))
    goals_against = float(inputs.get("goals_against", 0))
    shots_on_target_against = float(inputs.get("shots_on_target_against", 0))

    minutes_denominator = minutes if minutes > 0 else None
    matches_denominator = matches_played if matches_played > 0 else None
    sota_denominator = shots_on_target_against if shots_on_target_against > 0 else None

    row = {
        "age": age,
        "age_squared": age ** 2,
        "age_group": age_group(age),
        "age_peak_distance": abs(age - 27),
        "is_peak_age": int(24 <= age <= 29),
        "height_m": float(inputs["height_m"]),
        "preferred_foot": inputs["preferred_foot"],
        "pos_category": pos_category,
        **position_flags(pos_category),
        "nationality": inputs["nationality"],
        "club": inputs["club"],
        "league": league,
        "league_rank": int(inputs["league_rank"]),
        "season": season,
        "club_total_mv_mio": club_total_mv_mio,
        "club_total_mv_log": math.log1p(club_total_mv_mio),
        "club_total_mv_rank_league_season": rank,
        "club_total_mv_pct_league_season": pct,
        "prev_season_mv": prev_season_mv,
        "prev_season_mv_log": math.log1p(prev_season_mv),
        "prev_mv_category": "NoHistory" if prev_season_mv == 0 else create_market_value_category(prev_season_mv),
        "prev_mv_distance_to_10": prev_season_mv - 10,
        "prev_mv_distance_to_30": prev_season_mv - 30,
        "two_seasons_ago_mv": two_seasons_ago_mv,
        "two_seasons_ago_mv_log": math.log1p(two_seasons_ago_mv),
        "two_seasons_ago_mv_category": (
            "NoHistory" if two_seasons_ago_mv == 0 else create_market_value_category(two_seasons_ago_mv)
        ),
        "has_prev_mv": int(prev_season_mv > 0),
        "mv_history_count": int(inputs["mv_history_count"]),
        "prev_growth_rate": prev_growth_rate,
        "prev_growth_rate_clipped": min(max(prev_growth_rate, -1), 3),
        "prev_mv_to_club_total_ratio": prev_season_mv / club_total_mv_mio if club_total_mv_mio > 0 else 0,
        "age_prev_mv_interaction": age * math.log1p(prev_season_mv),
        "matches_played": matches_played,
        "starts": starts,
        "minutes": minutes,
        "goals": goals,
        "assists": assists,
        "non_penalty_goals": non_penalty_goals,
        "yellow_cards": yellow_cards,
        "red_cards": red_cards,
        "shots_total": shots_total,
        "shots_on_target": shots_on_target,
        "fouls_committed": fouls_committed,
        "fouls_drawn": fouls_drawn,
        "saves": saves,
        "clean_sheets": clean_sheets,
        "goals_against": goals_against,
        "has_performance_stats": int(minutes > 0 or matches_played > 0),
        "goals_per_90": goals / minutes_denominator * 90 if minutes_denominator else 0,
        "assists_per_90": assists / minutes_denominator * 90 if minutes_denominator else 0,
        "goal_assist_per_90": (goals + assists) / minutes_denominator * 90 if minutes_denominator else 0,
        "shots_per_90": shots_total / minutes_denominator * 90 if minutes_denominator else 0,
        "shots_on_target_per_90": shots_on_target / minutes_denominator * 90 if minutes_denominator else 0,
        "cards_per_90": (yellow_cards + red_cards) / minutes_denominator * 90 if minutes_denominator else 0,
        "starts_rate": starts / matches_denominator if matches_denominator else 0,
        "save_pct": saves / sota_denominator if sota_denominator else 0,
        "clean_sheet_pct": clean_sheets / matches_denominator if matches_denominator else 0,
    }

    feature_columns = [column for column in model_columns if column != TARGET_COLUMN]
    return pd.DataFrame([{column: row.get(column, 0) for column in feature_columns}])


def predict_category(model, label_encoder, input_df):
    prediction = model.predict(input_df)
    if prediction.dtype.kind in {"i", "u"}:
        label = label_encoder.inverse_transform(prediction)[0]
    else:
        label = prediction[0]

    probabilities = None
    if hasattr(model, "predict_proba"):
        probability_values = model.predict_proba(input_df)[0]
        classes = getattr(model, "classes_", label_encoder.transform(label_encoder.classes_))
        labels = label_encoder.inverse_transform(classes) if pd.Series(classes).dtype.kind in {"i", "u"} else classes
        probabilities = pd.DataFrame(
            {"category": labels, "probability": probability_values}
        ).sort_values("probability", ascending=False)
    return label, probabilities
