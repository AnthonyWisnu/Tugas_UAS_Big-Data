import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "transfermarkt_dataset_clean.csv"
FBREF_STATS_FILE = PROJECT_ROOT / "data" / "interim" / "fbref_player_stats.csv"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PERFORMANCE_MODEL_FILE = PROJECT_ROOT / "data" / "model" / "players_model_with_performance.csv"
MATCHING_RESULT_FILE = INTERIM_DIR / "player_matching_result.csv"
UNMATCHED_PLAYERS_FILE = INTERIM_DIR / "unmatched_players.csv"

TARGET_COLUMN = "market_value_category"
MATCH_THRESHOLD = 88

PERFORMANCE_FEATURES = [
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

DERIVED_FEATURES = [
    "goals_per_90",
    "assists_per_90",
    "goal_assist_per_90",
    "shots_per_90",
    "shots_on_target_per_90",
    "cards_per_90",
    "starts_rate",
    "save_pct",
    "clean_sheet_pct",
]

ALIASES = {
    "matches_played": ["standard_playing_time_mp", "standard_mp", "playing_time_mp", "mp"],
    "starts": ["standard_playing_time_starts", "standard_starts", "playing_time_starts", "starts"],
    "minutes": ["standard_playing_time_min", "standard_min", "playing_time_min", "min"],
    "goals": ["standard_performance_gls", "standard_gls", "gls"],
    "assists": ["standard_performance_ast", "standard_ast", "ast"],
    "non_penalty_goals": ["standard_performance_g_pk", "standard_performance_npg", "standard_npg", "npg"],
    "yellow_cards": ["misc_performance_crdy", "misc_crdy", "crdy"],
    "red_cards": ["misc_performance_crdr", "misc_crdr", "crdr"],
    "shots_total": ["shooting_standard_sh", "shooting_sh", "sh"],
    "shots_on_target": ["shooting_standard_sot", "shooting_sot", "sot"],
    "fouls_committed": ["misc_performance_fls", "misc_fls", "fls"],
    "fouls_drawn": ["misc_performance_fld", "misc_fld", "fld"],
    "saves": ["keeper_performance_saves", "keeper_saves", "saves"],
    "clean_sheets": ["keeper_performance_cs", "keeper_cs", "cs"],
    "goals_against": ["keeper_performance_ga", "keeper_ga", "ga"],
    "shots_on_target_against": ["keeper_performance_sota", "keeper_sota", "sota"],
}


def normalize_key(value):
    if pd.isna(value):
        return ""
    text = unidecode(str(value)).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def find_column(columns, aliases):
    normalized_columns = {normalize_key(column): column for column in columns}
    for alias in aliases:
        alias_key = normalize_key(alias)
        if alias_key in normalized_columns:
            return normalized_columns[alias_key]
    for norm_column, source_column in normalized_columns.items():
        for alias in aliases:
            if norm_column.endswith(normalize_key(alias)):
                return source_column
    return None


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def extract_performance_features(fbref_df):
    result = fbref_df.copy()
    for feature in PERFORMANCE_FEATURES + ["shots_on_target_against"]:
        source = find_column(result.columns, ALIASES.get(feature, [feature]))
        result[feature] = safe_numeric(result[source]) if source else np.nan

    key_columns = [
        "fbref_player_name",
        "fbref_club",
        "fbref_league",
        "season",
        "player_name_key",
        "club_key",
    ]
    existing_keys = [column for column in key_columns if column in result.columns]
    feature_columns = PERFORMANCE_FEATURES + ["shots_on_target_against"]
    result = result[existing_keys + feature_columns].copy()

    for column in feature_columns:
        result[column] = result[column].fillna(0)
    return result


def add_derived_features(df):
    df = df.copy()
    minutes = df["minutes"].replace(0, np.nan)
    matches = df["matches_played"].replace(0, np.nan)
    sota = df.get("shots_on_target_against", pd.Series(0, index=df.index)).replace(0, np.nan)

    df["goals_per_90"] = df["goals"] / minutes * 90
    df["assists_per_90"] = df["assists"] / minutes * 90
    df["goal_assist_per_90"] = (df["goals"] + df["assists"]) / minutes * 90
    df["shots_per_90"] = df["shots_total"] / minutes * 90
    df["shots_on_target_per_90"] = df["shots_on_target"] / minutes * 90
    df["cards_per_90"] = (df["yellow_cards"] + df["red_cards"]) / minutes * 90
    df["starts_rate"] = df["starts"] / matches
    df["save_pct"] = df["saves"] / sota
    df["clean_sheet_pct"] = df["clean_sheets"] / matches

    for column in DERIVED_FEATURES:
        df[column] = df[column].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def build_tm_matching_frame(clean_df, model_df):
    if len(clean_df) != len(model_df):
        raise ValueError("Clean dataset and model dataset row counts differ. Regenerate preprocessing first.")
    metadata_columns = ["player_id", "player_name", "club", "league", "season"]
    metadata = clean_df[metadata_columns].copy()
    metadata["tm_player_name_key"] = metadata["player_name"].map(normalize_key)
    metadata["tm_club_key"] = metadata["club"].map(normalize_key)
    metadata["row_id"] = np.arange(len(metadata))
    model_without_duplicate_metadata = model_df.drop(
        columns=[column for column in metadata_columns if column in model_df.columns],
        errors="ignore",
    )
    return pd.concat([metadata, model_without_duplicate_metadata.reset_index(drop=True)], axis=1)


def match_players(tm_df, fbref_df):
    fbref_lookup = {}
    for season, group in fbref_df.groupby("season"):
        choices = group["player_name_key"].dropna().unique().tolist()
        fbref_lookup[season] = (choices, group)

    matched_rows = []
    audit_rows = []
    performance_columns = PERFORMANCE_FEATURES + DERIVED_FEATURES

    for _, row in tm_df.iterrows():
        season = row["season"]
        player_key = row["tm_player_name_key"]
        club_key = row["tm_club_key"]
        match_info = {
            "row_id": row["row_id"],
            "player_id": row["player_id"],
            "player_name": row["player_name"],
            "club": row["club"],
            "season": season,
            "matched": False,
            "match_score": 0,
            "fbref_player_name": "",
            "fbref_club": "",
        }

        output = row.to_dict()
        for column in performance_columns:
            output[column] = 0
        output["has_performance_stats"] = 0

        if season in fbref_lookup and player_key:
            choices, group = fbref_lookup[season]
            match = process.extractOne(player_key, choices, scorer=fuzz.WRatio)
            if match and match[1] >= MATCH_THRESHOLD:
                candidates = group[group["player_name_key"] == match[0]].copy()
                if club_key and "club_key" in candidates.columns:
                    candidates["club_score"] = candidates["club_key"].map(
                        lambda value: fuzz.WRatio(club_key, value)
                    )
                    candidates = candidates.sort_values("club_score", ascending=False)
                    best_candidate = candidates.iloc[0]
                    club_score = float(best_candidate.get("club_score", 0))
                else:
                    best_candidate = candidates.iloc[0]
                    club_score = 0

                for column in performance_columns:
                    output[column] = best_candidate.get(column, 0)
                output["has_performance_stats"] = 1
                match_info.update(
                    {
                        "matched": True,
                        "match_score": float(match[1]),
                        "club_score": club_score,
                        "fbref_player_name": best_candidate.get("fbref_player_name", ""),
                        "fbref_club": best_candidate.get("fbref_club", ""),
                    }
                )

        matched_rows.append(output)
        audit_rows.append(match_info)

    return pd.DataFrame(matched_rows), pd.DataFrame(audit_rows)


def build_performance_dataset(clean_df, model_df, fbref_raw):
    fbref_features = extract_performance_features(fbref_raw)
    fbref_features = add_derived_features(fbref_features)
    tm_df = build_tm_matching_frame(clean_df, model_df)
    merged_df, audit_df = match_players(tm_df, fbref_features)

    leakage_columns = ["player_id", "player_name", "player_url", "market_value_mio", "market_value_str"]
    model_output = merged_df.drop(columns=[column for column in leakage_columns if column in merged_df.columns])
    helper_columns = ["tm_player_name_key", "tm_club_key", "row_id"]
    model_output = model_output.drop(columns=[column for column in helper_columns if column in model_output.columns])
    return model_output, audit_df


def merge_performance(
    clean_file=CLEAN_DATA_FILE,
    model_file=None,
    fbref_file=FBREF_STATS_FILE,
):
    clean_df = pd.read_csv(require_file(clean_file))
    if model_file:
        model_df = pd.read_csv(require_file(model_file))
    else:
        from src.preprocessing.clean_dataset import MODEL_FEATURE_COLUMNS, TARGET_COLUMN

        model_columns = [
            column
            for column in MODEL_FEATURE_COLUMNS + [TARGET_COLUMN]
            if column in clean_df.columns
        ]
        model_df = clean_df[model_columns].copy()
    fbref_raw = pd.read_csv(require_file(fbref_file))

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    model_output, audit_df = build_performance_dataset(clean_df, model_df, fbref_raw)

    PERFORMANCE_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    model_output.to_csv(PERFORMANCE_MODEL_FILE, index=False)
    audit_df.to_csv(MATCHING_RESULT_FILE, index=False)
    audit_df[~audit_df["matched"]].to_csv(UNMATCHED_PLAYERS_FILE, index=False)

    print(f"Saved model dataset with performance: {PERFORMANCE_MODEL_FILE}")
    print(f"Saved matching audit: {MATCHING_RESULT_FILE}")
    print(f"Saved unmatched players: {UNMATCHED_PLAYERS_FILE}")
    print(f"Rows: {len(model_output)}")
    print(f"Matched rows: {int(audit_df['matched'].sum())}")
    return model_output, audit_df


def parse_args():
    parser = argparse.ArgumentParser(description="Merge FBref performance data into model dataset.")
    parser.add_argument("--clean-file", default=str(CLEAN_DATA_FILE))
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--fbref-file", default=str(FBREF_STATS_FILE))
    return parser.parse_args()


def main():
    args = parse_args()
    merge_performance(
        clean_file=args.clean_file,
        model_file=args.model_file,
        fbref_file=args.fbref_file,
    )


if __name__ == "__main__":
    main()
