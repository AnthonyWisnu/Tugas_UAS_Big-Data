from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PLAYERS_FILE = PROJECT_ROOT / "data" / "raw" / "players_raw.csv"
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "transfermarkt_dataset_clean.csv"
MODEL_FILE = PROJECT_ROOT / "data" / "model" / "players_model.csv"

SEASON_MIN = 2017
SEASON_MAX = 2024
MIN_MARKET_VALUE_MIO = 5
TARGET_COLUMN = "market_value_category"

LEAKAGE_COLUMNS = {
    "market_value_mio",
    "market_value_str",
    "market_value_category",
    "value_category",
    "label",
    "target",
    "mv_growth_rate",
    "position_detail",
}

MODEL_FEATURE_COLUMNS = [
    "age",
    "age_squared",
    "age_group",
    "age_peak_distance",
    "is_peak_age",
    "height_m",
    "preferred_foot",
    "pos_category",
    "is_goalkeeper",
    "is_defender",
    "is_midfielder",
    "is_forward",
    "nationality",
    "club",
    "league",
    "league_rank",
    "season",
    "club_total_mv_mio",
    "club_total_mv_log",
    "club_total_mv_rank_league_season",
    "club_total_mv_pct_league_season",
    "prev_season_mv",
    "prev_season_mv_log",
    "prev_mv_category",
    "prev_mv_distance_to_10",
    "prev_mv_distance_to_30",
    "two_seasons_ago_mv",
    "two_seasons_ago_mv_log",
    "two_seasons_ago_mv_category",
    "has_prev_mv",
    "mv_history_count",
    "prev_growth_rate",
    "prev_growth_rate_clipped",
    "prev_mv_to_club_total_ratio",
    "age_prev_mv_interaction",
]


def parse_market_value(value):
    if value is None:
        return np.nan

    text = str(value).strip()
    if text == "" or text == "-" or text.lower() in {"nan", "none", "n/a"}:
        return np.nan

    normalized = (
        text.replace("\xa0", " ")
        .replace("\u202f", " ")
        .replace("\u20ac", "EUR")
        .strip()
    )
    normalized = re.sub(r"\s+", " ", normalized)
    normalized_lower = normalized.lower()

    number_match = re.search(r"(\d+(?:[.,]\d+)?)", normalized_lower)
    if not number_match:
        return np.nan

    try:
        amount = float(number_match.group(1).replace(",", "."))
    except ValueError:
        return np.nan

    unit_text = normalized_lower[number_match.end() :].strip()
    if unit_text.startswith("bn") or re.search(r"billion|milyar", normalized_lower):
        return amount * 1000
    if unit_text.startswith("m") or re.search(r"million|mio", normalized_lower):
        return amount
    if unit_text.startswith("k") or re.search(r"thousand|ribu", normalized_lower):
        return amount / 1000
    return amount


def create_market_value_category(value):
    if value < 10:
        return "Rendah"
    if value <= 30:
        return "Menengah"
    return "Tinggi"


def validate_raw_columns(df):
    required_columns = [
        "player_id",
        "player_name",
        "player_url",
        "pos_category",
        "position_detail",
        "age",
        "nationality",
        "height_m",
        "preferred_foot",
        "club",
        "club_total_mv_raw",
        "league",
        "league_rank",
        "season",
        "market_value_str",
        "market_value_mio",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required raw columns: {missing_columns}")


def fill_categorical(df, columns):
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown").astype(str).str.strip()
            df.loc[df[column] == "", column] = "Unknown"
    return df


def fill_numeric_by_group(df, column, group_column="pos_category"):
    df = df.copy()
    group_median = df.groupby(group_column, dropna=False)[column].transform("median")
    overall_median = df[column].median()
    df[column] = df[column].fillna(group_median).fillna(overall_median)
    return df


def add_historical_features_before_target_filter(df):
    df = df.copy()
    df = df.sort_values(["player_id", "season", "club"], kind="mergesort")
    grouped = df.groupby("player_id", sort=False)
    df["prev_season_mv"] = grouped["market_value_mio"].shift(1)
    df["two_seasons_ago_mv"] = grouped["market_value_mio"].shift(2)
    df["has_prev_mv"] = df["prev_season_mv"].notna().astype(int)
    df["mv_history_count"] = grouped.cumcount()
    df["prev_growth_rate"] = (
        (df["prev_season_mv"] - df["two_seasons_ago_mv"]) / df["two_seasons_ago_mv"]
    )
    df.loc[df["two_seasons_ago_mv"].isna(), "prev_growth_rate"] = np.nan
    df.loc[df["two_seasons_ago_mv"] == 0, "prev_growth_rate"] = np.nan
    return df


def add_safe_features(df):
    df = df.copy()

    df["is_goalkeeper"] = (df["pos_category"] == "Goalkeeper").astype(int)
    df["is_defender"] = (df["pos_category"] == "Defender").astype(int)
    df["is_midfielder"] = (df["pos_category"] == "Midfield").astype(int)
    df["is_forward"] = (df["pos_category"] == "Attack").astype(int)

    df["age_squared"] = df["age"] ** 2
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 20, 24, 28, 32, 100],
        labels=["U20", "21-24", "25-28", "29-32", "33plus"],
        include_lowest=True,
    ).astype(str)
    df["age_peak_distance"] = (df["age"] - 27).abs()
    df["is_peak_age"] = df["age"].between(24, 29, inclusive="both").astype(int)

    df["club_total_mv_log"] = np.log1p(df["club_total_mv_mio"])
    df["club_total_mv_rank_league_season"] = df.groupby(
        ["league", "season"]
    )["club_total_mv_mio"].rank(method="dense", ascending=False)
    df["club_total_mv_pct_league_season"] = df.groupby(
        ["league", "season"]
    )["club_total_mv_mio"].rank(method="average", pct=True)

    df["prev_season_mv"] = df["prev_season_mv"].fillna(0)
    df["two_seasons_ago_mv"] = df["two_seasons_ago_mv"].fillna(0)
    df["prev_growth_rate"] = df["prev_growth_rate"].fillna(0)
    df["prev_season_mv_log"] = np.log1p(df["prev_season_mv"])
    df["two_seasons_ago_mv_log"] = np.log1p(df["two_seasons_ago_mv"])
    df["prev_mv_category"] = df["prev_season_mv"].apply(
        lambda value: "NoHistory" if value == 0 else create_market_value_category(value)
    )
    df["two_seasons_ago_mv_category"] = df["two_seasons_ago_mv"].apply(
        lambda value: "NoHistory" if value == 0 else create_market_value_category(value)
    )
    df["prev_mv_distance_to_10"] = df["prev_season_mv"] - 10
    df["prev_mv_distance_to_30"] = df["prev_season_mv"] - 30
    df["prev_growth_rate_clipped"] = df["prev_growth_rate"].clip(lower=-1, upper=3)
    df["prev_mv_to_club_total_ratio"] = np.where(
        df["club_total_mv_mio"] > 0,
        df["prev_season_mv"] / df["club_total_mv_mio"],
        0,
    )
    df["age_prev_mv_interaction"] = df["age"] * df["prev_season_mv_log"]

    return df


def build_preprocessed_dataset(raw_file=RAW_PLAYERS_FILE):
    raw_file = Path(raw_file)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw players file not found: {raw_file}")

    raw_df = pd.read_csv(raw_file)
    validate_raw_columns(raw_df)

    df = raw_df.copy()
    report = {
        "raw_rows": len(df),
        "raw_columns": list(df.columns),
    }

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")
    df["market_value_mio"] = pd.to_numeric(df["market_value_mio"], errors="coerce")
    df["league_rank"] = pd.to_numeric(df["league_rank"], errors="coerce")
    df["club_total_mv_mio"] = df["club_total_mv_raw"].apply(parse_market_value)

    df = df[df["season"].between(SEASON_MIN, SEASON_MAX, inclusive="both")]
    report["rows_after_season_filter"] = len(df)

    df = df.dropna(subset=["market_value_mio"])
    report["rows_after_market_value_notna"] = len(df)

    duplicate_subset = ["player_id", "season"]
    before_dedup_all_values = len(df)
    df = df.sort_values(["player_id", "season", "club"], kind="mergesort")
    df = df.drop_duplicates(subset=duplicate_subset, keep="first")
    report["duplicates_removed_before_history"] = before_dedup_all_values - len(df)

    categorical_columns = [
        "player_id",
        "player_name",
        "player_url",
        "shirt_number",
        "pos_category",
        "position_detail",
        "nationality",
        "preferred_foot",
        "club",
        "league",
    ]
    df = fill_categorical(df, categorical_columns)

    for column in ["age", "height_m", "club_total_mv_mio"]:
        df = fill_numeric_by_group(df, column)

    df["season"] = df["season"].astype(int)
    df["league_rank"] = df["league_rank"].astype(int)

    df = add_historical_features_before_target_filter(df)
    report["rows_used_for_history"] = len(df)

    df = df[df["market_value_mio"] >= MIN_MARKET_VALUE_MIO].copy()
    report["rows_after_market_value_filter"] = len(df)

    df["market_value_category"] = df["market_value_mio"].apply(create_market_value_category)
    df["has_prev_mv"] = df["has_prev_mv"].astype(int)
    df["mv_history_count"] = df["mv_history_count"].astype(int)
    df = add_safe_features(df)

    clean_df = df.reset_index(drop=True)

    model_columns = [
        column
        for column in MODEL_FEATURE_COLUMNS + [TARGET_COLUMN]
        if column in clean_df.columns
    ]
    model_df = clean_df[model_columns].copy()

    leakage_in_model = sorted((set(model_df.columns) - {TARGET_COLUMN}) & LEAKAGE_COLUMNS)
    if leakage_in_model:
        raise ValueError(f"Leakage columns found in model dataset: {leakage_in_model}")

    report["duplicate_subset"] = duplicate_subset
    report["dropped_features"] = sorted(set(LEAKAGE_COLUMNS) | {"position_detail"})
    report["label_distribution"] = clean_df[TARGET_COLUMN].value_counts().to_dict()
    report["position_detail_null_raw"] = int(raw_df["position_detail"].isna().sum())
    report["position_detail_rows_raw"] = len(raw_df)
    report["clean_rows"] = len(clean_df)
    report["model_rows"] = len(model_df)
    report["model_columns"] = list(model_df.columns)

    return clean_df, model_df, report


def save_outputs(clean_df, model_df):
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_FILE, index=False)
    model_df.to_csv(MODEL_FILE, index=False)
    return PROCESSED_FILE, MODEL_FILE


def main():
    clean_df, model_df, report = build_preprocessed_dataset()
    processed_file, model_file = save_outputs(clean_df, model_df)

    print("Preprocessing selesai.")
    print(f"Processed file: {processed_file}")
    print(f"Model file    : {model_file}")
    print(f"Raw rows      : {report['raw_rows']}")
    print(f"Rows after market value not null: {report['rows_after_market_value_notna']}")
    print(f"Rows after market value filter  : {report['rows_after_market_value_filter']}")
    print(f"Clean rows    : {report['clean_rows']}")
    print(f"Model rows    : {report['model_rows']}")
    print(f"Duplicates removed before history: {report['duplicates_removed_before_history']}")
    print(f"Label distribution: {report['label_distribution']}")
    print(f"Dropped features: {report['dropped_features']}")
    return clean_df, model_df, report


if __name__ == "__main__":
    main()
