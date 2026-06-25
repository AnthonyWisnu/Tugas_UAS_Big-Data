from argparse import ArgumentParser
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_INPUT_FILE = PROJECT_ROOT / "data" / "model" / "players_model.csv"
CLEAN_INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "transfermarkt_dataset_clean.csv"
FBREF_INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "fbref_player_stats.csv"
MODEL_OUTPUT_FILE = PROJECT_ROOT / "data" / "model" / "players_model.csv"
MATCHING_AUDIT_FILE = PROJECT_ROOT / "data" / "interim" / "player_matching_result.csv"
UNMATCHED_FILE = PROJECT_ROOT / "data" / "interim" / "unmatched_players.csv"

TARGET_COLUMN = "market_value_category"
MATCH_KEYS = ["player_norm", "club_norm", "league_norm", "season"]
PLAYER_LEAGUE_SEASON_KEYS = ["player_norm", "league_norm", "season"]
PLAYER_SEASON_KEYS = ["player_norm", "season"]

PERFORMANCE_FEATURE_MAP = {
    "matches_played": "standard_playing_time_mp",
    "starts": "standard_playing_time_starts",
    "minutes": "standard_playing_time_min",
    "goals": "standard_performance_gls",
    "assists": "standard_performance_ast",
    "non_penalty_goals": "standard_performance_g_pk",
    "yellow_cards": "misc_performance_crdy",
    "red_cards": "misc_performance_crdr",
    "shots_total": "shooting_standard_sh",
    "shots_on_target": "shooting_standard_sot",
    "fouls_committed": "misc_performance_fls",
    "fouls_drawn": "misc_performance_fld",
    "saves": "keeper_performance_saves",
    "clean_sheets": "keeper_performance_cs",
    "goals_against": "keeper_performance_ga",
    "shots_on_target_against": "keeper_performance_sota",
}

FORBIDDEN_PERFORMANCE_FEATURES = {
    "xg",
    "non_penalty_xg",
    "xg_per_90",
    "non_penalty_xg_per_90",
    "aerial_won",
    "aerial_lost",
    "ball_recoveries",
}


def normalize_text(value):
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_league(value):
    text = normalize_text(value)
    replacements = {
        "eng premier league": "premier league",
        "premier league": "premier league",
        "es la liga": "la liga",
        "la liga": "la liga",
        "de bundesliga": "bundesliga",
        "bundesliga": "bundesliga",
        "it serie a": "serie a",
        "serie a": "serie a",
        "fr ligue 1": "ligue 1",
        "ligue 1": "ligue 1",
    }
    return replacements.get(text, text)


def normalize_season(value):
    if pd.isna(value):
        return np.nan
    text = str(value)
    match = re.search(r"(20\d{2})", text)
    if match:
        return int(match.group(1))
    numeric = pd.to_numeric(value, errors="coerce")
    return int(numeric) if pd.notna(numeric) else np.nan


def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Input file tidak ditemukan: {path}. Jalankan tahap yang sesuai, jangan scrape Transfermarkt ulang."
        )
    return path


def pick_first_existing(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def add_match_columns_transfermarkt(clean_df):
    df = clean_df.copy()
    required = ["player_name", "club", "league", "season"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing Transfermarkt metadata columns: {missing}")
    df["season"] = df["season"].map(normalize_season)
    df["player_norm"] = df["player_name"].map(normalize_text)
    df["club_norm"] = df["club"].map(normalize_text)
    df["league_norm"] = df["league"].map(normalize_league)
    return df


def add_match_columns_fbref(fbref_df):
    df = fbref_df.copy()
    player_column = pick_first_existing(df, ["player", "standard_player"])
    club_column = pick_first_existing(df, ["squad", "team", "standard_squad"])
    league_column = pick_first_existing(df, ["league", "standard_league"])
    if player_column is None or club_column is None:
        raise ValueError("FBref data must contain player and squad/team columns.")
    if league_column is None:
        df["league"] = ""
        league_column = "league"
    if "season" not in df.columns:
        season_column = pick_first_existing(df, ["fbref_season_label", "standard_season"])
        if season_column is None:
            raise ValueError("FBref data must contain season or fbref_season_label.")
        df["season"] = df[season_column].map(normalize_season)
    else:
        df["season"] = df["season"].map(normalize_season)

    df["player_norm"] = df[player_column].map(normalize_text)
    df["club_norm"] = df[club_column].map(normalize_text)
    df["league_norm"] = df[league_column].map(normalize_league)
    return df


def available_feature_map(fbref_df):
    return {
        feature: source
        for feature, source in PERFORMANCE_FEATURE_MAP.items()
        if source in fbref_df.columns
    }


def build_performance_frame(fbref_df):
    feature_map = available_feature_map(fbref_df)
    if not feature_map:
        raise ValueError("Tidak ada kolom performa valid dari FBref yang tersedia.")

    columns = MATCH_KEYS + list(feature_map.values())
    perf = fbref_df[columns].copy()
    for source in feature_map.values():
        perf[source] = pd.to_numeric(perf[source], errors="coerce")

    aggregation = {source: "sum" for source in feature_map.values()}
    perf = perf.groupby(MATCH_KEYS, dropna=False, as_index=False).agg(aggregation)
    perf = perf.rename(columns={source: feature for feature, source in feature_map.items()})
    return perf, list(feature_map.keys())


def unique_rows_by_keys(df, keys):
    counts = df.groupby(keys, dropna=False).size().reset_index(name="_key_count")
    unique_keys = counts[counts["_key_count"] == 1].drop(columns="_key_count")
    return df.merge(unique_keys, on=keys, how="inner")


def collect_matches(working, perf_df, performance_columns):
    matches = []
    remaining = working.copy()
    match_steps = [
        ("exact_player_club_league_season", MATCH_KEYS),
        ("unique_player_league_season", PLAYER_LEAGUE_SEASON_KEYS),
        ("unique_player_season", PLAYER_SEASON_KEYS),
    ]

    for method, keys in match_steps:
        if remaining.empty:
            break

        left = remaining
        right = perf_df
        if method != "exact_player_club_league_season":
            left = unique_rows_by_keys(remaining, keys)
            right = unique_rows_by_keys(perf_df, keys)

        matched = left[["_row_id"] + keys].merge(
            right[keys + performance_columns],
            on=keys,
            how="inner",
        )
        if matched.empty:
            continue

        matched = matched.drop_duplicates("_row_id", keep="first")
        matched["match_method"] = method
        matches.append(matched[["_row_id", "match_method"] + performance_columns])
        remaining = remaining[~remaining["_row_id"].isin(matched["_row_id"])].copy()

    if not matches:
        return pd.DataFrame(columns=["_row_id", "match_method"] + performance_columns)
    return pd.concat(matches, ignore_index=True)


def add_derived_features(df, available_features):
    df = df.copy()
    minutes = df["minutes"] if "minutes" in df.columns else pd.Series(0, index=df.index)
    per90_denominator = minutes.replace(0, np.nan) / 90

    if {"goals", "minutes"} <= set(df.columns):
        df["goals_per_90"] = (df["goals"] / per90_denominator).fillna(0)
    if {"assists", "minutes"} <= set(df.columns):
        df["assists_per_90"] = (df["assists"] / per90_denominator).fillna(0)
    if {"goals", "assists", "minutes"} <= set(df.columns):
        df["goal_assist_per_90"] = ((df["goals"] + df["assists"]) / per90_denominator).fillna(0)
    if {"shots_total", "minutes"} <= set(df.columns):
        df["shots_per_90"] = (df["shots_total"] / per90_denominator).fillna(0)
    if {"shots_on_target", "minutes"} <= set(df.columns):
        df["shots_on_target_per_90"] = (df["shots_on_target"] / per90_denominator).fillna(0)
    if {"yellow_cards", "red_cards", "minutes"} <= set(df.columns):
        df["cards_per_90"] = (
            (df["yellow_cards"] + df["red_cards"]) / per90_denominator
        ).fillna(0)
    if {"starts", "matches_played"} <= set(df.columns):
        df["starts_rate"] = np.where(df["matches_played"] > 0, df["starts"] / df["matches_played"], 0)
    if {"saves", "shots_on_target_against"} <= set(df.columns):
        df["save_pct"] = np.where(
            df["shots_on_target_against"] > 0,
            df["saves"] / df["shots_on_target_against"],
            0,
        )
    if {"clean_sheets", "matches_played"} <= set(df.columns):
        df["clean_sheet_pct"] = np.where(
            df["matches_played"] > 0,
            df["clean_sheets"] / df["matches_played"],
            0,
        )

    performance_columns = [
        column
        for column in df.columns
        if column in set(available_features)
        or column
        in {
            "goals_per_90",
            "assists_per_90",
            "goal_assist_per_90",
            "shots_per_90",
            "shots_on_target_per_90",
            "cards_per_90",
            "starts_rate",
            "save_pct",
            "clean_sheet_pct",
        }
    ]
    return df, performance_columns


def merge_performance(
    model_file=MODEL_INPUT_FILE,
    clean_file=CLEAN_INPUT_FILE,
    fbref_file=FBREF_INPUT_FILE,
    output_file=MODEL_OUTPUT_FILE,
):
    model_df = pd.read_csv(require_file(model_file))
    clean_df = pd.read_csv(require_file(clean_file))
    fbref_df = pd.read_csv(require_file(fbref_file))

    return merge_performance_frames(
        model_df=model_df,
        clean_df=clean_df,
        fbref_df=fbref_df,
        output_file=output_file,
        write_outputs=True,
    )


def merge_performance_frames(
    model_df,
    clean_df,
    fbref_df,
    output_file=MODEL_OUTPUT_FILE,
    write_outputs=True,
):
    model_df = model_df.copy()
    clean_df = clean_df.copy()
    fbref_df = fbref_df.copy()

    if len(model_df) != len(clean_df):
        raise ValueError("players_model.csv dan transfermarkt_dataset_clean.csv harus memiliki jumlah baris sama.")

    tm_meta = add_match_columns_transfermarkt(clean_df)
    fbref_match = add_match_columns_fbref(fbref_df)
    perf_df, base_features = build_performance_frame(fbref_match)

    working = model_df.reset_index(drop=True).copy()
    for key in MATCH_KEYS:
        working[key] = tm_meta[key].reset_index(drop=True)
    working["_row_id"] = np.arange(len(working))

    match_columns = [column for column in perf_df.columns if column not in MATCH_KEYS]
    matches = collect_matches(working, perf_df, match_columns)
    merged = working.merge(matches, on="_row_id", how="left")
    merged["match_method"] = merged["match_method"].fillna("unmatched")
    merged["has_performance_stats"] = (merged["match_method"] != "unmatched").astype(int)

    merged, performance_columns = add_derived_features(merged, base_features)
    for column in performance_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0)

    forbidden_present = sorted(FORBIDDEN_PERFORMANCE_FEATURES & set(merged.columns))
    if forbidden_present:
        raise ValueError(f"Forbidden unavailable performance features found: {forbidden_present}")

    audit_columns = ["_row_id", "has_performance_stats", "match_method"] + MATCH_KEYS
    audit = merged[audit_columns].copy()
    audit = audit.merge(
        clean_df[["player_name", "club", "league", "season"]].reset_index().rename(columns={"index": "_row_id"}),
        on="_row_id",
        how="left",
        suffixes=("_norm_key", ""),
    )
    unmatched = audit[audit["has_performance_stats"] == 0].copy()

    drop_columns = ["_row_id", "player_norm", "club_norm", "league_norm"]
    result = merged.drop(columns=[column for column in drop_columns if column in merged.columns])
    result = result.loc[:, ~result.columns.duplicated()].copy()

    output_file = Path(output_file)
    if write_outputs:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        MATCHING_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_file, index=False)
        audit.to_csv(MATCHING_AUDIT_FILE, index=False)
        unmatched.to_csv(UNMATCHED_FILE, index=False)

    report = {
        "rows": len(result),
        "matched_rows": int(result["has_performance_stats"].sum()),
        "unmatched_rows": int((result["has_performance_stats"] == 0).sum()),
        "match_method_counts": audit["match_method"].value_counts().to_dict(),
        "base_performance_features": base_features,
        "performance_columns": performance_columns,
        "output_file": str(output_file),
        "matching_audit_file": str(MATCHING_AUDIT_FILE),
        "unmatched_file": str(UNMATCHED_FILE),
    }
    return result, report


def parse_args():
    parser = ArgumentParser(description="Merge cached FBref performance stats into model dataset.")
    parser.add_argument("--model-file", default=str(MODEL_INPUT_FILE))
    parser.add_argument("--clean-file", default=str(CLEAN_INPUT_FILE))
    parser.add_argument("--fbref-file", default=str(FBREF_INPUT_FILE))
    parser.add_argument("--output", default=str(MODEL_OUTPUT_FILE))
    return parser.parse_args()


def main():
    args = parse_args()
    _, report = merge_performance(
        model_file=Path(args.model_file),
        clean_file=Path(args.clean_file),
        fbref_file=Path(args.fbref_file),
        output_file=Path(args.output),
    )
    print("Merge performa selesai.")
    print(f"Rows: {report['rows']}")
    print(f"Matched rows: {report['matched_rows']}")
    print(f"Unmatched rows: {report['unmatched_rows']}")
    print(f"Performance columns: {report['performance_columns']}")
    print(f"Output: {report['output_file']}")
    return report


if __name__ == "__main__":
    main()
