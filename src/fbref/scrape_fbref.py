import argparse
from functools import reduce
from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FBREF_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "fbref"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FBREF_OUTPUT_FILE = INTERIM_DIR / "fbref_player_stats.csv"

DEFAULT_LEAGUE = "Big 5 European Leagues Combined"
DEFAULT_SEASONS = ["2023-2024"]
ALL_SEASONS = [
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
]
STAT_TYPES = ["standard", "shooting", "misc", "keeper"]


def normalize_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def flatten_columns(df):
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name is not None:
        df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        flat_columns = []
        for column in df.columns:
            pieces = [str(part) for part in column if str(part) not in {"", "nan", "None"}]
            flat_columns.append("_".join(pieces))
        df.columns = flat_columns
    else:
        df.columns = [str(column) for column in df.columns]

    df.columns = [normalize_text(column) for column in df.columns]
    return df


def first_existing(columns, candidates):
    normalized = set(columns)
    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        if candidate_norm in normalized:
            return candidate_norm
    for column in columns:
        for candidate in candidates:
            if column.endswith(normalize_text(candidate)):
                return column
    return None


def add_key_columns(df, season, stat_type):
    df = df.copy()
    player_col = first_existing(df.columns, ["player", "players_player", "standard_player"])
    squad_col = first_existing(df.columns, ["squad", "team", "standard_squad"])
    league_col = first_existing(df.columns, ["comp", "league", "competition"])

    if player_col is None:
        raise ValueError(f"Player column not found for FBref stat type: {stat_type}")
    if squad_col is None:
        raise ValueError(f"Squad column not found for FBref stat type: {stat_type}")

    df["fbref_player_name"] = df[player_col].astype(str)
    df["fbref_club"] = df[squad_col].astype(str)
    df["fbref_league"] = df[league_col].astype(str) if league_col else DEFAULT_LEAGUE
    df["season_label"] = season
    df["season"] = int(str(season).split("-")[0])
    df["player_name_key"] = df["fbref_player_name"].map(normalize_text)
    df["club_key"] = df["fbref_club"].map(normalize_text)
    df["stat_type"] = stat_type
    return df


def rename_stat_columns(df, stat_type):
    key_columns = {
        "fbref_player_name",
        "fbref_club",
        "fbref_league",
        "season_label",
        "season",
        "player_name_key",
        "club_key",
        "stat_type",
    }
    renamed = {}
    for column in df.columns:
        if column in key_columns:
            continue
        renamed[column] = f"{stat_type}_{column}"
    return df.rename(columns=renamed)


def read_or_fetch_stat(season, stat_type, league=DEFAULT_LEAGUE, force_fetch=False):
    FBREF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = FBREF_CACHE_DIR / f"fbref_{season}_{stat_type}.csv"
    if cache_file.exists() and not force_fetch:
        return pd.read_csv(cache_file)

    import soccerdata as sd

    fbref = sd.FBref(
        leagues=league,
        seasons=season,
        no_cache=False,
        no_store=False,
        data_dir=FBREF_CACHE_DIR,
    )
    df = fbref.read_player_season_stats(stat_type=stat_type)
    df = flatten_columns(df)
    df = add_key_columns(df, season=season, stat_type=stat_type)
    df = rename_stat_columns(df, stat_type=stat_type)
    df.to_csv(cache_file, index=False)
    return df


def merge_stat_frames(frames):
    merge_keys = [
        "fbref_player_name",
        "fbref_club",
        "fbref_league",
        "season_label",
        "season",
        "player_name_key",
        "club_key",
    ]
    cleaned_frames = []
    for frame in frames:
        if "stat_type" in frame.columns:
            frame = frame.drop(columns=["stat_type"])
        duplicate_columns = [column for column in frame.columns if column.endswith("_level_0")]
        frame = frame.drop(columns=duplicate_columns, errors="ignore")
        cleaned_frames.append(frame)
    return reduce(
        lambda left, right: left.merge(right, on=merge_keys, how="outer"),
        cleaned_frames,
    )


def scrape_fbref(seasons, stat_types=None, league=DEFAULT_LEAGUE, force_fetch=False):
    stat_types = stat_types or STAT_TYPES
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    all_season_frames = []
    for season in seasons:
        stat_frames = []
        for stat_type in stat_types:
            print(f"Loading FBref {season} {stat_type}")
            stat_frames.append(
                read_or_fetch_stat(
                    season=season,
                    stat_type=stat_type,
                    league=league,
                    force_fetch=force_fetch,
                )
            )
        all_season_frames.append(merge_stat_frames(stat_frames))

    combined = pd.concat(all_season_frames, ignore_index=True)
    combined.to_csv(FBREF_OUTPUT_FILE, index=False)
    print(f"Saved FBref player stats: {FBREF_OUTPUT_FILE}")
    print(f"Rows: {len(combined)}")
    return combined


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape FBref player season stats via soccerdata.")
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--all-seasons", action="store_true")
    parser.add_argument("--stat-types", nargs="+", default=STAT_TYPES, choices=STAT_TYPES)
    parser.add_argument("--league", default=DEFAULT_LEAGUE)
    parser.add_argument("--force-fetch", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seasons = ALL_SEASONS if args.all_seasons else args.seasons
    scrape_fbref(
        seasons=seasons,
        stat_types=args.stat_types,
        league=args.league,
        force_fetch=args.force_fetch,
    )


if __name__ == "__main__":
    main()
