from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FBREF_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "fbref"
FBREF_INTERIM_FILE = PROJECT_ROOT / "data" / "interim" / "fbref_player_stats.csv"

DEFAULT_LEAGUE = "Big 5 European Leagues Combined"
DEFAULT_STAT_TYPES = ["standard", "shooting", "misc", "keeper"]
DEFAULT_VALIDATION_SEASONS = ["2023-2024"]
ALL_FBREF_SEASONS = [
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
]


def make_unique_columns(columns):
    seen = {}
    unique_columns = []
    for column in columns:
        base = str(column)
        count = seen.get(base, 0)
        unique_columns.append(base if count == 0 else f"{base}_{count}")
        seen[base] = count + 1
    return unique_columns


def flatten_columns(df, stat_type):
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
        df = df.reset_index()

    flattened = []
    for column in df.columns:
        if isinstance(column, tuple):
            parts = [str(part).strip().lower() for part in column if str(part).strip()]
            name = "_".join(parts)
        else:
            name = str(column).strip().lower()
        name = (
            name.replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace("+", "_")
            .replace("%", "pct")
        )
        while "__" in name:
            name = name.replace("__", "_")
        flattened.append(name.strip("_"))

    df.columns = make_unique_columns(flattened)
    metadata_columns = {
        "league",
        "season",
        "team",
        "squad",
        "player",
        "nation",
        "pos",
        "age",
        "born",
    }
    rename_map = {
        column: f"{stat_type}_{column}"
        for column in df.columns
        if column not in metadata_columns and not column.startswith(f"{stat_type}_")
    }
    df = df.rename(columns=rename_map)
    df.columns = make_unique_columns(df.columns)
    return df


def cache_path(season, stat_type):
    safe_season = str(season).replace("/", "-")
    return FBREF_CACHE_DIR / f"fbref_{safe_season}_{stat_type}.csv"


def read_or_fetch_stat(season, stat_type, league=DEFAULT_LEAGUE):
    FBREF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(season, stat_type)
    if path.exists():
        return pd.read_csv(path)

    try:
        import soccerdata as sd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Package soccerdata belum terinstall. Jalankan: pip install soccerdata"
        ) from exc

    fbref = sd.FBref(
        leagues=league,
        seasons=season,
        no_cache=False,
        no_store=False,
        data_dir=FBREF_CACHE_DIR,
    )
    df = fbref.read_player_season_stats(stat_type=stat_type)
    df = flatten_columns(df, stat_type)
    df["fbref_season_label"] = season
    df["season"] = int(str(season)[:4])
    df["fbref_stat_type"] = stat_type
    df.to_csv(path, index=False)
    return df


def merge_stat_frames(frames):
    if not frames:
        return pd.DataFrame()

    key_candidates = ["season", "league", "squad", "team", "player", "nation", "pos", "age", "born"]
    merged = None
    for frame in frames:
        frame = frame.copy()
        frame.columns = make_unique_columns(frame.columns)
        keys = [column for column in key_candidates if column in frame.columns]
        if merged is None:
            merged = frame
            continue
        merge_keys = [column for column in keys if column in merged.columns]
        if not merge_keys:
            merged = pd.concat([merged, frame], axis=1)
            merged.columns = make_unique_columns(merged.columns)
            continue
        overlap = [
            column
            for column in frame.columns
            if column in merged.columns and column not in merge_keys
        ]
        frame = frame.drop(columns=overlap)
        merged = merged.merge(frame, on=merge_keys, how="outer")
        merged.columns = make_unique_columns(merged.columns)
    return merged


def scrape_fbref(seasons=None, stat_types=None, league=DEFAULT_LEAGUE, output_file=FBREF_INTERIM_FILE):
    seasons = seasons or DEFAULT_VALIDATION_SEASONS
    stat_types = stat_types or DEFAULT_STAT_TYPES

    season_frames = []
    for season in seasons:
        frames = [read_or_fetch_stat(season, stat_type, league=league) for stat_type in stat_types]
        season_df = merge_stat_frames(frames)
        season_df["season"] = int(str(season)[:4])
        season_df["fbref_season_label"] = season
        season_frames.append(season_df)

    result = pd.concat(season_frames, ignore_index=True, sort=False)
    result.columns = make_unique_columns(result.columns)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    return result


def parse_args():
    parser = ArgumentParser(description="Scrape FBref player stats through soccerdata.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=DEFAULT_VALIDATION_SEASONS,
        help="FBref seasons such as 2023-2024. Use --all-seasons only when ready.",
    )
    parser.add_argument("--all-seasons", action="store_true")
    parser.add_argument("--stat-types", nargs="+", default=DEFAULT_STAT_TYPES)
    parser.add_argument("--league", default=DEFAULT_LEAGUE)
    parser.add_argument("--output", default=str(FBREF_INTERIM_FILE))
    return parser.parse_args()


def main():
    args = parse_args()
    seasons = ALL_FBREF_SEASONS if args.all_seasons else args.seasons
    df = scrape_fbref(
        seasons=seasons,
        stat_types=args.stat_types,
        league=args.league,
        output_file=Path(args.output),
    )
    print("FBref scraping selesai.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Output: {Path(args.output)}")
    return df


if __name__ == "__main__":
    main()
