# This line is a docstring: a short description of what this whole file (module) does.
"""Load and clean ATP match data (YYYY.csv season files) into one dataframe."""

# This import makes type hints like "list[Path] | None" work even on older Python versions.
from __future__ import annotations

# The logging module lets us print organized status messages instead of plain print() calls.
import logging
# Path is a modern, object-oriented way to work with file/folder paths.
from pathlib import Path

# numpy is imported for numerical operations (not used heavily yet, but will be for Elo math).
import numpy as np
# pandas is the core library for loading and manipulating tabular data (dataframes).
import pandas as pd

# Configure the logging system once: show INFO-level messages, formatted as plain text.
logging.basicConfig(level=logging.INFO, format="%(message)s")
# Create a logger object specific to this file, so log messages can be traced to their source.
logger = logging.getLogger(__name__)

# Comment block explaining why this constant exists before we define it.
# Columns we expect a raw match file to have. Used to fail loudly if the
# source schema changes instead of silently loading a misaligned dataframe.
# Note: this dataset (not the raw Sackmann tennis_atp repo directly, but a
# same-schema derivative) uses alphanumeric winner_id/loser_id (e.g. "GH92")
# rather than Sackmann's numeric player IDs, and adds an "indoor" column.
# EXPECTED_COLUMNS is a list of every column name our source CSVs must contain.
EXPECTED_COLUMNS = [
    # Tournament-level identifiers and metadata.
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "indoor", "tourney_date", "match_num",
    # Everything describing the winner of the match.
    "winner_id", "winner_seed", "winner_entry", "winner_name", "winner_hand",
    "winner_ht", "winner_ioc", "winner_age",
    # Everything describing the loser of the match.
    "loser_id", "loser_seed", "loser_entry", "loser_name", "loser_hand",
    "loser_ht", "loser_ioc", "loser_age",
    # The match result itself: final score, format, round reached, duration.
    "score", "best_of", "round", "minutes",
    # Winner's serve/return statistics for this match (aces, double faults, etc).
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms",
    "w_bpSaved", "w_bpFaced",
    # Loser's serve/return statistics for this match, mirroring the winner's.
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms",
    "l_bpSaved", "l_bpFaced",
    # World ranking and ranking points for both players at match time.
    "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
]

# Comment explaining why STAT_COLUMNS is split out as its own list.
# Serve/return stat columns are only populated from ~1991 onward and are
# frequently missing even within our 2010+ window (mostly Davis Cup ties
# and a handful of early-round matches at smaller events).
# STAT_COLUMNS holds just the point-by-point statistic columns (a subset of EXPECTED_COLUMNS).
STAT_COLUMNS = [
    # Winner's per-match serve/return counts.
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms",
    "w_bpSaved", "w_bpFaced",
    # Loser's per-match serve/return counts.
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms",
    "l_bpSaved", "l_bpFaced",
]


# def starts a function definition. This function is named find_match_files.
# It takes three inputs (parameters): where to look, and an optional year range.
# "-> list[Path]" tells readers/tools this function will hand back a list of file paths.
def find_match_files(
    raw_dir: Path, start_year: int = 2010, end_year: int | None = None
) -> list[Path]:
    # A docstring: describes what this function does, for anyone (including future you) reading it.
    """Return sorted main-tour YYYY.csv paths in raw_dir within [start_year, end_year].

    The glob "[0-9]" x4 + ".csv" matches "2023.csv" but NOT "2023_challenger.csv",
    "ATP_Database.csv", or "ongoing_tourneys.csv" — those other files are also
    in data/raw but aren't ATP main-tour season files, so we deliberately
    exclude them here rather than filtering them out later.

    end_year defaults to None, meaning "no upper bound" — this lets the
    pipeline automatically pick up new seasons dropped into data/raw
    without code changes.
    """
    # raw_dir.glob(...) searches the folder for filenames matching a pattern and returns them.
    # "[0-9][0-9][0-9][0-9].csv" means "exactly 4 digits, then .csv" — i.e. "2023.csv", not "2023_challenger.csv".
    # sorted(...) puts the results in order (alphabetical, which for 4-digit years means chronological).
    files = sorted(raw_dir.glob("[0-9][0-9][0-9][0-9].csv"))
    # Start an empty list that we'll fill with only the files inside our year range.
    selected = []
    # Loop over every file we found, one at a time, calling the current one "f".
    for f in files:
        # f.stem is the filename without its extension, e.g. "2023" from "2023.csv".
        # int(...) converts that text "2023" into the actual number 2023 so we can compare it.
        year = int(f.stem)
        # If this file's year is earlier than the start_year we asked for, skip it.
        if year < start_year:
            # "continue" jumps straight to the next loop iteration, skipping the rest of this one.
            continue
        # If an end_year was given AND this file's year is past it, skip it too.
        if end_year is not None and year > end_year:
            continue
        # This file passed both checks, so add it to our results list.
        selected.append(f)
    # After checking every file, if we found nothing at all in range...
    if not selected:
        # ...raise an error immediately instead of silently returning an empty list.
        # An f-string (the "f" before the quotes) lets us insert variables directly into the message.
        raise FileNotFoundError(
            f"No YYYY.csv main-tour files found in {raw_dir} "
            f"for years >= {start_year}. Did you copy the season CSVs "
            f"into data/raw?"
        )
    # Hand back the final filtered, sorted list of file paths to whoever called this function.
    return selected


# This function takes the same location/year inputs as find_match_files,
# but returns a pandas DataFrame (a table) instead of a list of file paths.
def load_matches(
    raw_dir: Path, start_year: int = 2010, end_year: int | None = None
) -> pd.DataFrame:
    # Docstring explaining the function's purpose and a design choice.
    """Read every matching atp_matches_YYYY.csv and concatenate into one DataFrame.

    Each file is one ATP season. We read them individually (rather than one
    big glob-read) so we can tag each row with its source file and catch a
    malformed year file without losing the traceback for which file broke.
    """
    # Reuse the function we just wrote to get the filtered, sorted list of files.
    files = find_match_files(raw_dir, start_year, end_year)
    # Start an empty list — we'll put one small table (one per file) into it.
    frames = []
    # Loop through each file path we got back.
    for f in files:
        # pd.read_csv reads one CSV file into a DataFrame (pandas' table object).
        # low_memory=False tells pandas to scan the whole file before guessing column types,
        # which avoids a warning/bug where a column looks like numbers in some rows and text in others.
        df = pd.read_csv(f, low_memory=False)
        # set(...) converts a list into a "set" — a collection with no duplicates, good for comparisons.
        # This line asks: "which columns does EXPECTED_COLUMNS have that this file's df does NOT have?"
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        # If that difference is non-empty, something is wrong with this file's structure.
        if missing:
            # Stop immediately with a clear error naming the file and exactly which columns are missing.
            raise ValueError(f"{f.name} is missing expected columns: {sorted(missing)}")
        # Add a new column to this table recording which file each row came from — useful for debugging later.
        df["source_file"] = f.name
        # Add this file's table to our running list of tables.
        frames.append(df)
        # Print a status line so we can watch progress while the script runs.
        # The ":," inside the f-string formats the number with comma thousands separators (e.g. 3,030).
        logger.info(f"  loaded {f.name}: {len(df):,} rows")

    # pd.concat stacks all the individual season tables into one big table, one on top of another.
    # ignore_index=True renumbers the rows 0, 1, 2, ... instead of keeping each file's original row numbers.
    combined = pd.concat(frames, ignore_index=True)
    # Log a summary of how many files and rows were combined.
    logger.info(f"Combined {len(files)} files into {len(combined):,} total rows")
    # Return the single combined table to whoever called this function.
    return combined


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Apply light, non-destructive cleaning to a combined matches DataFrame.

    Deliberately minimal at this stage: parse dates to real datetimes,
    coerce numeric columns that pandas sometimes reads as object dtype
    (because a handful of rows have missing values mixed with strings),
    drop exact duplicate rows, and sort chronologically. Anything more
    opinionated (e.g. dropping retirements) is left for the feature
    engineering step, since different models may want to handle them
    differently.
    """
    df = df.copy()

    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")

    # winner_id/loser_id are NOT included here: in this dataset they're
    # alphanumeric player codes (e.g. "GH92"), not Sackmann's numeric IDs.
    numeric_cols = [
        "draw_size", "match_num", "winner_ht", "winner_age",
        "loser_ht", "loser_age", "best_of", "minutes",
        "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
    ] + STAT_COLUMNS
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.drop_duplicates()
    n_dropped = before - len(df)
    if n_dropped:
        logger.info(f"Dropped {n_dropped} exact duplicate rows")

    df = df.sort_values("tourney_date", kind="stable").reset_index(drop=True)
    return df


def summarize_data(df: pd.DataFrame) -> dict:
    """Print and return a summary of the combined dataset.

    Returns a dict (rather than just printing) so downstream code — or a
    test — can assert on the numbers instead of scraping stdout.
    """
    n_players = pd.unique(pd.concat([df["winner_id"], df["loser_id"]])).shape[0]
    matches_by_surface = df["surface"].value_counts(dropna=False)
    missing_counts = df.isna().sum().sort_values(ascending=False)

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Date range: {df['tourney_date'].min().date()} to {df['tourney_date'].max().date()}")
    print(f"Unique players (winner or loser): {n_players:,}")
    print()
    print("Columns:")
    print(list(df.columns))
    print()
    print("Match counts by surface:")
    print(matches_by_surface.to_string())
    print()
    print("Missing value counts (top 20, non-zero only):")
    print(missing_counts[missing_counts > 0].head(20).to_string())
    print("=" * 60)

    return {
        "shape": df.shape,
        "date_range": (df["tourney_date"].min(), df["tourney_date"].max()),
        "n_players": n_players,
        "matches_by_surface": matches_by_surface,
        "missing_counts": missing_counts,
    }


def flag_data_quality_issues(df: pd.DataFrame) -> dict:
    """Check for known ATP data quality issues and print/return findings.

    These are issues that will corrupt Elo ratings or leak information into
    features if not handled explicitly downstream (not fixed here — this
    function only detects and reports).
    """
    issues = {}

    # Retirements / walkovers / defaults inflate a "winner's" rating if
    # treated as a normal win, since no full match was actually played.
    score_str = df["score"].astype(str)
    issues["retirements"] = int(score_str.str.contains("RET", na=False).sum())
    issues["walkovers"] = int(score_str.str.contains("W/O", na=False).sum())
    issues["defaults"] = int(score_str.str.contains("DEF", na=False).sum())

    # Missing surface can't be Elo-rated per-surface at all.
    issues["missing_surface"] = int(df["surface"].isna().sum())
    issues["unknown_surface_label"] = df.loc[
        df["surface"].notna() & ~df["surface"].isin(["Hard", "Clay", "Grass", "Carpet"]),
        "surface",
    ].unique().tolist()

    # Missing serve/return stats mean rows can't be used for point-level
    # (pA/pB) model features even though they're fine for Elo/Bradley-Terry.
    stat_missing = df[STAT_COLUMNS].isna().all(axis=1)
    issues["matches_missing_all_serve_stats"] = int(stat_missing.sum())
    issues["pct_missing_all_serve_stats"] = round(100 * stat_missing.mean(), 1)

    # Missing rank breaks ranking-delta features and can't be silently
    # imputed with 0 (0 would look like the #1 player).
    issues["missing_winner_rank"] = int(df["winner_rank"].isna().sum())
    issues["missing_loser_rank"] = int(df["loser_rank"].isna().sum())

    # Duplicate match keys would double-count a match in Elo updates. We
    # check this only among rows that HAVE a match_num: pandas treats
    # multiple NaNs as "duplicates" of each other, which would otherwise
    # make every match_num-less row in a tournament falsely count as a
    # duplicate of every other one. Missing match_num is tracked separately
    # below — as of 2026-07-21 it affects 10 recent (2025-2026) tournaments
    # whose match_num hasn't been backfilled yet by the data source.
    issues["missing_match_num"] = int(df["match_num"].isna().sum())
    keyed = df.dropna(subset=["match_num"])
    dup_key = keyed.duplicated(subset=["tourney_id", "match_num"], keep=False)
    issues["duplicate_tourney_match_num"] = int(dup_key.sum())

    # Sanity bounds — values outside these ranges are almost certainly data
    # entry errors, not real match outcomes. NaN is excluded explicitly
    # (age is often missing for lower-tier Davis Cup players) so it isn't
    # double-counted as both "missing" and "implausible".
    issues["implausible_age"] = int(
        (df["winner_age"].notna() & ~df["winner_age"].between(12, 45)).sum()
        + (df["loser_age"].notna() & ~df["loser_age"].between(12, 45)).sum()
    )
    # Retirements are excluded: a 5-minute RET is a legitimate short match,
    # not a data error. The upper bound is 700, not 500 — John Isner vs.
    # Nicolas Mahut (Wimbledon 2010, the longest match in tennis history,
    # 11:05 / 665 minutes) is a real outlier that a tighter bound would
    # incorrectly flag.
    score_str_local = df["score"].astype(str)
    is_incomplete = (
        score_str_local.str.contains("RET", na=False)
        | score_str_local.str.contains("W/O", na=False)
        | score_str_local.str.contains("DEF", na=False)
    )
    implausible_minutes_mask = (
        df["minutes"].notna() & ~is_incomplete & ~df["minutes"].between(10, 700)
    )
    issues["implausible_minutes"] = int(implausible_minutes_mask.sum())

    print("=" * 60)
    print("DATA QUALITY FLAGS")
    print("=" * 60)
    for key, value in issues.items():
        print(f"{key}: {value}")
    print("=" * 60)

    return issues


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Attach boolean quality-flag columns instead of dropping/imputing rows.

    We deliberately don't resolve these issues here: whether a retirement
    should be excluded is an Elo-vs-LightGBM decision, not a data-cleaning
    one, so each downstream consumer filters on these flags itself instead
    of the flags being baked silently into which rows even exist.
    """
    df = df.copy()
    score_str = df["score"].astype(str)

    df["is_retirement"] = score_str.str.contains("RET", na=False)
    df["is_walkover"] = score_str.str.contains("W/O", na=False)
    df["is_default"] = score_str.str.contains("DEF", na=False)
    df["is_incomplete_match"] = df["is_retirement"] | df["is_walkover"] | df["is_default"]

    df["has_serve_stats"] = ~df[STAT_COLUMNS].isna().all(axis=1)
    df["has_match_num"] = df["match_num"].notna()

    return df


def run_pipeline(
    raw_dir: Path, start_year: int = 2010, end_year: int | None = None
) -> pd.DataFrame:
    """End-to-end: load, clean, flag quality issues, summarize, return the DataFrame."""
    logger.info(f"Loading matches from {raw_dir} (years {start_year}-{end_year or 'present'})")
    df = load_matches(raw_dir, start_year, end_year)
    df = clean_matches(df)
    df = add_quality_flags(df)
    summarize_data(df)
    flag_data_quality_issues(df)
    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    matches = run_pipeline(raw_dir, start_year=2010)

    out_path = processed_dir / "atp_matches_combined.csv"
    matches.to_csv(out_path, index=False)
    logger.info(f"Saved cleaned dataset to {out_path}")
