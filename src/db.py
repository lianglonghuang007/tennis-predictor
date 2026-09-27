"""SQLite storage for serving live predictions.

The API reads from this database instead of the raw CSVs: a small, indexed file that answers
"give me this player's current stats" or "what's this pair's head-to-head" in milliseconds,
without re-running the feature pipeline on every request.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.features.elo import UNKNOWN_SURFACE
from src.features.rolling_stats import ROLLING_STAT_NAMES
from src.features.snapshot import ROLLING_WINDOWS, build_player_snapshots

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "tennis.db"

# Derived from the same constants the snapshot builder uses, so the table's
# columns can't drift out of sync with the DataFrame written into it.
ROLLING_COLUMNS = [f"{stat}_last{w}" for stat in ROLLING_STAT_NAMES for w in ROLLING_WINDOWS]


def get_connection(db_path: Path = DB_PATH, read_only: bool = False) -> sqlite3.Connection:
    """Open a connection with foreign-key enforcement on and rows readable by column name.

    read_only=True is for the API: it fails loudly if the file doesn't exist,
    instead of sqlite3's default of silently creating a new empty database.
    """
    if read_only:
        # FastAPI may open the connection and run the endpoint on different
        # worker threads; each connection still serves only one request at a
        # time, so disabling sqlite3's same-thread check is safe here.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(db_path)
    # SQLite ships with foreign-key checks OFF for backwards compatibility,
    # and the setting is per-connection, so it has to be turned on every time.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """Drop and recreate all tables. The database is a rebuildable cache of the CSVs, not a source of truth."""
    rolling_column_defs = ",\n            ".join(f"{col} REAL" for col in ROLLING_COLUMNS)
    conn.executescript(f"""
        DROP TABLE IF EXISTS matches;
        DROP TABLE IF EXISTS player_surface_stats;
        DROP TABLE IF EXISTS players;

        CREATE TABLE players (
            player_id       TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            hand            TEXT,
            ht              REAL,
            ioc             TEXT,
            age             REAL,
            rank            INTEGER,
            rank_points     REAL,
            last_match_date TEXT NOT NULL
        );

        CREATE TABLE player_surface_stats (
            player_id TEXT NOT NULL REFERENCES players(player_id),
            surface   TEXT NOT NULL,
            elo       REAL NOT NULL,
            {rolling_column_defs},
            PRIMARY KEY (player_id, surface)
        );

        CREATE TABLE matches (
            match_id     INTEGER PRIMARY KEY,
            tourney_date TEXT NOT NULL,
            surface      TEXT,
            winner_id    TEXT NOT NULL REFERENCES players(player_id),
            loser_id     TEXT NOT NULL REFERENCES players(player_id)
        );

        -- Head-to-head lookups filter on the (winner_id, loser_id) pair; without
        -- an index every lookup scans all ~47K rows.
        CREATE INDEX idx_matches_pair ON matches (winner_id, loser_id);
    """)
    conn.commit()


def load_database(conn: sqlite3.Connection, df: pd.DataFrame) -> dict[str, int]:
    """Fill the (already created) tables from the cleaned match DataFrame. Returns row counts per table.

    Rows with a missing player_id are skipped: the source data has a handful
    of matches with a blank winner/loser id, and a row with no id can't be
    looked up or referenced by anything.
    """
    players, player_surface_stats = build_player_snapshots(df)
    players = players.dropna(subset=["player_id"])
    player_surface_stats = player_surface_stats.dropna(subset=["player_id"])
    # SQLite has no date type; ISO-format text sorts and compares correctly.
    players["last_match_date"] = players["last_match_date"].dt.strftime("%Y-%m-%d")

    matches = df[["tourney_date", "surface", "winner_id", "loser_id"]].dropna(
        subset=["winner_id", "loser_id"]
    )
    matches["tourney_date"] = matches["tourney_date"].dt.strftime("%Y-%m-%d")
    # Same bucketing as the training-time H2H feature, so a live H2H lookup
    # counts exactly the matches the model saw counted during training.
    matches["surface"] = matches["surface"].fillna(UNKNOWN_SURFACE)

    tables = {
        "players": players,
        "player_surface_stats": player_surface_stats,
        "matches": matches,
    }
    # Order matters: players must exist before rows that reference them.
    for table, frame in tables.items():
        frame.to_sql(table, conn, if_exists="append", index=False)
    conn.commit()
    return {table: len(frame) for table, frame in tables.items()}
