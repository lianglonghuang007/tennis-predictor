"""FastAPI app serving player lookups and match win-probability predictions.

Run from the project root:
    ./.venv/bin/python -m uvicorn api.main:app --reload
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import lightgbm as lgb
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query

from api.schemas import HeadToHead, MatchBreakdown, PlayerDetail, PlayerSummary, Prediction, SurfaceStats
from src.db import get_connection
from src.features.elo import UNKNOWN_SURFACE
from src.prediction import (
    build_feature_frame,
    current_elo,
    elo_win_prob,
    match_breakdown,
    predict_win_prob,
    serve_point_prob,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "lightgbm_model.joblib"

app = FastAPI(title="ATP Match Win Probability API")


@lru_cache(maxsize=1)
def get_model() -> lgb.LGBMClassifier:
    """Load the trained model once, on first use, and reuse it for every later request."""
    return joblib.load(MODEL_PATH)


def get_db() -> Iterator[sqlite3.Connection]:
    """One read-only connection per request, closed once the response is sent."""
    conn = get_connection(read_only=True)
    try:
        yield conn
    finally:
        conn.close()


@app.get("/players", response_model=list[PlayerSummary])
def search_players(
    q: str = Query(min_length=2, description="Part of a player's name, case-insensitive"),
    limit: int = Query(default=10, ge=1, le=50),
    db: sqlite3.Connection = Depends(get_db),
) -> list[PlayerSummary]:
    """Players whose name contains `q`, best-ranked first; unranked/retired players last."""
    rows = db.execute(
        """
        SELECT player_id, name, ioc, rank, last_match_date
        FROM players
        WHERE name LIKE ?
        ORDER BY rank IS NULL, rank, name
        LIMIT ?
        """,
        (f"%{q}%", limit),
    ).fetchall()
    return [PlayerSummary(**dict(row)) for row in rows]


@app.get("/players/{player_id}", response_model=PlayerDetail)
def get_player(player_id: str, db: sqlite3.Connection = Depends(get_db)) -> PlayerDetail:
    """One player's profile plus current Elo and form on each surface they've played."""
    player = db.execute("SELECT * FROM players WHERE player_id = ?", (player_id,)).fetchone()
    if player is None:
        raise HTTPException(status_code=404, detail=f"No player with id {player_id!r}")

    # The "Unknown" bucket holds matches with no recorded surface: kept for the
    # model's features, but not a surface anyone can pick.
    surface_rows = db.execute(
        "SELECT * FROM player_surface_stats WHERE player_id = ? AND surface != ? ORDER BY surface",
        (player_id, UNKNOWN_SURFACE),
    ).fetchall()

    as_of = _data_as_of(db)
    surfaces = [
        SurfaceStats(**{**dict(row), "elo": current_elo(dict(row), as_of)}) for row in surface_rows
    ]
    return PlayerDetail(**dict(player), surfaces=surfaces)


def _data_as_of(db: sqlite3.Connection) -> pd.Timestamp:
    """The most recent match date in the data, which predictions and ratings are computed as of.

    Using today's date instead would make every player look inactive (and
    decay their Elo) for however long it's been since the data was refreshed.
    """
    return pd.Timestamp(db.execute("SELECT MAX(last_match_date) FROM players").fetchone()[0])


def _fetch_player(db: sqlite3.Connection, player_id: str) -> dict:
    row = db.execute("SELECT * FROM players WHERE player_id = ?", (player_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No player with id {player_id!r}")
    return dict(row)


def _fetch_surface_stats(db: sqlite3.Connection, player_id: str, surface: str) -> dict | None:
    row = db.execute(
        "SELECT * FROM player_surface_stats WHERE player_id = ? AND surface = ?", (player_id, surface)
    ).fetchone()
    return dict(row) if row is not None else None


def _h2h_wins(db: sqlite3.Connection, winner_id: str, loser_id: str, surface: str) -> int:
    return db.execute(
        "SELECT COUNT(*) FROM matches WHERE winner_id = ? AND loser_id = ? AND surface = ?",
        (winner_id, loser_id, surface),
    ).fetchone()[0]


@app.get("/predict", response_model=Prediction)
def predict(
    player1_id: str,
    player2_id: str,
    surface: Literal["Hard", "Clay", "Grass"],
    # A plain int with an explicit check: query values arrive as strings, and
    # Literal[3, 5] only matches the int 3/5, so "5" from a URL would be rejected.
    best_of: int = Query(default=3, description="3 or 5"),
    db: sqlite3.Connection = Depends(get_db),
    model: lgb.LGBMClassifier = Depends(get_model),
) -> Prediction:
    """Win probability for player 1 against player 2, plus Elo, head-to-head, and a scoreline breakdown."""
    if best_of not in (3, 5):
        raise HTTPException(status_code=422, detail="best_of must be 3 or 5")
    if player1_id == player2_id:
        raise HTTPException(status_code=400, detail="player1_id and player2_id must be different players")

    p1 = _fetch_player(db, player1_id)
    p2 = _fetch_player(db, player2_id)
    s1 = _fetch_surface_stats(db, player1_id, surface)
    s2 = _fetch_surface_stats(db, player2_id, surface)
    h2h_1 = _h2h_wins(db, player1_id, player2_id, surface)
    h2h_2 = _h2h_wins(db, player2_id, player1_id, surface)

    as_of = _data_as_of(db)

    features = build_feature_frame(p1, p2, s1, s2, h2h_1, h2h_2, surface, best_of, as_of)
    elo_1 = current_elo(s1, as_of)
    elo_2 = current_elo(s2, as_of)

    serve_1 = serve_point_prob(s1, s2)
    serve_2 = serve_point_prob(s2, s1)
    breakdown = (
        MatchBreakdown(**match_breakdown(serve_1, serve_2, best_of))
        if serve_1 is not None and serve_2 is not None
        else None
    )

    return Prediction(
        player_1=PlayerSummary(**p1),
        player_2=PlayerSummary(**p2),
        surface=surface,
        best_of=best_of,
        as_of=as_of.strftime("%Y-%m-%d"),
        player_1_win_prob=predict_win_prob(model, features),
        elo_win_prob=elo_win_prob(elo_1, elo_2),
        player_1_elo=elo_1,
        player_2_elo=elo_2,
        head_to_head=HeadToHead(player_1_wins=h2h_1, player_2_wins=h2h_2),
        breakdown=breakdown,
    )
