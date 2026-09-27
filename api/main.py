"""FastAPI app serving player lookups and match win-probability predictions.

Run from the project root:
    ./.venv/bin/python -m uvicorn api.main:app --reload
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

from fastapi import Depends, FastAPI, HTTPException, Query

from api.schemas import PlayerDetail, PlayerSummary, SurfaceStats
from src.db import get_connection
from src.features.elo import UNKNOWN_SURFACE

app = FastAPI(title="ATP Match Win Probability API")


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

    return PlayerDetail(
        **dict(player),
        surfaces=[SurfaceStats(**dict(row)) for row in surface_rows],
    )
