"""FastAPI app serving player lookups and match win-probability predictions.

Run from the project root:
    ./.venv/bin/python -m uvicorn api.main:app --reload
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

from fastapi import Depends, FastAPI, Query

from api.schemas import PlayerSummary
from src.db import get_connection

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
