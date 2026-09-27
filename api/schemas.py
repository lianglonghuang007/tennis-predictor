"""Response shapes for the API. FastAPI validates every response against these and uses them to
generate the interactive docs at /docs.
"""

from __future__ import annotations

from pydantic import BaseModel


class PlayerSummary(BaseModel):
    """One search result: enough to show in a picker and tell same-named players apart."""

    player_id: str
    name: str
    ioc: str | None
    rank: int | None
    last_match_date: str


class SurfaceStats(BaseModel):
    """A player's current rating and recent form on one surface. Form fields are None when the
    player has no matches with recorded serve stats on that surface."""

    surface: str
    elo: float
    serve_win_pct_last10: float | None
    serve_win_pct_last20: float | None
    serve_win_pct_last50: float | None
    return_win_pct_last10: float | None
    return_win_pct_last20: float | None
    return_win_pct_last50: float | None
    bp_conversion_last10: float | None
    bp_conversion_last20: float | None
    bp_conversion_last50: float | None
    bp_saved_pct_last10: float | None
    bp_saved_pct_last20: float | None
    bp_saved_pct_last50: float | None


class PlayerDetail(PlayerSummary):
    """Full profile for one player, including per-surface stats."""

    hand: str | None
    ht: float | None
    age: float | None
    rank_points: float | None
    surfaces: list[SurfaceStats]
