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
    elo: float  # decayed toward 1500 for time since the player's last match on this surface
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


class MatchBreakdown(BaseModel):
    """Scoreline detail from the point-level engines, driven by each player's serve/return form.

    Its match win probability comes from a different model than the headline
    LightGBM number, so the two won't match exactly.
    """

    player_1_serve_point_prob: float
    player_2_serve_point_prob: float
    player_1_hold_prob: float
    player_2_hold_prob: float
    player_1_set_win_prob: float
    player_1_match_win_prob: float
    straight_sets_prob: float
    went_the_distance_prob: float
    set_score_probs: dict[str, float]


class HeadToHead(BaseModel):
    """Past results between the two players on the requested surface."""

    player_1_wins: int
    player_2_wins: int


class Prediction(BaseModel):
    player_1: PlayerSummary
    player_2: PlayerSummary
    surface: str
    best_of: int
    as_of: str
    player_1_win_prob: float
    elo_win_prob: float
    player_1_elo: float
    player_2_elo: float
    head_to_head: HeadToHead
    breakdown: MatchBreakdown | None
