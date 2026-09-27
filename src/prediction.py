"""Turn two players' stored snapshots into a live match prediction.

Rebuilds the exact feature row the LightGBM model was trained on, from
current state rather than a historical match, and pairs the model's headline
probability with the Markov/Monte Carlo scoreline breakdown.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.engine.markov import prob_win_game, prob_win_match, prob_win_set
from src.engine.monte_carlo import simulate_many_matches, summarize_simulations
from src.features.build_features import ROUND_ORDER, add_diff_features
from src.features.elo import decay_rating
from src.features.rolling_stats import ROLLING_STAT_NAMES
from src.features.snapshot import ROLLING_WINDOWS
from src.models.lightgbm_model import select_features

# A hypothetical matchup has no real tournament, so the match-level features
# the model was trained on are filled with the most common values for the
# chosen format in recent seasons: best-of-5 is effectively always a Grand
# Slam (128 draw); best-of-3 uses a Masters event (96 draw), the most common
# top-tier best-of-3 level. Indoor, match number, and seeds are left missing,
# which LightGBM saw often enough in training to handle.
MATCH_CONTEXT_BY_BEST_OF = {
    5: {"tourney_level": "G", "draw_size": 128.0},
    3: {"tourney_level": "M", "draw_size": 96.0},
}
DEFAULT_ROUND = "R32"

PLAYER_ATTRIBUTES = ["hand", "ht", "ioc", "age", "rank", "rank_points"]

N_SIMULATIONS = 5_000


def current_elo(surface_stats: dict | None, as_of: pd.Timestamp) -> float:
    """A player's surface Elo decayed up to `as_of`, matching how training-time pre-match Elo is computed."""
    if surface_stats is None:
        return 1500.0
    if surface_stats["elo_last_date"] is None:
        return surface_stats["elo"]
    return decay_rating(surface_stats["elo"], pd.Timestamp(surface_stats["elo_last_date"]), as_of)


def build_feature_frame(
    p1: dict,
    p2: dict,
    p1_surface: dict | None,
    p2_surface: dict | None,
    p1_h2h_wins: int,
    p2_h2h_wins: int,
    surface: str,
    best_of: int,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """One oriented (player_1/player_2) row with the same columns build_feature_table produces for a real match.

    p1/p2 are rows from the players table, p1_surface/p2_surface rows from
    player_surface_stats (None if the player has never played on `surface`).
    """
    context = MATCH_CONTEXT_BY_BEST_OF[best_of]
    row = {
        "surface": surface,
        "draw_size": context["draw_size"],
        "tourney_level": context["tourney_level"],
        "indoor": np.nan,
        "match_num": np.nan,
        "best_of": float(best_of),
        "round_number": ROUND_ORDER[DEFAULT_ROUND],
    }

    sides = {
        1: (p1, p1_surface, p1_h2h_wins, p2_h2h_wins),
        2: (p2, p2_surface, p2_h2h_wins, p1_h2h_wins),
    }
    for n, (player, surface_stats, wins, losses) in sides.items():
        prefix = f"player_{n}_"
        row[prefix + "seed"] = np.nan
        for attr in PLAYER_ATTRIBUTES:
            row[prefix + attr] = player[attr]
        row[prefix + "elo_pre"] = current_elo(surface_stats, as_of)
        row[prefix + "days_rest"] = float((as_of - pd.Timestamp(player["last_match_date"])).days)
        row[prefix + "h2h_wins_pre"] = wins
        row[prefix + "h2h_losses_pre"] = losses
        for stat in ROLLING_STAT_NAMES:
            for w in ROLLING_WINDOWS:
                col = f"{stat}_last{w}"
                row[prefix + col] = surface_stats[col] if surface_stats is not None else np.nan

    df = pd.DataFrame([row])
    # SQLite returns missing values as None; the model expects NaN.
    df = df.fillna(np.nan).infer_objects()
    return add_diff_features(df)


def _swap_sides(df: pd.DataFrame) -> pd.DataFrame:
    """The same match with player_1 and player_2 exchanged."""
    def swap(col: str) -> str:
        if col.startswith("player_1_"):
            return "player_2_" + col.removeprefix("player_1_")
        if col.startswith("player_2_"):
            return "player_1_" + col.removeprefix("player_2_")
        return col

    swapped = df.rename(columns=swap)
    for col in ("rank_diff", "age_diff", "elo_pre_diff"):
        swapped[col] = -df[col]
    return swapped[df.columns]


def predict_win_prob(model: lgb.LGBMClassifier, features: pd.DataFrame) -> float:
    """P(player_1 wins), averaged over both orientations of the matchup.

    A tree model isn't exactly symmetric: P(A beats B) with A as player_1 and
    1 - P(B beats A) with B as player_1 can differ slightly. Averaging the two
    guarantees P(A beats B) + P(B beats A) = 1, whichever player is listed first.
    """
    X, _ = select_features(pd.concat([features, _swap_sides(features)], ignore_index=True))
    p = model.predict_proba(X[model.feature_name_])[:, 1]
    return float((p[0] + (1 - p[1])) / 2)


def elo_win_prob(elo_1: float, elo_2: float) -> float:
    """P(player_1 wins) implied by the Elo gap alone."""
    return 1.0 / (1.0 + 10.0 ** ((elo_2 - elo_1) / 400.0))


def serve_point_prob(server_stats: dict | None, returner_stats: dict | None) -> float | None:
    """P(server wins a point on serve) against this specific returner.

    Averages the server's own serve-point win rate with the rate the returner
    concedes on return (1 - return win %), using the longest window available.
    None when either player has no recorded serve/return stats on the surface.
    """
    if server_stats is None or returner_stats is None:
        return None
    for w in sorted(ROLLING_WINDOWS, reverse=True):
        serve = server_stats[f"serve_win_pct_last{w}"]
        ret = returner_stats[f"return_win_pct_last{w}"]
        if serve is not None and ret is not None and not (np.isnan(serve) or np.isnan(ret)):
            return (serve + (1.0 - ret)) / 2.0
    return None


def match_breakdown(p1_serve: float, p2_serve: float, best_of: int) -> dict:
    """Scoreline-level detail from the Markov engine (exact) and Monte Carlo simulation (path-dependent)."""
    sims = simulate_many_matches(p1_serve, p2_serve, best_of=best_of, n_sims=N_SIMULATIONS, seed=42)
    summary = summarize_simulations(sims)
    set_scores = (
        (sims["sets_a"].astype(str) + "-" + sims["sets_b"].astype(str))
        .value_counts(normalize=True)
        .sort_index()
    )
    return {
        "player_1_serve_point_prob": p1_serve,
        "player_2_serve_point_prob": p2_serve,
        "player_1_hold_prob": prob_win_game(p1_serve),
        "player_2_hold_prob": prob_win_game(p2_serve),
        "player_1_set_win_prob": prob_win_set(p1_serve, p2_serve),
        "player_1_match_win_prob": prob_win_match(p1_serve, p2_serve, best_of=best_of),
        "straight_sets_prob": float(summary["p_straight_sets"]),
        "went_the_distance_prob": float(summary["p_went_the_distance"]),
        "set_score_probs": {score: float(p) for score, p in set_scores.items()},
    }
