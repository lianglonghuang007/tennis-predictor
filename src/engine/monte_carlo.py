"""Monte Carlo match simulation: point-by-point simulation for path-dependent
metrics the analytical Markov engine (markov.py) can't give directly — e.g.
the probability of a specific set score or a match going the distance, not
just who wins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.markov import _set_server, _tiebreak_server


def simulate_game(p: float, rng: np.random.Generator) -> bool:
    """Simulate one game point by point; return True if the server wins.

    Same deuce/advantage rule as markov.prob_win_game's closed form (first to
    4+ points, win by 2) — but drawn point by point from real randomness
    here, rather than computed as a probability.
    """
    server_pts, returner_pts = 0, 0
    while True:
        if rng.random() < p:
            server_pts += 1
        else:
            returner_pts += 1
        if server_pts >= 4 and server_pts - returner_pts >= 2:
            return True
        if returner_pts >= 4 and returner_pts - server_pts >= 2:
            return False


def simulate_tiebreak(p_a: float, p_b: float, first_server: str, rng: np.random.Generator) -> str:
    """Simulate a tiebreak point by point; return 'A' or 'B'.

    Reuses markov._tiebreak_server for serve alternation, so the simulated
    and analytical engines share one definition of the scoring rules instead
    of two separate (and possibly drifting) implementations.
    """
    other = "B" if first_server == "A" else "A"
    a, b = 0, 0
    while True:
        server = _tiebreak_server(a, b, first_server, other)
        p_a_wins_point = p_a if server == "A" else (1 - p_b)
        if rng.random() < p_a_wins_point:
            a += 1
        else:
            b += 1
        if a >= 7 and a - b >= 2:
            return "A"
        if b >= 7 and b - a >= 2:
            return "B"


def simulate_set(
    p_a_serve: float, p_b_serve: float, first_server: str, rng: np.random.Generator
) -> tuple[str, int, int]:
    """Simulate one set; return (winner, games_a, games_b) — the actual score, not just who won."""
    other = "B" if first_server == "A" else "A"
    games_a, games_b = 0, 0
    while True:
        if games_a == 6 and games_b == 6:
            tiebreak_server = _set_server(games_a, games_b, first_server, other)
            winner = simulate_tiebreak(p_a_serve, p_b_serve, tiebreak_server, rng)
            if winner == "A":
                return "A", games_a + 1, games_b
            return "B", games_a, games_b + 1
        if games_a >= 6 and games_a - games_b >= 2:
            return "A", games_a, games_b
        if games_b >= 6 and games_b - games_a >= 2:
            return "B", games_a, games_b

        server = _set_server(games_a, games_b, first_server, other)
        server_p = p_a_serve if server == "A" else p_b_serve
        game_winner = server if simulate_game(server_p, rng) else ("B" if server == "A" else "A")
        if game_winner == "A":
            games_a += 1
        else:
            games_b += 1


def simulate_match(
    p_a_serve: float, p_b_serve: float, best_of: int, first_server: str, rng: np.random.Generator
) -> dict:
    """Simulate one full match; return the winner plus the full path (every set's score)."""
    sets_needed = best_of // 2 + 1
    sets_a, sets_b = 0, 0
    set_scores: list[tuple[int, int]] = []
    server_for_this_set, other = first_server, ("B" if first_server == "A" else "A")

    while sets_a < sets_needed and sets_b < sets_needed:
        winner, games_a, games_b = simulate_set(p_a_serve, p_b_serve, server_for_this_set, rng)
        set_scores.append((games_a, games_b))
        if winner == "A":
            sets_a += 1
        else:
            sets_b += 1
        # Who serves first alternates set to set, same as real matches —
        # note this is one respect in which simulation is MORE faithful to
        # the real rules than prob_win_match's documented simplification.
        server_for_this_set, other = other, server_for_this_set

    return {
        "winner": "A" if sets_a > sets_b else "B",
        "sets_a": sets_a,
        "sets_b": sets_b,
        "set_scores": set_scores,
        "n_sets": len(set_scores),
    }


def simulate_many_matches(
    p_a_serve: float,
    p_b_serve: float,
    best_of: int = 3,
    first_server: str = "A",
    n_sims: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run n_sims independent match simulations; return one row of outcomes per simulated match."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_sims):
        result = simulate_match(p_a_serve, p_b_serve, best_of, first_server, rng)
        rows.append({
            "a_wins": result["winner"] == "A",
            "sets_a": result["sets_a"],
            "sets_b": result["sets_b"],
            "n_sets": result["n_sets"],
            "went_the_distance": result["n_sets"] == best_of,
            "straight_sets": min(result["sets_a"], result["sets_b"]) == 0,
        })
    return pd.DataFrame(rows)


def summarize_simulations(sims: pd.DataFrame) -> dict:
    """Summarize path-dependent metrics the analytical engine can't give directly."""
    return {
        "p_a_wins_match": sims["a_wins"].mean(),
        "p_straight_sets": sims["straight_sets"].mean(),
        "p_went_the_distance": sims["went_the_distance"].mean(),
        "mean_sets_played": sims["n_sets"].mean(),
    }
