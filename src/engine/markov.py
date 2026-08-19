"""Analytical point -> game -> set -> match win-probability engine.

Pure probability/combinatorics — no trained model dependency. Every function
here takes each player's P(win a point on their own serve) as an input and
derives everything else from first principles, which makes this module
independently testable against known closed-form and symmetry results.
"""

from __future__ import annotations

import math


def prob_win_game(p: float) -> float:
    """Probability the server wins a single game, given p = P(win one point on serve).

    Points are assumed i.i.d. Bernoulli(p) — a standard simplifying
    assumption in analytical tennis models (it ignores momentum/clutch
    effects within a game).

    Derivation: the server can win outright at 4-0, 4-1, or 4-2 (reaching 4
    points before the opponent reaches 3, without ever hitting deuce), or the
    game can reach 3-3 (deuce), after which it becomes a "win by 2"
    sub-problem with its own clean closed form.
    """
    # P(win 4-j), for j = 0, 1, 2 opponent points: comb(3+j, j) counts the
    # number of point orderings with exactly j opponent points among the
    # first 3+j points, given the (4+j)-th point must be the server's
    # clinching win.
    p_win_outright = sum(
        math.comb(3 + j, j) * (p ** 4) * ((1 - p) ** j)
        for j in range(3)
    )
    # P(reach 3-3): exactly 3 points each among the first 6 points.
    p_reach_deuce = math.comb(6, 3) * (p ** 3) * ((1 - p) ** 3)
    # From deuce, let d = P(win from deuce). Next 2 points either win it
    # outright (p^2), lose it outright ((1-p)^2), or split and return to
    # deuce (2p(1-p)) — solving d = p^2 + 2p(1-p)*d gives this closed form.
    p_win_from_deuce = (p ** 2) / (p ** 2 + (1 - p) ** 2)
    return p_win_outright + p_reach_deuce * p_win_from_deuce


def _tiebreak_server(a: int, b: int, first_server: str, other: str) -> str:
    """Who serves the next point in a tiebreak at score a-b (first_server served point 1).

    Real tiebreak serving rule: the first server serves 1 point, then players
    alternate serving 2 points each (1 / 2 / 2 / 2 / ...).
    """
    points_played = a + b
    switches_so_far = (points_played + 1) // 2
    return first_server if switches_so_far % 2 == 0 else other


def prob_win_tiebreak(
    p_a: float, p_b: float, first_server: str = "A", target: int = 7, max_extra_points: int = 60
) -> float:
    """Probability player A wins a tiebreak, given each player's own-serve point-win probability.

    Computed via BOTTOM-UP dynamic programming (an explicit table, filled in
    order), not top-down recursion. A first attempt at this used plain
    recursion with memoization and hit Python's recursion limit: even though
    memoization prevents recomputing any (a, b) state twice, a "win by 2"
    tiebreak has no upper bound on how long two evenly-matched players can
    stay tied, and Python doesn't optimize deep call chains the way some
    other languages do — the number of DISTINCT states was small, but the
    raw call-stack DEPTH chasing the tied-forever tail wasn't. Filling the
    table bottom-up (highest scores first, working back to 0-0) computes the
    exact same values with a simple loop instead of nested function calls,
    so there's no call stack to overflow.

    max_extra_points bounds the table: beyond `target + max_extra_points`
    combined points, remaining probability is treated as split 50/50. For any
    realistic point-win probability (not exactly 0 or 1), the chance of a
    tiebreak reaching even 20 extra points is already astronomically small,
    so 60 is a very safe margin, not a meaningful approximation.
    """
    other = "B" if first_server == "A" else "A"
    max_score = target + max_extra_points
    values: dict[tuple[int, int], float] = {}

    # Fill in order of DECREASING total points played, so that by the time we
    # compute values[(a, b)], both values[(a + 1, b)] and values[(a, b + 1)]
    # — which it depends on — already exist in the table.
    for total in range(2 * max_score, -1, -1):
        lo = max(0, total - max_score)
        hi = min(total, max_score)
        for a in range(lo, hi + 1):
            b = total - a
            if a >= target and a - b >= 2:
                values[(a, b)] = 1.0
            elif b >= target and b - a >= 2:
                values[(a, b)] = 0.0
            elif a == max_score or b == max_score:
                values[(a, b)] = 0.5  # table boundary; see max_extra_points note above
            else:
                server = _tiebreak_server(a, b, first_server, other)
                p_a_wins_point = p_a if server == "A" else (1 - p_b)
                values[(a, b)] = (
                    p_a_wins_point * values[(a + 1, b)] + (1 - p_a_wins_point) * values[(a, b + 1)]
                )

    return values[(0, 0)]


def _set_server(games_a: int, games_b: int, first_server: str, other: str) -> str:
    """Who serves the next game at score games_a-games_b (first_server served game 1).

    Unlike a tiebreak, set serving simply alternates every single game.
    """
    return first_server if (games_a + games_b) % 2 == 0 else other


def prob_win_set(p_a_serve: float, p_b_serve: float, first_server: str = "A") -> float:
    """Probability player A wins a set (first to 6 games, win by 2, 7-point tiebreak at 6-6).

    Recursion + a fresh memo dict, for the same reason as prob_win_tiebreak:
    who's serving alternates game-to-game, so which player's prob_win_game
    result applies depends on the running score.
    """
    other = "B" if first_server == "A" else "A"
    p_a_holds = prob_win_game(p_a_serve)  # P(A wins a game A is serving)
    p_b_holds = prob_win_game(p_b_serve)  # P(B wins a game B is serving)
    memo: dict[tuple[int, int], float] = {}

    def f(games_a: int, games_b: int) -> float:
        if games_a == 6 and games_b == 6:
            # The set is decided by a single tiebreak "game" at 6-6, not by
            # further (games_a, games_b) recursion.
            tiebreak_server = _set_server(games_a, games_b, first_server, other)
            return prob_win_tiebreak(p_a_serve, p_b_serve, first_server=tiebreak_server)
        if games_a >= 6 and games_a - games_b >= 2:
            return 1.0
        if games_b >= 6 and games_b - games_a >= 2:
            return 0.0
        if (games_a, games_b) in memo:
            return memo[(games_a, games_b)]

        server = _set_server(games_a, games_b, first_server, other)
        p_a_wins_game = p_a_holds if server == "A" else (1 - p_b_holds)

        result = p_a_wins_game * f(games_a + 1, games_b) + (1 - p_a_wins_game) * f(games_a, games_b + 1)
        memo[(games_a, games_b)] = result
        return result

    return f(0, 0)


def prob_win_match(
    p_a_serve: float, p_b_serve: float, best_of: int = 3, first_server: str = "A"
) -> float:
    """Probability player A wins a best-of-`best_of` (3 or 5) match.

    Simplifying assumption, documented rather than hidden: one set win
    probability is computed (assuming `first_server` serves first in that
    set) and treated as i.i.d. across every set in the match. In reality the
    player who serves first alternates set-to-set, which has a small
    second-order effect on each set's exact probability — real analytical
    tennis models commonly make this same simplification, since the effect
    is typically well under a percentage point.
    """
    p_set = prob_win_set(p_a_serve, p_b_serve, first_server=first_server)
    sets_needed = best_of // 2 + 1

    # Standard "win a best-of-N series" formula: sum, over every possible
    # match length k, of P(exactly sets_needed - 1 wins in the first k - 1
    # sets) x P(win the k-th, clinching set).
    prob = 0.0
    for k in range(sets_needed, best_of + 1):
        ways = math.comb(k - 1, sets_needed - 1)
        prob += ways * (p_set ** sets_needed) * ((1 - p_set) ** (k - sets_needed))
    return prob
