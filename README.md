# ATP Tennis Match Win Probability

Predicts ATP match win probabilities from historical match data. Surface-specific Elo ratings and
engineered form features feed a Bradley-Terry baseline and a LightGBM model; a separate analytical
Markov chain engine converts a per-point win probability into game/set/match probabilities.

**Status:** data pipeline, feature engineering, both models, and the Markov engine are complete.
FastAPI backend, SQLite storage, a React frontend, and bookmaker-odds benchmarking are planned but
not yet built.

**Stack:** Python, pandas, numpy, scikit-learn, LightGBM, Optuna, SHAP, matplotlib.

## Data Pipeline

**Source:** ATP main-tour match data, 2010–present (season files, one row per match). Not read
directly from Sackmann's `tennis_atp` repo — this project uses a same-schema derivative dataset,
so the loader is written against the actual files rather than assumed conventions (see gotchas
below).

**Code:** [`src/data_loader.py`](src/data_loader.py)

- `find_match_files` — locates `data/raw/YYYY.csv` season files in a year range, excluding
  `_challenger` and aggregate files (`ATP_Database.csv`, `ongoing_tourneys.csv`) also present in
  `data/raw`.
- `load_matches` — reads and concatenates season files, validating columns against the expected
  schema so a source format change fails loudly instead of silently misaligning data.
- `clean_matches` — parses tournament dates, coerces numeric columns, drops exact duplicate rows,
  sorts chronologically. Deliberately minimal: doesn't drop or impute anything that a downstream
  model might want to treat differently.
- `add_quality_flags` — attaches boolean columns (`is_retirement`, `is_walkover`, `is_default`,
  `is_incomplete_match`, `has_serve_stats`, `has_match_num`) instead of resolving quality issues
  in the base dataset. Whether to exclude a retirement, for example, is a modeling decision (Elo
  down-weights/excludes it; a stats-based feature might not) rather than something to bake into
  "the cleaned data."
- `summarize_data` / `flag_data_quality_issues` — print + return summary stats and quality checks.

**Output:** `data/processed/atp_matches_combined.csv` — 47,398 matches, 2010-01-03 to
2026-07-19, 1,778 unique players.

**Data quality findings (2010–present):**

| Issue | Count | Handling |
|---|---|---|
| Retirements | 1,398 | Flagged (`is_retirement`), not dropped |
| Walkovers | 299 | Flagged (`is_walkover`), not dropped |
| Defaults | 10 | Flagged (`is_default`), not dropped |
| Missing surface | 143 | Flagged via `surface.isna()`; excluded from surface-specific Elo |
| Missing all serve/return stats | 3,002 (6.3%) | Flagged (`has_serve_stats`); mostly Davis Cup ties and early rounds at smaller events |
| Missing winner/loser rank | 314 / 787 | Left as NaN — not imputable with 0 |
| Missing `match_num` | 489 | 10 recent (2025–2026) tournaments not yet backfilled by the data source |
| Duplicate `(tourney_id, match_num)` | 2 | **Data bug in the source**: `tourney_id "2026-416"` is reused for both the Munich and Rome Masters 2026 events. Requires a synthetic tournament key before any per-tournament join. |

Two gotchas specific to this dataset (vs. the standard Sackmann schema) that the loader accounts
for: player IDs (`winner_id`/`loser_id`) are alphanumeric (e.g. `"GH92"`), not numeric, so they're
excluded from numeric coercion; and there's an extra `indoor` column not present in the original
Sackmann format.

Sanity-bound checks (`implausible_age`, `implausible_minutes`) return 0 flagged rows once NaN ages
(common for lower-tier Davis Cup players) and legitimate outliers — retirements, and the 665-minute
John Isner–Nicolas Mahut 2010 Wimbledon match, the longest in tennis history — are excluded from
the bounds rather than treated as data errors.

**Run it:**

```bash
./.venv/bin/python src/data_loader.py
```

## Feature Engineering

**Code:** [`src/features/`](src/features/)

- **Player re-orientation** (`build_features.py`) — the source data always lists the match winner
  in `winner_*` columns, which would let a model trivially learn "player_1 always wins" instead of
  anything about tennis. Each match's winner/loser is randomly (seeded, reproducible) reassigned to
  `player_1`/`player_2`, with `player_1_won` as the resulting label.
- **Surface-specific Elo with time decay** (`elo.py`) — a per-player, per-surface rating updated
  chronologically after each match, with inactive players' ratings decaying toward the mean
  (half-life based) between matches. Walkovers are excluded from rating updates.
- **Rolling serve/return form** (`rolling_stats.py`) — trailing 10/20/50-match averages of serve
  win %, return win %, break-point conversion, and break-point save %, computed per surface with a
  strict `shift`-before-`rolling` pattern so a match's features never include that match's own
  result.
- **Head-to-head, days rest, round encoding, opponent-adjusted "edge" features** (`build_features.py`)
  — surface-specific H2H record and days since each player's last match, both computed as
  running, point-in-time-correct state; tournament round mapped to an ordinal scale; and a
  "serve edge" feature comparing one player's recent serve form directly against the specific
  opponent's recent return form, rather than against the field in general.

Every feature is constructed so that a given match's row only uses information available strictly
*before* that match — enforced throughout via chronological processing and `shift`-based rolling
windows, not just at the final train/test split.

**Train/val/test split** (`time_based_split`): chronological, not random — train through
2024-06-24, validation 2024-07-01–2025-06-30, test 2025-07-14 onward. A random split would let a
model train on matches chronologically after ones it's evaluated on, which no real deployment
could ever do.

## Models

**Code:** [`src/models/`](src/models/), evaluation utilities in [`src/evaluation.py`](src/evaluation.py)

### Baseline: Bradley-Terry (logistic regression)

Logistic regression on three standardized diff features: Elo rating gap, ranking gap, age gap.

### Primary: LightGBM

Gradient-boosted trees over the full 62-feature set, with categorical features (surface, hand,
tournament level, nationality) handled natively rather than one-hot encoded. Hyperparameters tuned
via Optuna against time-based cross-validation (`sklearn.TimeSeriesSplit`, 5 expanding-window
folds) rather than random K-fold, for the same point-in-time reason as the train/test split.
Feature importance computed via SHAP.

### Results (held-out test set, 2025-07-14 onward)

| Model | Accuracy | Brier score | Log-loss |
|---|---|---|---|
| Naive rank-only baseline (`rank_1 < rank_2`) | 64.1% (n=2,914) | — | — |
| Bradley-Terry baseline | 63.3% (n=2,914) | 0.221 | 0.633 |
| **LightGBM** | **65.9% (n=2,952)** | **0.214** | **0.615** |

LightGBM's edge over the naive rank-only baseline (65.0% vs. 64.1% on the matched row subset) is
statistically significant (McNemar's test, p ≈ 0.017) despite being under a full point — a ranking
system is already a strong aggregate signal, so a wide margin over it isn't the expected outcome
for a well-behaved model.

Top SHAP features: ranking gap and Elo gap dominate, followed by nationality, the serve-edge
features, and raw ranking/points — confirming the engineered opponent-adjusted features carry real
signal rather than being redundant with Elo/rank alone.

## Analytical Engine (Markov Chain)

**Code:** [`src/engine/markov.py`](src/engine/markov.py)

Pure probability/combinatorics, with no trained-model dependency: given each player's probability
of winning a point on their own serve, computes exact point → game → set → match win probabilities
using closed-form and dynamic-programming solutions to the actual tennis scoring rules (deuce,
alternating tiebreak service, best-of-3/5).

- `prob_win_game(p)` — closed-form solution, verified against the standard reference value
  (`p=0.6` → 73.57% game win probability).
- `prob_win_tiebreak`, `prob_win_set` — dynamic programming over the score state, since the server
  alternates mid-tiebreak and mid-set in a way that has no single closed-form expression.
- `prob_win_match` — combines set probabilities via the standard best-of-N series formula; verified
  that best-of-5 amplifies a stronger player's edge more than best-of-3 does, matching the real
  reason Grand Slam finals are played best-of-5.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Python 3.12. Always use `./.venv/bin/python`, not a bare `python3`.

## Roadmap

- FastAPI backend + SQLite storage
- Benchmark against bookmaker-implied probabilities (tennis-data.co.uk odds)
- pytest coverage for the Markov engine and core pipeline functions
- React frontend
- Deployment (Render/Railway)
