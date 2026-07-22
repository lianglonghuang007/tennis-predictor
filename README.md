# ATP Tennis Match Win Probability

Full-stack app predicting ATP match win probabilities: surface-specific Elo ratings feed a
Bradley-Terry baseline and a LightGBM model estimating each player's serve-point win probability,
which drives an analytical Markov chain engine for game/set/match win probability. React frontend,
FastAPI backend, SQLite storage, benchmarked against bookmaker-implied odds.

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
  probably should down-weight/exclude it; a stats-based feature might not care) — not something
  to bake into "the cleaned data."
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
| Duplicate `(tourney_id, match_num)` | 2 | **Genuine data bug**: `tourney_id "2026-416"` is reused for both the Munich and Rome Masters 2026 events. Needs a synthetic tournament key before any per-tournament join, or these two events will collide. |

Two gotchas specific to this dataset (vs. the standard Sackmann schema) that the loader accounts
for: player IDs (`winner_id`/`loser_id`) are alphanumeric (e.g. `"GH92"`), not numeric, so they're
excluded from numeric coercion; and there's an extra `indoor` column not present in the original
Sackmann format.

Sanity-bound checks (`implausible_age`, `implausible_minutes`) came back at 0 after excluding NaN
ages (common for lower-tier Davis Cup players) and retirements/the John Isner–Nicolas Mahut 2010
Wimbledon match (665 minutes, the longest match in tennis history) from the bounds — both were
initially flagging real data, not errors, which is worth knowing before trusting a sanity check at
face value.

**Run it:**

```bash
./.venv/bin/python src/data_loader.py
```
