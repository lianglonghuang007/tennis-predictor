"""Rebuild the SQLite database the API serves from.

Run after the data pipeline (src/data_loader.py) whenever new match data
arrives. Like train_models.py, this is an occasional batch job, not something
run per API request.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import DB_PATH, create_schema, get_connection, load_database

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    processed_path = project_root / "data" / "processed" / "atp_matches_combined.csv"

    logger.info(f"Loading {processed_path}")
    df = pd.read_csv(processed_path, low_memory=False, parse_dates=["tourney_date"])

    # Build into a temporary file and swap it in only once it's complete, so
    # a running API never reads a half-built database mid-rebuild.
    tmp_path = DB_PATH.with_suffix(".db.tmp")
    tmp_path.unlink(missing_ok=True)

    conn = get_connection(tmp_path)
    try:
        create_schema(conn)
        counts = load_database(conn, df)
    finally:
        conn.close()

    tmp_path.replace(DB_PATH)

    for table, n in counts.items():
        logger.info(f"  {table}: {n:,} rows")
    logger.info(f"Saved database to {DB_PATH}")


if __name__ == "__main__":
    main()
