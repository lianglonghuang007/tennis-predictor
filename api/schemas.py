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
