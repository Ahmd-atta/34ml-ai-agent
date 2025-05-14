"""
Persistent publish queue              memory/schedule.json

[
  {
    "post_id"      : "uuid-string",
    "channel"      : "LinkedIn",
    "text"         : "… frozen copy …",
    "scheduled_for": "2025-05-25"
  }
]
"""

from __future__ import annotations
import json, datetime, itertools
from pathlib import Path
from typing import List, Dict

_STORE = Path("memory/schedule.json")
_STORE.parent.mkdir(parents=True, exist_ok=True)


# ── I/O helpers ────────────────────────────────────────────────────
def _load() -> List[Dict]:
    return json.load(_STORE.open()) if _STORE.exists() else []


def _save(rows: List[Dict]):
    json.dump(rows, _STORE.open("w"), indent=2)


# ── public API ────────────────────────────────────────────────────
def add_to_queue(post_id: str, channel: str, text: str, iso_date: str) -> str:
    """Returns '' on success or an error string on clash / bad date."""
    try:
        datetime.date.fromisoformat(iso_date)
    except ValueError:
        return f"Date '{iso_date}' is not ISO-8601 (YYYY-MM-DD)."

    rows = _load()
    clash = any(
        r["channel"].lower() == channel.lower() and r["scheduled_for"] == iso_date
        for r in rows
    )
    if clash:
        return f"{channel} already has a post on {iso_date}."

    rows.append(
        {
            "post_id": post_id,
            "channel": channel,
            "text": text,
            "scheduled_for": iso_date,
        }
    )
    _save(rows)
    return ""


def remove_from_queue(post_id: str, iso_date: str | None = None) -> bool:
    """
    Delete a queued post.  If iso_date is None, delete first matching id.
    Returns True when something was removed.
    """
    rows = _load()
    before = len(rows)
    rows = [
        r
        for r in rows
        if not (r["post_id"] == post_id and (iso_date is None or iso_date == r["scheduled_for"]))
    ]
    if len(rows) != before:
        _save(rows)
        return True
    return False


def get_queue(channel: str | None = None) -> List[Dict]:
    rows = _load()
    if channel:
        rows = [r for r in rows if r["channel"].lower() == channel.lower()]
    # sort ascending by date, then channel
    return sorted(rows, key=lambda r: (r["scheduled_for"], r["channel"].lower()))


def map_post_id_to_date() -> dict[str, str]:
    """Utility: {post_id: iso_date} for quick lookup."""
    return {r["post_id"]: r["scheduled_for"] for r in _load()}