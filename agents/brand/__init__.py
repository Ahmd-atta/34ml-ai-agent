# agents/brand/__init__.py
"""
Expose get_brand() so other modules can grab the cached brand profile.
"""

from pathlib import Path
import json
from .profiler import build_profile

_BRAND_JSON = Path("memory/brand.json")


def get_brand() -> dict:
    """
    Return the brand profile as a dict.
    If memory/brand.json doesn't exist yet, build it first.
    """
    if not _BRAND_JSON.exists():
        return build_profile(force=False)
    return json.load(_BRAND_JSON.open())