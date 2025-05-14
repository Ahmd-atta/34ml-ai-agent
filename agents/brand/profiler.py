#C:\Users\HP\social-media-ai-agent\agents\brand\profiler.py

"""
Brand Profiler
==============
Extracts 34ML’s tone, audience, and writing rules from the website
content.  Result is cached in memory/brand.json.

Usage (package style, from repo root):
    python -m agents.brand.profiler            # prints JSON
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# project-level import – works when run with  -m  or inside other code
from kb import get_query_engine               # noqa: E402

load_dotenv()

JSON_PATH = Path("memory/brand.json")
JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

PROMPT = """
You are a brand-voice analyst.

Given the CONTEXT below, output *ONLY valid JSON* with:
• "tone"        : array of 3-5 adjectives
• "audience"    : 1-2 full sentences
• "style_rules" : short bullet list (max 6 bullets)

CONTEXT
=======
{context}
=======
"""


def _generate_profile(context: str) -> Dict[str, List[str] | str]:
    """Call Gemini and robust-parse the JSON block."""
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        cache=False,
    )
    raw = llm.invoke(PROMPT.format(context=context)).content.strip()
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError(f"LLM did not return JSON.\n---\n{raw}\n---")
    return json.loads(match.group())


def build_profile(force: bool = False):
    """Create (or load) brand.json."""
    if JSON_PATH.exists() and not force:
        return json.load(JSON_PATH.open())

    ctx = get_query_engine(top_k=8).query(
        "Summarise 34ML's writing style, customers, and product area in one paragraph."
    )
    data = _generate_profile(str(ctx))
    json.dump(data, JSON_PATH.open("w"), indent=2)
    return data


def load_profile():
    """Load cached profile, building it if missing."""
    if not JSON_PATH.exists():
        return build_profile()
    return json.load(JSON_PATH.open())


# manual run
if __name__ == "__main__":
    print(build_profile())