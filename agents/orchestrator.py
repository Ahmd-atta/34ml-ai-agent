"""
Orchestrator node
─────────────────
Reads `state["user_input"]` and decides which agent node should run next.

Sets
----
state["route"]      : "generate" | "scheduler" | "kb" | "end"
state["channel"]    : "Instagram" | "LinkedIn" | "Facebook" | ...
state["with_image"] : bool
"""

import re
from typing import Dict, List

# ---------- patterns -------------------------------------------------
_PAT_SCHED   = re.compile(r"\b(show|schedule|remove|unschedule)\b", re.I)
_PAT_HISTORY = re.compile(r"\bshow\s+history\b", re.I)

# Every supported channel + aliases
_CHANNEL_RE = (
    r"(instagram|insta|ig|"
    r"linkedin|li|"
    r"facebook|fb|"
    r"twitter|tweet|x)"
)

# “write linkedin post …”, “create insta post with image …”, etc.
_PAT_POST = re.compile(
    rf"""
    \b(?:write|create|draft|make)      # action verb
    \s+(?:a\s+|new\s+)?                # optional fillers
    {_CHANNEL_RE}                      # channel   (capt. group 1)
    \s+post                            # the word ‘post’
    (.*?)                              # the rest of the sentence
    """,
    re.I | re.X,
)

_ALIAS_MAP = {
    "insta": "instagram",
    "ig": "instagram",
    "li": "linkedin",
    "fb": "facebook",
    "tweet": "twitter",
    "x": "twitter",
}


def orchestrator(state: Dict) -> Dict:
    user: str = state.get("user_input", "")

    # ---------- empty or None ----------
    if not user.strip():
        state["route"] = "end"
        return state

    # ---------- keep conversation history ----------
    history: List[Dict] = state.get("conversation_history", [])
    if not history or history[-1].get("user") != user:
        history.append({"user": user, "bot": ""})
    state["conversation_history"] = history

    # ---------- routing decisions ----------
    text_low = user.lower()

    if _PAT_HISTORY.search(text_low) or _PAT_SCHED.search(text_low):
        state["route"] = "scheduler"
        return state

    m = _PAT_POST.search(text_low)
    if m:
        ch_raw = m.group(1).lower()
        channel = _ALIAS_MAP.get(ch_raw, ch_raw).capitalize()  # “instagram” → “Instagram”
        state["channel"] = channel
        state["with_image"] = "with image" in text_low
        state["route"] = "generate"
        return state

    # fallback → KB search / random chat
    state["route"] = "kb"
    return state