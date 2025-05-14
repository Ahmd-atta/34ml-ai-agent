"""
Channel-aware offline Scheduler.

Data files
----------
memory/posts.json       – all approved posts
memory/schedule.json    – scheduled items

Supported commands
------------------
show queue | show <channel> queue
show posts | show <channel> posts
show scheduled posts | show scheduled <channel> posts
show history
schedule last [<channel>] post for|on <date>
schedule <id-prefix> for|on <date>
remove last [<channel>]
remove <id-prefix> [from <date>]
help | ?
"""

from __future__ import annotations
import re, shlex
from typing import List, Dict
from dateutil import parser as dparse          # python-dateutil

from memory.post_store import _load as load_posts
from memory.schedule_store import (
    add_to_queue,
    get_queue,
    remove_from_queue,
    map_post_id_to_date,
)

# ────────────────────────────────────────────────────────────────
# Channel aliases / helpers
# ────────────────────────────────────────────────────────────────
_ALIAS = {
    "instagram": "Instagram",
    "insta": "Instagram",
    "ig": "Instagram",
    "linkedin": "LinkedIn",
    "linked": "LinkedIn",
    "linked in": "LinkedIn",
    "linkdin": "LinkedIn",
    "twitter": "X",
    "x": "X",
}
def norm_ch(word: str | None) -> str | None:
    if not word:
        return None
    return _ALIAS.get(word.lower().strip())

# ────────────────────────────────────────────────────────────────
# Small utils
# ────────────────────────────────────────────────────────────────
def iso(text: str) -> str | None:
    try:
        return dparse.parse(text, fuzzy=True).date().isoformat()
    except Exception:
        return None

def latest(posts: List[Dict], channel: str | None) -> Dict | None:
    cand = [p for p in posts if channel is None or p["channel"] == channel]
    return max(cand, key=lambda p: p["datetime"], default=None)

def fmt(rows: List[Dict]) -> str:
    if not rows:
        return "Nothing found."
    return "\n".join(
        f"- {r['when']} - {r['channel']} - {r['id'][:8]}... \"{r['text'][:60]}...\"{' Image: ' + r['image_url'] if r.get('image_url') else ''}"
        for r in rows
    )

# ────────────────────────────────────────────────────────────────
# MAIN entry – called by LangChain as a Tool
# ────────────────────────────────────────────────────────────────
def scheduler_tool(user_request: str, state: Dict | None = None) -> str:
    toks = shlex.split(user_request.lower())
    if not toks:
        return "Unrecognised scheduler command. Type 'help'."

    posts      = load_posts()
    sched_map  = map_post_id_to_date()
    cmd        = toks[0]

    # --------------------------- show history ---------------------
    if cmd == "show" and "history" in toks:
        history = state.get("conversation_history", []) if state else []
        if not history:
            return "No conversation history found."
        # Limit to last 10 entries and format clearly
        result = ["=== Conversation History (Last 10 Interactions) ==="]
        for item in history[-10:]:
            if item["user"] and item["user"].lower() != "show history":
                result.append(f"User: {item['user']}")
            if item["bot"]:
                # Truncate long bot responses for readability
                bot_text = item["bot"][:200] + "..." if len(item["bot"]) > 200 else item["bot"]
                result.append(f"Bot: {bot_text}")
        return "\n".join(result) or "No conversation history found."

    # --------------------------- help -----------------------------
    if cmd in {"help", "?"}:
        return (
            "Scheduler commands:\n"
            "  show queue | show <channel> queue\n"
            "  show posts | show <channel> posts\n"
            "  show scheduled posts | show scheduled <channel> posts\n"
            "  show history\n"
            "  schedule last [<channel>] post for <date>\n"
            "  schedule <id> for <date>\n"
            "  remove last [<channel>] | remove <id> [from <date>]\n"
        )

    # --------------------------- SHOW -----------------------------
    if cmd == "show":
        # ----- queue or scheduled -----
        if "queue" in toks or ("scheduled" in toks and "posts" in toks):
            ch = next((norm_ch(t) for t in toks if norm_ch(t)), None)
            rows = get_queue(ch)
            rows_fmt = [
                {"when": r["scheduled_for"], "channel": r["channel"],
                 "id": r["post_id"], "text": r["text"], "image_url": r.get("image_url")}
                for r in rows
            ]
            return fmt(sorted(rows_fmt, key=lambda x: (x["when"], x["channel"])))

        # ----- posts (all or filtered) -----
        if "posts" in toks:
            ch          = next((norm_ch(t) for t in toks if norm_ch(t)), None)
            only_sched  = "scheduled" in toks
            rows: List[Dict] = []
            for p in posts:
                if ch and p["channel"] != ch:
                    continue
                if only_sched and p["id"] not in sched_map:
                    continue
                when = p["datetime"][:10]
                if p["id"] in sched_map:
                    when += f" [SCHEDULED {sched_map[p['id']]}]"
                rows.append(
                    {"when": when, "channel": p["channel"],
                     "id": p["id"], "text": p["text"], "image_url": p.get("image_url")}
                )
            return fmt(rows)

    # --------------------------- SCHEDULE -------------------------
    if cmd == "schedule":
        # schedule last [channel] post for/on DATE
        if toks[1] == "last":
            # detect optional channel token(s)
            ch = next((norm_ch(t) for t in toks[2:] if norm_ch(t)), None)
            # find index of "for" or "on"
            try:
                idx = toks.index("for")
            except ValueError:
                idx = toks.index("on")
            date_expr = " ".join(toks[idx + 1:])
            post = latest(posts, ch)
            if not post:
                return "No posts available."
            date_iso = iso(date_expr)
            if not date_iso:
                return f"Couldn't parse date '{date_expr}'."
            err = add_to_queue(post["id"], post["channel"], post["text"], date_iso)
            return "Scheduled." if not err else f"Error: {err}"

        # schedule <id> for/on DATE
        pid = toks[1]
        post = next((p for p in posts if p["id"].startswith(pid)), None)
        if not post:
            return f"Post '{pid}' not found."
        try:
            idx = toks.index("for")
        except ValueError:
            idx = toks.index("on")
        date_expr = " ".join(toks[idx + 1:])
        date_iso  = iso(date_expr)
        if not date_iso:
            return f"Couldn't parse date '{date_expr}'."
        err = add_to_queue(post["id"], post["channel"], post["text"], date_iso)
        return "Scheduled." if not err else f"Error: {err}"

    # --------------------------- REMOVE ---------------------------
    if cmd in {"remove", "unschedule"}:
        if toks[1] == "last":
            ch = next((norm_ch(t) for t in toks[2:] if norm_ch(t)), None)
            cand = [p for p in posts if p["id"] in sched_map]
            target = latest(cand, ch)
            if not target:
                return "Nothing to remove."
            remove_from_queue(target["id"])
            return "Removed."

        pid = toks[1]
        try:
            idx = toks.index("from")
            date_iso = iso(" ".join(toks[idx + 1:]))
        except ValueError:
            date_iso = None
        ok = remove_from_queue(pid, date_iso)
        return "Removed." if ok else "Nothing matched."

    # --------------------------- fallback -------------------------
    return "Unrecognised scheduler command. Type 'help' for list."