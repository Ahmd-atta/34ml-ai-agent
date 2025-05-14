# memory/post_store.py
import json, uuid, datetime
from pathlib import Path
from typing import List, Dict
from memory.similarity import add_vector      # keeps LT-memory updated

POSTS_PATH = Path("memory/posts.json")
POSTS_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── private loader ─────────────────────────────────────────────────
def _load() -> List[Dict]:
    return json.load(POSTS_PATH.open()) if POSTS_PATH.exists() else []


# ── public save ────────────────────────────────────────────────────
def save_post(channel: str, text: str, image_url: str = None, image_path: str = None) -> str | None:
    """
    Save the approved post and return its UUID.
    If the text already exists, returns None.
    """
    text = text.strip()
    data = _load()
    if any(p["text"] == text for p in data):
        return None  # duplicate

    post_id = str(uuid.uuid4())
    data.append(
        {
            "id": post_id,
            "datetime": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "channel": channel,
            "text": text,
            "image_url": image_url,
            "image_path": image_path
        }
    )
    json.dump(data, POSTS_PATH.open("w"), indent=2)

    add_vector(text)          # embed into long-term memory
    return post_id