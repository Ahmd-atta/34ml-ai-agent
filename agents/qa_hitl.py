# agents/qa_hitl.py
"""
Human-in-the-Loop approval.
A = approve, E = paste-edit, R = reject, quit/exit bubbles to CLI.
"""

import re
from memory.post_store import save_post

_PLACEHOLDER = re.compile(r"$$[^$$]+\]")   # detects [anything]


def _has_placeholder(text: str) -> bool:
    return bool(_PLACEHOLDER.search(text))


def approve_or_edit(draft: str, channel: str, image_url: str = None, image_path: str = None) -> str | None:
    current = draft.strip()

    while True:
        print("\n--- DRAFT -----------------------------------------")
        print(current)
        if image_url:
            print(f"Generated image: {image_url}")
        print("----------------------------------------------------")
        action = input("[A]pprove  [E]dit  [R]eject  (or 'quit')? ").strip().lower()

        if action in {"quit", "exit"}:
            return None

        if action.startswith("r"):
            print("❌  Rejected")
            return None

        if action.startswith("e"):
            print("Paste the full, edited text below (blank line to finish):")
            lines: list[str] = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            current = "\n".join(lines).strip()
            continue

        if action.startswith("a"):
            if _has_placeholder(current):
                print("⚠️  Draft still has placeholders like [Client Name]. Edit before approving.")
                continue
            save_post(channel, current, image_url, image_path)
            print("✅  Saved & approved")
            return current