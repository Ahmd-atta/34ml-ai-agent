"""
LangGraph wrapper-nodes
────────────────────────────────────────
Each node receives and returns the shared `state` dict that flows through
the graph.

Keys the generator may add / update
-----------------------------------
draft          : str   – text draft to be approved
image_url      : str|None
image_done     : bool  – True once a DALL·E image is generated
waiting_for_qa : bool
channel        : str   – “Instagram”, “LinkedIn”, …

Scheduler and KB nodes simply drop their textual result into
state["result"].
"""

from typing import Dict, List
import logging

from tools.generator  import generator_tool
from tools.scheduler  import scheduler_tool
from tools.rag_tool   import rag_search

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  GENERATOR NODE                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
def generator_node(state: Dict) -> Dict:
    """
    Call the content-generator tool and merge its output back into state.
    """
    logger.debug("generator_node: input keys %s", list(state.keys()))

    gen_out = generator_tool(state)   # ← call real logic
    state.update(gen_out)             # draft / image_url / channel / …

    # ------- conversation history -------
    hist = state.get("conversation_history", [])
    if hist and hist[-1]["user"] == state["user_input"] and not hist[-1]["bot"]:
        # prefer the draft; else whatever the node returned as result
        bot_reply = gen_out.get("draft") or gen_out.get("result")
        hist[-1]["bot"] = bot_reply
    state["conversation_history"] = _dedup_history(hist)

    return state


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SCHEDULER NODE                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
def scheduler_node(state: Dict) -> Dict:
    result = scheduler_tool(state["user_input"], state=state)
    state["result"] = result

    hist = state.get("conversation_history", [])
    if hist and hist[-1]["user"].lower() != "show history" and not hist[-1]["bot"]:
        hist[-1]["bot"] = result
    state["conversation_history"] = _dedup_history(hist)

    return state


# ╔══════════════════════════════════════════════════════════════════╗
# ║  KB / RAG NODE                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
def kb_node(state: Dict) -> Dict:
    answer = str(rag_search(state["user_input"]))
    state["result"] = answer

    hist = state.get("conversation_history", [])
    if hist and hist[-1]["user"] == state["user_input"] and not hist[-1]["bot"]:
        hist[-1]["bot"] = answer
    state["conversation_history"] = _dedup_history(hist)

    return state


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Helper: de-duplicate history                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
def _dedup_history(hist: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Keep only the most recent occurrence of each *user* message.
    Trim to max 50 entries.
    """
    if not hist:
        return hist

    dedup, seen = [], set()
    for entry in reversed(hist):
        user_txt = entry["user"].lower()
        if user_txt not in seen:
            seen.add(user_txt)
            dedup.append(entry)

    return list(reversed(dedup))[:50]