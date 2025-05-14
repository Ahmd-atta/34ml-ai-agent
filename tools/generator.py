"""
Content-generator tool (LangGraph node)

• Detects channel & “with image”.
• Uses RAG facts + brand tone.
• Guards against duplicates.
• Generates image once (DALL·E-3). Flag `image_done` prevents re-calls.
• Returns: draft, image_url, channel, waiting_for_qa, image_done.
"""

from __future__ import annotations
import os, re
from typing import Dict
from dotenv import load_dotenv

from agents.brand       import get_brand
from tools.rag_tool     import rag_search
from memory.similarity  import too_similar
from tools.image_agent  import create_image
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ── brand consts ─────────────────────────────────────────────
b       = get_brand()
TONE    = ", ".join(b["tone"])
RULES   = "; ".join(b["style_rules"])
AUD     = b["audience"]

# ── LLM (Gemini-1.5-Flash) ───────────────────────────────────
_LLM = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    cache=False,
)

# ── regex helpers ────────────────────────────────────────────
_PAT_CH_INST = re.compile(r"\binstagram|insta|ig\b", re.I)
_PAT_CH_LINK = re.compile(r"\blinkedin|li\b", re.I)
_PAT_CH_FB   = re.compile(r"\bfacebook|fb\b", re.I)
_NEEDS_CLIENT = re.compile(r"\b(client|case\s*study|testimonial)\b", re.I)

def _detect_channel(msg: str) -> str:
    if _PAT_CH_INST.search(msg): return "Instagram"
    if _PAT_CH_LINK.search(msg): return "LinkedIn"
    if _PAT_CH_FB.search(msg):   return "Facebook"
    return "LinkedIn"

# ─────────────────────────────────────────────────────────────
def generator_tool(state: Dict) -> Dict:
    user_msg   = state["user_input"]

    channel    = state.get("channel", _detect_channel(user_msg))
    with_image = state.get("with_image", False) or "with image" in user_msg.lower()

    topic = re.sub(r"\bwith\s+image\b", "", user_msg, flags=re.I).strip()
    if too_similar(topic):
        topic += " (fresh angle, avoid repeating earlier posts)"

    facts = rag_search(f"List 3 short facts about 34ML relevant to '{topic}'.", top_k=6)

    placeholder_rule = (
        "If you mention a client, write it as [Client Name]."
        if _NEEDS_CLIENT.search(topic) else
        "Avoid [Client Name] placeholders."
    )

    prompt = f"""Write a {channel} post.
Audience: {AUD}
Tone: {TONE}
Style rules: {RULES}

Facts about 34ML:
{facts}

Topic: {topic}

{placeholder_rule}
Return ONLY the post text."""
    draft = _LLM.invoke(prompt).content.strip()

    # ------- image (only once) ---------------------------------
    image_url  = state.get("image_url")
    image_done = state.get("image_done", False)

    if with_image and not image_done:
        try:
            img_prompt = f"Create an engaging {channel} image about '{topic}'."
            image_url  = create_image(img_prompt, channel.lower())["url"]
            image_done = True
            #  ⇓⇓⇓  write flag into state so a 2nd pass in same cycle sees it
            state["image_url"]  = image_url
            state["image_done"] = True
        except Exception as e:
            draft += f"\n\n(Note: image generation error – {e})"

    # ------- return --------------------------------------------
    return {
        "draft"        : draft,
        "image_url"    : image_url,
        "image_done"   : image_done,
        "channel"      : channel,
        "waiting_for_qa": True,
        "result"       : None,
    }