"""
ui_gradio.py  – 34ML Social-Media AI Agent
──────────────────────────────────────────
• Chat pane
• HITL approve / edit / reject / quit
• Scheduler help (type “help”)
• Shows generated image
"""

import uuid, re, logging, gradio as gr
from langgraph.checkpoint.memory import MemorySaver
from build_graph import get_runner
from memory.post_store import save_post

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ────────────────────────── LangGraph runner
runner = get_runner(checkpointer=MemorySaver())

# ────────────────────────── constants
RESET_KEYS = {
    "waiting_for_qa": False,
    "draft": None,
    "approved": None,
    # keep image_url / image_done until QA finishes
}

FULL_HELP = (
    "Scheduler commands:\n"
    "  show queue | show <channel> queue\n"
    "  show posts | show <channel> posts\n"
    "  show scheduled posts | show scheduled <channel> posts\n"
    "  show history\n"
    "  schedule last [<channel>] post for <date>\n"
    "  schedule <id> for <date>\n"
    "  remove last [<channel>] | remove <id> [from <date>]\n\n"
    "Generation examples:\n"
    "  write instagram post with image about <topic>\n"
    "  write linkedin post about <topic>\n\n"
    "HITL while a draft is shown:\n"
    "  approve (a) | edit <text> (e <text>) | reject (r) | quit (q)"
)

# ────────────────────────── helper
def _invoke_graph(thread_id: str, extra: dict):
    payload = {**RESET_KEYS, **extra}
    return runner.invoke(
        payload,
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
    )

# ────────────────────────── main callback
def chat_callback(
    history,
    user_msg,
    thread_state,
    qa_flag,
    cur_draft,
    cur_img,
    cur_channel,
    img_done,
):
    """Main chat / HITL handler."""
    thread_id = thread_state or str(uuid.uuid4())
    msg = user_msg.strip()
    if not msg:
        return history, "", thread_id, qa_flag, cur_draft, cur_img, cur_channel, img_done

    # ---------------- HELP
    if msg.lower() == "help":
        history.append((msg, FULL_HELP))
        return history, "", thread_id, qa_flag, cur_draft, cur_img, cur_channel, img_done

    # ---------------- HITL phase
    if qa_flag:
        cmd = msg.lower()
        if cmd in {"approve", "a", "reject", "r", "quit", "q"} or cmd.startswith(("edit ", "e ")):
            # determine final text
            if cmd in {"approve", "a"}:
                final_text = cur_draft
            elif cmd.startswith(("edit ", "e ")):
                final_text = msg[5:] if cmd.startswith("edit ") else msg[2:]
            else:  # reject / quit
                final_text = None

            if final_text:
                save_post(cur_channel or "LinkedIn", final_text, cur_img, None)
                history.append((msg, "✅ Saved & approved"))
            else:
                history.append((msg, "Draft discarded."))

            # tell graph QA finished and clear image flags
            _invoke_graph(thread_id, {
                                    "approved": bool(final_text),
                                    "image_url": None,
                                    "image_done": False})
            return history, "", thread_id, False, None, None, None, False

        history.append((msg, "Invalid QA command. Use approve/edit/reject/quit."))
        return history, "", thread_id, True, cur_draft, cur_img, cur_channel, img_done

    # ---------------- normal user turn
    state_out = _invoke_graph(thread_id, {"user_input": msg})

    draft      = state_out.get("draft")
    img_url    = state_out.get("image_url")
    channel    = state_out.get("channel")
    img_done   = state_out.get("image_done", False)
    qa_needed  = state_out.get("waiting_for_qa", False)
    result_txt = state_out.get("result")

    # If generator returned dict inside result, unpack
    if isinstance(result_txt, dict):
        draft   = result_txt.get("draft", draft)
        img_url = result_txt.get("image_url", img_url)
        result_txt = result_txt.get("text") or result_txt.get("message")

    if draft or qa_needed:
        reply = f"--- DRAFT ---\n{draft}\n\n"
        if img_url:
            reply += f"Generated image: {img_url}\n\n"
        reply += "[A]pprove  [E]dit  [R]eject  [Q]uit"
        history.append((msg, reply))
        return history, "", thread_id, True, draft, img_url, channel, img_done

    # regular non-draft answer
    if result_txt is None:
        result_txt = "No response."
    history.append((msg, result_txt))
    return history, "", thread_id, False, None, None, None, img_done

# ────────────────────────── UI definition
demo = gr.Blocks(title="34ML Social-Media Agent")
with demo:
    # session state
    thread_state  = gr.State(None)
    qa_state      = gr.State(False)
    cur_draft     = gr.State(None)
    cur_img_url   = gr.State(None)
    cur_channel   = gr.State(None)
    img_done      = gr.State(False)

    gr.Markdown("## 34ML Social-Media AI Agent")

    with gr.Row():
        with gr.Column(scale=7):
            chatbox = gr.Chatbot(height=550)
            textbox = gr.Textbox(placeholder="Type a command …", show_label=False)
        with gr.Column(scale=3):
            image   = gr.Image(label="Generated image", visible=False)
            gr.Markdown("Type `help` for full command list.")

    # submit chain
    textbox.submit(
        chat_callback,
        inputs=[chatbox, textbox, thread_state, qa_state,
                cur_draft, cur_img_url, cur_channel, img_done],
        outputs=[chatbox, textbox, thread_state, qa_state,
                 cur_draft, cur_img_url, cur_channel, img_done],
    ).then(
        # keep channel unchanged; just update image preview
        lambda url_val, chan_val: (
            gr.update(value=url_val or None, visible=bool(url_val)),
            chan_val,
        ),
        inputs=[cur_img_url, cur_channel],
        outputs=[image,      cur_channel],
    )

    gr.Markdown("---\nScrape → RAG → Draft → Human approval → Schedule")

if __name__ == "__main__":
    demo.launch()