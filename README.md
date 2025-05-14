# 34ML Social-Media AI Agent 🚀

Scrape 34ML.com → extract brand identity → draft posts → human approval → schedule content – all from a local CLI.

---

## 0 Executive Summary

The 34ML Social-Media AI Agent automates content creation for LinkedIn and Instagram. A one-off scrape of `34ml.com` builds a FAISS vector knowledge base (MiniLM-L6-v2) for RAG-powered post generation. Brand tone, audience, and style are auto-extracted to `memory/brand.json`. The CLI generates posts (text via Gemini, images via DALL·E 3), enforces human-in-the-loop (HITL) approval, and prevents near-duplicate content using a similarity guard. Posts are stored in `memory/posts.json` and scheduled via `memory/schedule.json`. A LangGraph multi-agent orchestrator with checkpointing drives the workflow, running offline except for Gemini and OpenAI API calls.

---
```
## 1 Feature Matrix

| Phase | Feature                                                            | Status |
|------:|------------------------------------------------------------------ -|:------:|
| 1     | Repo bootstrap + secrets handling (.env)                           | ✅     |
| 2     | Gemini-powered CLI for post generation                             | ✅     |
| 3     | Web scraper → MiniLM embeddings → FAISS KB (RAG)                   | ✅     |
| 4     | KBSearch tool + RAG-aware content generation                       | ✅     |
| 5     | Brand profiler (tone, audience, style) → `brand.json`              | ✅     |
| 6     | ContentGenerator + QA/HITL + `posts.json` storage                  | ✅     |
| 7     | Long-term similarity guard (MiniLM vectors in FAISS)               | ✅     |
| 8     | Scheduler agent (queue, per-channel limits, remove, help)          | ✅     |
| 9     | LangGraph multi-agent orchestrator + DALL·E 3 image generation     | ✅     |
| 10    | Final documentation + optional gradio UI                           | ✅     |

---
```
## 2 Architecture (Phases 1–9)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           build_graph.py (LangGraph)                        │
│                                                                             │
│            ┌──────────┐        ┌────────────┐        ┌───────┐              │
│   CLI →    │orchestr. │──────▶ │generator   │──────▶ │  END  │             │
│            └────┬─────┘        └────┬───────┘        └──┬────┘              │
│                 │                  (draft + QA/HITL)     │                  │
│                 │                   image_agent (DALL·E 3)                  │
│                 │                                        │                  │
│                 │        ┌────────────┐                  │                  │
│                 ├──────▶ │scheduler   │──────▶───────────┘                 │
│                 │        └────┬───────┘                                     │
│                 │             │  (show/add/rm queue)                        │
│                 │             ▼                                             │
│                 │        result → CLI                                       │
│                 │                                                           │
│                 │        ┌────────────┐                                     │
│                 └──────▶ │kb (RAG)    │──────▶───────────┐                 │
│                          └────────────┘                  │                  │
│                                                          │                  │
└───────────────────────────────────────────────────────────┘                  │
                                                                               │
┌──────────────┐ scrape+clean  ┌──────────────┐ embeddings  ┌─────────────┐    │
│   Scraper    ├──────────────▶│   Chunks     ├────────────▶│  FAISS KB   │◀───┘
└──────────────┘               └──────────────┘             └─────────────┘

• generator    = RAG + similarity guard + QA/HITL + save_post + image_agent
• image_agent  = DALL·E 3 image generation → data/images/
• scheduler    = offline queue manager (show/schedule/remove)
• kb           = vector KB search over scraped pages
• END          = graph terminates; CLI prints state["result"]

Data Plane
----------
scraper → chunks → MiniLM embeddings → FAISS KB (RAG)
approved post → MiniLM vector → FAISS dup-guard
image_agent → PNG → data/images/

Storage
-------
posts.json        = approved posts (id, datetime, channel, text, image_url, image_path)
schedule.json     = scheduled posts (post_id, channel, text, scheduled_for, image_url)
brand.json        = tone, audience, style
vector_store/     = FAISS RAG index
lstm_vectors/     = FAISS dup-guard embeddings
```

### Workflow & Agent Communication
The workflow is a LangGraph `StateGraph` (`build_graph.py`) with four nodes: `orchestrator`, `generator`, `scheduler`, and `kb`. Agents communicate via a shared `GraphState` dictionary, persisted by `MemorySaver`.

1. **Orchestrator (`agents/orchestrator.py`)**:
   - Receives CLI input (e.g., “write instagram post about our new AI feature with image”).
   - Uses regex (`_PAT_POST`, `_PAT_SCHED`) to route to `generator` (post creation), `scheduler` (queue commands), or `kb` (company queries).
   - Sets `state["channel"]` (e.g., “Instagram”) and `state["with_image"]` (True/False).
   - Communication: Updates `state["result"]` or routes to next node.

2. **Generator (`tools/generator.py`)**:
   - Detects channel aliases (e.g., “insta” → “Instagram”) using `PATTERNS`.
   - Queries FAISS KB (`rag_tool.py`) for 34ML facts.
   - Checks similarity guard (`similarity.py`) to avoid duplicates.
   - Generates draft via Gemini (`gemini-1.5-flash-latest`).
   - If `with_image`, calls `image_agent` (`tools/image_agent.py`) to generate a DALL·E 3 image.
   - Runs QA/HITL (`qa_hitl.py`) for approval/edit/rejection.
   - Saves approved posts to `posts.json` (`post_store.py`).
   - Communication: Updates `state["result"]` with draft or error, returns to `END`.

3. **Image Agent (`tools/image_agent.py`)**:
   - Called by `generator` when `with_image=True`.
   - Generates image via DALL·E 3 (`client.images.generate`), saves to `data/images/<channel>_<uuid>.png`.
   - Returns `image_url` and `image_path` to `generator` for QA/HITL and storage.
   - Communication: Appends image metadata to `state["result"]`.

4. **Scheduler (`tools/scheduler.py`)**:
   - Handles commands (`show queue`, `schedule last`, `remove`).
   - Reads/writes `schedule.json` (`schedule_store.py`), enforcing one post per channel per day.
   - Displays `image_url` for posts with images.
   - Communication: Updates `state["result"]` with queue or confirmation.

5. **KB (`tools/rag_tool.py`)**:
   - Searches FAISS KB (`vector_store/`) for 34ML facts during post generation.
   - Communication: Provides context to `generator` via `state["result"]`.

**Data Flow**:
- CLI input → `orchestrator` → (`generator` + `image_agent` | `scheduler` | `kb`) → `END` → CLI output.
- Persistent state (`posts.json`, `schedule.json`, `brand.json`) ensures session continuity.
- FAISS (`vector_store/`, `lstm_vectors/`) supports RAG and duplicate detection.

**Key Tech**:
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`.
- **LLM**: Gemini 1.5 Flash (`langchain-google-genai`).
- **Image Generation**: OpenAI DALL·E 3 (`openai==1.51.2`).
- **Orchestration**: LangGraph (`langgraph==0.4.3`), checkpointed by `MemorySaver`.
- **Storage**: JSON + FAISS, no external DBs.

---

## 3 Quick Start

```bash
git clone https://github.com/your-username/social-media-ai-agent.git
cd social-media-ai-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows; use source .venv/bin/activate for Linux/Mac
pip install -r requirements.txt

cp .env.sample .env
# Edit .env with:
# GOOGLE_API_KEY=your_google_api_key
# OPENAI_API_KEY=your_openai_api_key

# One-off ingestion
python build_kb.py https://34ml.com/
python -m agents.brand.profiler

python app.py  # Start CLI
```

---

## 4 CLI Cheat-Sheet

```
help                           List scheduler commands
show queue                     List scheduled posts
show instagram posts           List approved Instagram posts
show scheduled linkedin posts  List scheduled LinkedIn posts
schedule last instagram post for next Friday
remove last linkedin           Remove last LinkedIn post
write instagram post about our new AI feature with image
write linkedin post about our new AI feature
```

---

## 5 Repository Map

```
app.py                     • CLI entry point, LangGraph runner, conversation history
build_graph.py             • LangGraph StateGraph construction
build_kb.py                • Scrape 34ml.com, build FAISS vector KB
agents/
│ orchestrator.py          • Routes inputs to generator/scheduler/kb
│ qa_hitl.py               • HITL approval/edit/rejection, placeholder enforcement
│ brand/
│   profiler.py            • Extracts tone/audience/style to brand.json
tools/
│ generator.py             • Post generation (RAG, similarity guard, image agent)
│ image_agent.py           • DALL·E 3 image generation
│ scheduler.py             • Queue management (show/schedule/remove)
│ rag_tool.py              • FAISS KB search for RAG
memory/
│ vector_store/            • FAISS RAG index
│ lstm_vectors/            • FAISS duplicate-guard embeddings
│ posts.json               • Approved posts
│ schedule.json            • Scheduled posts
│ brand.json               • Brand tone/audience/style
│ post_store.py            • Post storage logic
│ schedule_store.py        • Schedule storage logic
│ similarity.py            • Duplicate detection
data/
│ raw/                     • Cached HTML/text from scraper
│ images/                  • DALL·E 3 images (<channel>_<uuid>.png)
.env                       • GOOGLE_API_KEY, OPENAI_API_KEY
requirements.txt           • Dependencies
```

---

## 6 Demo Flow (3 min)

```text
python app.py
You: write instagram post about our new AI feature with image
--- DRAFT ---
🚀 Level up your business with 34ML's new AI-powered feature! ...
Generated image: https://oaidalleapiprodscus.blob.core.windows.net/...
[A]pprove [E]dit [R]eject? Approve
Bot: ✅ Saved & approved
You: schedule last instagram post for next Friday
Bot: Scheduled.
You: show queue
  - 2025-05-16 - Instagram - fbd46ba7... "🚀 Level up..." Image: https://oaidalle...
You: remove last instagram
Bot: Removed.
You: quit
```

Ensures no duplicates, no scheduling conflicts, and full human approval.

---

## 7 Cost Management

- **DALL·E 3**: ~$0.04 per 1024x1024 standard-quality image.
- **Gemini**: Free tier available (`console.cloud.google.com`).
- **Monitoring**: Check `platform.openai.com/usage` for Image Generations.
- **Previous Issue**: A $0.17 Chat Completions charge occurred on May 13, 2025. Contact OpenAI support (`help.openai.com`) for a refund, referencing the 11:36 AM EEST call.
- **Best Practices**:
  - Limit image tests to 1–2 per session (~$0.08).
  - Monitor logs for `DALL·E 3 API response` to confirm endpoint usage.

---

## 8 Installation

1. **Clone Repository**:
   ```bash
   git clone [https://github.com/your-username/social-media-ai-agent.git](https://github.com/Ahmd-atta/34ml-ai-agent)
   cd social-media-ai-agent
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows; source .venv/bin/activate for Linux/Mac
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Key dependencies:
   - `openai==1.51.2`
   - `langchain-community==0.3.23`
   - `langchain-google-genai`
   - `sentence-transformers`
   - `langgraph==0.4.3`
   - `requests`
   - `pillow`
   - `python-dotenv`

4. **Configure Environment**:
   Create `.env`:
   ```bash
   GOOGLE_API_KEY=your_google_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```
   - Google API key: `console.cloud.google.com`.
   - OpenAI API key: `platform.openai.com`.

5. **Create Directories**:
   ```bash
   mkdir -p data/raw data/images memory
   ```

---

## 9 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/YourFeature`.
3. Commit changes: `git commit -m 'Add YourFeature'`.
4. Push: `git push origin feature/YourFeature`.
5. Open a pull request.

---

## 14 Contact

For questions, contact the 34ML team at `support@34ml.com`.

---

Made with 💜 by Ahmed Khaled for the 34ML ML Challenge.
