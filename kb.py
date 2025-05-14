# kb.py
"""
Central place for:
1. Building / loading the persisted FAISS vector store (RAG memory)
2. Registering the default embedding model (MiniLM, local)
3. Registering the default LLM (Gemini-1.5-flash via LangChain)
4. Returning a ready-to-use QueryEngine with adjustable top-k

Works without any OpenAI key.
"""

from pathlib import Path
import os
from typing import Optional

# --- Load env so GOOGLE_API_KEY is visible no matter who imports kb.py ----
from dotenv import load_dotenv

load_dotenv()  # must be before we construct the LLM

# ---------------- Llama-Index / LangChain imports -------------------------
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- Constants & shared singletons --------------------------
INDEX_DIR = Path("memory/vector_store")
INDEX_DIR.mkdir(parents=True, exist_ok=True)  # ensure folder exists

# Local MiniLM embedder (384-d, small, free)
_EMBED = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# Gemini LLM (no cache so temperature makes a difference)
_LLM = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.8,          # bump up for more variety
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    cache=False,               # TURN OFF LangChain response cache
)

# Register as global defaults so any Llama-Index call picks them up
Settings.embed_model = _EMBED
Settings.llm = _LLM


# -------------------------------------------------------------------------
def build_or_load(docs: Optional[list] = None) -> VectorStoreIndex:
    """
    If the vector store already exists, load & return it.
    Otherwise `docs` must be supplied to build, persist, and return.
    """
    if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
        return load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(INDEX_DIR)),
            embed_model=_EMBED,  # explicit, although Settings already set
        )

    if docs is None:
        raise ValueError("Need `docs` to build a new index")

    index = VectorStoreIndex.from_documents(docs, embed_model=_EMBED)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    return index


def get_query_engine(top_k: int = 5):
    """
    Convenience wrapper that returns a RetrieverQueryEngine
    with `similarity_top_k` chunks.  Uses persisted index.
    """
    index = build_or_load()
    return index.as_query_engine(similarity_top_k=top_k)


# Small manual test (run: python kb.py)
if __name__ == "__main__":
    engine = get_query_engine(5)
    print(engine.query("What does 34ml do?"))