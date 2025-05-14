# memory/similarity.py
"""
Long-term similarity guard for approved posts.
Uses the same MiniLM embedder as RAG to keep things simple.
Stores 384-d vectors in a FAISS IndexFlatIP file:  memory/lstm_vectors/faiss.index
"""

from pathlib import Path
import faiss, numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

INDEX_PATH = Path("memory/lstm_vectors/faiss.index")
EMBED = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def _load_index() -> faiss.IndexFlatIP:
    if INDEX_PATH.exists():
        return faiss.read_index(str(INDEX_PATH))
    return faiss.IndexFlatIP(384)

def _save_index(idx: faiss.IndexFlatIP):
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(INDEX_PATH))

def _embed(text: str) -> np.ndarray:
    vec = EMBED.get_text_embedding(text)
    return np.asarray(vec, dtype="float32")

def add_vector(text: str):
    idx = _load_index()
    idx.add(_embed(text).reshape(1, -1))
    _save_index(idx)

def too_similar(text: str, threshold: float = 0.85) -> bool:
    idx = _load_index()
    if idx.ntotal == 0:
        return False
    vec = _embed(text).reshape(1, -1)
    D, _ = idx.search(vec, 1)
    return bool(D[0][0] >= threshold)