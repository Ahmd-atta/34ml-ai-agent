#agent/scraper.py

from pathlib import Path
from typing import List
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import Document

RAW_DIR = Path("data/raw")

def scrape(base_url: str, depth: int = 1) -> List[Document]:
    """
    Download `base_url` (plus internal links later) and return a list of
    Llama-Index Document objects.  Also saves clean text copies.
    """
    reader = SimpleWebPageReader(html_to_text=True)
    docs = reader.load_data([base_url])      # updated API (no max_depth)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for i, doc in enumerate(docs):
        (RAW_DIR / f"page_{i}.txt").write_text(doc.text, encoding="utf-8")

    return docs