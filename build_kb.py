# build_kb.py
import sys
from agents.scraper import scrape
from kb import build_or_load

if len(sys.argv) < 2:
    print("Usage: python build_kb.py https://your-company.com")
    sys.exit(1)

url = sys.argv[1]
docs = scrape(url)                   # already tested
index = build_or_load(docs)

print(f"✅ Vector store ready with {len(index.docstore.docs)} documents")