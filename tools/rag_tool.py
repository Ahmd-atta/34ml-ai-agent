# tools/rag_tool.py
from kb import get_query_engine

def rag_search(question: str, top_k: int = 5) -> str:
    """
    LangChain-compatible function; given a question, returns an
    evidence-grounded answer from the vector store.
    """
    engine = get_query_engine(top_k)
    return str(engine.query(question))