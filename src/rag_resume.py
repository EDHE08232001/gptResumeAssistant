# ./src/rag_resume.py
import os, json
from pathlib import Path
import numpy as np
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_DIR = PROJECT_ROOT / "RAG"

CHUNKS_JSON = RAG_DIR / "resume_chunks.json"
VECTORS_NPY = RAG_DIR / "resume_vectors.npy"

EMBED_MODEL = "text-embedding-3-small"

_client = None
_chunks = None
_vectors = None

def _client_get() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def _load_index():
    global _chunks, _vectors
    if _chunks is None or _vectors is None:
        if not CHUNKS_JSON.exists() or not VECTORS_NPY.exists():
            raise RuntimeError(
                "RAG index not found. Run: python ./RAG/build_resume_index.py"
            )
        _chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
        _vectors = np.load(VECTORS_NPY).astype(np.float32)  # already normalized
    return _chunks, _vectors

def _embed_query(q: str) -> np.ndarray:
    client = _client_get()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Returns list of {score, text}.
    """
    chunks, vectors = _load_index()
    qv = _embed_query(query)

    # cosine similarity since all vectors normalized
    scores = vectors @ qv  # [n]
    top_idx = np.argsort(scores)[-k:][::-1]

    results = []
    for i in top_idx:
        results.append({"score": float(scores[i]), "text": chunks[int(i)]})
    return results
