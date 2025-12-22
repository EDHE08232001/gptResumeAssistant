# ./RAG/build_resume_index.py
import json
import hashlib
import re
import os
from pathlib import Path
from typing import List

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Add it to a .env file or export it in your shell."
    )

client = OpenAI(api_key=API_KEY)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

RESUME_MD = PROJECT_ROOT / "public" / "Edward_He_s_Resume.md"

OUT_DIR = BASE_DIR
CHUNKS_JSON = OUT_DIR / "resume_chunks.json"
VECTORS_NPY = OUT_DIR / "resume_vectors.npy"
META_JSON = OUT_DIR / "resume_meta.json"

# ---------------------------------------------------------------------
# Embedding config
# ---------------------------------------------------------------------
EMBED_MODEL = "text-embedding-3-small"  # strong, cheap, stable
EMBED_BATCH_SIZE = 64                  # safe default

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_markdown(
    md: str,
    max_chars: int = 1800,
    overlap_chars: int = 250,
) -> List[str]:
    """
    Paragraph-based chunker with light overlap.
    Stable, predictable, resume-safe.
    """
    blocks = [b.strip() for b in md.split("\n\n") if b.strip()]
    chunks = []
    cur = ""

    for b in blocks:
        if not cur:
            cur = b
            continue

        if len(cur) + 2 + len(b) <= max_chars:
            cur += "\n\n" + b
        else:
            chunks.append(cur)
            tail = cur[-overlap_chars:] if overlap_chars > 0 else ""
            cur = (tail + "\n\n" + b).strip()

    if cur:
        chunks.append(cur)

    return [normalize(c) for c in chunks if c.strip()]

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Batched embedding → float32 matrix [n, d]
    """
    all_vectors = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )
        all_vectors.extend(item.embedding for item in resp.data)

    vectors = np.asarray(all_vectors, dtype=np.float32)
    return vectors

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    if not RESUME_MD.exists():
        raise FileNotFoundError(f"Missing resume file: {RESUME_MD}")

    md_raw = RESUME_MD.read_text(encoding="utf-8")
    md = normalize(md_raw)
    chunks = chunk_markdown(md)

    print(f"[INFO] Resume characters : {len(md):,}")
    print(f"[INFO] Chunks generated  : {len(chunks)}")

    vectors = embed_texts(chunks)

    # L2 normalize once → cosine similarity == dot product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    vectors = vectors / norms

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Persist chunks
    CHUNKS_JSON.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(VECTORS_NPY, vectors)

    # Metadata (deterministic + audit-friendly)
    meta = {
        "source_file": str(RESUME_MD),
        "source_sha256": file_sha256(RESUME_MD),
        "embed_model": EMBED_MODEL,
        "num_chunks": len(chunks),
        "vector_dim": int(vectors.shape[1]),
        "chunk_sha256": [
            hashlib.sha256(c.encode("utf-8")).hexdigest() for c in chunks
        ],
    }

    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] Wrote {CHUNKS_JSON}")
    print(f"[OK] Wrote {VECTORS_NPY}")
    print(f"[OK] Wrote {META_JSON}")

if __name__ == "__main__":
    main()