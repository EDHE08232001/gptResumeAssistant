# ./RAG/build_resume_index.py
import os, json, hashlib, re
from pathlib import Path
import numpy as np
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RESUME_MD = PROJECT_ROOT / "public" / "Edward_He_s_Resume.md"

OUT_DIR = BASE_DIR
CHUNKS_JSON = OUT_DIR / "resume_chunks.json"
VECTORS_NPY = OUT_DIR / "resume_vectors.npy"
META_JSON = OUT_DIR / "resume_meta.json"

EMBED_MODEL = "text-embedding-3-small"  # strong + cheap default :contentReference[oaicite:2]{index=2}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def normalize(text: str) -> str:
    # collapse excessive whitespace while preserving structure
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_markdown(md: str, max_chars: int = 1800, overlap_chars: int = 250) -> list[str]:
    """
    Simple, robust chunker:
      1) split on blank lines (paragraph-ish)
      2) pack into chunks up to max_chars
      3) add small character overlap for continuity
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
            # overlap tail
            tail = cur[-overlap_chars:] if overlap_chars > 0 else ""
            cur = (tail + "\n\n" + b).strip()

    if cur:
        chunks.append(cur)

    # final cleanup
    return [normalize(c) for c in chunks if c.strip()]

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Batches embeddings and returns float32 matrix [n, d].
    """
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    vectors = np.array([item.embedding for item in resp.data], dtype=np.float32)
    return vectors

def main():
    if not RESUME_MD.exists():
        raise FileNotFoundError(f"Missing resume: {RESUME_MD}")

    md = normalize(RESUME_MD.read_text(encoding="utf-8"))
    chunks = chunk_markdown(md)

    print(f"Resume chars: {len(md):,}")
    print(f"Chunks: {len(chunks)}")

    vectors = embed_texts(chunks)
    # L2-normalize once so cosine similarity becomes dot-product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    vectors = vectors / norms

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_JSON.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(VECTORS_NPY, vectors)

    meta = {
        "source_file": str(RESUME_MD),
        "source_sha256": file_sha256(RESUME_MD),
        "embed_model": EMBED_MODEL,
        "num_chunks": len(chunks),
        "vector_dim": int(vectors.shape[1]),
    }
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {CHUNKS_JSON}")
    print(f"Wrote: {VECTORS_NPY}")
    print(f"Wrote: {META_JSON}")

if __name__ == "__main__":
    main()
