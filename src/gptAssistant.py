# ./src/gptAssistant.py
"""
gptAssistant.py
Service layer: RAG over Edward He's resume markdown stored in ./public,
with vectors stored in ./RAG.
"""

from typing import List, Dict
import os
from openai import OpenAI
from src.rag_resume import retrieve

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Set it in your environment.")

client = OpenAI(api_key=api_key)

MODEL = "gpt-5-nano"  # fast + cheap :contentReference[oaicite:5]{index=5}

def _last_user_text(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return m["content"]
    return ""

def generate_reply(messages: List[Dict[str, str]]) -> str:
    user_q = _last_user_text(messages)
    rag_hits = retrieve(user_q, k=5)

    context = "\n\n".join(
        [f"[Chunk score={h['score']:.3f}]\n{h['text']}" for h in rag_hits]
    )

    system_prompt = (
        "You are Edward He's professional resume assistant. "
        "Answer questions about Edward using ONLY the retrieved resume context below. "
        "If the answer is not supported by the context, say you are not sure.\n\n"
        "=== RETRIEVED RESUME CONTEXT ===\n"
        f"{context}\n"
        "=== END CONTEXT ==="
    )

    # Responses API (recommended) :contentReference[oaicite:6]{index=6}
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            *messages,
        ],
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
        max_output_tokens=500,
    )
    return resp.output_text
