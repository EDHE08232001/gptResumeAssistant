"""
gptAssistant.py
Service layer: loads Edward He's resume and sends it in a system prompt
so every GPT reply has that context.
"""

from typing import List, Dict
import os, pathlib
from openai import OpenAI
from PyPDF2 import PdfReader

# --- Load API key and create client ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Set it in a .env file or export it."
    )

client = OpenAI(api_key=api_key)

# --- Load and cache resume text once ---
_resume_text: str | None = None

def _load_resume_text() -> str:
    """Extract and cache the text from public/Edward_He_s_Resume.pdf."""
    global _resume_text
    if _resume_text is None:
        pdf_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "public" / "Edward_He_s_Resume.pdf"
        )
        reader = PdfReader(str(pdf_path))
        _resume_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return _resume_text


def generate_reply(messages: List[Dict[str, str]]) -> str:
    """
    Call OpenAI and return the assistant's reply,
    always including Edward He's resume in the system prompt.
    """
    resume_text = _load_resume_text()

    system_prompt = (
        "You are Edward He's personal assistant. "
        "Use the details from his resume to answer questions about his skills, "
        "education, and experience. Respond factually and professionally.\n\n"
        "----- RESUME CONTENT -----\n"
        f"{resume_text}\n"
        "--------------------------"
    )

    completion = client.chat.completions.create(
        model="gpt-5-nano",  # Use a valid model name
        messages=[{"role": "system", "content": system_prompt}] + messages
    )
    return completion.choices[0].message.content