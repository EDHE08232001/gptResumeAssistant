"""
main.py
FastAPI backend for a React chatbot – sends Edward He's resume as context.
"""

from dotenv import load_dotenv
load_dotenv()  # Load OPENAI_API_KEY before importing gptAssistant

import os
import asyncio
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.gptAssistant import generate_reply

# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------
app = FastAPI(
    title="GPT Resume Assistant API",
    version="1.0.0",
)

# ---------------------------------------------------------------------
# CORS (lock down in production)
# ---------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str

# ---------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------
@app.on_event("startup")
def startup_checks() -> None:
    """
    Fail fast if critical resources are missing.
    """
    required_env = ["OPENAI_API_KEY"]
    for key in required_env:
        if not os.getenv(key):
            raise RuntimeError(f"Missing required environment variable: {key}")

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """
    Return a GPT reply with Edward He's resume included as context.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    messages_dicts = [m.model_dump() for m in req.messages]

    # Run blocking LLM call off the event loop
    reply = await asyncio.to_thread(generate_reply, messages_dicts)

    return ChatResponse(reply=reply)

# ---------------------------------------------------------------------
# Local verification
# ---------------------------------------------------------------------
if __name__ == "__main__":
    async def run_quick_test():
        test_request = ChatRequest(
            messages=[
                Message(
                    role="user",
                    content="Please summarize my resume."
                )
            ]
        )
        response = await chat_endpoint(test_request)
        print("=== Quick Resume Summary Test ===")
        print(response.reply)

    asyncio.run(run_quick_test())