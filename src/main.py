"""
main.py
FastAPI backend for a React chatbot – sends Edward He's resume as context.
"""

from dotenv import load_dotenv
load_dotenv()  # Load OPENAI_API_KEY before importing gptAssistant

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from gptAssistant import generate_reply
import asyncio

app = FastAPI(title="GPT Resume Assistant API")

# Allow frontend to talk to backend (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Return a GPT reply with Edward He's resume included as system context."""
    messages_dicts = [m.model_dump() for m in req.messages]
    reply = generate_reply(messages_dicts)
    return ChatResponse(reply=reply)


# -------------------------------------------------------------------------
# Quick local verification:
# Run `python3 ./src/main.py` and it will summarize Edward He's resume.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    async def run_quick_test():
        test_request = ChatRequest(
            messages=[Message(role="user", content="Please summarize my resume.")]
        )
        response = await chat_endpoint(test_request)
        print("=== Quick Resume Summary Test ===")
        print(response.reply)

    asyncio.run(run_quick_test())