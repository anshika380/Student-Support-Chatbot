"""
main.py
-------
FastAPI backend exposing the Student Support Chatbot as a REST API.

Endpoints:
    POST /api/chat            -> send a message, get a bot response
    GET  /api/categories      -> list all FAQ categories
    GET  /api/faqs/{category} -> list FAQs under a category
    GET  /api/health          -> health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid

from chatbot_engine import chatbot

app = FastAPI(
    title="AI Chatbot for Student Support Services",
    description="Semantic-search based chatbot (Sentence-BERT) that answers common student queries.",
    version="1.0.0",
)

# Allow the frontend (served separately) to call this API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory chat history per session (fine for a demo/internship project;
# swap for a database like PostgreSQL/Supabase for production use)
chat_sessions = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    confidence: float
    category: Optional[str]
    matched_question: Optional[str]
    suggestions: List[str]


@app.get("/api/health")
def health_check():
    return {"status": "ok", "backend": chatbot.backend}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = req.session_id or str(uuid.uuid4())
    chat_sessions.setdefault(session_id, []).append({"role": "user", "content": req.message})

    result = chatbot.get_response(req.message)

    chat_sessions[session_id].append({"role": "bot", "content": result["response"]})

    return ChatResponse(
        session_id=session_id,
        response=result["response"],
        confidence=result["confidence"],
        category=result["category"],
        matched_question=result["matched_question"],
        suggestions=result["suggestions"],
    )


@app.get("/api/categories")
def categories():
    return {"categories": chatbot.get_categories()}


@app.get("/api/faqs/{category}")
def faqs_by_category(category: str):
    results = chatbot.get_faqs_by_category(category)
    if not results:
        raise HTTPException(status_code=404, detail=f"No FAQs found for category '{category}'.")
    return {"category": category, "faqs": results}


@app.get("/api/history/{session_id}")
def get_history(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": session_id, "history": chat_sessions[session_id]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
