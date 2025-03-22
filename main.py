import os
import openai
import asyncio
import logging
import uuid
import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logging.info(f"Using openai package version: {openai.__version__}")

# Configure CORS for testing (update allowed origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Set OpenAI API key and system prompt from environment variables.
openai.api_key = os.getenv("OPENAI_API_KEY")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a successful CEO giving business advice.")

# Global in-memory dictionary to store session history.
# Each session_id maps to a dict with keys: "messages" (a list of message dicts) and "created_at" (a timestamp).
session_history: Dict[str, Dict[str, Any]] = {}

# Chat request model now includes an optional session_id.
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

def generate_session_title(messages: List[Dict[str, str]]) -> str:
    """
    Generate a short title for a session based on the first user message.
    If no user message is found, returns "Untitled Session".
    """
    for msg in messages:
        if msg["role"] == "user":
            words = msg["content"].split()
            if words:
                # Use the first 3 words as the title.
                return " ".join(words[:3])
    return "Untitled Session"

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.options("/chat")
async def options_chat():
    return {}

@app.post("/chat")
async def chat(request: ChatRequest):
    if not openai.api_key:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key is missing. Please check your environment variables."
        )
    try:
        # Use an existing session or create a new one if none provided.
        session_id = request.session_id
        if session_id is None or session_id not in session_history:
            session_id = str(uuid.uuid4())
            session_history[session_id] = {
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
                "created_at": datetime.datetime.now().isoformat()
            }
        
        # Append the user's message.
        session_history[session_id]["messages"].append({"role": "user", "content": request.message})
        logging.info(f"Session {session_id} - Received message: {request.message}")
        
        # Use the entire conversation history as context.
        conversation = session_history[session_id]["messages"]
        
        # Call the synchronous OpenAI API wrapped in asyncio.to_thread to avoid blocking.
        response = await asyncio.to_thread(
            lambda: openai.ChatCompletion.create(
                model="gpt-4",
                messages=conversation
            )
        )
        ai_response = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Append the assistant's response to the session history.
        session_history[session_id]["messages"].append({"role": "assistant", "content": ai_response})
        logging.info(f"Session {session_id} - Response sent: {ai_response}")
        
        # Return the session id along with the reply.
        return {"session_id": session_id, "response": ai_response}
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Endpoint to list all sessions with a title, preview, and creation timestamp.
@app.get("/sessions")
async def get_sessions():
    sessions = []
    for session_id, data in session_history.items():
        messages = data.get("messages", [])
        title = generate_session_title(messages)
        preview = messages[-1]["content"] if messages else ""
        created_at = data.get("created_at", "")
        sessions.append({"session_id": session_id, "title": title, "preview": preview, "created_at": created_at})
    return {"sessions": sessions}

# Endpoint to retrieve the full conversation for a specific session.
@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id in session_history:
        data = session_history[session_id]
        return {"session_id": session_id, "created_at": data.get("created_at", ""), "messages": data.get("messages", [])}
    else:
        raise HTTPException(status_code=404, detail="Session not found")
