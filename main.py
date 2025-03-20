import os
import openai
import asyncio
import logging
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logging.info(f"Using openai package version: {openai.__version__}")

# Configure CORS for testing (update for production)
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
# Each session_id maps to a list of messages (each message is a dict with "role" and "content")
session_history: Dict[str, List[Dict[str, str]]] = {}

# Chat request model includes an optional session_id.
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

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
        # Use existing session or create a new one.
        session_id = request.session_id
        if session_id is None or session_id not in session_history:
            session_id = str(uuid.uuid4())
            # Start conversation with system prompt.
            session_history[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Append the user's message.
        session_history[session_id].append({"role": "user", "content": request.message})
        logging.info(f"Session {session_id} - Received message: {request.message}")
        
        # Use the entire conversation history as context.
        conversation = session_history[session_id]
        
        # Call the synchronous OpenAI API wrapped in asyncio.to_thread to avoid blocking.
        response = await asyncio.to_thread(
            lambda: openai.ChatCompletion.create(
                model="gpt-4",
                messages=conversation
            )
        )
        ai_response = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Append the assistant's response to the session history.
        session_history[session_id].append({"role": "assistant", "content": ai_response})
        logging.info(f"Session {session_id} - Response sent: {ai_response}")
        
        # Return the session id along with the reply.
        return {"session_id": session_id, "response": ai_response}
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
