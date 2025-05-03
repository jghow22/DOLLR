import os
import openai
import asyncio
import logging
import uuid
import datetime
import sqlite3
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any

# Initialize FastAPI app
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logging.info(f"Using openai package version: {openai.__version__}")

# Configure CORS for testing (update allowed origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
    allow_headers=["*"],
)

# Set OpenAI API key and configuration from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4")  # Allow configuring the model
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "100"))  # Maximum sessions to return
MAX_TOKEN_LIMIT = int(os.getenv("MAX_TOKEN_LIMIT", "8192"))  # Token limit for context window
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))  # Default temperature for responses

# Enhanced system prompt
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """
You are an elite business consultant with deep expertise spanning finance, marketing, strategy, operations, and leadership. Your guidance is sought by entrepreneurs and executives at all stages of business development.

Your areas of expertise include:
1. Financial Analysis & Planning:
   - Cash flow management, financial modeling, and investment evaluation
   - Fundraising strategies (VC, angel investment, bootstrapping)
   - Pricing strategies and revenue optimization

2. Market Research & Strategy:
   - Market sizing and competitive analysis
   - Customer segmentation and targeting
   - Product-market fit evaluation

3. Growth & Marketing:
   - Go-to-market strategies
   - Digital marketing optimization and channel selection
   - Sales funnel optimization and conversion strategies

4. Operations & Scaling:
   - Process optimization and automation
   - Team structure and hiring strategies
   - Supply chain and vendor management

5. Leadership & Management:
   - Executive decision-making frameworks
   - Team building and organizational culture
   - Crisis management and turnaround strategies

In your responses:
- Prioritize actionable, specific advice over generic statements
- Support recommendations with business principles and relevant examples
- Consider both short-term wins and long-term sustainability
- When appropriate, outline step-by-step implementation plans
- Be honest about limitations and risks in your recommended approaches

Your goal is to provide the kind of strategic insight and practical guidance that would typically cost thousands of dollars in consulting fees.
""")

# Global in-memory dictionary to store session history as a fallback
session_history: Dict[str, Dict[str, Any]] = {}

# Database management
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    # Use memory by default for compatibility
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize the database tables if they don't exist"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Create sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT,
            last_activity TEXT
        )
        ''')
        # Create messages table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
        ''')
        conn.commit()

# Request/response models
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
                # Use the first 3-5 words as the title
                title = " ".join(words[:min(5, len(words))])
                if len(title) > 40:
                    title = title[:37] + "..."
                return title
    return "Untitled Session"

# Initialize the database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    logging.info("Database initialized")

# Endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "model": MODEL}

@app.options("/chat")
async def options_chat():
    """Handle preflight requests for CORS"""
    return {}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Process a chat message and return the AI response"""
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
        
        # Call the OpenAI API
        response = await asyncio.to_thread(
            lambda: openai.ChatCompletion.create(
                model=MODEL,
                messages=conversation,
                temperature=DEFAULT_TEMPERATURE
            )
        )
        
        ai_response = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Append the assistant's response to the session history.
        session_history[session_id]["messages"].append({"role": "assistant", "content": ai_response})
        logging.info(f"Session {session_id} - Response sent: {ai_response[:100]}...")
        
        # Return the session id along with the reply.
        return {"session_id": session_id, "response": ai_response}
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/sessions")
async def get_sessions():
    """Retrieve a list of all chat sessions"""
    try:
        sessions = []
        for session_id, data in session_history.items():
            messages = data.get("messages", [])
            title = generate_session_title(messages)
            
            # Get the most recent message as preview
            preview = ""
            for msg in reversed(messages):
                if msg["role"] != "system":
                    preview = msg["content"]
                    break
            
            created_at = data.get("created_at", "")
            
            sessions.append({
                "session_id": session_id,
                "title": title,
                "preview": preview[:100] + "..." if len(preview) > 100 else preview,
                "created_at": created_at
            })
        
        # Sort by created_at (newest first)
        sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return {"sessions": sessions[:MAX_SESSIONS]}
    except Exception as e:
        logging.error(f"Error in get_sessions endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Retrieve the full conversation for a specific session"""
    if session_id in session_history:
        data = session_history[session_id]
        return {
            "session_id": session_id,
            "created_at": data.get("created_at", ""),
            "messages": data.get("messages", [])
        }
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    if session_id in session_history:
        del session_history[session_id]
        return {"status": "success", "message": "Session deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/stats")
async def get_stats():
    """Get usage statistics"""
    try:
        session_count = len(session_history)
        message_count = sum(len(data.get("messages", [])) for data in session_history.values())
        
        # Calculate sessions in last 24 hours
        now = datetime.datetime.now()
        recent_sessions = 0
        for data in session_history.values():
            created_at = data.get("created_at", "")
            try:
                created_datetime = datetime.datetime.fromisoformat(created_at)
                if (now - created_datetime).total_seconds() < 86400:  # 24 hours in seconds
                    recent_sessions += 1
            except (ValueError, TypeError):
                pass
        
        avg_messages = message_count / session_count if session_count > 0 else 0
        
        return {
            "total_sessions": session_count,
            "total_messages": message_count,
            "avg_messages_per_session": round(avg_messages, 2),
            "sessions_last_24h": recent_sessions,
            "model": MODEL
        }
    except Exception as e:
        logging.error(f"Error in stats endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
