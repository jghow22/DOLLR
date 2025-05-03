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

# Database management
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect('sessions.db')
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
        # Use an existing session or create a new one
        session_id = request.session_id
        current_time = datetime.datetime.now().isoformat()
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create new session if needed
            if session_id is None:
                session_id = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO sessions (session_id, created_at, last_activity) VALUES (?, ?, ?)",
                    (session_id, current_time, current_time)
                )
            else:
                # Check if session exists
                cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
                if not cursor.fetchone():
                    session_id = str(uuid.uuid4())
                    cursor.execute(
                        "INSERT INTO sessions (session_id, created_at, last_activity) VALUES (?, ?, ?)",
                        (session_id, current_time, current_time)
                    )
                else:
                    # Update last activity timestamp
                    cursor.execute(
                        "UPDATE sessions SET last_activity = ? WHERE session_id = ?",
                        (current_time, session_id)
                    )
            
            # Get conversation history
            conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
            cursor.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            )
            for row in cursor.fetchall():
                conversation.append({"role": row["role"], "content": row["content"]})
            
            # Add the user message to conversation and save to database
            conversation.append({"role": "user", "content": request.message})
            cursor.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, "user", request.message, current_time)
            )
            
            logging.info(f"Session {session_id} - Received message: {request.message}")
            
            # Call OpenAI API
            response = await asyncio.to_thread(
                lambda: openai.ChatCompletion.create(
                    model=MODEL,
                    messages=conversation,
                    temperature=DEFAULT_TEMPERATURE
                )
            )
            
            ai_response = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Save AI response to database
            cursor.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, "assistant", ai_response, current_time)
            )
            
            conn.commit()
            logging.info(f"Session {session_id} - Response sent: {ai_response[:100]}...")
            
            return {"session_id": session_id, "response": ai_response}
            
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/sessions")
async def get_sessions():
    """Retrieve a list of all chat sessions"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Get all sessions with their first user message
            cursor.execute('''
                SELECT s.session_id, s.created_at, m.content as first_message
                FROM sessions s
                LEFT JOIN (
                    SELECT session_id, content, MIN(timestamp) as first_timestamp
                    FROM messages
                    WHERE role = 'user'
                    GROUP BY session_id
                ) m ON s.session_id = m.session_id
                ORDER BY s.last_activity DESC
                LIMIT ?
            ''', (MAX_SESSIONS,))
            
            sessions = []
            for row in cursor.fetchall():
                session_id = row["session_id"]
                messages = [{"role": "user", "content": row["first_message"]}] if row["first_message"] else []
                title = generate_session_title(messages) if messages else "Untitled Session"
                
                # Get the most recent message as preview
                cursor.execute(
                    "SELECT content FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT 1",
                    (session_id,)
                )
                latest = cursor.fetchone()
                preview = latest["content"] if latest else ""
                
                sessions.append({
                    "session_id": session_id,
                    "title": title,
                    "preview": preview[:100] + "..." if len(preview) > 100 else preview,
                    "created_at": row["created_at"]
                })
                
            return {"sessions": sessions}
    except Exception as e:
        logging.error(f"Error in get_sessions endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Retrieve the full conversation for a specific session"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Check if session exists
            cursor.execute("SELECT session_id, created_at FROM sessions WHERE session_id = ?", (session_id,))
            session = cursor.fetchone()
            
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Get all messages for the session
            cursor.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            )
            
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for row in cursor.fetchall():
                messages.append({
                    "role": row["role"],
                    "content": row["content"]
                })
            
            return {
                "session_id": session_id,
                "created_at": session["created_at"],
                "messages": messages
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_session endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session and all its messages"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Check if session exists
            cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Delete all messages for the session
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            # Delete the session
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            
            return {"status": "success", "message": "Session deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in delete_session endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get usage statistics"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Get total sessions count
            cursor.execute("SELECT COUNT(*) as session_count FROM sessions")
            session_count = cursor.fetchone()["session_count"]
            
            # Get total message count
            cursor.execute("SELECT COUNT(*) as message_count FROM messages")
            message_count = cursor.fetchone()["message_count"]
            
            # Get average messages per session
            avg_messages = message_count / session_count if session_count > 0 else 0
            
            # Get sessions created in the last 24 hours
            cursor.execute(
                "SELECT COUNT(*) as recent_sessions FROM sessions WHERE created_at > ?",
                ((datetime.datetime.now() - datetime.timedelta(days=1)).isoformat(),)
            )
            recent_sessions = cursor.fetchone()["recent_sessions"]
            
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
