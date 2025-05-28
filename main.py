import os
import openai
import asyncio
import logging
import uuid
import datetime
import base64
import io
import json
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from typing import Optional, List, Dict, Any, Union
from contextlib import contextmanager
from pydantic import BaseModel, Field, EmailStr
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import sqlite3
import hashlib
import secrets
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Google Sheets Database Class
class GoogleSheetsDB:
    def __init__(self, credentials_json: str, spreadsheet_name: str):
        """Initialize the Google Sheets database connection.
        
        Args:
            credentials_json: The service account credentials JSON as a string
            spreadsheet_name: Name of the Google Sheet to use as database
        """
        self.scope = ['https://spreadsheets.google.com/feeds',
                      'https://www.googleapis.com/auth/drive']
        
        try:
            # Authenticate with Google using the JSON string
            credentials_dict = json.loads(credentials_json)
            credentials = ServiceAccountCredentials.from_json_keyfile_dict(
                credentials_dict, self.scope
            )
            self.client = gspread.authorize(credentials)
            
            # Open the spreadsheet
            try:
                self.spreadsheet = self.client.open(spreadsheet_name)
                logging.info(f"Successfully connected to Google Sheet: {spreadsheet_name}")
            except gspread.exceptions.SpreadsheetNotFound:
                # Create the spreadsheet if it doesn't exist
                self.spreadsheet = self.client.create(spreadsheet_name)
                logging.info(f"Created new Google Sheet: {spreadsheet_name}")
                
                # Initialize the sheets structure
                self._initialize_sheets()
        except Exception as e:
            logging.error(f"Error connecting to Google Sheets: {str(e)}")
            raise
    
    def _initialize_sheets(self):
        """Create the initial structure of the spreadsheet with required sheets and headers."""
        # Setup conversations sheet
        try:
            conversations_sheet = self.spreadsheet.add_worksheet(title="conversations", rows=1000, cols=6)
            conversations_sheet.append_row([
                "id", "session_id", "timestamp", "role", "content", "model"
            ])
            logging.info("Created conversations sheet")
        except Exception as e:
            logging.warning(f"Could not create conversations sheet: {str(e)}")
        
        # Setup sessions sheet
        try:
            sessions_sheet = self.spreadsheet.add_worksheet(title="sessions", rows=1000, cols=5)
            sessions_sheet.append_row([
                "session_id", "created_at", "last_activity", "title", "industry"
            ])
            logging.info("Created sessions sheet")
        except Exception as e:
            logging.warning(f"Could not create sessions sheet: {str(e)}")
        
        # Setup users sheet
        try:
            users_sheet = self.spreadsheet.add_worksheet(title="users", rows=1000, cols=5)
            users_sheet.append_row([
                "user_id", "email", "name", "created_at", "last_login"
            ])
            logging.info("Created users sheet")
        except Exception as e:
            logging.warning(f"Could not create users sheet: {str(e)}")
    
    def _get_sheet(self, sheet_name: str):
        """Get a worksheet by name, creating it if it doesn't exist."""
        try:
            return self.spreadsheet.worksheet(sheet_name)
        except gspread.exceptions.WorksheetNotFound:
            # If the sheet doesn't exist, create it with appropriate headers
            if sheet_name == "conversations":
                sheet = self.spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=6)
                sheet.append_row(["id", "session_id", "timestamp", "role", "content", "model"])
            elif sheet_name == "sessions":
                sheet = self.spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=5)
                sheet.append_row(["session_id", "created_at", "last_activity", "title", "industry"])
            elif sheet_name == "users":
                sheet = self.spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=5)
                sheet.append_row(["user_id", "email", "name", "created_at", "last_login"])
            else:
                sheet = self.spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=10)
            return sheet
    
    # --- Conversation Methods ---
    
    def add_message(self, session_id: str, role: str, content: str, model: str = "gpt-4") -> str:
        """Add a new message to the conversations sheet.
        
        Args:
            session_id: The ID of the session this message belongs to
            role: Either 'user' or 'assistant'
            content: The message content
            model: The AI model used (for assistant messages)
            
        Returns:
            The message ID
        """
        sheet = self._get_sheet("conversations")
        timestamp = datetime.datetime.now().isoformat()
        message_id = f"msg_{timestamp.replace(':', '_')}_{session_id[-6:]}"
        
        # Update the sessions last_activity
        self.update_session_activity(session_id)
        
        # Add the new message
        sheet.append_row([
            message_id,
            session_id,
            timestamp,
            role,
            content,
            model if role == "assistant" else ""
        ])
        
        return message_id
    
    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a specific session."""
        sheet = self._get_sheet("conversations")
        
        # Get all rows
        all_records = sheet.get_all_records()
        
        # Filter for the session and sort by timestamp
        session_messages = [
            {"id": row["id"], 
             "role": row["role"], 
             "content": row["content"],
             "timestamp": row["timestamp"]}
            for row in all_records if row["session_id"] == session_id
        ]
        
        # Sort by timestamp
        session_messages.sort(key=lambda x: x["timestamp"])
        
        return session_messages
    
    # --- Session Methods ---
    
    def create_session(self, session_id: str, industry: Optional[str] = None) -> None:
        """Create a new chat session."""
        sheet = self._get_sheet("sessions")
        timestamp = datetime.datetime.now().isoformat()
        
        sheet.append_row([
            session_id,
            timestamp,  # created_at
            timestamp,  # last_activity
            "New Conversation",  # title
            industry or ""  # industry
        ])
    
    def update_session_activity(self, session_id: str) -> None:
        """Update the last_activity timestamp for a session."""
        sheet = self._get_sheet("sessions")
        
        # Find the row with this session_id
        cell = sheet.find(session_id)
        if cell:
            timestamp = datetime.datetime.now().isoformat()
            sheet.update_cell(cell.row, 3, timestamp)  # Column 3 is last_activity
    
    def update_session_title(self, session_id: str, title: str) -> None:
        """Update the title of a session."""
        sheet = self._get_sheet("sessions")
        
        # Find the row with this session_id
        cell = sheet.find(session_id)
        if cell:
            sheet.update_cell(cell.row, 4, title)  # Column 4 is title
    
    def get_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the most recent sessions, sorted by last_activity."""
        sheet = self._get_sheet("sessions")
        
        # Get all rows
        all_records = sheet.get_all_records()
        
        # Sort by last_activity (newest first)
        sorted_records = sorted(
            all_records, 
            key=lambda x: x["last_activity"] if x["last_activity"] else "", 
            reverse=True
        )
        
        return sorted_records[:limit]
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific session by ID."""
        sheet = self._get_sheet("sessions")
        
        # Get all rows
        all_records = sheet.get_all_records()
        
        # Find the matching session
        for record in all_records:
            if record["session_id"] == session_id:
                return record
        
        return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        # First delete the session record
        sessions_sheet = self._get_sheet("sessions")
        cell = sessions_sheet.find(session_id)
        if cell:
            sessions_sheet.delete_row(cell.row)
            
            # Now delete all messages for this session
            messages_sheet = self._get_sheet("conversations")
            rows_to_delete = []
            
            # Find all messages with this session_id
            session_id_cells = messages_sheet.findall(session_id)
            for cell in session_id_cells:
                # Only delete if it's in the session_id column (column 2)
                if cell.col == 2:
                    rows_to_delete.append(cell.row)
            
            # Delete rows in reverse order to avoid changing indices
            for row in sorted(rows_to_delete, reverse=True):
                messages_sheet.delete_row(row)
            
            return True
        
        return False
    
    # --- User Methods ---
    
    def add_user(self, user_id: str, email: str, name: str) -> None:
        """Add a new user to the users sheet."""
        sheet = self._get_sheet("users")
        timestamp = datetime.datetime.now().isoformat()
        
        sheet.append_row([
            user_id,
            email,
            name,
            timestamp,  # created_at
            timestamp   # last_login
        ])
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Find a user by email address."""
        sheet = self._get_sheet("users")
        
        # Get all rows
        all_records = sheet.get_all_records()
        
        # Find the matching user
        for record in all_records:
            if record["email"] == email:
                return record
        
        return None
    
    def update_user_login(self, user_id: str) -> None:
        """Update the last_login timestamp for a user."""
        sheet = self._get_sheet("users")
        
        # Find the row with this user_id
        cell = sheet.find(user_id)
        if cell:
            timestamp = datetime.datetime.now().isoformat()
            sheet.update_cell(cell.row, 5, timestamp)  # Column 5 is last_login

# Initialize FastAPI app
app = FastAPI(title="AI CEO Business Consultant API")

# Setup logging
logging.basicConfig(level=logging.INFO)
logging.info(f"Using openai package version: {openai.__version__}")

# Configure CORS for testing (update allowed origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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

# Industry-specific prompts
INDUSTRY_PROMPTS = {
    "tech": """
You are a technology industry expert consultant with deep knowledge of SaaS, marketplaces, mobile apps, enterprise software, and consumer tech. You understand tech business models, pricing strategies, user acquisition, and product-market fit for tech companies.
Your specialized guidance covers:
- SaaS metrics (CAC, LTV, churn, MRR/ARR)
- Tech startup fundraising (pre-seed to Series C+)
- Growth hacking and user acquisition strategies
- Product development frameworks
- Tech team hiring and management
- Tech platform selection and scaling
""",
    "healthcare": """
You are a healthcare industry consultant with expertise in healthcare delivery, medical devices, pharmaceuticals, digital health, and healthcare regulations. You understand reimbursement models, clinical workflows, and healthcare compliance.
Your specialized guidance covers:
- Healthcare business models and revenue cycle management
- Regulatory pathways (FDA, HIPAA, etc.)
- Healthcare market access strategies
- Patient acquisition and engagement
- Healthcare operations optimization
- Value-based care implementation
""",
    "retail": """
You are a retail business consultant with expertise in both online and brick-and-mortar retail operations. You understand inventory management, merchandising, retail analytics, customer experience, and omnichannel strategies.
Your specialized guidance covers:
- Retail pricing and promotion strategies
- Store operations and layout optimization
- Inventory management and logistics
- Retail customer analytics and personalization
- E-commerce and omnichannel integration
- Retail staffing and customer service excellence
""",
    "finance": """
You are a financial services industry consultant with expertise in banking, investments, insurance, fintech, and financial regulations. You understand financial product development, risk management, and compliance requirements.
Your specialized guidance covers:
- Financial product design and pricing
- Risk assessment and management frameworks
- Financial regulatory compliance
- Customer acquisition in financial services
- Banking operations and efficiency
- Fintech innovation and implementation
""",
    "manufacturing": """
You are a manufacturing industry consultant with expertise in production processes, supply chain management, quality control, and lean manufacturing. You understand production economics, inventory management, and manufacturing technology.
Your specialized guidance covers:
- Manufacturing process optimization
- Supply chain resilience and management
- Quality control systems and certifications
- Cost reduction strategies in manufacturing
- Factory layout and workflow design
- Manufacturing technology implementation
"""
}

# Global in-memory dictionary to store session history (fallback if Google Sheets fails)
session_history: Dict[str, Dict[str, Any]] = {}
# Global in-memory dictionary to store user data (fallback if Google Sheets fails)
users_db: Dict[str, Dict[str, Any]] = {}
# Global in-memory token storage
tokens_db: Dict[str, str] = {}

# Initialize Google Sheets DB
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "dollrDB1")

# Initialize sheets_db as None first (will be set if credentials exist)
sheets_db = None

if GOOGLE_CREDS_JSON:
    try:
        sheets_db = GoogleSheetsDB(GOOGLE_CREDS_JSON, GOOGLE_SHEET_NAME)
        logging.info("Google Sheets database initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Google Sheets database: {str(e)}")
        # sheets_db remains None, will fall back to in-memory storage
else:
    logging.warning("GOOGLE_CREDS_JSON environment variable not set. Using in-memory storage instead.")

# Database management (fallback SQLite in-memory)
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    # Use memory by default for compatibility with Render
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
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT UNIQUE,
                password_hash TEXT,
                name TEXT,
                created_at TEXT
            )
        ''')
        conn.commit()

# Request/response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    industry: Optional[str] = None

class SessionAnalysisResponse(BaseModel):
    opportunities: List[str]
    challenges: List[str]
    actions: List[str]

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class CompetitorRequest(BaseModel):
    company_name: str
    industry: str
    specific_questions: Optional[List[str]] = None

class ContentRequest(BaseModel):
    content_type: str  # "email", "social", "blog", "presentation"
    topic: str
    target_audience: str
    tone: str = "professional"
    length: str = "medium"  # "short", "medium", "long"

class FinancialData(BaseModel):
    data_type: str  # "cash_flow", "profit_loss", "balance_sheet", "metrics"
    data: Dict[str, Union[float, List[float]]]
    labels: Optional[List[str]] = None
    title: str
    description: Optional[str] = None

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

def extract_insights(messages: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """
    Analyze conversation and extract business insights.
    Returns opportunities, challenges, and recommended actions.
    """
    # Combine all assistant messages to analyze
    assistant_content = ""
    for msg in messages:
        if msg["role"] == "assistant":
            assistant_content += msg["content"] + "\n\n"
    
    # Simple pattern matching for insights
    opportunities = re.findall(r'opportunity|potential|growth|advantage|chance', 
                               assistant_content, re.IGNORECASE)
    challenges = re.findall(r'challenge|problem|issue|difficulty|obstacle|risk', 
                           assistant_content, re.IGNORECASE)
    actions = re.findall(r'should|recommend|suggest|implement|step|action|plan', 
                        assistant_content, re.IGNORECASE)
    
    # If we don't find enough insights with simple pattern matching,
    # we can use the OpenAI API to generate them
    if len(opportunities) < 2 or len(challenges) < 2 or len(actions) < 2:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use a smaller model for this task
                messages=[
                    {"role": "system", "content": "You are a business analysis AI. Extract business insights from the text."},
                    {"role": "user", "content": f"Analyze this business conversation and extract: 1) Key business opportunities, 2) Business challenges, and 3) Recommended actions. Format as JSON with 'opportunities', 'challenges', and 'actions' arrays.\n\n{assistant_content}"}
                ],
                temperature=0.3
            )
            result = response.choices[0].message.content
            try:
                # Try to extract JSON from the response
                json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
                if json_match:
                    result = json_match.group(1)
                insights = json.loads(result)
                if isinstance(insights, dict):
                    return {
                        "opportunities": insights.get("opportunities", [])[:5],
                        "challenges": insights.get("challenges", [])[:5],
                        "actions": insights.get("actions", [])[:5]
                    }
            except (json.JSONDecodeError, AttributeError):
                pass
        except Exception as e:
            logging.error(f"Error extracting insights with OpenAI: {str(e)}")
    
    # Fallback to our simple pattern matching
    return {
        "opportunities": opportunities[:5],
        "challenges": challenges[:5],
        "actions": actions[:5]
    }

def generate_summary(messages: List[Dict[str, str]]) -> str:
    """Generate a summary of the conversation."""
    # Extract user questions and core topics
    user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
    if not user_messages:
        return "No conversation data available."
    
    # Simple approach: use first user message as basis for summary
    first_msg = user_messages[0]
    if len(first_msg) > 150:
        return first_msg[:150] + "..."
    
    # More advanced: try to use OpenAI to generate a summary
    try:
        all_content = "\n".join(user_messages)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Summarize this business conversation in 2-3 sentences."},
                {"role": "user", "content": all_content}
            ],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating summary: {str(e)}")
        # Fallback to simple approach
        return f"Conversation about: {first_msg[:100]}..."

def get_system_prompt(industry: Optional[str] = None) -> str:
    """Get the appropriate system prompt based on industry."""
    if industry and industry.lower() in INDUSTRY_PROMPTS:
        return INDUSTRY_PROMPTS[industry.lower()]
    return SYSTEM_PROMPT

# Simple security
security = HTTPBearer()
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token not in tokens_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return tokens_db[token]

# Initialize the database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    logging.info("In-memory SQLite database initialized")

# Endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    db_status = "google_sheets" if sheets_db else "in_memory"
    return {"status": "ok", "model": MODEL, "database": db_status}

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
        current_time = datetime.datetime.now().isoformat()
        
        # If sheets_db is available, use it; otherwise fall back to in-memory
        if sheets_db:
            if session_id is None or not sheets_db.get_session(session_id):
                # Create a new session
                session_id = str(uuid.uuid4())
                # Get appropriate system prompt
                system_prompt = get_system_prompt(request.industry)
                
                # Create the session in Google Sheets
                sheets_db.create_session(session_id, request.industry)
                
                # Add the system message
                sheets_db.add_message(session_id, "system", system_prompt)
                
            # Add the user message to Google Sheets
            sheets_db.add_message(session_id, "user", request.message)
            
            # Retrieve all messages for this session to build context
            messages = sheets_db.get_session_messages(session_id)
            conversation = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            
        else:
            # Fallback to in-memory storage
            if session_id is None or session_id not in session_history:
                session_id = str(uuid.uuid4())
                # Get appropriate system prompt
                system_prompt = get_system_prompt(request.industry)
                session_history[session_id] = {
                    "messages": [{"role": "system", "content": system_prompt}],
                    "created_at": current_time,
                    "industry": request.industry
                }
            
            # Append the user's message.
            session_history[session_id]["messages"].append({"role": "user", "content": request.message})
            
            # Use the entire conversation history as context.
            conversation = session_history[session_id]["messages"]
        
        logging.info(f"Session {session_id} - Received message: {request.message}")
        
        # Call the OpenAI API
        response = await asyncio.to_thread(
            lambda: openai.ChatCompletion.create(
                model=MODEL,
                messages=conversation,
                temperature=DEFAULT_TEMPERATURE
            )
        )
        
        ai_response = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Store the assistant's response
        if sheets_db:
            sheets_db.add_message(session_id, "assistant", ai_response, MODEL)
            
            # Generate and update the session title if it's a new session
            session = sheets_db.get_session(session_id)
            if session and session["title"] == "New Conversation":
                # Use the first user message as the title
                title = request.message[:50] + ("..." if len(request.message) > 50 else "")
                sheets_db.update_session_title(session_id, title)
        else:
            # Fallback to in-memory storage
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
        if sheets_db:
            # Get sessions from Google Sheets
            sheet_sessions = sheets_db.get_sessions(MAX_SESSIONS)
            
            sessions = []
            for session in sheet_sessions:
                # Get the first user message for preview
                messages = sheets_db.get_session_messages(session["session_id"])
                preview = ""
                for msg in messages:
                    if msg["role"] == "user":
                        preview = msg["content"]
                        break
                
                sessions.append({
                    "session_id": session["session_id"],
                    "title": session["title"] or "Untitled Session",
                    "preview": preview[:100] + "..." if len(preview) > 100 else preview,
                    "created_at": session["created_at"],
                    "industry": session["industry"]
                })
            
            return {"sessions": sessions}
        else:
            # Fallback to in-memory storage
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
                industry = data.get("industry", "")
                sessions.append({
                    "session_id": session_id,
                    "title": title,
                    "preview": preview[:100] + "..." if len(preview) > 100 else preview,
                    "created_at": created_at,
                    "industry": industry
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
    try:
        if sheets_db:
            # Get session data from Google Sheets
            session = sheets_db.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Get messages for this session
            messages = sheets_db.get_session_messages(session_id)
            conversation = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            
            return {
                "session_id": session_id,
                "created_at": session.get("created_at", ""),
                "messages": conversation,
                "industry": session.get("industry", "")
            }
        else:
            # Fallback to in-memory storage
            if session_id in session_history:
                data = session_history[session_id]
                return {
                    "session_id": session_id,
                    "created_at": data.get("created_at", ""),
                    "messages": data.get("messages", []),
                    "industry": data.get("industry", "")
                }
            else:
                raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving session: {str(e)}")

@app.get("/session/{session_id}/analysis")
async def analyze_session(session_id: str):
    """Analyze a session and extract key insights"""
    try:
        if sheets_db:
            # Get session messages from Google Sheets
            session = sheets_db.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            messages = sheets_db.get_session_messages(session_id)
            conversation = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            insights = extract_insights(conversation)
            return SessionAnalysisResponse(**insights)
        else:
            # Fallback to in-memory storage
            if session_id in session_history:
                data = session_history[session_id]
                messages = data.get("messages", [])
                insights = extract_insights(messages)
                return SessionAnalysisResponse(**insights)
            else:
                raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error analyzing session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing session: {str(e)}")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    try:
        if sheets_db:
            # Delete session from Google Sheets
            success = sheets_db.delete_session(session_id)
            if success:
                return {"status": "success", "message": "Session deleted"}
            else:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            # Fallback to in-memory storage
            if session_id in session_history:
                del session_history[session_id]
                return {"status": "success", "message": "Session deleted"}
            else:
                raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@app.post("/session/{session_id}/export")
async def export_session(session_id: str):
    """Export a session as a formatted report"""
    try:
        messages = []
        if sheets_db:
            # Get messages from Google Sheets
            session = sheets_db.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            sheet_messages = sheets_db.get_session_messages(session_id)
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in sheet_messages]
        else:
            # Fallback to in-memory storage
            if session_id in session_history:
                data = session_history[session_id]
                messages = data.get("messages", [])
            else:
                raise HTTPException(status_code=404, detail="Session not found")
        
        # Generate a summary and extract key insights
        summary = generate_summary(messages)
        insights = extract_insights(messages)
        
        # Format the conversation
        conversation = []
        for msg in messages:
            if msg["role"] != "system":  # Skip the system prompt
                conversation.append({
                    "role": "You" if msg["role"] == "user" else "AI CEO",
                    "content": msg["content"]
                })
        
        # Create the report structure
        report = {
            "title": generate_session_title(messages),
            "date": datetime.datetime.now().strftime("%B %d, %Y"),
            "summary": summary,
            "insights": insights,
            "conversation": conversation
        }
        return report
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error exporting session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exporting session: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get usage statistics"""
    try:
        if sheets_db:
            # Get stats from Google Sheets
            all_sessions = sheets_db.get_sessions(1000)  # Get up to 1000 sessions
            session_count = len(all_sessions)
            
            # Count messages
            message_count = 0
            recent_sessions = 0
            industries = {}
            
            now = datetime.datetime.now()
            
            for session in all_sessions:
                # Add to industry breakdown
                industry = session.get("industry", "general")
                if not industry:
                    industry = "general"
                industries[industry] = industries.get(industry, 0) + 1
                
                # Check if session is recent (last 24 hours)
                try:
                    created_at = session.get("created_at", "")
                    if created_at:
                        created_datetime = datetime.datetime.fromisoformat(created_at)
                        if (now - created_datetime).total_seconds() < 86400:  # 24 hours in seconds
                            recent_sessions += 1
                except (ValueError, TypeError):
                    pass
                
                # Get messages for this session
                try:
                    messages = sheets_db.get_session_messages(session["session_id"])
                    message_count += len(messages)
                except Exception as e:
                    logging.error(f"Error getting messages for session {session['session_id']}: {str(e)}")
        else:
            # Fallback to in-memory storage
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
            
            # Calculate industry breakdown
            industries = {}
            for data in session_history.values():
                industry = data.get("industry", "general")
                if not industry:
                    industry = "general"
                industries[industry] = industries.get(industry, 0) + 1
        
        avg_messages = message_count / session_count if session_count > 0 else 0
        return {
            "total_sessions": session_count,
            "total_messages": message_count,
            "avg_messages_per_session": round(avg_messages, 2),
            "sessions_last_24h": recent_sessions,
            "industry_breakdown": industries,
            "model": MODEL,
            "database": "google_sheets" if sheets_db else "in_memory"
        }
    except Exception as e:
        logging.error(f"Error in stats endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/analyze/document")
async def analyze_document(file: UploadFile = File(...)):
    """Analyze a business document and provide insights"""
    try:
        contents = await file.read()
        file_extension = file.filename.split('.')[-1].lower()
        # Extract text based on file type
        text = ""
        if file_extension in ['txt', 'md']:
            text = contents.decode('utf-8')
        elif file_extension in ['pdf']:
            # For PDF, we would use a library like PyPDF2 or pdfminer
            # This is a placeholder
            text = "PDF content extraction would happen here."
        elif file_extension in ['docx']:
            # For DOCX, we would use python-docx
            # This is a placeholder
            text = "DOCX content extraction would happen here."
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        # Use OpenAI to analyze the document
        response = await asyncio.to_thread(
            lambda: openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a business document analyst. Analyze the following business document and provide insights."},
                    {"role": "user", "content": f"Please analyze this business document and provide key insights, opportunities, risks, and recommendations:\n\n{text[:5000]}"}  # Limit to first 5000 chars
                ],
                temperature=0.5
            )
        )
        analysis = response.choices[0].message.content
        return {
            "filename": file.filename,
            "analysis": analysis
        }
    except Exception as e:
        logging.error(f"Error analyzing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing document: {str(e)}")

@app.post("/research/competitor")
async def research_competitor(request: CompetitorRequest):
    """Research a competitor and provide analysis"""
    try:
        # Create a prompt for the competitor analysis
        prompt = f"""
        Please provide a detailed competitor analysis for '{request.company_name}' in the {request.industry} industry. 
        Include:
        1. Company overview and background
        2. Products/services
        3. Target market
        4. Strengths and weaknesses
        5. Market positioning
        6. Business model
        7. Key differentiators
        {f"Also address these specific questions: {', '.join(request.specific_questions)}" if request.specific_questions else ""}
        """
        
        # Use OpenAI to generate the analysis
        response = await asyncio.to_thread(
            lambda: openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a competitive intelligence expert. Provide detailed competitor analysis based on publicly available information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
        )
        analysis = response.choices[0].message.content
        return {
            "company": request.company_name,
            "industry": request.industry,
            "analysis": analysis
        }
    except Exception as e:
        logging.error(f"Error researching competitor: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error researching competitor: {str(e)}")

@app.post("/generate/content")
async def generate_content(request: ContentRequest):
    """Generate business content like marketing copy, emails, or presentations"""
    try:
        # Adjust instructions based on content type
        instructions = {
            "email": "Create a professional business email",
            "social": "Create engaging social media content",
            "blog": "Write a business blog post",
            "presentation": "Create an outline for a business presentation with slide content"
        }
        instruction = instructions.get(request.content_type, "Create professional business content")
        
        prompt = f"""
        {instruction} about {request.topic} for {request.target_audience}.
        Tone should be {request.tone}.
        Length: {request.length}.
        """
        
        # Use OpenAI to generate the content
        response = await asyncio.to_thread(
            lambda: openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert business content creator specializing in marketing, communications, and presentations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
        )
        content = response.choices[0].message.content
        return {
            "content_type": request.content_type,
            "topic": request.topic,
            "content": content
        }
    except Exception as e:
        logging.error(f"Error generating content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")

@app.post("/visualize/financial")
async def visualize_financial(data: FinancialData):
    """Generate charts and visualizations from financial data"""
    try:
        # Create a figure
        plt.figure(figsize=(10, 6))
        
        # Different visualization based on data type
        if data.data_type == "cash_flow":
            # For cash flow, we might want a line chart
            if isinstance(list(data.data.values())[0], list):
                # Multiple series
                for key, values in data.data.items():
                    plt.plot(data.labels or range(len(values)), values, label=key)
                plt.legend()
            else:
                # Single series
                plt.plot(data.labels or list(data.data.keys()), list(data.data.values()))
            plt.title(data.title)
            plt.ylabel("Amount")
            if data.labels:
                plt.xlabel("Period")
        elif data.data_type == "profit_loss":
            # For P&L, we might want a bar chart
            plt.bar(data.labels or list(data.data.keys()), list(data.data.values()))
            plt.title(data.title)
            plt.ylabel("Amount")
        elif data.data_type == "metrics":
            # For metrics, we might want a radar chart
            # This is simplified - a real implementation would be more complex
            categories = data.labels or list(data.data.keys())
            values = list(data.data.values())
            
            # Create a radar chart
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            values.append(values[0])
            angles.append(angles[0])
            categories.append(categories[0])
            
            plt.polar(angles, values)
            plt.fill(angles, values, alpha=0.3)
            plt.xticks(angles[:-1], categories[:-1])
            plt.title(data.title)
        else:
            # Default to a bar chart
            plt.bar(data.labels or list(data.data.keys()), list(data.data.values()))
            plt.title(data.title)
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to base64 for easy embedding
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return {
            "chart": f"data:image/png;base64,{img_str}",
            "title": data.title,
            "description": data.description or "Financial visualization"
        }
    except Exception as e:
        logging.error(f"Error generating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

# User management endpoints
@app.post("/user/register", response_model=Token)
async def register_user(user: UserCreate):
    """Register a new user"""
    try:
        # Check if email already exists
        if sheets_db:
            existing_user = sheets_db.get_user_by_email(user.email)
            if existing_user:
                raise HTTPException(status_code=400, detail="Email already registered")
            
            # Create user
            user_id = str(uuid.uuid4())
            hashed_password = hashlib.sha256(user.password.encode()).hexdigest()
            
            # Add user to Google Sheets
            sheets_db.add_user(user_id, user.email, user.name)
        else:
            # Fallback to in-memory storage
            if any(u.get("email") == user.email for u in users_db.values()):
                raise HTTPException(status_code=400, detail="Email already registered")
            
            # Create user
            user_id = str(uuid.uuid4())
            hashed_password = hashlib.sha256(user.password.encode()).hexdigest()
            users_db[user_id] = {
                "user_id": user_id,
                "email": user.email,
                "password_hash": hashed_password,
                "name": user.name,
                "created_at": datetime.datetime.now().isoformat()
            }
        
        # Create token
        token = secrets.token_hex(32)
        tokens_db[token] = user_id
        
        return {"access_token": token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error registering user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error registering user: {str(e)}")

@app.post("/user/login", response_model=Token)
async def login_user(user: UserLogin):
    """Login a user"""
    try:
        # Find user by email
        found_user = None
        found_user_id = None
        
        if sheets_db:
            found_user = sheets_db.get_user_by_email(user.email)
            if found_user:
                found_user_id = found_user.get("user_id")
        else:
            # Fallback to in-memory storage
            for user_id, u in users_db.items():
                if u.get("email") == user.email:
                    found_user = u
                    found_user_id = user_id
                    break
        
        if not found_user:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Check password
        hashed_password = hashlib.sha256(user.password.encode()).hexdigest()
        if found_user.get("password_hash") != hashed_password:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Update last login if using Google Sheets
        if sheets_db:
            sheets_db.update_user_login(found_user_id)
        
        # Create token
        token = secrets.token_hex(32)
        tokens_db[token] = found_user_id
        
        return {"access_token": token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error logging in user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error logging in user: {str(e)}")

@app.get("/user/me")
async def get_current_user_info(user_id: str = Depends(get_current_user)):
    """Get current user information"""
    try:
        if sheets_db:
            # This would require a method to get user by ID from Google Sheets
            # For now, we'll fallback to in-memory data
            if user_id not in users_db:
                raise HTTPException(status_code=404, detail="User not found")
            user_data = users_db[user_id].copy()
            # Remove sensitive information
            user_data.pop("password_hash", None)
            return user_data
        else:
            if user_id not in users_db:
                raise HTTPException(status_code=404, detail="User not found")
            user_data = users_db[user_id].copy()
            # Remove sensitive information
            user_data.pop("password_hash", None)
            return user_data
    except Exception as e:
        logging.error(f"Error getting user info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting user info: {str(e)}")

@app.get("/user/sessions")
async def get_user_sessions(user_id: str = Depends(get_current_user)):
    """Get all sessions for the current user"""
    try:
        # In a real implementation, we would filter sessions by user_id
        # For this demo, we'll just return all sessions
        
        if sheets_db:
            # Get sessions from Google Sheets
            sheet_sessions = sheets_db.get_sessions(MAX_SESSIONS)
            
            sessions = []
            for session in sheet_sessions:
                # Get the first user message for preview
                messages = sheets_db.get_session_messages(session["session_id"])
                preview = ""
                for msg in messages:
                    if msg["role"] == "user":
                        preview = msg["content"]
                        break
                
                sessions.append({
                    "session_id": session["session_id"],
                    "title": session["title"] or "Untitled Session",
                    "preview": preview[:100] + "..." if len(preview) > 100 else preview,
                    "created_at": session["created_at"],
                    "industry": session["industry"]
                })
            
            return {"sessions": sessions}
        else:
            # Fallback to in-memory storage
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
                industry = data.get("industry", "")
                sessions.append({
                    "session_id": session_id,
                    "title": title,
                    "preview": preview[:100] + "..." if len(preview) > 100 else preview,
                    "created_at": created_at,
                    "industry": industry
                })
            # Sort by created_at (newest first)
            sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return {"sessions": sessions[:MAX_SESSIONS]}
    except Exception as e:
        logging.error(f"Error in get_user_sessions endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Financial calculator endpoints
@app.post("/calculate/roi")
async def calculate_roi(
    initial_investment: float = Form(...),
    net_profit: float = Form(...),
    investment_period: Optional[int] = Form(1)  # Default to 1 year
):
    """Calculate Return on Investment (ROI)"""
    try:
        roi = (net_profit - initial_investment) / initial_investment * 100
        annual_roi = roi / investment_period
        return {
            "roi": round(roi, 2),
            "annual_roi": round(annual_roi, 2),
            "investment_period": investment_period,
            "initial_investment": initial_investment,
            "net_profit": net_profit
        }
    except ZeroDivisionError:
        raise HTTPException(status_code=400, detail="Initial investment cannot be zero")
    except Exception as e:
        logging.error(f"Error calculating ROI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating ROI: {str(e)}")

@app.post("/calculate/breakeven")
async def calculate_breakeven(
    fixed_costs: float = Form(...),
    price_per_unit: float = Form(...),
    variable_cost_per_unit: float = Form(...)
):
    """Calculate Break-Even Point"""
    try:
        contribution_margin = price_per_unit - variable_cost_per_unit
        if contribution_margin <= 0:
            raise HTTPException(
                status_code=400, 
                detail="Contribution margin must be positive (price must exceed variable cost)"
            )
        breakeven_units = fixed_costs / contribution_margin
        breakeven_revenue = breakeven_units * price_per_unit
        return {
            "breakeven_units": round(breakeven_units, 2),
            "breakeven_revenue": round(breakeven_revenue, 2),
            "contribution_margin": round(contribution_margin, 2),
            "fixed_costs": fixed_costs,
            "price_per_unit": price_per_unit,
            "variable_cost_per_unit": variable_cost_per_unit
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error calculating break-even point: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating break-even point: {str(e)}")

@app.post("/calculate/cash-burn")
async def calculate_cash_burn(
    starting_cash: float = Form(...),
    monthly_expenses: float = Form(...),
    monthly_revenue: float = Form(0)  # Default to zero for pre-revenue
):
    """Calculate Cash Burn Rate and Runway"""
    try:
        monthly_burn = monthly_expenses - monthly_revenue
        if monthly_burn <= 0:
            return {
                "cash_burn_rate": 0,
                "runway_months": float('inf'),
                "is_profitable": True,
                "monthly_profit": -monthly_burn
            }
        runway_months = starting_cash / monthly_burn
        return {
            "cash_burn_rate": round(monthly_burn, 2),
            "runway_months": round(runway_months, 2),
            "runway_years": round(runway_months / 12, 2),
            "is_profitable": False,
            "starting_cash": starting_cash,
            "monthly_expenses": monthly_expenses,
            "monthly_revenue": monthly_revenue
        }
    except ZeroDivisionError:
        return {
            "cash_burn_rate": 0,
            "runway_months": float('inf'),
            "is_profitable": True,
            "monthly_profit": monthly_revenue
        }
    except Exception as e:
        logging.error(f"Error calculating cash burn: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating cash burn: {str(e)}")
