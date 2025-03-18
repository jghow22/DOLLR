import os
import openai
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import ChatCompletion  # Import ChatCompletion directly

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logging.info(f"Using openai package version: {openai.__version__}")

# Configure CORS: temporarily allow all origins for testing purposes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing; update to your specific domain for production.
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Set your OpenAI API key from environment variables.
openai.api_key = os.getenv("OPENAI_API_KEY")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a successful CEO giving business advice.")

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
async def health():
    return {"status": "ok"}

# Explicitly handle OPTIONS requests for /chat.
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
        logging.info(f"Received message: {request.message}")
        # Log the acreate method from the directly imported ChatCompletion
        logging.info("Using method: " + str(ChatCompletion.acreate))
        # Use the new async interface via the directly imported ChatCompletion.
        response = await ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message}
            ]
        )
        ai_response = response['choices'][0]['message']['content']
        logging.info("Response sent")
        return {"response": ai_response}
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
