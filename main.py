import os
import openai
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Configure CORS: temporarily allow all origins for testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to your specific domain for production.
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
        # Use the new async interface for chat completions.
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message}
            ]
        )
        # Access the response content.
        ai_response = response['choices'][0]['message']['content']
        logging.info("Response sent")
        return {"response": ai_response}
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
