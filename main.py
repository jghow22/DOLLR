import os
import openai
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logging.info(f"Using openai package version: {openai.__version__}")

# Attempt to unwrap ChatCompletion if it is wrapped by APIRemovedInV1Proxy.
try:
    ChatCompletion = openai.ChatCompletion.__wrapped__
    logging.info("Successfully unwrapped ChatCompletion.")
except AttributeError:
    ChatCompletion = openai.ChatCompletion
    logging.info("Could not unwrap ChatCompletion; using as is.")

# Configure CORS (update allow_origins for production).
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

class ChatRequest(BaseModel):
    message: str

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
        logging.info(f"Received message: {request.message}")
        logging.info("Using method: " + str(ChatCompletion.acreate))
        # Call the async API (using the unwrapped ChatCompletion if available)
        response = await ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message}
            ]
        )
        ai_response = response["choices"][0]["message"]["content"]
        logging.info("Response sent")
        return {"response": ai_response}
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
