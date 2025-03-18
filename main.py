import os
import openai
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for Wix frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your specific Wix site domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    if not openai.api_key:
        return {"response": "OpenAI API key is missing. Please check your environment variables."}

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a successful CEO giving business advice."},
                {"role": "user", "content": request.message}
            ]
        )
        return {"response": response['choices'][0]['message']['content']}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}
