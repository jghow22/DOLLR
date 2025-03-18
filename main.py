from fastapi import FastAPI, Request
from pydantic import BaseModel
import openai
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all frontend origins (for Wix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API Key (replace with your actual key)
openai.api_key = "your-openai-api-key"

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a successful CEO giving business advice."},
                      {"role": "user", "content": request.message}]
        )
        return {"response": response['choices'][0]['message']['content']}
    except Exception as e:
        return {"response": "I'm currently unavailable. Please try again later."}
