import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수에서 API 키를 가져옵니다 (보안 유지)
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
nvidia_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_question = request.message.strip()
    if not user_question:
        raise HTTPException(status_code=400, detail="메시지를 입력해주세요.")

    try:
        response = await nvidia_client.chat.completions.create(
            model="z-ai/glm-5.2",
            messages=[
                {
                    "role": "system", 
                    "content": "당신은 웹사이트에서 사용자를 보좌하는 다정하고 싹싹한 'AI 개인 비서'입니다. 어미는 항상 '~요', '~죠', '~해 드릴게요 😊'와 같이 부드럽게 끝마치세요."
                },
                {"role": "user", "content": user_question}
            ],
            temperature=0.82, 
            max_tokens=1500
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))