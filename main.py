import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
from supabase import create_client, Client

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수 설정
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

nvidia_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class RoomCreateRequest(BaseModel):
    user_id: str

class ChatRequest(BaseModel):
    room_id: str
    message: str

# 1. 사용자의 채팅방 목록 불러오기
@app.get("/api/rooms/{user_id}")
async def get_rooms(user_id: str):
    try:
        res = supabase.table("chat_rooms").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return {"rooms": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. 새 채팅방 생성
@app.post("/api/rooms")
async def create_room(req: RoomCreateRequest):
    try:
        res = supabase.table("chat_rooms").insert({"user_id": req.user_id, "title": "새로운 대화"}).execute()
        return {"room": res.data[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. 특정 채팅방의 메시지 내역 불러오기
@app.get("/api/messages/{room_id}")
async def get_messages(room_id: str):
    try:
        res = supabase.table("chat_messages").select("*").eq("room_id", room_id).order("created_at", desc=False).execute()
        return {"messages": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. 메시지 전송 및 AI 응답 처리 (이전 대화 맥락 포함)
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    room_id = req.room_id
    user_msg = req.message.strip()

    if not user_msg:
        raise HTTPException(status_code=400, detail="메시지를 입력해주세요.")

    try:
        # 사용자 메시지 DB 저장
        supabase.table("chat_messages").insert({"room_id": room_id, "sender": "user", "content": user_msg}).execute()

        # 과거 대화 내역 조회 (최근 10개 맥락 유지)
        history_res = supabase.table("chat_messages").select("sender, content").eq("room_id", room_id).order("created_at", desc=False).execute()
        
        formatted_messages = [
            {"role": "system", "content": "당신은 사용자를 보좌하는 다정하고 싹싹한 'AI 개인 비서'입니다. 어미는 항상 '~요', '~죠', '~해 드릴게요 😊'와 같이 정중하게 끝마치세요."}
        ]
        for msg in history_res.data[-10:]:
            role = "user" if msg["sender"] == "user" else "assistant"
            formatted_messages.append({"role": role, "content": msg["content"]})

        response = await nvidia_client.chat.completions.create(
            model="z-ai/glm-5.2",
            messages=formatted_messages,
            temperature=0.82,
            max_tokens=1500
        )
        ai_answer = response.choices[0].message.content

        # AI 답변 DB 저장
        supabase.table("chat_messages").insert({"room_id": room_id, "sender": "bot", "content": ai_answer}).execute()

        # 첫 질문일 경우 채팅방 제목 자동 업데이트
        if len(history_res.data) <= 1:
            new_title = user_msg[:15] + "..." if len(user_msg) > 15 else user_msg
            supabase.table("chat_rooms").update({"title": new_title}).eq("id", room_id).execute()

        return {"answer": ai_answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
