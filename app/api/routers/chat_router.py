from fastapi import APIRouter

from app.schemas import ServiceRequest, ServiceResponse
from app.services.chatbot_service import chatbot

router = APIRouter()


@router.post("/chat", status_code=200, description="", response_model=ServiceResponse)
async def start_process(req: ServiceRequest):
    return chatbot.invoke(req)


@router.get("/info", status_code=200, description="", response_model=ServiceResponse)
async def get_info():
    return chatbot.get_info()
