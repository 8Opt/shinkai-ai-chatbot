from fastapi import APIRouter

from app.schemas import ServiceResponse, ServiceRequest
from app.services.chatbot_service import Pipeline

router = APIRouter()


@router.post("/invoke", status_code=200, description="", response_model=ServiceResponse)
async def start_process(req: ServiceRequest):
    return Pipeline().invoke(req)


@router.get("/info", status_code=200, description="", response_model=ServiceResponse)
async def get_info():
    return Pipeline().get_info()
