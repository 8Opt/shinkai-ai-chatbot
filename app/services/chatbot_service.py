from app.schemas import ServiceRequest, ServiceResponse
from app.common import setup_logger
from app.services.base import BaseService

logger = setup_logger("[AI_SERVICE]")


class ChatbotService(BaseService):
    def invoke(self, data: ServiceRequest) -> ServiceResponse:
        pass
