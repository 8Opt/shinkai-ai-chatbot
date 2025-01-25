from abc import ABC, abstractmethod
from typing import Any
from app.core.config import settings

class BaseService(ABC): 

    @abstractmethod
    def invoke(self, payload: Any) -> Any: 
        raise NotImplementedError 
    


    def get_info(self):
        data_send = {"ip": settings.HOST, "port": str(settings.PORT), "data": {}}
        return data_send
