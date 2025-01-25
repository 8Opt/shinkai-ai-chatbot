from fastapi import FastAPI

from app.api import api_router
from app.core.setup_lifspan import lifespan


app = FastAPI(
    title="Shinkai's Chatbot",
    docs_url="/",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api/v1")
