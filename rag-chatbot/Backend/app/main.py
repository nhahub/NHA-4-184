import os
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.core.rate_limiter import limiter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from app.middleware.logging import LoggingMiddleware
from app.core.logging_config import setup_logging
from app.db.session import engine
from app.db.models import Base
from app.api import auth, chat, feedback
from fastapi.responses import Response
from app.mlops.metrics import get_metrics, get_metrics_content_type

# Create tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="RAG Customer Support Chatbot",
    description="AI-powered support using Retrieval-Augmented Generation",
    version="1.0.0"
)

environment = os.getenv("ENVIRONMENT", "development")
setup_logging(environment)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "fallback-secret-key-change-this"),
    same_site="lax",
    https_only=False,
    max_age=3600
)

app.add_middleware(LoggingMiddleware)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(feedback.router)

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "RAG Chatbot API is running"}

@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics endpoint — exposes all app metrics."""
    return Response(
        content=get_metrics(),
        media_type=get_metrics_content_type()
    )