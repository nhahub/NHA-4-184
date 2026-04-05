import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from app.db.session import engine
from app.db.models import Base
from app.api import auth, chat, feedback

# Create tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="RAG Customer Support Chatbot",
    description="AI-powered support using Retrieval-Augmented Generation",
    version="1.0.0"
)

app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "your-secret-key"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(feedback.router)

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "RAG Chatbot API is running 🚀"}