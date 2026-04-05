import time
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from starlette.requests import Request
from app.core.rate_limiter import limiter
from app.db.session import get_db
from app.db.models import User, Conversation, ChatMessage
from app.models.request import ChatRequest
from app.models.response import (
    ChatResponse, SourceChunk,
    ConversationListItem, ConversationDetail
)
from app.core.security import get_current_user
from app.nlp.retrieval import Retriever
from app.nlp.generator import Generator

router = APIRouter(prefix="/chat", tags=["Chat"])

retriever = Retriever()
generator = Generator()


@router.post("/ask", response_model=ChatResponse)
@limiter.limit("10/minute")
def ask_question(
    request: Request,
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start_time = time.time()

    # Retrieve context
    result = retriever.get_answer_with_context(
        question=body.question,
        n_results=body.n_results
    )

    # Generate answer using LLM
    answer = generator.generate_answer(
        question=body.question,
        retrieved_chunks=result["all_results"]
    )

    elapsed = round(time.time() - start_time, 3)

    # If conversation_id provided, use existing conversation; otherwise create new one
    if body.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == body.conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Create new conversation with first question as title
        title = body.question[:100]
        conversation = Conversation(user_id=current_user.id, title=title)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # Save message to DB
    context_text = "\n---\n".join([r["text"] for r in result["all_results"]])
    message = ChatMessage(
        conversation_id=conversation.id,
        user_query=body.question,
        retrieved_context=context_text,
        llm_response=answer,
        response_time=elapsed
    )
    db.add(message)
    db.commit()

    # Build sources
    sources = []
    for r in result["all_results"]:
        sources.append(SourceChunk(
            chunk_id=r["chunk_id"],
            text=r["text"],
            distance=r["distance"],
            category=r["metadata"].get("issue_area", "")
        ))

    return ChatResponse(
        answer=answer,
        confidence=result["confidence"],
        matched_question=result["matched_question"],
        category=result["category"],
        sources=sources,
        conversation_id=conversation.id
    )


# GET all conversations for the logged-in user
@router.get("/history", response_model=list[ConversationListItem])
def get_all_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(Conversation.updated_at.desc()).all()

    return conversations


# GET one conversation with all its messages
@router.get("/history/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation
