import time
import logging
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

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
from app.nlp.transcriber import Transcriber
from app.mlops.tracker import tracker
from app.mlops.metrics import (
    chat_requests_total, chat_response_seconds, retrieval_confidence
)

router = APIRouter(prefix="/chat", tags=["Chat"])

retriever = Retriever()
generator = Generator()
transcriber = Transcriber()


@router.post("/ask", response_model=ChatResponse)
@limiter.limit("10/minute")
def ask_question(
    request: Request,
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    logger.info(f"Chat query started: user_id={current_user.id}, conversation_id={body.conversation_id}, question_len={len(body.question)}")

    if not body.question.strip():
        logger.warning(f"Chat query failed: empty question, user_id={current_user.id}")
        chat_requests_total.labels(status="error").inc()
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start_time = time.time()
    retrieval_start = time.time()

    # Retrieve context
    result = retriever.get_answer_with_context(
        question=body.question,
        n_results=body.n_results
    )

    retrieval_time = round((time.time() - retrieval_start) * 1000, 2)
    logger.info(f"Retrieval completed: chunks_found={len(result['all_results'])}, confidence={result['confidence']}, duration_ms={retrieval_time}")

    gen_start = time.time()
    # Generate answer using LLM
    try:
        answer = generator.generate_answer(
            question=body.question,
            retrieved_chunks=result["all_results"]
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {str(e)}", extra={
            "user_id":current_user.id,
            "question_len": len(body.question)
        }, exc_info=True)
        chat_requests_total.labels(status="error").inc()
        raise HTTPException(status_code=500, detail="Failed to generate answer")

    gen_time = round((time.time() - gen_start) * 1000, 2)
    logger.info(f"LLM generation completed: answer_len={len(answer)}, duration_ms={gen_time}")

    elapsed = round(time.time() - start_time, 3)

    # If conversation_id provided, use existing conversation; otherwise create new one
    if body.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == body.conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        if not conversation:
            logger.warning(f"Chat query failed: conversation not found, conversation_id={body.conversation_id}, user_id={current_user.id}")
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Create new conversation with first question as title
        title = body.question[:100]
        conversation = Conversation(user_id=current_user.id, title=title)
        db.add(conversation)
        try:
            db.commit()
            db.refresh(conversation)
            logger.info(f"New conversation created: conversation_id={conversation.id}, user_id={current_user.id}")
        except Exception as e:
            logger.error(f"Failed to create conversation: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to create conversation")

    # Save message to DB
    db_start = time.time()
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

    db_time = round((time.time() - db_start) * 1000, 2)
    logger.info(f"Chat message saved: conversation_id={conversation.id}, duration_ms={db_time}")

    # Build sources
    sources = []
    for r in result["all_results"]:
        sources.append(SourceChunk(
            chunk_id=r["chunk_id"],
            text=r["text"],
            distance=r["distance"],
            category=r["metadata"].get("issue_area", "")
        ))

    # --- Log to MLflow ---
    tracker.log_chat_query(
        question=body.question,
        answer=answer,
        confidence=result["confidence"],
        response_time=elapsed,
        n_results=body.n_results,
        sources_count=len(sources),
        category=result["category"],
        user_id=current_user.id,
        conversation_id=conversation.id
    )

    # --- Record Prometheus metrics ---
    chat_requests_total.labels(status="success").inc()
    chat_response_seconds.observe(elapsed)
    retrieval_confidence.observe(result["confidence"])

    logger.info(f"Chat query completed: user_id={current_user.id}, conversation_id={conversation.id}, total_duration_ms={elapsed}")

    return ChatResponse(
        answer=answer,
        confidence=result["confidence"],
        matched_question=result["matched_question"],
        category=result["category"],
        sources=sources,
        conversation_id=conversation.id
    )

@router.post("/voice")
@limiter.limit("5/minute")
async def voice_query(
    request: Request,
    audio: UploadFile = File(...),
    conversation_id: int = Form(0),
    n_results: int = Form(3),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Voice query started: user_id={current_user.id}, filename={audio.filename}, content_type={audio.content_type}")

    # Read audio file
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Transcribe with Whisper
    try:
        transcription = transcriber.transcribe(audio_bytes, filename=audio.filename)
    except Exception as e:
        logger.error(f"Voice transcription failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to transcribe audio")

    question = transcription["text"]
    if not question.strip():
        raise HTTPException(status_code=400, detail="Could not understand audio")

    logger.info(f"Voice transcribed: text='{question[:100]}', user_id={current_user.id}")

    # From here, same logic as /chat/ask
    start_time = time.time()

    result = retriever.get_answer_with_context(question=question, n_results=n_results)

    try:
        answer = generator.generate_answer(question=question, retrieved_chunks=result["all_results"])
    except Exception as e:
        logger.error(f"LLM generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate answer")

    elapsed = round(time.time() - start_time, 3)

    # Handle conversation
    conv_id = conversation_id if conversation_id != 0 else None
    if conv_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == conv_id,
            Conversation.user_id == current_user.id
        ).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation = Conversation(user_id=current_user.id, title=question[:100])
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # Save message
    context_text = "\n---\n".join([r["text"] for r in result["all_results"]])
    message = ChatMessage(
        conversation_id=conversation.id,
        user_query=question,
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

    # Log to MLflow
    tracker.log_chat_query(
        question=question, answer=answer,
        confidence=result["confidence"], response_time=elapsed,
        n_results=n_results, sources_count=len(sources),
        category=result["category"], user_id=current_user.id,
        conversation_id=conversation.id
    )

    # Prometheus metrics
    chat_requests_total.labels(status="success").inc()
    chat_response_seconds.observe(elapsed)
    retrieval_confidence.observe(result["confidence"])

    return {
        "transcription": question,
        "answer": answer,
        "confidence": result["confidence"],
        "matched_question": result["matched_question"],
        "category": result["category"],
        "sources": sources,
        "conversation_id": conversation.id
    }



# GET all conversations for the logged-in user
@router.get("/history", response_model=list[ConversationListItem])
def get_all_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(Conversation.updated_at.desc()).all()

    logger.info(f"Retrieved conversation history: user_id={current_user.id}, count={len(conversations)}")

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
        logger.warning(f"Conversation not found: conversation_id={conversation_id}, user_id={current_user.id}")
        raise HTTPException(status_code=404, detail="Conversation not found")

    logger.info(f"Retrieved conversation: conversation_id={conversation_id}, user_id={current_user.id}, message_count={len(conversation.messages)}")

    return conversation
