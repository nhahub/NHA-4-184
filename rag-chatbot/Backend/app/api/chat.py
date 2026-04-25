import time
import logging
import json
import base64
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
from app.nlp.router import LLMRouter
from app.nlp.tts import TextToSpeech
from app.mlops.tracker import tracker
from app.mlops.metrics import (
    chat_requests_total, chat_response_seconds, retrieval_confidence
)

router = APIRouter(prefix="/chat", tags=["Chat"])

retriever = Retriever()
generator = Generator()
transcriber = Transcriber()
llm_router = LLMRouter()
tts = TextToSpeech()

# Confidence thresholds
HIGH_CONFIDENCE = 0.5
LOW_CONFIDENCE = 0.2


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

    # Step 1: Load or create conversation
    if body.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == body.conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        if not conversation:
            logger.warning(f"Chat query failed: conversation not found, conversation_id={body.conversation_id}, user_id={current_user.id}")
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
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

    # Step 2: Load conversation history
    conversation_history = []
    if body.conversation_id and conversation:
        previous_messages = db.query(ChatMessage).filter(
            ChatMessage.conversation_id == conversation.id
        ).order_by(ChatMessage.created_at.desc()).limit(6).all()
        previous_messages.reverse()
        for msg in previous_messages:
            conversation_history.append({"role": "user", "content": msg.user_query})
            conversation_history.append({"role": "assistant", "content": msg.llm_response})
        logger.info(f"Loaded conversation history: messages={len(previous_messages)}, conversation_id={conversation.id}")

    # Step 3: LLM Router — classify intent
    route = llm_router.classify(body.question)
    intent = route["intent"].lower()
    logger.info(f"Router decision: intent={intent}, user_id={current_user.id}")

    # ============================================
    # CASUAL PATH — Skip retrieval entirely
    # ============================================
    if intent == "casual":
        try:
            answer = generator.direct_answer(body.question, conversation_history)
        except Exception as e:
            logger.error(f"Direct answer failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate answer")

        elapsed = round(time.time() - start_time, 3)

        # Save message to DB
        message = ChatMessage(
            conversation_id=conversation.id,
            user_query=body.question,
            retrieved_context="[CASUAL - No retrieval needed]",
            llm_response=answer,
            response_time=elapsed
        )
        db.add(message)
        db.commit()

        chat_requests_total.labels(status="success").inc()
        chat_response_seconds.observe(elapsed)

        logger.info(f"Casual query completed: user_id={current_user.id}, pipeline=casual, duration={elapsed}")

        return ChatResponse(
            answer=answer,
            confidence=1.0,
            matched_question="",
            category="General",
            sources=[],
            conversation_id=conversation.id,
            intent="casual",
            pipeline="casual",
            verified=None
        )

    # ============================================
    # SUPPORT PATH — Full RAG pipeline
    # ============================================

    # Step 4: Rewrite vague follow-up questions
    search_query = body.question
    if conversation_history:
        search_query = generator.rewrite_query(body.question, conversation_history)

    # Step 5: Retrieve context from ChromaDB
    retrieval_start = time.time()
    result = retriever.get_answer_with_context(
        question=search_query,
        n_results=body.n_results
    )
    retrieval_time = round((time.time() - retrieval_start) * 1000, 2)
    logger.info(f"Retrieval completed: chunks_found={len(result['all_results'])}, confidence={result['confidence']}, duration_ms={retrieval_time}")

    # Step 6: Generate answer using LLM
    gen_start = time.time()
    try:
        answer = generator.generate_answer(
            question=body.question,
            retrieved_chunks=result["all_results"],
            conversation_history=conversation_history
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {str(e)}", exc_info=True)
        chat_requests_total.labels(status="error").inc()
        raise HTTPException(status_code=500, detail="Failed to generate answer")

    gen_time = round((time.time() - gen_start) * 1000, 2)
    logger.info(f"LLM generation completed: answer_len={len(answer)}, duration_ms={gen_time}")

    # Step 7: Determine confidence level and verify if needed
    confidence = result["confidence"]
    verified = None
    pipeline = "medium_confidence"

    if confidence >= HIGH_CONFIDENCE:
        pipeline = "high_confidence"
        logger.info(f"High confidence path: confidence={confidence}")

    elif confidence < LOW_CONFIDENCE:
        # Low confidence — verify the answer
        context_text = "\n".join([r["text"] for r in result["all_results"]])
        verification = generator.verify_answer(body.question, answer, context_text)
        verified = verification["is_valid"]

        if verified:
            pipeline = "low_confidence_verified"
            logger.info(f"Low confidence VERIFIED: confidence={confidence}")
        else:
            pipeline = "low_confidence_rejected"
            answer = "I'm sorry, I don't have enough information to answer that question accurately. Could you please rephrase or provide more details?"
            logger.info(f"Low confidence REJECTED: confidence={confidence}")

    elapsed = round(time.time() - start_time, 3)

    # Step 8: Save message to DB
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

    # Log to MLflow
    tracker.log_chat_query(
        question=body.question, answer=answer,
        confidence=confidence, response_time=elapsed,
        n_results=body.n_results, sources_count=len(sources),
        category=result["category"], user_id=current_user.id,
        conversation_id=conversation.id
    )

    # Prometheus metrics
    chat_requests_total.labels(status="success").inc()
    chat_response_seconds.observe(elapsed)
    retrieval_confidence.observe(confidence)

    logger.info(f"Support query completed: user_id={current_user.id}, pipeline={pipeline}, confidence={confidence}, duration={elapsed}")

    return ChatResponse(
        answer=answer,
        confidence=confidence,
        matched_question=result["matched_question"],
        category=result["category"],
        sources=sources,
        conversation_id=conversation.id,
        intent="support",
        pipeline=pipeline,
        verified=verified
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

    start_time = time.time()

    # Step 1: Load or create conversation
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

    # Step 2: Load conversation history (last 6 messages)
    conversation_history = []
    if conv_id and conversation:
        previous_messages = db.query(ChatMessage).filter(
            ChatMessage.conversation_id == conversation.id
        ).order_by(ChatMessage.created_at.desc()).limit(6).all()

        previous_messages.reverse()

        for msg in previous_messages:
            conversation_history.append({"role": "user", "content": msg.user_query})
            conversation_history.append({"role": "assistant", "content": msg.llm_response})

        logger.info(f"Loaded conversation history: messages={len(previous_messages)}, conversation_id={conversation.id}")

    # Step 3: LLM Router — classify intent
    route = llm_router.classify(question)
    intent = route["intent"].lower()
    logger.info(f"Voice router decision: intent={intent}, user_id={current_user.id}")

    # ============================================
    # CASUAL PATH — Skip retrieval entirely
    # ============================================
    if intent == "casual":
        try:
            answer = generator.direct_answer(question, conversation_history)
        except Exception as e:
            logger.error(f"Voice direct answer failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate answer")

        elapsed = round(time.time() - start_time, 3)

        # Save message to DB
        message = ChatMessage(
            conversation_id=conversation.id,
            user_query=question,
            retrieved_context="[CASUAL - No retrieval needed]",
            llm_response=answer,
            response_time=elapsed
        )
        db.add(message)
        db.commit()

        chat_requests_total.labels(status="success").inc()
        chat_response_seconds.observe(elapsed)

        logger.info(f"Voice casual query completed: user_id={current_user.id}, pipeline=casual, duration={elapsed}")

        response_data = {
            "transcription": question,
            "answer": answer,
            "confidence": 1.0,
            "matched_question": "",
            "category": "General",
            "sources": [],
            "conversation_id": conversation.id,
            "intent": "casual",
            "pipeline": "casual",
            "verified": None,
            "audio_base64": None
        }

        # TTS: Convert answer to speech
        if tts.is_available:
            try:
                tts_audio = tts.synthesize(answer)
                response_data["audio_base64"] = base64.b64encode(tts_audio).decode("utf-8")
            except Exception as e:
                logger.error(f"Voice TTS failed (casual): {str(e)}")

        return response_data

    # ============================================
    # SUPPORT PATH — Full RAG pipeline
    # ============================================

    # Step 4: Rewrite vague follow-up questions for better retrieval
    search_query = question
    if conversation_history:
        search_query = generator.rewrite_query(question, conversation_history)

    # Step 5: Retrieve context from ChromaDB
    retrieval_start = time.time()
    result = retriever.get_answer_with_context(question=search_query, n_results=n_results)
    retrieval_time = round((time.time() - retrieval_start) * 1000, 2)
    logger.info(f"Voice retrieval completed: chunks_found={len(result['all_results'])}, confidence={result['confidence']}, duration_ms={retrieval_time}")

    # Step 6: Generate answer using LLM
    gen_start = time.time()
    try:
        answer = generator.generate_answer(
            question=question,
            retrieved_chunks=result["all_results"],
            conversation_history=conversation_history
        )
    except Exception as e:
        logger.error(f"Voice LLM generation failed: {str(e)}", exc_info=True)
        chat_requests_total.labels(status="error").inc()
        raise HTTPException(status_code=500, detail="Failed to generate answer")

    gen_time = round((time.time() - gen_start) * 1000, 2)
    logger.info(f"Voice LLM generation completed: answer_len={len(answer)}, duration_ms={gen_time}")

    # Step 7: Determine confidence level and verify if needed
    confidence = result["confidence"]
    verified = None
    pipeline = "medium_confidence"

    if confidence >= HIGH_CONFIDENCE:
        pipeline = "high_confidence"
        logger.info(f"Voice high confidence path: confidence={confidence}")

    elif confidence < LOW_CONFIDENCE:
        # Low confidence — verify the answer
        context_text = "\n".join([r["text"] for r in result["all_results"]])
        verification = generator.verify_answer(question, answer, context_text)
        verified = verification["is_valid"]

        if verified:
            pipeline = "low_confidence_verified"
            logger.info(f"Voice low confidence VERIFIED: confidence={confidence}")
        else:
            pipeline = "low_confidence_rejected"
            answer = "I'm sorry, I don't have enough information to answer that question accurately. Could you please rephrase or provide more details?"
            logger.info(f"Voice low confidence REJECTED: confidence={confidence}")

    elapsed = round(time.time() - start_time, 3)

    # Step 8: Save message to DB
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
        confidence=confidence, response_time=elapsed,
        n_results=n_results, sources_count=len(sources),
        category=result["category"], user_id=current_user.id,
        conversation_id=conversation.id
    )

    # Prometheus metrics
    chat_requests_total.labels(status="success").inc()
    chat_response_seconds.observe(elapsed)
    retrieval_confidence.observe(confidence)

    logger.info(f"Voice support query completed: user_id={current_user.id}, pipeline={pipeline}, confidence={confidence}, duration={elapsed}")

    response_data = {
        "transcription": question,
        "answer": answer,
        "confidence": confidence,
        "matched_question": result["matched_question"],
        "category": result["category"],
        "sources": [s.dict() for s in sources],
        "conversation_id": conversation.id,
        "intent": "support",
        "pipeline": pipeline,
        "verified": verified,
        "audio_base64": None
    }

    # TTS: Convert answer to speech
    if tts.is_available:
        try:
            tts_audio = tts.synthesize(answer)
            response_data["audio_base64"] = base64.b64encode(tts_audio).decode("utf-8")
        except Exception as e:
            logger.error(f"Voice TTS failed (support): {str(e)}")

    return response_data



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
