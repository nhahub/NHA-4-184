from fastapi import APIRouter, HTTPException
from app.models.request import ChatRequest
from app.models.response import ChatResponse, SourceChunk
from app.nlp.retrieval import Retriever
from app.nlp.generator import Generator

router = APIRouter(prefix="/chat", tags=["Chat"])

retriever = Retriever()
generator = Generator()


@router.post("/ask", response_model=ChatResponse)
def ask_question(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = retriever.get_answer_with_context(
        question=request.question,
        n_results=request.n_results
    )

    # Generate answer using LLM
    answer = generator.generate_answer(
        question=request.question,
        retrieved_chunks=result["all_results"]
    )

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
        sources=sources
    )
