import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db
from app.db.models import Ticket, TicketResponse, User
from app.models.request import TicketResponseRequest, UpdateTicketStatusRequest
from app.models.response import TicketOut, UserTicketOut
from app.nlp.retrieval import Retriever

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tickets", tags=["Tickets"])

# One shared instance — same embedder model as the rest of the app
_retriever = Retriever()


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Dependency: raises 403 if the user is not an admin."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# ── User: see my own tickets (MUST be before /{ticket_id}) ─────────────
@router.get("/user/mine", response_model=list[UserTicketOut])
def get_my_tickets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    tickets = db.query(Ticket).filter(
        Ticket.user_id == current_user.id
    ).order_by(Ticket.created_at.desc()).all()

    result = []
    for t in tickets:
        last_response = (
            db.query(TicketResponse)
            .filter(TicketResponse.ticket_id == t.id)
            .order_by(TicketResponse.created_at.desc())
            .first()
        )
        result.append(UserTicketOut(
            id=t.id,
            conversation_id=t.conversation_id,
            question=t.question,
            status=t.status,
            created_at=t.created_at,
            answer=last_response.answer if last_response else None,
        ))
    return result


# ── Admin: see all tickets ─────────────────────────────────────────────
@router.get("/", response_model=list[TicketOut])
def get_all_tickets(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    tickets = db.query(Ticket).order_by(Ticket.created_at.desc()).all()
    return tickets


# ── Admin: see one ticket in detail ───────────────────────────────────
@router.get("/{ticket_id}", response_model=TicketOut)
def get_ticket(
    ticket_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


# ── Admin: respond to a ticket + inject into ChromaDB ─────────────────
@router.post("/{ticket_id}/respond")
def respond_to_ticket(
    ticket_id: int,
    body: TicketResponseRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    # 1. Save admin response to DB
    response = TicketResponse(
        ticket_id=ticket_id,
        admin_id=admin.id,
        answer=body.answer
    )
    db.add(response)

    # 2. Mark ticket as resolved
    ticket.status = "resolved"
    ticket.updated_at = datetime.utcnow()
    db.commit()
    logger.info(f"Ticket {ticket_id} resolved by admin {admin.id}")

    # 3. Add Q&A to ChromaDB with correct embedding (same model as all docs)
    try:
        document = f"Question: {ticket.question}\nAnswer: {body.answer}"
        # Compute embedding using our Embedder — same all-MiniLM-L6-v2 model
        embedding = _retriever.embedder.embed_text(document)
        _retriever.vector_db.add_document(
            doc_id=f"ticket_{ticket_id}",
            document=document,
            embedding=embedding,
            metadata={
                "issue_area": "Human Agent",
                "source": "human_agent",
                "ticket_id": str(ticket_id)
            }
        )
        logger.info(f"Ticket {ticket_id} answer added to ChromaDB successfully")
    except Exception as e:
        logger.error(f"Failed to add ticket answer to ChromaDB: {e}", exc_info=True)
        # Don't crash — DB is already saved, ChromaDB failure is non-fatal

    return {"message": "Response sent and knowledge base updated ✅"}


# ── Admin: update ticket status only ──────────────────────────────────
@router.patch("/{ticket_id}/status")
def update_ticket_status(
    ticket_id: int,
    body: UpdateTicketStatusRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    if body.status not in ("open", "in_progress", "resolved"):
        raise HTTPException(status_code=400, detail="Status must be: open, in_progress, or resolved")

    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket.status = body.status
    ticket.updated_at = datetime.utcnow()
    db.commit()
    return {"message": f"Ticket status updated to '{body.status}'"}
