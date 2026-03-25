from pydantic import BaseModel, model_validator, field_validator
from typing import Optional

class RegisterRequest(BaseModel):
    username: str
    password: str
    confirm_password: str

    @field_validator("password")
    @classmethod
    def password_length(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        if len(v) > 72:
            raise ValueError("Password must be at most 72 characters")
        return v

    @model_validator(mode="after")
    def passwords_must_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[int] = None
    n_results: Optional[int] = 3

class FeedbackRequest(BaseModel):
    message_id: int
    rating: int  # 1 = thumbs up, -1 = thumbs down
    comment: Optional[str] = None

    @field_validator("rating")
    @classmethod
    def rating_must_be_valid(cls, v):
        if v not in (1, -1):
            raise ValueError("Rating must be 1 (thumbs up) or -1 (thumbs down)")
        return v
