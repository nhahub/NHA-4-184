from pydantic import BaseModel, model_validator, field_validator


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