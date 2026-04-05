import os
import secrets
from authlib.integrations.starlette_client import OAuth
from starlette.requests import Request
from starlette.responses import RedirectResponse
import random
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import User, OTPCode
from app.models.request import (
    RegisterRequest, LoginRequest,
    ForgotPasswordRequest, VerifyOTPRequest, ResetPasswordRequest
)
from app.models.response import TokenResponse, UserResponse, OTPResponse, ResetTokenResponse
from app.core.security import hash_password, verify_password, create_access_token
from app.utils.email import send_otp_email


oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=201)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    # Check username not taken
    if db.query(User).filter(User.username == request.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    # Check email not taken
    if db.query(User).filter(User.email == request.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        username=request.username,
        email=request.email,
        hashed_password=hash_password(request.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == request.username).first()

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    token = create_access_token(data={"sub": str(user.id)})
    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user)
    )


# ==================== FORGOT PASSWORD ====================

@router.post("/forgot-password", response_model=OTPResponse)
def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Email not found")

    # Generate 6-digit OTP
    otp_code = str(random.randint(100000, 999999))

    # Delete any old OTPs for this user
    db.query(OTPCode).filter(OTPCode.user_id == user.id).delete()

    # Save new OTP
    otp = OTPCode(
        user_id=user.id,
        code=otp_code,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5)
    )
    db.add(otp)
    db.commit()

    # Send OTP email
    send_otp_email(to_email=user.email, otp_code=otp_code)

    return OTPResponse(message="OTP sent to your email")


@router.post("/verify-otp", response_model=ResetTokenResponse)
def verify_otp(request: VerifyOTPRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Email not found")

    # Find valid OTP
    otp = db.query(OTPCode).filter(
        OTPCode.user_id == user.id,
        OTPCode.code == request.otp,
        OTPCode.is_used == False
    ).first()

    if not otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    if otp.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="OTP has expired")

    # Mark OTP as used
    otp.is_used = True
    db.commit()

    # Generate reset token (short-lived, 10 minutes)
    reset_token = create_access_token(data={"sub": str(user.id), "purpose": "reset"})
    
    return ResetTokenResponse(
        reset_token=reset_token,
        user=UserResponse.model_validate(user)
    )


@router.post("/reset-password")
def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    import jwt
    from app.core.security import SECRET_KEY, ALGORITHM

    try:
        payload = jwt.decode(request.reset_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("purpose") != "reset":
            raise HTTPException(status_code=400, detail="Invalid reset token")
        user_id = int(payload.get("sub"))
    except (jwt.ExpiredSignatureError, jwt.DecodeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update password
    user.hashed_password = hash_password(request.new_password)
    db.commit()

    return {"message": "Password reset successfully"}

# ==================== GOOGLE OAUTH ====================

@router.get("/google/login")
async def google_login(request: Request):
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
        raise HTTPException(status_code=400, detail="Google authentication failed")

    user_info = token.get("userinfo")
    if not user_info:
        raise HTTPException(status_code=400, detail="Could not get user info from Google")

    email = user_info.get("email")
    google_id = user_info.get("sub")
    name = user_info.get("name", "")
    picture = user_info.get("picture", "")

    # Check if user already exists (by Google ID or email)
    user = db.query(User).filter(
        (User.google_id == google_id) | (User.email == email)
    ).first()

    if user:
        # Update Google info if needed
        if not user.google_id:
            user.google_id = google_id
            user.profile_picture = picture
            db.commit()
    else:
        # Create new user
        username = email.split("@")[0] + "_" + secrets.token_hex(3)
        user = User(
            username=username,
            email=email,
            google_id=google_id,
            profile_picture=picture,
            hashed_password=None
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # Generate JWT token
    access_token = create_access_token(data={"sub": str(user.id)})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "profile_picture": user.profile_picture
        }
    }

