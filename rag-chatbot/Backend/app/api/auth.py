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
