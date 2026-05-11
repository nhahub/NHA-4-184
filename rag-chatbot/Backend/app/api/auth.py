import os
import secrets
import httpx
import time
import logging
import redis
from authlib.integrations.starlette_client import OAuth
from starlette.requests import Request
from starlette.responses import RedirectResponse
import random
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.rate_limiter import limiter
from app.core.security import get_current_user
from app.db.session import get_db
from app.db.models import User, OTPCode
from app.models.request import (
    RegisterRequest, LoginRequest,
    ForgotPasswordRequest, VerifyOTPRequest, ResetPasswordRequest
)
from app.models.response import TokenResponse, UserResponse, OTPResponse, ResetTokenResponse
from app.core.security import hash_password, verify_password, create_access_token
from app.utils.email import send_otp_email
from app.mlops.metrics import auth_login_total, auth_register_total


logger = logging.getLogger(__name__)

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

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
@limiter.limit("5/minute")
def register(request: Request,body: RegisterRequest, db: Session = Depends(get_db)):
    # Check username not taken
    if db.query(User).filter(User.username == body.username).first():
        logger.warning(f"Registration failed: username already taken, username={body.username}")
        auth_register_total.labels(status="failed").inc()
        raise HTTPException(status_code=400, detail="Username already taken")

    # Check email not taken
    if db.query(User).filter(User.email == body.email).first():
        logger.warning(f"Registration failed: email already exists, email={body.email}")
        auth_register_total.labels(status="failed").inc()
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        username=body.username,
        email=body.email,
        hashed_password=hash_password(body.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f"User registered: user_id={user.id}, username={user.username}, email={user.email}")
    auth_register_total.labels(status="success").inc()
    return user


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
def login(request: Request,body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == body.username).first()

    if not user or not verify_password(body.password, user.hashed_password):
        logger.warning(f"Login failed: invalid credentials for username={body.username}")
        auth_login_total.labels(status="failed").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    token = create_access_token(data={"sub": str(user.id)})
    logger.info(f"User login successful: user_id={user.id}, username={user.username}")
    auth_login_total.labels(status="success").inc()
    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user)
    )


# ==================== FORGOT PASSWORD ====================

@router.post("/forgot-password", response_model=OTPResponse)
@limiter.limit("3/minute")
def forgot_password(request: Request,body: ForgotPasswordRequest, db: Session = Depends(get_db)):
    start_time = time.time()
    user = db.query(User).filter(User.email == body.email).first()
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
    duration_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(f"Password reset requested: user_id={user.id}, email={user.email}, otp_sent=true, duration_ms={duration_ms}")

    return OTPResponse(message="OTP sent to your email")


@router.post("/verify-otp", response_model=ResetTokenResponse)
@limiter.limit("5/minute")
def verify_otp(request: Request,body: VerifyOTPRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user:
        logger.warning(f"OTP verification failed: email not found, email={body.email}")
        raise HTTPException(status_code=404, detail="Email not found")

    # Find valid OTP
    otp = db.query(OTPCode).filter(
        OTPCode.user_id == user.id,
        OTPCode.code == body.otp,
        OTPCode.is_used == False
    ).first()

    if not otp:
        logger.warning(f"OTP verification failed: invalid otp for user_id={user.id}")
        raise HTTPException(status_code=400, detail="Invalid OTP")

    if otp.expires_at < datetime.now(timezone.utc):
        logger.warning(f"OTP verification failed: otp expired for user_id={user.id}")
        raise HTTPException(status_code=400, detail="OTP has expired")

    # Mark OTP as used
    otp.is_used = True
    db.commit()

    # Generate reset token (short-lived, 10 minutes)
    reset_token = create_access_token(data={"sub": str(user.id), "purpose": "reset"})
    
    logger.info(f"OTP verification successful: user_id={user.id}")

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
            logger.warning(f"Password reset failed: invalid purpose in token")
            raise HTTPException(status_code=400, detail="Invalid reset token")
        user_id = int(payload.get("sub"))
    except (jwt.ExpiredSignatureError, jwt.DecodeError, ValueError):
        logger.warning(f"Password reset failed: invalid or expired reset token")
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        logger.warning(f"Password reset failed: user not found for user_id={user_id}")
        raise HTTPException(status_code=404, detail="User not found")

    # Update password
    user.hashed_password = hash_password(request.new_password)
    db.commit()
    logger.info(f"Password reset completed: user_id={user.id}")

    return {"message": "Password reset successfully"}

# ==================== GOOGLE OAUTH ====================

@router.get("/google/login")
async def google_login(request: Request):
    logger.info("Google OAuth login initiated")

    state = secrets.token_urlsafe(32)
    redis_client.setex(f"oauth:{state}", 300, "1")  # expires in 5 mins

    client_id = os.getenv("GOOGLE_CLIENT_ID")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/auth/google/callback")

    google_auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&response_type=code"
        f"&scope=openid email profile"
        f"&state={state}"
    )

    return RedirectResponse(url=google_auth_url)


@router.get("/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    try:
        state_in_request = request.query_params.get("state")
        code = request.query_params.get("code")

        if not state_in_request or not redis_client.exists(f"oauth:{state_in_request}"):
            raise HTTPException(status_code=400, detail="Invalid state parameter")
        redis_client.delete(f"oauth:{state_in_request}")

        redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/auth/google/callback")
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                    "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                }
            )
            token_data = token_response.json()

        async with httpx.AsyncClient() as client:
            user_response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {token_data['access_token']}"}
            )
            user_info = user_response.json()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Google OAuth callback failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Google authentication failed: {str(e)}")

    email = user_info.get("email")
    google_id = user_info.get("id")
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
        logger.info(f"Google OAuth: existing user logged in, user_id={user.id}")
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
        logger.info(f"Google OAuth: new user registered, user_id={user.id}")

    # Generate JWT token
    access_token = create_access_token(data={"sub": str(user.id)})

    # Redirect to frontend with token
    frontend_url = f"http://localhost:3000/auth/callback?access_token={access_token}"
    return RedirectResponse(url=frontend_url)

@router.get("/me")
def get_current_user_info(current_user: User = Depends(get_current_user)):
    logger.info(f"User info retrieved: user_id={current_user.id}")
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "profile_picture": current_user.profile_picture
    }

