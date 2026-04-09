import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


def send_otp_email(to_email: str, otp_code: str):
    logger.info(f"Sending OTP email: recipient={to_email}")
    
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        logger.error(f"Email configuration missing: EMAIL_ADDRESS={bool(EMAIL_ADDRESS)}, EMAIL_PASSWORD={bool(EMAIL_PASSWORD)}")
        raise ValueError("Email credentials not configured in environment variables")
    
    subject = "Password Reset OTP - RAG Chatbot"
    body = f"""
    <html>
    <body>
        <h2>Password Reset Request</h2>
        <p>Your OTP code is:</p>
        <h1 style="color: #4CAF50; letter-spacing: 5px;">{otp_code}</h1>
        <p>This code will expire in <strong>5 minutes</strong>.</p>
        <p>If you didn't request this, please ignore this email.</p>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email
    msg.attach(MIMEText(body, "html"))

    try:
        logger.debug(f"Connecting to Gmail SMTP server...")
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            logger.debug(f"SMTP connection established, authenticating...")
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            logger.debug(f"Authentication successful, sending email...")
            server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())
        logger.info(f"OTP email sent successfully: recipient={to_email}")
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"Email authentication failed: {str(e)}", exc_info=True)
        raise
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error sending OTP email: {str(e)}, recipient={to_email}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Failed to send OTP email: {str(e)}, recipient={to_email}, error_type={type(e).__name__}", exc_info=True)
        raise
