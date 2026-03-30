import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


def send_otp_email(to_email: str, otp_code: str):
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

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())
