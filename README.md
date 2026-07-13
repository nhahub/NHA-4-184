# 📦 BrownBox AI — RAG Customer Support Chatbot

An AI-powered customer support chatbot built with **Retrieval-Augmented Generation (RAG)** that delivers accurate, context-aware answers from a custom knowledge base. When the system lacks confidence, it automatically escalates to human agents via a built-in **ticket system**.

> **DEPI Final Project** — Full-stack AI application with MLOps monitoring, voice support, and human-in-the-loop feedback.

---

## 📑 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [RAG Pipeline](#-rag-pipeline)
- [Data Flow](#-data-flow)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#1-clone-the-repository)
  - [Backend Setup (Docker)](#2-backend-setup-docker)
  - [Backend Setup (Local)](#3-backend-setup-local-development)
  - [Frontend Setup](#4-frontend-setup)
  - [Create an Admin User](#5-create-an-admin-user)
- [Environment Variables](#-environment-variables)
- [API Endpoints](#-api-endpoints)
- [Monitoring & MLOps](#-monitoring--mlops)
- [Human Agent Ticket System](#-human-agent-ticket-system)
- [Team](#-team)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **RAG Chatbot** | Retrieves relevant context from ChromaDB and generates answers using Groq LLM |
| 🗣️ **Voice Input/Output** | Speech-to-text via OpenAI Whisper + text-to-speech via ElevenLabs |
| 🎫 **Human Agent Tickets** | Auto-creates support tickets when AI confidence is too low |
| 🔄 **Dynamic Learning** | Admin answers get injected back into the knowledge base |
| 📊 **MLOps Monitoring** | Prometheus metrics, MLflow experiment tracking, Streamlit dashboard |
| 🔐 **Authentication** | JWT + Google OAuth 2.0 with password reset via OTP email |
| 💬 **Conversation History** | Full multi-turn conversation memory with 6-message context window |
| 🧠 **Query Rewriting** | LLM rewrites vague follow-up questions for better retrieval |
| ✅ **Answer Verification** | Low-confidence answers are verified before being shown to users |
| 👍 **Feedback & Retraining** | Users rate answers; admins can retrain with corrected Q&A pairs |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Docker Compose                              │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Frontend │  │ Backend  │  │PostgreSQL│  │  Redis   │           │
│  │ Next.js  │  │ FastAPI  │  │   DB     │  │  Cache   │           │
│  │ :3000    │  │ :8000    │  │ :5432    │  │ :6379    │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │              │              │              │                 │
│       └──── REST ────┘              └──── SQL ─────┘                 │
│                      │                                               │
│              ┌───────┴────────┐                                     │
│              │   NLP Pipeline  │                                     │
│              │  ┌────────────┐ │    ┌──────────┐                    │
│              │  │ Embedder   │ │    │ ChromaDB │                    │
│              │  │ MiniLM-L6  │─┼───▶│ VectorDB │                    │
│              │  └────────────┘ │    └──────────┘                    │
│              │  ┌────────────┐ │                                     │
│              │  │ Generator  │ │    ┌──────────┐                    │
│              │  │ Groq LLM   │─┼───▶│ Whisper  │                    │
│              │  └────────────┘ │    │ (Voice)  │                    │
│              │  ┌────────────┐ │    └──────────┘                    │
│              │  │ LLM Router │ │                                     │
│              │  └────────────┘ │    ┌──────────┐                    │
│              └─────────────────┘    │ElevenLabs│                    │
│                                     │  (TTS)   │                    │
│  ┌──────────┐  ┌──────────┐        └──────────┘                    │
│  │Streamlit │  │  MLflow  │  ┌──────────┐                          │
│  │Dashboard │  │ Tracking │  │Prometheus│                          │
│  │ :8501    │  │ :5000    │  │  :9090   │                          │
│  └──────────┘  └──────────┘  └──────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 RAG Pipeline

The core RAG pipeline processes every user query through the following steps:

```
User Question
      │
      ▼
┌─────────────┐
│  LLM Router │──── "casual" ──▶ Direct LLM Answer (no retrieval)
│  (Intent)   │
└──────┬──────┘
       │ "support"
       ▼
┌─────────────┐
│Query Rewrite│ ◄── Uses last 6 messages as context
│  (if needed) │     to resolve follow-up questions
└──────┬──────┘
       ▼
┌─────────────┐     ┌──────────┐
│  Embedder   │────▶│ ChromaDB │
│ MiniLM-L6-v2│     │  Search  │
└─────────────┘     └────┬─────┘
                         │ Top-K chunks
                         ▼
                  ┌─────────────┐
                  │  Generator  │ ◄── Groq LLM (llama-3.3-70b)
                  │ (Answer Gen)│
                  └──────┬──────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Confidence Check    │
              │                     │
              │ ≥ 0.5  → Return    │ HIGH confidence
              │ 0.2-0.5 → Return   │ MEDIUM confidence
              │ < 0.2  → Verify ↓  │ LOW confidence
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  Answer Verifier    │
              │                     │
              │ ✅ Valid → Return   │
              │ ❌ Invalid → Ticket │ ← Human Agent Escalation
              └─────────────────────┘
```

### Confidence Thresholds

| Confidence | Pipeline | Action |
|-----------|----------|--------|
| `≥ 0.5` | `high_confidence` | Return answer directly |
| `0.2 – 0.5` | `medium_confidence` | Return answer with lower score |
| `< 0.2` + verified | `low_confidence_verified` | Answer passed verification, return it |
| `< 0.2` + not verified | `human_agent` | Create support ticket, notify user |

---

## 📊 Data Flow

### 1. Data Ingestion (One-time)
```
CSV Data ──▶ Text Chunking ──▶ Embedding (MiniLM-L6-v2) ──▶ ChromaDB
                                                              (Persistent)
```

### 2. Query Processing
```
User ──▶ Frontend ──▶ Backend API ──▶ Router ──▶ Retrieval ──▶ LLM ──▶ Response
                                                    │
                                              ChromaDB Search
                                          (embedding similarity)
```

### 3. Human Agent Feedback Loop
```
Low Confidence Query
        │
        ▼
  Ticket Created ──▶ Admin Dashboard ──▶ Admin Responds
                                              │
                                              ▼
                                    Answer Added to ChromaDB
                                    (with correct embedding)
                                              │
                                              ▼
                                    Future queries get the
                                    answer automatically ✅
```

---

## 🛠 Tech Stack

### Backend
| Technology | Purpose |
|-----------|---------|
| **FastAPI** | REST API framework |
| **PostgreSQL 16** | Relational database (users, conversations, tickets) |
| **ChromaDB** | Vector database for document embeddings |
| **Sentence Transformers** | `all-MiniLM-L6-v2` for text embedding |
| **Groq API** | LLM inference (`llama-3.3-70b-versatile`) |
| **OpenAI Whisper** | Speech-to-text transcription |
| **ElevenLabs** | Text-to-speech synthesis |
| **Redis** | OAuth state caching |
| **SQLAlchemy** | ORM for database operations |

### Frontend
| Technology | Purpose |
|-----------|---------|
| **Next.js 16** | React framework with App Router |
| **TypeScript** | Type-safe JavaScript |
| **Tailwind CSS 4** | Utility-first styling |
| **Radix UI** | Accessible component primitives |
| **Lucide React** | Icon library |
| **Axios** | HTTP client |

### MLOps & Monitoring
| Technology | Purpose |
|-----------|---------|
| **Prometheus** | Metrics collection (response times, confidence scores) |
| **MLflow** | Experiment tracking for chat queries |
| **Streamlit** | Real-time monitoring dashboard |
| **Docker Compose** | Container orchestration (6 services) |

---

## 📁 Project Structure

```
rag-chatbot/
├── Backend/
│   ├── app/
│   │   ├── api/                  # API route handlers
│   │   │   ├── auth.py           # Auth endpoints (register, login, Google OAuth, OTP)
│   │   │   ├── chat.py           # Chat endpoints (/ask, /voice, /history)
│   │   │   ├── feedback.py       # Feedback and admin retraining
│   │   │   └── tickets.py        # Human agent ticket management
│   │   ├── core/                 # Security, rate limiting, logging config
│   │   ├── db/                   # Database models and session management
│   │   │   ├── models.py         # SQLAlchemy models (User, Conversation, Ticket, etc.)
│   │   │   └── session.py        # DB connection and session factory
│   │   ├── middleware/           # Request logging middleware
│   │   ├── mlops/                # MLflow tracker and Prometheus metrics
│   │   ├── models/               # Pydantic request/response schemas
│   │   ├── nlp/                  # NLP pipeline components
│   │   │   ├── embedder.py       # Sentence Transformer embedding
│   │   │   ├── generator.py      # Groq LLM answer generation + verification
│   │   │   ├── ingestion.py      # CSV data ingestion into ChromaDB
│   │   │   ├── retrieval.py      # Vector similarity search + answer extraction
│   │   │   ├── router.py         # Intent classifier (casual vs support)
│   │   │   ├── transcriber.py    # Whisper speech-to-text
│   │   │   ├── tts.py            # ElevenLabs text-to-speech
│   │   │   └── vector_db.py      # ChromaDB wrapper (search, add, count)
│   │   ├── utils/                # Email utilities
│   │   └── main.py               # FastAPI app entry point
│   ├── monitoring/               # Streamlit dashboard
│   ├── ingest_data.py            # Data ingestion script
│   ├── requirements.txt          # Python dependencies
│   └── .env                      # Environment variables (not committed)
│
├── frontend/
│   ├── app/                      # Next.js pages (App Router)
│   │   ├── auth/                 # OAuth callback page
│   │   ├── chat/                 # Main chat interface
│   │   ├── login/                # Login page
│   │   ├── register/             # Registration page
│   │   ├── tickets/              # Admin ticket management page
│   │   └── forgot-password/      # Password reset flow
│   ├── components/               # React components
│   │   └── chat/                 # Chat UI components (sidebar, messages, input)
│   ├── hooks/                    # Custom React hooks (useAuth, useChat, useFeedback)
│   ├── services/                 # API service layer (auth, chat, tickets)
│   ├── types/                    # TypeScript type definitions
│   └── package.json              # Node.js dependencies
│
├── data/
│   └── processed/                # Processed CSV data for ingestion
│       └── qa_chunks.csv         # Question-Answer pairs dataset
│
├── docker-compose.yml            # 6-service orchestration
├── Dockerfile                    # Backend Docker image
├── entrypoint.sh                 # Auto-ingestion on first run
├── prometheus.yml                # Prometheus scrape config
└── .gitignore                    # Git ignore rules
```

---

## 🚀 Getting Started

### Prerequisites

- **Docker** and **Docker Compose** installed ([Get Docker](https://docs.docker.com/get-docker/))
- **Node.js 18+** and **npm** ([Get Node.js](https://nodejs.org/))
- **Git** ([Get Git](https://git-scm.com/))
- API Keys (see [Environment Variables](#-environment-variables))

---

### 1. Clone the Repository

```bash
git clone https://github.com/nhahub/NHA-4-184.git
cd NHA-4-184/rag-chatbot
```

---

### 2. Backend Setup (Docker)

This is the **recommended** approach. Docker handles everything automatically.

#### Step 1: Create the `.env` file

```bash
cp Backend/.env.example Backend/.env
```

Edit `Backend/.env` and fill in your API keys:

```env
# ============ Database ============
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_project

# ============ JWT Secret ============
SECRET_KEY=your-secret-key-here

# ============ Groq LLM API ============
GROQ_API_KEY=your-groq-api-key

# ============ Email (for OTP) ============
EMAIL_ADDRESS=your-email@gmail.com
EMAIL_PASSWORD=your-app-password

# ============ Google OAuth ============
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback

# ============ ElevenLabs TTS ============
ELEVENLABS_API_KEY=your-elevenlabs-key
ELEVENLABS_VOICE_ID=EXAVITQu4vr4xnSDxMaL
```

#### Step 2: Build and start all services

```bash
docker compose up --build -d
```

This starts **6 containers**:

| Service | URL | Description |
|---------|-----|-------------|
| Backend API | http://localhost:8000 | FastAPI + Swagger docs at `/docs` |
| PostgreSQL | localhost:5432 | Relational database |
| Redis | localhost:6379 | OAuth state cache |
| Streamlit | http://localhost:8501 | Monitoring dashboard |
| MLflow | http://localhost:5000 | Experiment tracking UI |
| Prometheus | http://localhost:9090 | Metrics collection |

#### Step 3: Verify it's running

```bash
# Check all containers are healthy
docker compose ps

# Test the API
curl http://localhost:8000/
# Expected: {"status":"ok","message":"RAG Chatbot API is running"}
```

> **Note:** On the first run, the `entrypoint.sh` script automatically ingests the CSV data into ChromaDB. This only happens once — subsequent starts skip ingestion.

---

### 3. Backend Setup (Local Development)

If you prefer running without Docker:

```bash
# Create virtual environment
cd Backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Make sure PostgreSQL and Redis are running locally
# Then run data ingestion
python ingest_data.py

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### 4. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at **http://localhost:3000**.

---

### 5. Create an Admin User

After registering a user (via the UI or Google OAuth), promote them to admin:

```bash
# Using Docker
docker exec rag-postgres psql -U postgres -d rag_project \
  -c "UPDATE users SET is_admin = true WHERE email = 'your-email@example.com';"

# Verify
docker exec rag-postgres psql -U postgres -d rag_project \
  -c "SELECT id, username, email, is_admin FROM users;"
```

After logging out and back in, the admin will see a **"Support Tickets"** link in the sidebar.

---

## 🔐 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | ✅ | PostgreSQL connection string |
| `SECRET_KEY` | ✅ | JWT token signing secret |
| `GROQ_API_KEY` | ✅ | Groq API key for LLM (llama-3.3-70b) |
| `EMAIL_ADDRESS` | ✅ | Gmail address for OTP emails |
| `EMAIL_PASSWORD` | ✅ | Gmail App Password (not your real password) |
| `GOOGLE_CLIENT_ID` | ⚠️ | Google OAuth client ID (optional if not using Google login) |
| `GOOGLE_CLIENT_SECRET` | ⚠️ | Google OAuth client secret |
| `GOOGLE_REDIRECT_URI` | ⚠️ | OAuth callback URL |
| `ELEVENLABS_API_KEY` | ⚠️ | ElevenLabs API key (optional — voice output disabled without it) |
| `ELEVENLABS_VOICE_ID` | ⚠️ | ElevenLabs voice ID |
| `REDIS_HOST` | Docker only | Redis hostname (set automatically in Docker) |
| `REDIS_PORT` | Docker only | Redis port (set automatically in Docker) |

---

## 📡 API Endpoints

### Authentication (`/auth`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register a new user |
| POST | `/auth/login` | Login and get JWT token |
| GET | `/auth/me` | Get current user info |
| GET | `/auth/google/login` | Start Google OAuth flow |
| GET | `/auth/google/callback` | Google OAuth callback |
| POST | `/auth/forgot-password` | Send OTP email |
| POST | `/auth/verify-otp` | Verify OTP code |
| POST | `/auth/reset-password` | Reset password with token |

### Chat (`/chat`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat/ask` | Send a text question (full RAG pipeline) |
| POST | `/chat/voice` | Send audio file (Whisper → RAG → TTS) |
| GET | `/chat/history` | List all conversations |
| GET | `/chat/history/{id}` | Get conversation with all messages |

### Feedback (`/feedback`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/feedback/` | Submit feedback (thumbs up/down) |
| GET | `/feedback/retrain-candidates` | Admin: get negative feedback for retraining |
| POST | `/feedback/retrain/{id}` | Admin: retrain with corrected answer |

### Tickets (`/tickets`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/tickets/user/mine` | User: see my tickets |
| GET | `/tickets/` | Admin: list all tickets |
| GET | `/tickets/{id}` | Admin: ticket details |
| POST | `/tickets/{id}/respond` | Admin: answer ticket + add to knowledge base |
| PATCH | `/tickets/{id}/status` | Admin: update ticket status |

---

## 📈 Monitoring & MLOps

### Prometheus Metrics (`:9090`)
- `chat_requests_total` — Total chat requests (success/error)
- `chat_response_seconds` — Response time histogram
- `retrieval_confidence` — Retrieval confidence distribution

### MLflow Tracking (`:5000`)
Every chat query is logged with:
- Question, answer, confidence score
- Response time, sources count
- User ID, conversation ID

### Streamlit Dashboard (`:8501`)
Real-time monitoring showing:
- Active users and conversations
- Average response times
- Confidence score trends
- Error rates

---

## 🎫 Human Agent Ticket System

### How It Works

1. **User asks a question** the bot can't answer confidently
2. **Confidence < 0.2** triggers the answer verification step
3. **Verification fails** → a support ticket is automatically created
4. **User sees:** *"A support ticket has been created. A human agent will respond shortly."*
5. **Admin opens** the Support Tickets dashboard
6. **Admin responds** with the correct answer
7. **Answer is injected** into ChromaDB with the correct embedding
8. **Next time** anyone asks the same question → the bot answers correctly

### Admin Ticket Dashboard

Admins see the ticket panel in the sidebar. From there they can:
- View all open/resolved tickets
- Respond to tickets
- Track ticket statistics



## 👥 Team

| Name | Role |
|------|------|
| Ahmed Elsenosy | Full-Stack AI Developer & Team Lead |
| Shrouk Eissa | Data & AI Developer |
| Youssef Ashraf | Backend & AI Developer |
| Ashraqat Effat | Frontend Developer & MLOps |
| Fatma Shehata | Frontend Developer & MLOps |

---

## 📄 License

This project was built as a final project for the **Digital Egypt Pioneers Initiative (DEPI)** program.
