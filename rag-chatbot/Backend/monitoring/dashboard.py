import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# ==================== CONFIG ====================

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag_project")
engine = create_engine(DATABASE_URL)

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "Admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "ADMIN123")
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def run_query(query: str) -> pd.DataFrame:
    """Execute SQL query and return as DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="RAG Chatbot Monitor",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #475569;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOGIN GATE ====================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Admin Login")
    st.markdown("Enter your credentials to access the dashboard.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

    st.stop()

# ==================== SIDEBAR NAVIGATION ====================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Analytics", "🔧 Retrain Knowledge Base"])

if st.sidebar.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.rerun()

# ==================== PAGE 1: ANALYTICS ====================

if page == "📊 Analytics":

    st.title("RAG Chatbot Monitoring Dashboard")
    st.markdown("Real-time insights from your customer support chatbot")
    st.divider()

    # --- TOP KPIs ---
    kpi_data = run_query("""
        SELECT 
            COUNT(*) as total_queries,
            ROUND(AVG(response_time)::numeric, 3) as avg_response_time,
            ROUND(MIN(response_time)::numeric, 3) as min_response_time,
            ROUND(MAX(response_time)::numeric, 3) as max_response_time
        FROM chat_messages
    """)

    user_count = run_query("SELECT COUNT(*) as total FROM users")
    conversation_count = run_query("SELECT COUNT(*) as total FROM conversations")

    feedback_stats = run_query("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) as negative
        FROM feedback
    """)

    total_queries = int(kpi_data["total_queries"].iloc[0]) if not kpi_data.empty else 0
    avg_time = float(kpi_data["avg_response_time"].iloc[0]) if kpi_data["avg_response_time"].iloc[0] else 0
    total_users = int(user_count["total"].iloc[0]) if not user_count.empty else 0
    total_conversations = int(conversation_count["total"].iloc[0]) if not conversation_count.empty else 0
    total_feedback = int(feedback_stats["total"].iloc[0]) if not feedback_stats.empty else 0
    positive_fb = int(feedback_stats["positive"].iloc[0]) if feedback_stats["positive"].iloc[0] else 0
    negative_fb = int(feedback_stats["negative"].iloc[0]) if feedback_stats["negative"].iloc[0] else 0
    satisfaction_rate = round((positive_fb / total_feedback * 100), 1) if total_feedback > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Queries", total_queries)
    with col2:
        st.metric("Avg Response Time", f"{avg_time}s")
    with col3:
        st.metric("Total Users", total_users)
    with col4:
        st.metric("Conversations", total_conversations)
    with col5:
        st.metric("Satisfaction", f"{satisfaction_rate}%")

    st.divider()

    # --- ROW 1: CHARTS ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Response Time Trend")
        
        response_trend = run_query("""
            SELECT 
                DATE(created_at) as date,
                ROUND(AVG(response_time)::numeric, 3) as avg_time,
                COUNT(*) as query_count
            FROM chat_messages
            GROUP BY DATE(created_at)
            ORDER BY date
        """)
        
        if not response_trend.empty:
            fig = px.line(
                response_trend,
                x="date",
                y="avg_time",
                markers=True,
                labels={"date": "Date", "avg_time": "Avg Response Time (s)"},
            )
            fig.update_layout(
                template="plotly_dark",
                height=350,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            fig.update_traces(line_color="#38bdf8", line_width=3)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet. Send some chat messages first!")

    with col_right:
        st.subheader("User Satisfaction")
        
        if total_feedback > 0:
            fig = px.pie(
                values=[positive_fb, negative_fb],
                names=["Positive", "Negative"],
                color_discrete_sequence=["#22c55e", "#ef4444"],
                hole=0.4
            )
            fig.update_layout(
                template="plotly_dark",
                height=350,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feedback submitted yet!")

    st.divider()

    # --- ROW 2: CHARTS ---
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("Daily Query Volume")
        
        daily_volume = run_query("""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as queries
            FROM chat_messages
            GROUP BY DATE(created_at)
            ORDER BY date
        """)
        
        if not daily_volume.empty:
            fig = px.bar(
                daily_volume,
                x="date",
                y="queries",
                labels={"date": "Date", "queries": "Number of Queries"},
            )
            fig.update_layout(
                template="plotly_dark",
                height=350,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            fig.update_traces(marker_color="#a78bfa")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet.")

    with col_right2:
        st.subheader("Top Issue Categories")
        
        category_data = run_query("""
            SELECT 
                COALESCE(
                    SUBSTRING(retrieved_context FROM 'issue_area["\\s:]+([^"\\\\n]+)'),
                    'Unknown'
                ) as category,
                COUNT(*) as count
            FROM chat_messages
            WHERE retrieved_context IS NOT NULL
            GROUP BY category
            ORDER BY count DESC
            LIMIT 10
        """)
        
        if not category_data.empty and not (category_data["category"] == "Unknown").all():
            fig = px.bar(
                category_data[category_data["category"] != "Unknown"],
                x="count",
                y="category",
                orientation="h",
                labels={"count": "Queries", "category": "Category"},
            )
            fig.update_layout(
                template="plotly_dark",
                height=350,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            fig.update_traces(marker_color="#fb923c")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category data available yet.")

    st.divider()

    # --- ROW 3: RESPONSE TIME DISTRIBUTION ---
    st.subheader("Response Time Distribution")

    response_dist = run_query("""
        SELECT response_time FROM chat_messages WHERE response_time IS NOT NULL
    """)

    if not response_dist.empty:
        fig = px.histogram(
            response_dist,
            x="response_time",
            nbins=20,
            labels={"response_time": "Response Time (seconds)", "count": "Frequency"},
        )
        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig.update_traces(marker_color="#38bdf8")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No response time data yet.")

    st.divider()

    # --- ROW 4: RECENT CONVERSATIONS ---
    st.subheader("Recent Conversations")

    recent = run_query("""
        SELECT 
            c.id as conv_id,
            c.title,
            u.username,
            c.created_at,
            COUNT(m.id) as message_count,
            ROUND(AVG(m.response_time)::numeric, 3) as avg_time
        FROM conversations c
        JOIN users u ON c.user_id = u.id
        LEFT JOIN chat_messages m ON m.conversation_id = c.id
        GROUP BY c.id, c.title, u.username, c.created_at
        ORDER BY c.created_at DESC
        LIMIT 15
    """)

    if not recent.empty:
        st.dataframe(
            recent.rename(columns={
                "conv_id": "ID",
                "title": "Title",
                "username": "User",
                "created_at": "Created At",
                "message_count": "Messages",
                "avg_time": "Avg Time (s)"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No conversations yet.")

# ==================== PAGE 2: RETRAIN ====================

elif page == "🔧 Retrain Knowledge Base":

    st.title("🔧 Knowledge Base Retraining")
    st.markdown("Review negative feedback and add corrected answers to improve the chatbot.")
    st.divider()

    # --- Fetch negative feedback ---
    negative_data = run_query("""
        SELECT 
            cm.id as message_id,
            cm.user_query as question,
            cm.llm_response as bad_answer,
            f.comment,
            cm.created_at
        FROM feedback f
        JOIN chat_messages cm ON f.chat_message_id = cm.id
        WHERE f.rating = -1 AND (f.is_retrained IS NOT TRUE)
        ORDER BY cm.created_at DESC
    """)

    if negative_data.empty:
        st.success("✅ No negative feedback found! The chatbot is performing well.")
        st.stop()

    st.subheader(f"📋 Bad Feedback ({len(negative_data)} items)")

    st.dataframe(
        negative_data.rename(columns={
            "message_id": "ID",
            "question": "User Question",
            "bad_answer": "Bad Answer",
            "comment": "User Comment",
            "created_at": "Date"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --- Retrain form ---
    st.subheader("✏️ Submit Corrected Answer")

    selected_id = st.selectbox(
        "Select message to correct",
        options=negative_data["message_id"].tolist(),
        format_func=lambda x: f"ID {x}: {negative_data[negative_data['message_id']==x]['question'].iloc[0][:80]}..."
    )

    if selected_id:
        selected_row = negative_data[negative_data["message_id"] == selected_id].iloc[0]

        st.markdown(f"**Question:** {selected_row['question']}")
        st.markdown(f"**Bad Answer:** {selected_row['bad_answer']}")

        correct_answer = st.text_area(
            "Write the correct answer:",
            height=150,
            placeholder="Enter the correct answer that should be given for this question..."
        )

        if st.button("✅ Add to Knowledge Base", type="primary"):
            if not correct_answer.strip():
                st.warning("Please write a correct answer first.")
            else:
                try:
                    response = requests.post(
                        f"{API_BASE}/feedback/admin/retrain",
                        params={
                            "message_id": selected_id,
                            "correct_answer": correct_answer.strip()
                        }
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Added to knowledge base! Doc ID: {result['doc_id']}")
                        st.balloons()
                    else:
                        st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
                except requests.ConnectionError:
                    st.error("❌ Cannot connect to backend. Make sure the API is running on port 8000.")

# ==================== FOOTER ====================

st.divider()
st.markdown(
    f"<p style='text-align:center; color:#64748b;'>Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True
)
