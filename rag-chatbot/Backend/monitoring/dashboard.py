import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# ==================== CONFIG ====================

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag_project")
engine = create_engine(DATABASE_URL)


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

st.title("RAG Chatbot Monitoring Dashboard")
st.markdown("Real-time insights from your customer support chatbot")
st.divider()

# ==================== TOP KPIs ====================

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

# ==================== ROW 1: CHARTS ====================

col_left, col_right = st.columns(2)

# --- Response Time Over Time ---
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

# --- User Satisfaction Pie ---
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

# ==================== ROW 2: CHARTS ====================

col_left2, col_right2 = st.columns(2)

# --- Query Volume Per Day ---
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

# --- Top Question Categories ---
with col_right2:
    st.subheader("Top Issue Categories")
    
    category_data = run_query("""
        SELECT 
            COALESCE(
                SUBSTRING(retrieved_context FROM 'issue_area["\s:]+([^"\\n]+)'),
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

# ==================== ROW 3: RESPONSE TIME DISTRIBUTION ====================

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

# ==================== ROW 4: RECENT CONVERSATIONS ====================

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

# ==================== FOOTER ====================

st.divider()
st.markdown(
    f"<p style='text-align:center; color:#64748b;'>Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True
)
