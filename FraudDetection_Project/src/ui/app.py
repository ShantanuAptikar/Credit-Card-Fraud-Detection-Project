"""
Streamlit UI - Card Fraud Detection Admin Dashboard
Credit & Debit Card Fraud Detection System
Authors: Karan Sumbe, Isha Ghokane, Shantanu Aptikar, Shreya Pawar
Guide: Prof. Aradhana Pawar & Dr. Lakshmikant Malphedwar

Run with: streamlit run ui/app.py
"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

if STREAMLIT_AVAILABLE:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from models.fraud_classifier import FraudClassifier
    from models.face_recognition import FaceRecognitionEngine
    from database import DatabaseManager

    # ── PAGE CONFIG ─────────────────────────────────────────────────
    st.set_page_config(
        page_title="SecurePay Admin Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ── CUSTOM CSS ───────────────────────────────────────────────────
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .metric-card { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

    # ── HEADER ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0">🛡️ SecurePay — Admin Dashboard</h1>
        <p style="margin:5px 0 0 0; opacity:0.85">CNN Biometric + ML Transaction Analysis | Real-Time Monitoring</p>
    </div>
    """, unsafe_allow_html=True)

    # ── SESSION STATE INIT ───────────────────────────────────────────
    if "db_manager" not in st.session_state:
        _db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "fraud_detection.db")
        st.session_state.db_manager = DatabaseManager(db_path=_db_path)

    if "classifier" not in st.session_state:
        with st.spinner("Loading ML classifier..."):
            clf = FraudClassifier(model_path="./models/")
            if not clf.load_model():
                clf.train(use_synthetic=True)
                clf.save_model()
        st.session_state.classifier = clf

    if "face_engine" not in st.session_state:
        st.session_state.face_engine = FaceRecognitionEngine(
            db_manager=st.session_state.db_manager
        )

    # ── SIDEBAR ──────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("", ["📊 Dashboard & Analytics", "⚙️ System Settings"])
        st.divider()
        st.markdown("**System Status**")
        st.success("✅ ML Model: Ready")
        st.success("✅ Face Engine: Ready")
        st.success("✅ Database: Connected")
        history_count = len(st.session_state.db_manager.get_history())
        st.info(f"📝 Transactions: {history_count}")

    # ── PAGE: DASHBOARD ──────────────────────────────────────────────
    if "Dashboard" in page:

        history = st.session_state.db_manager.get_history()
        stats = st.session_state.db_manager.get_stats()

        if not history:
            st.info("No transactions processed yet. Run transactions via the SecurePay web app to see analytics here.")
        else:
            df = pd.DataFrame(history)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            # --- KPI Metrics Row ---
            total = stats.get("total_transactions", 0)
            approved = stats.get("approved", 0)
            blocked = stats.get("blocked", 0)
            held = stats.get("held_for_review", 0)
            avg_latency = stats.get("avg_latency_ms", 0)
            approval_rate = round((approved / total) * 100, 1) if total > 0 else 0

            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Total Transactions", total)
            k2.metric("✅ Approved", approved)
            k3.metric("🚫 Blocked", blocked)
            k4.metric("⚠️ Held for Review", held)
            k5.metric("⚡ Avg Latency", f"{avg_latency}ms")
            k6.metric("Approval Rate", f"{approval_rate}%")

            st.divider()

            # --- ROW 1: Decision Pie Chart + Hourly Volume ---
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Transaction Decisions")
                decision_counts = df["decision"].value_counts().reset_index()
                decision_counts.columns = ["Decision", "Count"]
                colors = {
                    "APPROVED": "#10b981",
                    "BLOCKED_ML_FRAUD": "#ef4444",
                    "BLOCKED_BIOMETRIC_FAILURE": "#f97316",
                    "HELD_FOR_REVIEW": "#f59e0b"
                }
                fig_pie = px.pie(
                    decision_counts, values="Count", names="Decision",
                    color="Decision",
                    color_discrete_map=colors,
                    hole=0.4
                )
                fig_pie.update_traces(textinfo="percent+label")
                fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.markdown("#### Transaction Volume Over Time")
                df_time = df.copy()
                df_time["date"] = df_time["timestamp"].dt.date
                vol = df_time.groupby(["date", "decision"]).size().reset_index(name="count")
                fig_vol = px.bar(
                    vol, x="date", y="count", color="decision",
                    color_discrete_map=colors,
                    barmode="stack",
                    labels={"date": "Date", "count": "Count", "decision": "Decision"}
                )
                fig_vol.update_layout(margin=dict(t=10, b=10))
                st.plotly_chart(fig_vol, use_container_width=True)

            # --- ROW 2: Fraud Score Distribution + Latency Trend ---
            col3, col4 = st.columns(2)

            with col3:
                st.markdown("#### Fraud Score Distribution")
                if "risk_score" in df.columns and df["risk_score"].notna().any():
                    fig_hist = px.histogram(
                        df[df["risk_score"].notna()],
                        x="risk_score", nbins=20,
                        color_discrete_sequence=["#3b82f6"],
                        labels={"risk_score": "ML Fraud Score"}
                    )
                    fig_hist.add_vline(x=0.65, line_dash="dash", line_color="red", annotation_text="Block Threshold")
                    fig_hist.add_vline(x=0.45, line_dash="dash", line_color="orange", annotation_text="Review Threshold")
                    fig_hist.update_layout(margin=dict(t=10, b=10))
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.info("No fraud score data available yet.")

            with col4:
                st.markdown("#### API Latency Over Time (ms)")
                if "latency_ms" in df.columns and df["latency_ms"].notna().any():
                    fig_lat = px.line(
                        df[df["latency_ms"].notna()],
                        x="timestamp", y="latency_ms",
                        color_discrete_sequence=["#8b5cf6"],
                        labels={"latency_ms": "Latency (ms)", "timestamp": "Time"}
                    )
                    fig_lat.add_hline(y=1000, line_dash="dash", line_color="red", annotation_text="1s SLA")
                    fig_lat.update_layout(margin=dict(t=10, b=10))
                    st.plotly_chart(fig_lat, use_container_width=True)
                else:
                    st.info("No latency data available yet.")

            # --- ROW 3: Face Similarity Scores ---
            st.divider()
            st.markdown("#### Face Biometric Similarity Scores (Gate 1)")
            if "face_similarity" in df.columns and df["face_similarity"].notna().any():
                fig_face = px.scatter(
                    df[df["face_similarity"].notna()],
                    x="timestamp", y="face_similarity",
                    color="decision",
                    color_discrete_map=colors,
                    labels={"face_similarity": "Face Similarity Score", "timestamp": "Time"}
                )
                fig_face.add_hline(y=0.70, line_dash="dash", line_color="orange", annotation_text="Match Threshold")
                fig_face.update_layout(margin=dict(t=10, b=10))
                st.plotly_chart(fig_face, use_container_width=True)
            else:
                st.info("No face similarity data available yet.")

            # --- ROW 4: Recent Transaction Table ---
            st.divider()
            st.markdown("#### Recent Transactions")
            display_cols = ["tx_id", "user_id", "amount", "decision", "risk_score", "face_similarity", "latency_ms", "timestamp"]
            display_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[display_cols].sort_values("timestamp", ascending=False).head(50), use_container_width=True)

    # ── PAGE: SETTINGS ───────────────────────────────────────────────
    elif "Settings" in page:
        st.subheader("⚙️ System Settings")
        clf = st.session_state.classifier

        st.markdown("**Model Configuration**")
        new_fraud_threshold = st.slider("Fraud Rejection Threshold", 0.5, 0.95, clf.fraud_threshold, 0.01)
        new_review_threshold = st.slider("Hold-for-Review Threshold", 0.3, 0.6, clf.review_threshold, 0.01)
        new_face_threshold = st.slider("Face Match Threshold", 0.7, 0.95, st.session_state.face_engine.MATCH_THRESHOLD, 0.01)

        if st.button("Apply Settings"):
            clf.fraud_threshold = new_fraud_threshold
            clf.review_threshold = new_review_threshold
            st.session_state.face_engine.MATCH_THRESHOLD = new_face_threshold
            st.success("Settings updated successfully.")

        st.divider()
        if st.button("🔄 Retrain Model (Synthetic Data)"):
            with st.spinner("Retraining model..."):
                metrics = clf.train(use_synthetic=True)
                clf.save_model()
            st.success(f"Model retrained! Accuracy: {metrics['accuracy']:.3f} | AUC: {metrics['auc_roc']:.3f}")

        st.divider()
        st.markdown("**System Information**")
        st.json({
            "model_type": clf.model_type,
            "fraud_threshold": clf.fraud_threshold,
            "review_threshold": clf.review_threshold,
            "face_match_threshold": st.session_state.face_engine.MATCH_THRESHOLD,
            "model_trained": clf.is_trained,
            "demo_mode": st.session_state.face_engine.is_demo_mode
        })

else:
    print("Streamlit not installed. Run: pip install streamlit")
    print("Then launch the UI with: streamlit run ui/app.py")
