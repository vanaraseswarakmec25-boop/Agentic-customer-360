import streamlit as st
import requests
import json

# Dashboard configuration
st.set_page_config(
    page_title="Agentic Customer 360 - Operations Desk",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Agentic Customer 360 — Real-Time Operations Desk")
st.markdown("Monitor edge swarms, vector memory retrieval, multi-agent debates, and policy guardrails live.")

# Sidebar - Test Payload Generator
st.sidebar.header("🧪 Event Simulator")
customer_id = st.sidebar.text_input("Customer ID", value="cust_101")
event_type = st.sidebar.selectbox("Event Type", ["support_ticket", "account_anomaly", "billing_inquiry"])
category = st.sidebar.selectbox("Category", ["technical", "billing", "general"])

ticket_text = st.sidebar.text_area(
    "Ticket / Escalation Text",
    value="My app keeps freezing at checkout and I want a full refund!"
)

weekly_logins = st.sidebar.number_input("Weekly Logins", value=2, min_value=0)
baseline_avg_logins = st.sidebar.number_input("Baseline Avg Logins", value=15, min_value=0)
amount = st.sidebar.number_input("Transaction Amount ($)", value=450.0)
avg_spend = st.sidebar.number_input("Avg Spend ($)", value=100.0)
std_dev = st.sidebar.number_input("Std Dev ($)", value=20.0)

backend_url = st.sidebar.text_input("FastAPI Endpoint", value="http://127.0.0.1:8000/events/stream")

# Main Dashboard View
if st.sidebar.button("🚀 Trigger Event Stream"):
    payload = {
        "event_id": f"evt_{customer_id}_01",
        "customer_id": customer_id,
        "event_type": event_type,
        "payload": {
            "ticket_text": ticket_text,
            "weekly_logins": weekly_logins,
            "baseline_avg_logins": baseline_avg_logins,
            "amount": amount,
            "avg_spend": avg_spend,
            "std_dev": std_dev,
            "category": category
        }
    }

    with st.spinner("Processing event through Swarm & MAS Pipeline..."):
        try:
            response = requests.post(backend_url, json=payload)
            if response.status_code == 200:
                data = response.json()
                
                # --- Top Status Banner ---
                status = data.get("status", "UNKNOWN")
                if status == "PROCESSED":
                    st.success(f"Status: {status} — Processed & Persisted to Memory")
                elif status == "HITL_REQUIRED":
                    st.warning(f"Status: {status} — Escalated for Human Review")
                else:
                    st.error(f"Status: {status} — Fast-Path Guardrail Triggered")

                col1, col2, col3 = st.columns(3)

                # --- Column 1: Shared State Board ---
                with col1:
                    st.subheader("📊 Swarm State Board")
                    state = data.get("shared_state_board", {})
                    st.metric("Usage Trend", state.get("usage_trend", "N/A"))
                    st.metric("Sentiment Score", round(state.get("sentiment_score", 0.0), 2))
                    st.metric("Transaction Z-Score", state.get("transaction_anomaly_score", 0.0))

                # --- Column 2: ChromaDB Memory ---
                with col2:
                    st.subheader("🧠 Vector Memory")
                    mem_count = data.get("retrieved_memories_count", 0)
                    st.metric("Episodic Memories Found", mem_count)
                    if mem_count > 0:
                        st.info("Relevant historical context fetched via Cosine Similarity (≥ 0.78).")

                # --- Column 3: Debate & Action ---
                with col3:
                    st.subheader("⚔️ Agent Debate")
                    debate = data.get("debate", {})
                    # CORRECT:
                    st.write("**Conflict Detected:**", debate.get("conflict_detected", False))
                    st.write("**Strategy:**", debate.get("resolution_strategy", "N/A"))
                    st.caption(f"Reasoning: {debate.get('reasoning', '')}")

                st.divider()

                # --- Draft Refinement Passes ---
                st.subheader("📝 Multi-Pass Sequential Draft (Round-Robin)")
                drafts = data.get("drafts", {})
                t1, t2, t3 = st.tabs(["Pass 1: Operational Base", "Pass 2: Tone Adjusted", "Pass 3: Final Draft"])
                with t1:
                    st.code(drafts.get("raw_draft", ""), language="markdown")
                with t2:
                    st.code(drafts.get("tone_adjusted_draft", ""), language="markdown")
                with t3:
                    st.code(drafts.get("compliance_checked_draft", ""), language="markdown")

                # --- Final Policy & HITL Panel ---
                st.subheader("🛡️ Critique-Refiner & HITL Safety Guardrail")
                refiner = data.get("final_output", {})
                st.json(refiner)

            else:
                st.error(f"Backend Error ({response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")