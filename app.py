import json
import requests
import streamlit as st

# Dashboard configuration
st.set_page_config(
    page_title="Agentic Customer 360 - Operations Desk",
    page_icon="⚡",
    layout="wide",
)

tab_desk, tab_hitl = st.tabs(
    ["⚡ Operations Desk", "🚨 HITL Review Queue (Human Approval)"]
)

with tab_desk:
    st.title("⚡ Agentic Customer 360 — Real-Time Operations Desk")
    st.caption(
        "Monitor edge swarms, vector memory retrieval, multi-agent debates, and policy guardrails live."
    )

    # Sidebar - Test Payload Generator
    st.sidebar.header("🧪 Event Simulator")
    customer_id = st.sidebar.text_input("Customer ID", value="cust_101")
    event_type = st.sidebar.selectbox(
        "Event Type", ["support_ticket", "account_anomaly", "billing_inquiry"]
    )
    category = st.sidebar.selectbox("Category", ["technical", "billing", "general"])

    ticket_text = st.sidebar.text_area(
        "Ticket / Escalation Text",
        value="My app keeps freezing at checkout and I want a full refund!",
    )

    extracted_trait = st.sidebar.text_input(
        "Extracted Trait (Semantic Profile)",
        value="Customer holds VIP status and demands priority resolution.",
    )

    weekly_logins = st.sidebar.number_input("Weekly Logins", value=2, min_value=0)
    baseline_avg_logins = st.sidebar.number_input(
        "Baseline Avg Logins", value=15, min_value=0
    )
    amount = st.sidebar.number_input("Transaction Amount ($)", value=450.0)
    avg_spend = st.sidebar.number_input("Avg Spend ($)", value=100.0)
    std_dev = st.sidebar.number_input("Std Dev ($)", value=20.0)

    backend_url = st.sidebar.text_input(
        "FastAPI Endpoint", value="http://127.0.0.1:8000/events/stream"
    )

    # Main Dashboard View
    if st.sidebar.button("🚀 Trigger Event Stream"):
        payload_data = {
            "ticket_text": ticket_text,
            "weekly_logins": weekly_logins,
            "baseline_avg_logins": baseline_avg_logins,
            "amount": amount,
            "avg_spend": avg_spend,
            "std_dev": std_dev,
            "category": category,
        }

        if extracted_trait:
            payload_data["extracted_trait"] = extracted_trait

        payload = {
            "event_id": f"evt_{customer_id}_01",
            "customer_id": customer_id,
            "event_type": event_type,
            "payload": payload_data,
        }

        with st.spinner("Processing event through Swarm & MAS Pipeline..."):
            try:
                response = requests.post(backend_url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    agent_outputs = data.get("agent_outputs", {})

                    status = agent_outputs.get(
                        "status", data.get("status", "UNKNOWN")
                    )
                    if status == "PROCESSED":
                        st.success(
                            f"Status: {status} — Processed & Persisted to Memory"
                        )
                    elif status == "HITL_REQUIRED":
                        st.warning(
                            f"Status: {status} — Escalated for Human Review"
                        )
                    else:
                        st.error(
                            f"Status: {status} — Fast-Path Guardrail Triggered"
                        )

                    col1, col2, col3 = st.columns(3)

                    # --- Column 1: Shared State Board ---
                    with col1:
                        st.subheader("📊 Swarm State Board")
                        state = data.get("shared_state_board", {})
                        st.metric("Usage Trend", state.get("usage_trend", "N/A"))
                        st.metric(
                            "Sentiment Score",
                            round(state.get("sentiment_score", 0.0), 2),
                        )
                        st.metric(
                            "Transaction Z-Score",
                            state.get("transaction_anomaly_score", 0.0),
                        )

                    # --- Column 2: Dual Vector Memory ---
                    with col2:
                        st.subheader("🧠 Dual Vector Memory")

                        episodic_data = data.get("episodic_memory", {})
                        epi_count = episodic_data.get("count", 0)

                        semantic_data = data.get("semantic_memory", {})
                        sem_count = semantic_data.get("count", 0)

                        st.markdown(f"**Episodic Memories Found:** `{epi_count}`")
                        st.markdown(f"**Semantic Traits Found:** `{sem_count}`")

                        if sem_count > 0:
                            st.info("🧬 **Semantic Profile Facts:**")
                            for idx, fact in enumerate(
                                semantic_data.get("facts", []), 1
                            ):
                                trait_text = (
                                    fact.get("fact", "")
                                    if isinstance(fact, dict)
                                    else str(fact)
                                )
                                st.write(f"• **Trait #{idx}:** {trait_text}")

                        if epi_count > 0:
                            st.caption("📜 **Episodic Interaction Logs:**")
                            for idx, mem in enumerate(
                                episodic_data.get("memories", []), 1
                            ):
                                mem_text = (
                                    mem.get("summary", "")
                                    if isinstance(mem, dict)
                                    else str(mem)
                                )
                                st.caption(f"**Memory #{idx}:** {mem_text}")

                    # --- Column 3: Debate & Action ---
                    with col3:
                        st.subheader("⚔️ Agent Debate")
                        st.write(
                            "**Conflict Detected:**",
                            agent_outputs.get("conflict_detected", False),
                        )
                        st.write(
                            "**Strategy:**",
                            agent_outputs.get("resolution_strategy", "N/A"),
                        )

                    st.divider()

                    # --- Draft Refinement Passes ---
                    st.subheader(
                        "📝 Multi-Pass Sequential Draft (Round-Robin)"
                    )
                    drafts = agent_outputs.get("drafts", {})
                    t1, t2, t3 = st.tabs([
                        "Pass 1: Operational Base",
                        "Pass 2: Tone Adjusted",
                        "Pass 3: Final Draft",
                    ])
                    with t1:
                        st.code(
                            drafts.get("raw_draft", "No draft generated."),
                            language="markdown",
                        )
                    with t2:
                        st.code(
                            drafts.get(
                                "tone_adjusted_draft", "No draft generated."
                            ),
                            language="markdown",
                        )
                    with t3:
                        st.code(
                            drafts.get(
                                "compliance_checked_draft", "No draft generated."
                            ),
                            language="markdown",
                        )

                    st.subheader(
                        "🛡️ Critique-Refiner & HITL Safety Guardrail"
                    )
                    refiner_output = agent_outputs.get("final_response", {})
                    st.write(refiner_output)

                else:
                    st.error(
                        f"Backend Error ({response.status_code}): {response.text}"
                    )
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")

with tab_hitl:
    st.header("🚨 Human-in-the-Loop Review Desk")
    st.caption(
        "Review and modify draft responses flagged for policy, sentiment, or legal compliance before dispatch."
    )

    if st.button("🔄 Refresh Queue"):
        st.rerun()

    try:
        q_resp = requests.get("http://127.0.0.1:8000/hitl/queue")
        if q_resp.status_code == 200:
            q_data = q_resp.json()
            pending_count = q_data.get("pending_count", 0)
            queue = q_data.get("queue", [])

            st.metric("Pending Approval Count", pending_count)
            st.divider()

            if pending_count == 0:
                st.info("🎉 No tickets currently require human intervention.")
            else:
                for item in queue:
                    t_id = item["ticket_id"]
                    with st.expander(
                        f"Ticket `{t_id}` | Customer: `{item['customer_id']}`",
                        expanded=True,
                    ):
                        st.write(f"**Flag Reason:** {item['reason']}")

                        edited_response = st.text_area(
                            "Review & Edit Draft before Sending:",
                            value=item["final_response"],
                            key=f"text_{t_id}",
                            height=120,
                        )

                        col_app, col_rej, _ = st.columns([1, 1, 3])

                        if col_app.button("✅ Approve & Dispatch", key=f"app_{t_id}"):
                            res = requests.post(
                                f"http://127.0.0.1:8000/hitl/action/{t_id}",
                                json={
                                    "action": "APPROVE",
                                    "final_text": edited_response,
                                },
                            )
                            if res.status_code == 200:
                                st.success(f"Approved and sent for `{t_id}`!")
                                st.rerun()

                        if col_rej.button("❌ Reject Response", key=f"rej_{t_id}"):
                            res = requests.post(
                                f"http://127.0.0.1:8000/hitl/action/{t_id}",
                                json={"action": "REJECT"},
                            )
                            if res.status_code == 200:
                                st.warning(f"Rejected `{t_id}`.")
                                st.rerun()
        else:
            st.error(f"Error fetching queue: {q_resp.status_code}")

    except requests.exceptions.ConnectionError:
        st.error(
            "FastAPI backend is offline. Ensure `uvicorn main:app --reload` is running."
        )