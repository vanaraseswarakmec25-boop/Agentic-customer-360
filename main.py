import asyncio
import json
import os
import re
import time
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from memory import (
    add_episodic_memory,
    add_semantic_memory,
    retrieve_customer_context,
)
from schemas import CustomerState, EventInput, InferredEventOutput
from swarms import (
    run_agent_debate,
    run_critique_refiner,
    run_fastpath_guardrail,
    run_round_robin_drafting,
    support_swarm_worker,
    transaction_swarm_worker,
    usage_swarm_worker,
)

# Load environment variables
load_dotenv()

app = FastAPI(title="Agentic Customer 360 Ingestion Desk")

# Global in-memory storage for HITL pending review tickets
hitl_queue = {}

# Global storage for tracking customer event timelines in strict chronological order
customer_event_timelines = {}


def parse_iso_timestamp(ts_str: str) -> datetime:
    """Parses standard ISO timestamps for chronological sorting."""
    if not ts_str:
        return datetime.now()
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return datetime.now()


def sanitize_pii(text: str) -> str:
    """Masks SSNs, Credit Cards, and Emails before sending text to LLM agents."""
    if not isinstance(text, str):
        return text
    # Mask Credit Cards
    text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[REDACTED_CARD]', text)
    # Mask SSNs
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', text)
    # Mask Emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', text)
    return text


def log_inferred_event(
    customer_id: str,
    inferred_state: str,
    confidence_level: float,
    action_decided: str,
):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "customer_id": customer_id,
        "inferred_customer_state": inferred_state,
        "confidence_level": confidence_level,
        "action_decided": action_decided,
    }
    with open("inferred_events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


@app.get("/")
def read_root():
    return {"message": "Agentic Customer 360 FastAPI Backend is Running!"}


# --- HITL QUEUE ENDPOINTS ---
@app.get("/hitl/queue")
async def get_hitl_queue():
    """Retrieve all tickets currently pending human review."""
    return {"pending_count": len(hitl_queue), "queue": list(hitl_queue.values())}


@app.get("/hitl/review/{ticket_id}")
async def review_hitl_ticket(ticket_id: str):
    """Allows human approver to inspect reasoning, draft choices, and memory context."""
    if ticket_id not in hitl_queue:
        raise HTTPException(status_code=404, detail="Ticket ID not found in review queue.")
    
    ticket = hitl_queue[ticket_id]
    return {
        "ticket_id": ticket_id,
        "customer_id": ticket["customer_id"],
        "timestamp": ticket["timestamp"],
        "why_flagged": ticket["reason"],
        "draft_options": ticket["drafts"],
        "recommended_response": ticket["final_response"],
        "status": ticket["status"],
    }


@app.post("/hitl/action/{ticket_id}")
async def resolve_hitl_ticket(ticket_id: str, action_data: dict):
    """Handle human decisions: approve original, approve edited text, or reject."""
    if ticket_id not in hitl_queue:
        return {"status": "ERROR", "message": "Ticket ID not found in queue."}

    ticket = hitl_queue.pop(ticket_id)
    action = action_data.get("action", "APPROVE")
    final_text = action_data.get("final_text", ticket.get("final_response", ""))

    if action == "APPROVE":
        add_episodic_memory(
            customer_id=ticket["customer_id"],
            event_type="hitl_human_approval",
            summary=f"Human Approved Response: {final_text}",
            category="compliance",
            metadata={"reviewer_action": "APPROVED", "ticket_id": ticket_id},
        )
        log_inferred_event(
            customer_id=ticket["customer_id"],
            inferred_state="Human Verified",
            confidence_level=1.00,
            action_decided="HUMAN_APPROVED",
        )
        return {
            "status": "SUCCESS",
            "message": f"Ticket {ticket_id} approved and dispatched.",
            "final_text": final_text,
        }
    else:
        add_episodic_memory(
            customer_id=ticket["customer_id"],
            event_type="hitl_human_rejection",
            summary="Human rejected AI response. Ticket closed/escalated offline.",
            category="compliance",
            metadata={"reviewer_action": "REJECTED", "ticket_id": ticket_id},
        )
        log_inferred_event(
            customer_id=ticket["customer_id"],
            inferred_state="Draft Rejected",
            confidence_level=1.00,
            action_decided="HUMAN_REJECTED",
        )
        return {"status": "SUCCESS", "message": f"Ticket {ticket_id} rejected."}


# --- INGESTION ENDPOINT ---
@app.post("/events/stream")
async def process_event_stream(event: EventInput):
    start_time = time.time()
    try:
        raw_text = (
            event.payload.get("ticket_text")
            or event.payload.get("message")
            or event.payload.get("text")
            or event.payload.get("issue")
            or (event.payload.get("ticket", {}).get("text") if isinstance(event.payload.get("ticket"), dict) else "")
            or ""
        )
        category = event.payload.get("category", "general")

        # Deduplication & Chronological Sequencing
        raw_ts = event.payload.get("timestamp") or datetime.now().isoformat()
        event_dt = parse_iso_timestamp(raw_ts)
        timeline = customer_event_timelines.setdefault(event.customer_id, [])

        event_id = getattr(event, "event_id", f"EVT-{time.time()}")
        if any(e.get("event_id") == event_id for e in timeline):
            return {
                "status": "SKIPPED_DUPLICATE",
                "message": f"Event {event_id} already processed. Ignored to prevent double counting."
            }

        timeline.append({
            "event_id": event_id,
            "timestamp": event_dt,
            "event_type": event.event_type,
            "payload": event.payload
        })
        timeline.sort(key=lambda x: x["timestamp"])

        # PII Masking Guardrail
        ticket_text = sanitize_pii(raw_text)

        # Stage A: Fast-Path Guardrail
        if run_fastpath_guardrail(ticket_text):
            output = InferredEventOutput(
                customer_id=event.customer_id,
                timestamp=datetime.now().isoformat(),
                detected_life_event="Legal / Fraud Threat Flagged",
                action_taken="HALTED: Escalated directly to Human Legal Queue",
                reasoning="Synchronous Regex Guardrail matched high-risk keywords.",
                hitl_approved=False,
                status="LEGAL_ESCALATION",
            )
            log_inferred_event(
                customer_id=event.customer_id,
                inferred_state="Legal Risk Flagged",
                confidence_level=0.99,
                action_decided="HALTED_LEGAL_ESCALATION",
            )
            return {"status": "HALTED_LEGAL_ESCALATION", "data": output}

        # Stage B: Shared State Initialization
        state = CustomerState(customer_id=event.customer_id)

        # Stage C: Run Scoped Swarm Workers
        await asyncio.gather(
            usage_swarm_worker(event.payload, state),
            support_swarm_worker({"ticket_text": ticket_text, "category": category}, state),
            transaction_swarm_worker(event.payload, state),
        )

        extracted_trait = event.payload.get("extracted_trait")
        if extracted_trait:
            add_semantic_memory(
                customer_id=event.customer_id,
                fact=extracted_trait,
                category=category,
            )

        # Stage D: Retrieve Dual Memory Context
        context = await retrieve_customer_context(
            customer_id=event.customer_id,
            query_text=ticket_text or event.event_type,
            category=category,
        )
        past_memories = context["episodic"]
        semantic_facts = context["semantic"]

        # Stage E: Multi-Agent Debate & Refinement
        debate_result = await run_agent_debate(
            state=state,
            memories=past_memories,
            semantic_facts=semantic_facts,
        )
        draft_result = await run_round_robin_drafting(
            state, debate_result.resolution_strategy
        )
        refiner_result = await run_critique_refiner(draft_result)

        # Stage F: Store Episodic Memory
        add_episodic_memory(
            customer_id=event.customer_id,
            event_type=event.event_type,
            summary=ticket_text,
            category=category,
            metadata={"resolution": debate_result.resolution_strategy},
        )

        status_flag = (
            "PROCESSED" if refiner_result.approved else "HITL_REQUIRED"
        )
        ticket_id = (
            f"TICK-{event.customer_id}-{datetime.now().strftime('%M%S')}"
        )

        # --- UPDATED DOMAIN-SPECIFIC EVENT CLASSIFICATION ---
        text_lower = ticket_text.lower() if ticket_text else ""
        payload = event.payload or {}
        event_type = getattr(event, "event_type", "")
        amount = payload.get("amount", 0)
        is_international = payload.get("is_international", False)

        # 1. Direct LLM/Agent override
        if extracted_trait:
            inferred_trait = extracted_trait

        # 2. Financial Anomaly / High-Value Events (e.g. EVT_000382: $12k transfer)
        elif event_type in ["outbound_transfer", "wire_transfer"] and amount >= 10000:
            inferred_trait = "HIGH_VALUE_TRANSFER"
        elif event_type == "purchase" and (amount >= 5000 or is_international):
            inferred_trait = "SUSPICIOUS_TRANSACTION"
        elif state.transaction_anomaly_score and float(state.transaction_anomaly_score) > 3.0:
            inferred_trait = "TRANSACTION_ANOMALY"

        # 3. Usage & Churn Trends from Swarm Workers
        elif state.usage_trend and float(state.usage_trend) < -0.5:
            inferred_trait = "CHURN_RISK"

        # 4. Text & Sentiment Keyword Fallbacks (for support messages/tickets)
        elif any(w in text_lower for w in ["cancel", "awful", "terrible", "leaving", "frustrated", "bad"]):
            inferred_trait = "CHURN_RISK"
        elif any(w in text_lower for w in ["payment", "charged", "refund", "billing", "invoice"]):
            inferred_trait = "BILLING_ISSUE"
        elif any(w in text_lower for w in ["error", "bug", "crash", "locked", "slow", "down"]):
            inferred_trait = "TECHNICAL_ISSUE"
        elif state.sentiment_score and float(state.sentiment_score) < -0.3:
            inferred_trait = "AT_RISK"

        # 5. Low-level routine telemetry (e.g. app logins)
        elif event_type == "login":
            inferred_trait = "ROUTINE_ACTIVITY"

        # 6. Fallback
        else:
            inferred_trait = "STABLE"

        # --- UPDATED GUARDRAIL & HITL ROUTING ---
        # Force HITL review for high-risk states or refiner rejections
        HIGH_RISK_STATES = {"HIGH_VALUE_TRANSFER", "SUSPICIOUS_TRANSACTION", "CHURN_RISK"}
        
        if inferred_trait in HIGH_RISK_STATES or not refiner_result.approved:
            status_flag = "HITL_REQUIRED"
        else:
            status_flag = "PROCESSED"

        if status_flag == "HITL_REQUIRED":
            hitl_queue[ticket_id] = {
                "ticket_id": ticket_id,
                "customer_id": state.customer_id,
                "timestamp": datetime.now().isoformat(),
                "reason": debate_result.reasoning,
                "drafts": draft_result,
                "final_response": refiner_result.final_output,
                "status": status_flag,
            }
            log_inferred_event(
                customer_id=event.customer_id,
                inferred_state=inferred_trait,
                confidence_level=0.85,
                action_decided="HITL_REQUIRED",
            )
        else:
            log_inferred_event(
                customer_id=event.customer_id,
                inferred_state=inferred_trait,
                confidence_level=0.95,
                action_decided="AUTO_DISPATCH",
            )

        latency_ms = round((time.time() - start_time) * 1000, 2)
        calc_confidence = 0.95 if status_flag == "PROCESSED" else 0.85

        return {
            "status": "SWARM_PROCESSING_COMPLETE",
            "customer_id": state.customer_id,
            "metrics": {
                "latency_ms": latency_ms,
                "confidence_score": calc_confidence,
            },
            "shared_state_board": {
                "usage_trend": state.usage_trend,
                "sentiment_score": state.sentiment_score,
                "transaction_anomaly_score": state.transaction_anomaly_score,
            },
            "episodic_memory": {
                "count": len(past_memories),
                "memories": past_memories,
            },
            "semantic_memory": {
                "count": len(semantic_facts),
                "facts": semantic_facts,
            },
            "agent_outputs": {
                "conflict_detected": debate_result.conflict_detected,
                "resolution_strategy": debate_result.resolution_strategy,
                "drafts": draft_result,
                "final_response": refiner_result.final_output,
                "status": status_flag,
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"DEBUG ERROR: {type(e).__name__}: {str(e)}"
        )