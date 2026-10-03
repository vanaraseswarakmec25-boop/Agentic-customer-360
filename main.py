import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
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


@app.get("/")
def read_root():
    return {"message": "Agentic Customer 360 FastAPI Backend is Running!"}


# --- HITL QUEUE ENDPOINTS ---
@app.get("/hitl/queue")
async def get_hitl_queue():
    """Retrieve all tickets currently pending human review."""
    return {"pending_count": len(hitl_queue), "queue": list(hitl_queue.values())}


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
        return {"status": "SUCCESS", "message": f"Ticket {ticket_id} rejected."}


# --- INGESTION ENDPOINT ---
@app.post("/events/stream")
async def process_event_stream(event: EventInput):
    ticket_text = event.payload.get(
        "ticket_text", event.payload.get("message", "")
    )
    category = event.payload.get("category", "general")

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
        return {"status": "HALTED_LEGAL_ESCALATION", "data": output}

    # Stage B: Shared State Initialization
    state = CustomerState(customer_id=event.customer_id)

    # Stage C: Run Non-LLM Swarm Workers
    await asyncio.gather(
        usage_swarm_worker(event.payload, state),
        support_swarm_worker(event.payload, state),
        transaction_swarm_worker(event.payload, state),
    )

    # Save Semantic Trait BEFORE Context Retrieval
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

    status_flag = "PROCESSED" if refiner_result.approved else "HITL_REQUIRED"
    ticket_id = f"TICK-{event.customer_id}-{datetime.now().strftime('%M%S')}"

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

    return {
        "status": "SWARM_PROCESSING_COMPLETE",
        "customer_id": state.customer_id,
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