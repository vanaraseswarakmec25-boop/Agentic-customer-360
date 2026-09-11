import asyncio
from datetime import datetime
from fastapi import FastAPI
from schemas import EventInput, CustomerState, InferredEventOutput
from swarms import (
    run_fastpath_guardrail,
    usage_swarm_worker,
    support_swarm_worker,
    transaction_swarm_worker,
    run_agent_debate,
    run_round_robin_drafting,
    run_critique_refiner,
)
from memory import query_episodic_memory, add_episodic_memory

# 1. Initialize the FastAPI application
app = FastAPI(title="Agentic Customer 360 Ingestion Desk")


# Root check endpoint
@app.get("/")
def read_root():
    return {"message": "Agentic Customer 360 FastAPI Backend is Running!"}


# 2. Endpoint that runs Fast-Path Guardrails and Parallel Swarm Workers
@app.post("/events/stream")
async def process_event_stream(event: EventInput):
    ticket_text = event.payload.get("ticket_text", "")

    # Stage A: Fast-Path Guardrail (Synchronous Regex Scan)
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

    # Stage B: Shared State Board Initialization
    state = CustomerState(customer_id=event.customer_id)

    # Stage C: Run Non-LLM Swarm Workers in Parallel using asyncio.gather
    await asyncio.gather(
        usage_swarm_worker(event.payload, state),
        support_swarm_worker(event.payload, state),
        transaction_swarm_worker(event.payload, state),
    )
    category = event.payload.get("category", "general")
    past_memories = query_episodic_memory(
        customer_id=event.customer_id,
        query_text=ticket_text or event.event_type,
        category=category,
        similarity_threshold=0.78
    )
    debate_result = await run_agent_debate(state, past_memories)
    draft_result = await run_round_robin_drafting(state, debate_result.resolution_strategy)
    refiner_result = await run_critique_refiner(draft_result)

    add_episodic_memory(
        customer_id=event.customer_id,
        event_type=event.event_type,
        summary=refiner_result.final_output,
        category=category,
        metadata={"resolution": debate_result.resolution_strategy}
    )

    status_flag = "PROCESSED" if refiner_result.approved else "HITL_REQUIRED"

    # Return the populated State Board (before passing to Groq LLM)
    return {
        "status": "SWARM_PROCESSING_COMPLETE",
        "customer_id": state.customer_id,
        "shared_state_board": {
            "usage_trend": state.usage_trend,
            "sentiment_score": state.sentiment_score,
            "transaction_anomaly_score": state.transaction_anomaly_score,
        },
    }