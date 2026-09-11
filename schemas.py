from pydantic import BaseModel
from typing import Dict, Any, Optional

# Input payload schema
class EventInput(BaseModel):
    event_id: str
    customer_id: str
    event_type: str
    payload: Dict[str, Any]

# State board schema
class CustomerState(BaseModel):
    customer_id: str
    usage_trend: str = "normal"
    sentiment_score: float = 0.0
    transaction_anomaly_score: float = 0.0

# Inferred output for guardrails
class InferredEventOutput(BaseModel):
    customer_id: str
    timestamp: str
    detected_life_event: str
    action_taken: str
    reasoning: str
    hitl_approved: bool
    status: str

# Multi-Agent Pipeline Schemas (MISSING MODELS)
class DebateResult(BaseModel):
    conflict_detected: bool
    reasoning: str
    resolution_strategy: str

class DraftMessage(BaseModel):
    raw_draft: str
    tone_adjusted_draft: str
    compliance_checked_draft: str

class RefinerOutput(BaseModel):
    approved: bool
    feedback: str
    final_output: str