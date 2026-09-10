from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class EventInput(BaseModel):
    event_id: str
    customer_id: str
    event_type: str
    payload: Dict[str, Any] = {}

class CustomerState(BaseModel):
    customer_id: str
    usage_trend: str = "normal"
    sentiment_score: float = 0.0
    transaction_anomaly_score: float = 0.0

class InferredEventOutput(BaseModel):
    event_id: str
    customer_id: str
    status: str
    state_snapshot: CustomerState
    timestamp: str