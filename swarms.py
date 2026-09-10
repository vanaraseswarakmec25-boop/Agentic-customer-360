# swarms.py
import re
from schemas import CustomerState
from vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
LEGAL_FRAUD_PATTERN = r"\b(sue|suing|lawsuit|attorney|lawyer|court|litigation|fraud|hacked|stolen)\b"

def run_fastpath_guardrail(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(LEGAL_FRAUD_PATTERN, text, re.IGNORECASE))

async def usage_swarm_worker(payload: dict, state: CustomerState):
    weekly_logins = payload.get("weekly_logins", 10)
    baseline_avg = payload.get("baseline_avg_logins", 10)
    if baseline_avg > 0 and weekly_logins < (0.5 * baseline_avg):
        state.usage_trend = "severe_drop_off_50%"
    else:
        state.usage_trend = "normal"

async def support_swarm_worker(payload: dict, state: CustomerState):
    ticket_text = payload.get("ticket_text", "")
    if ticket_text:
        scores = analyzer.polarity_scores(ticket_text)
        state.sentiment_score = scores["compound"]

async def transaction_swarm_worker(payload: dict, state: CustomerState):
    amount = payload.get("amount", 0.0)
    avg_spend = payload.get("avg_spend", 100.0)
    std_dev = payload.get("std_dev", 20.0)
    if std_dev > 0:
        z_score = (amount - avg_spend) / std_dev
        if z_score > 3.0:
            state.transaction_anomaly_score = round(z_score, 2)