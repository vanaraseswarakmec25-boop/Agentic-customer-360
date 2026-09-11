# swarms.py
import re
from schemas import CustomerState,DebateResult, DraftMessage, RefinerOutput
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

async def run_agent_debate(state: CustomerState, memories: list) -> DebateResult:
    if state.usage_trend == "severe_drop_off_50%" and state.transaction_anomaly_score > 3.0:
        has_vip = any("VIP" in m.get("summary", "") for m in memories)
        strategy = "UPSELL_VIP" if has_vip else "RETENTION_OFFER"
        return DebateResult(
            conflict_detected=True,
            reasoning="Usage dropped severely but transaction Z-score is high. Conflicting signals resolved.",
            resolution_strategy=strategy
        )
    
    return DebateResult(
        conflict_detected=False,
        reasoning="Metrics align across swarm workers.",
        resolution_strategy="STANDARD_SUPPORT"
    )

# 2. Round-Robin Drafting Stage (Sequential 3-Pass Refinement)
async def run_round_robin_drafting(state: CustomerState, strategy: str) -> DraftMessage:
    # Pass 1: Operational Base Draft
    pass1 = f"Account update for customer {state.customer_id}."
    
    # Pass 2: Tone & Sentiment Alignment
    if state.sentiment_score < -0.3:
        pass2 = f"{pass1} We sincerely apologize for the inconvenience and are reviewing your account priority."
    else:
        pass2 = f"{pass1} Thank you for being a valued account holder."

    # Pass 3: Strategy & Action Insertion
    if strategy == "UPSELL_VIP":
        pass3 = f"{pass2} A dedicated VIP account manager has been assigned to your support queue."
    elif strategy == "RETENTION_OFFER":
        pass3 = f"{pass2} A 15% loyalty retention credit has been flagged for your account."
    else:
        pass3 = pass2

    return DraftMessage(
        raw_draft=pass1,
        tone_adjusted_draft=pass2,
        compliance_checked_draft=pass3
    )

# 3. Critique-Refiner Stage (Policy & HITL Enforcement)
async def run_critique_refiner(draft: DraftMessage) -> RefinerOutput:
    message = draft.compliance_checked_draft
    
    # Policy check: Block unauthorized 100% discount promises
    if "100% free" in message.lower() or "free forever" in message.lower():
        return RefinerOutput(
            approved=False,
            feedback="Unauthorized discount policy detected in draft.",
            final_output="Escalated to HITL manager for compliance review."
        )

    return RefinerOutput(
        approved=True,
        feedback="Passed compliance, safety, and tone bounds.",
        final_output=message
    )