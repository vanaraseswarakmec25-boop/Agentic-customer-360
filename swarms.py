import re
import os
import json
from dotenv import load_dotenv
from groq import Groq
from schemas import CustomerState, DebateResult, DraftMessage, RefinerOutput
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

analyzer = SentimentIntensityAnalyzer()
LEGAL_FRAUD_PATTERN = (
    r"\b(sue|suing|lawsuit|attorney|lawyer|court|litigation|fraud|hacked|stolen)\b"
)

def query_groq(system_prompt: str, user_prompt: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Groq API Error: {e}")
        return {}

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

# 1. Multi-Agent Debate Stage (Updated Signature & Dual Memory Inspection)
async def run_agent_debate(
    state: CustomerState, memories: list, semantic_facts: list = None
) -> DebateResult:
    semantic_facts = semantic_facts or []

    system_prompt = """
    You are an AI Strategy Agent resolving customer disputes.
    Analyze customer state, episodic history, and semantic traits to determine if there is a operational conflict.
    
    Return JSON only with these exact keys:
    - "conflict_detected": boolean
    - "reasoning": string explaining the decision
    - "resolution_strategy": string (choose from: "UPSELL_VIP", "RETENTION_OFFER", "STANDARD_SUPPORT", "TECHNICAL_ESCALATION")
    """

    user_prompt = f"""
    Customer ID: {state.customer_id}
    Sentiment Score: {state.sentiment_score}
    Usage Trend: {state.usage_trend}
    Transaction Z-Score: {state.transaction_anomaly_score}
    Semantic Traits: {semantic_facts}
    Episodic Logs: {memories}
    """
    res_json = query_groq(system_prompt, user_prompt)

    if (
        state.usage_trend == "severe_drop_off_50%"
        and state.transaction_anomaly_score > 3.0
    ):
        # Check episodic memories for VIP mention
        has_vip_episodic = any(
            "VIP"
            in (m.get("summary", "") if isinstance(m, dict) else str(m))
            for m in memories
        )

        # Check semantic memory facts for VIP mention
        has_vip_semantic = any(
            "VIP"
            in (f.get("fact", "") if isinstance(f, dict) else str(f))
            for f in semantic_facts
        )

        has_vip = has_vip_episodic or has_vip_semantic
        strategy = "UPSELL_VIP" if has_vip else "RETENTION_OFFER"

        return DebateResult(
            conflict_detected=True,
            reasoning="Usage dropped severely but transaction Z-score is high. Conflicting signals resolved.",
            resolution_strategy=strategy,
        )

    return DebateResult(
        conflict_detected=False,
        reasoning="Metrics align across swarm workers.",
        resolution_strategy="STANDARD_SUPPORT",
    )

# 2. Round-Robin Drafting Stage (Deterministic Control Flow + Groq Generation)
async def run_round_robin_drafting(
    state: CustomerState, strategy: str
) -> DraftMessage:
    system_prompt = "You are a customer communications writer. Return JSON only with a single key 'draft'."

    # Pass 1: Operational Base Draft
    p1 = query_groq(
        system_prompt,
        f"Draft a concise operational update message for customer {state.customer_id}."
    )
    pass1 = p1.get("draft", f"Account update for customer {state.customer_id}.")

    # Pass 2: Tone & Sentiment Alignment (Your Exact If/Else Loop -> Groq Prompting)
    if state.sentiment_score < -0.3:
        p2_prompt = f"Rewrite this draft to express sincere apologies for the inconvenience and state that we are reviewing their account priority: '{pass1}'"
    else:
        p2_prompt = f"Rewrite this draft to include a brief thank you for being a valued account holder: '{pass1}'"

    p2 = query_groq(system_prompt, p2_prompt)
    pass2 = p2.get("draft", pass1)

    # Pass 3: Strategy & Action Insertion (Your Exact If/Else Loop -> Groq Prompting)
    if strategy == "UPSELL_VIP":
        p3_prompt = f"Integrate a clear notice into this message stating that a dedicated VIP account manager has been assigned to their support queue: '{pass2}'"
        p3 = query_groq(system_prompt, p3_prompt)
        pass3 = p3.get("draft", pass2)
    elif strategy == "RETENTION_OFFER":
        p3_prompt = f"Integrate a clear notice into this message stating that a 15% loyalty retention credit has been flagged for their account: '{pass2}'"
        p3 = query_groq(system_prompt, p3_prompt)
        pass3 = p3.get("draft", pass2)
    else:
        pass3 = pass2

    return DraftMessage(
        raw_draft=pass1,
        tone_adjusted_draft=pass2,
        compliance_checked_draft=pass3,
    )

# 3. Critique-Refiner Stage (Policy & HITL Enforcement)
async def run_critique_refiner(draft: DraftMessage) -> RefinerOutput:
    message = draft.compliance_checked_draft

    # Direct Policy Guardrail: Catch refunds, credits, discounts, or escalation flags
    flag_keywords = [
        "100% free",
        "free forever",
        "refund",
        "credit",
        "discount",
        "cancel",
        "compensation",
    ]
    if any(keyword in message.lower() for keyword in flag_keywords):
        return RefinerOutput(
            approved=False,
            feedback="Financial waiver or retention override detected in draft. Escalated for human authorization.",
            final_output=message,
        )

    system_prompt = """
    You are a Safety & Policy Auditor for an enterprise customer service team.
    Evaluate the draft response and return JSON with:
    - "approved": boolean (Set to FALSE if the draft offers financial compensation, refunds, accounts overrides, or handles high churn risk)
    - "feedback": string
    - "final_output": string
    """
    user_prompt = f"Audit this draft for customer service safety, accuracy, and tone: '{message}'"
    res_json = query_groq(system_prompt, user_prompt)

    if not res_json:
        return RefinerOutput(
            approved=False,  # Safe default to HITL if LLM call fails
            feedback="Audit failed to execute. Routing to human review for safety.",
            final_output=message,
        )

    return RefinerOutput(
        approved=res_json.get("approved", False),
        feedback=res_json.get(
            "feedback", "Escalated for safety and policy check."
        ),
        final_output=res_json.get("final_output", message),
    )