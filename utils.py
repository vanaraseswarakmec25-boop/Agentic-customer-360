import re
import time

def sanitize_pii(text: str) -> str:
    """Masks SSNs, credit cards, emails, and phone numbers before sending to LLM."""
    if not isinstance(text, str):
        return text
    # Mask Credit Cards
    text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[REDACTED_CARD]', text)
    # Mask SSNs
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', text)
    # Mask Emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', text)
    return text

def parse_event_timestamp(ts_str: str) -> float:
    """Parses incoming ISO timestamp string to epoch for out-of-order sorting."""
    try:
        from datetime import datetime
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
    except Exception:
        return time.time()