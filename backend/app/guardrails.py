"""Rule-based input guardrail. Cheap, fast, runs BEFORE any LLM call (so blocked
requests cost nothing). It is a first line of defence, not a complete one - the
generate prompt and the validate step add further layers."""
import re

MAX_QUESTION_CHARS = 500

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |the )?(previous|prior|above|system)",
    r"(reveal|show|print|repeat) (me )?(your |the )?(system )?(prompt|instructions)",
    r"you are now\b",
    r"\bjailbreak\b",
    r"developer mode",
    r"pretend (that )?you (are|have no)",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def check_question(question: str) -> tuple[bool, str]:
    """Return (allowed, reason). reason is empty when allowed."""
    q = (question or "").strip()
    if not q:
        return False, "Please enter a question."
    if len(q) > MAX_QUESTION_CHARS:
        return False, f"Question is too long (max {MAX_QUESTION_CHARS} characters)."
    for pattern in _COMPILED:
        if pattern.search(q):
            return False, "This looks like an attempt to override the assistant's instructions."
    return True, ""
