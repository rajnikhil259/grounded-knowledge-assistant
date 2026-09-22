"""Pydantic models = the *contract* for structured LLM output and for the API."""
from pydantic import BaseModel


class Answer(BaseModel):
    """What the generate step must return (structured output, not free text)."""
    found_in_context: bool      # False => the passages do not contain the answer
    answer: str
    source_ids: list[int]       # numbers of the passages used, e.g. [1, 3]
    confidence: float           # 0.0 - 1.0, model's self-reported confidence


class Verdict(BaseModel):
    """What the validate step must return."""
    supported: bool             # True only if EVERY claim is backed by the passages
    unsupported_claims: list[str]
    reason: str


class AskRequest(BaseModel):
    question: str
    collection: str = "default"
    top_k: int | None = None
