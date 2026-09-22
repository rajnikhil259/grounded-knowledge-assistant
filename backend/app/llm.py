"""Thin wrapper around the Gemini API. Everything that talks to the model lives here,
so swapping providers later means changing only this file."""
import time
from functools import lru_cache

from google import genai
from google.genai import types
from pydantic import BaseModel

from .config import EMBED_DIM, EMBED_MODEL, GEMINI_API_KEY, LLM_MODEL

EMBED_BATCH = 50


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Put it in backend/.env")
    return genai.Client(api_key=GEMINI_API_KEY)


def _with_retry(fn, attempts: int = 4):
    """Free tiers rate-limit (HTTP 429) and sometimes return 503. Back off and retry."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", None)
            if code in (429, 500, 503) and i < attempts - 1:
                time.sleep(2 ** (i + 1))   # 2s, 4s, 8s
                continue
            raise


def embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Turn texts into vectors.
    task_type: 'RETRIEVAL_DOCUMENT' for chunks, 'RETRIEVAL_QUERY' for questions -
    the model embeds them slightly differently so questions land near their answers."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        resp = _with_retry(
            lambda b=batch: _client().models.embed_content(
                model=EMBED_MODEL,
                contents=b,
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=EMBED_DIM
                ),
            )
        )
        vectors.extend(e.values for e in resp.embeddings)
    return vectors


def generate_structured(system: str, prompt: str, schema: type[BaseModel]):
    """Call the LLM and force the reply to match `schema` (structured output).
    Returns (parsed_object, input_tokens, output_tokens)."""
    resp = _with_retry(
        lambda: _client().models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
    )
    parsed = resp.parsed if isinstance(resp.parsed, schema) else schema.model_validate_json(resp.text)
    usage = resp.usage_metadata
    in_tok = (usage.prompt_token_count or 0) if usage else 0
    # "thinking" tokens are billed as output tokens too
    out_tok = ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0
    return parsed, in_tok, out_tok
