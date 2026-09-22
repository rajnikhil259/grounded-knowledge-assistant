"""Step-by-step self-test. Run from the backend folder:  python check_setup.py
Each check prints OK or tells you exactly what to fix."""
import sys

from app import config


def step(name, fn):
    try:
        detail = fn()
        print(f"[ OK ] {name}" + (f"  ->  {detail}" if detail else ""))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}\n        {type(e).__name__}: {e}")
        return False


def check_key():
    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY.startswith("paste_"):
        raise RuntimeError("GEMINI_API_KEY missing. Copy .env.example to backend/.env and add your key.")
    return "key found"


def check_embed():
    from app import llm
    v = llm.embed(["hello world"], "RETRIEVAL_QUERY")[0]
    return f"{config.EMBED_MODEL}, vector length {len(v)} (expected {config.EMBED_DIM})"


def check_llm():
    from app import llm
    from app.schemas import Verdict
    obj, i, o = llm.generate_structured("Reply as JSON.", "Is 2+2=4? supported=true if yes.", Verdict)
    return f"{config.LLM_MODEL}, structured output parsed, tokens in/out = {i}/{o}"


def check_db():
    from app.db import get_conn, init_db
    init_db()
    with get_conn() as conn:
        v = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
    return f"connected, pgvector {v[0]}"


if __name__ == "__main__":
    results = [step("API key", check_key)]
    if results[0]:
        results += [step("Embeddings", check_embed), step("LLM structured output", check_llm)]
    results.append(step("Database + pgvector", check_db))
    sys.exit(0 if all(results) else 1)
