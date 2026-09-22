"""Retrieval = find the chunks whose meaning is closest to the question."""
import numpy as np

from . import llm
from .db import get_conn


def search(question: str, collection: str, top_k: int) -> list[dict]:
    """Embed the question, then ask Postgres for the top_k nearest chunks.

    `<=>` is pgvector's cosine DISTANCE operator (0 = identical direction).
    similarity = 1 - distance, so higher is better (max 1.0)."""
    qvec = np.array(llm.embed([question], "RETRIEVAL_QUERY")[0], dtype=np.float32)
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, source, page, content, 1 - (embedding <=> %s) AS similarity
            FROM chunks
            WHERE collection = %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (qvec, collection, qvec, top_k),
        ).fetchall()
    return [
        {"id": r[0], "source": r[1], "page": r[2], "content": r[3], "similarity": float(r[4])}
        for r in rows
    ]
