"""Observability: log every query to Postgres and aggregate for the stats page."""
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M
from .db import get_conn


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimated USD cost. The free tier bills $0, this shows what it WOULD cost at paid prices."""
    return (input_tokens * PRICE_INPUT_PER_M + output_tokens * PRICE_OUTPUT_PER_M) / 1_000_000


def log_query(r: dict) -> None:
    # r["retry_log"] is a plain Python list ([] if nothing was ever rejected).
    # Jsonb(...) tells psycopg to store it as a real JSONB value, not a Python-repr string.
    params = {**r, "retry_log": Jsonb(r.get("retry_log", []))}
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO query_logs
               (collection, question, answer, status, latency_ms, retrieval_ms, input_tokens,
                output_tokens, est_cost_usd, chunks_retrieved, top_similarity,
                validation_passed, retries, llm_calls, retry_log)
               VALUES (%(collection)s, %(question)s, %(answer)s, %(status)s, %(latency_ms)s,
                %(retrieval_ms)s, %(input_tokens)s, %(output_tokens)s, %(est_cost_usd)s,
                %(chunks_retrieved)s, %(top_similarity)s, %(validation_passed)s,
                %(retries)s, %(llm_calls)s, %(retry_log)s)""",
            params,
        )


def get_retries(limit: int = 20) -> list[dict]:
    """Every past question where the validator rejected at least one draft -
    exactly the data you'd pull up if asked 'what did the validator actually catch?'"""
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT created_at, question, answer, retries, retry_log
               FROM query_logs
               WHERE retries > 0
               ORDER BY id DESC LIMIT %s""",
            (limit,),
        )
        return cur.fetchall()


def get_stats() -> dict:
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT
                 count(*)                                                    AS total_queries,
                 coalesce(round(avg(latency_ms)), 0)                         AS avg_latency_ms,
                 coalesce(round(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)), 0) AS p95_latency_ms,
                 coalesce(sum(input_tokens), 0)                              AS input_tokens,
                 coalesce(sum(output_tokens), 0)                             AS output_tokens,
                 coalesce(sum(est_cost_usd), 0)::float                       AS est_cost_usd,
                 count(*) FILTER (WHERE status = 'answered')                 AS answered,
                 count(*) FILTER (WHERE status = 'not_found')                AS not_found,
                 count(*) FILTER (WHERE status = 'unverified')               AS unverified,
                 count(*) FILTER (WHERE status = 'blocked')                  AS blocked,
                 coalesce(avg(validation_passed::int) FILTER (WHERE validation_passed IS NOT NULL), 0)::float
                                                                             AS validation_pass_rate
               FROM query_logs"""
        )
        summary = cur.fetchone()
        cur.execute(
            """SELECT created_at, question, status, latency_ms, input_tokens, output_tokens,
                      est_cost_usd::float AS est_cost_usd, validation_passed, retries
               FROM query_logs ORDER BY id DESC LIMIT 15"""
        )
        recent = cur.fetchall()
    return {"summary": summary, "recent": recent}