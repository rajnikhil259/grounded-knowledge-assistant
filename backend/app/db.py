"""Database layer: PostgreSQL + pgvector.

Two tables:
  chunks     - the knowledge base (text + its embedding vector)
  query_logs - one row per question (latency, tokens, cost, validation result)
"""
import psycopg
from pgvector.psycopg import register_vector

from .config import DATABASE_URL, EMBED_DIM

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS chunks (
    id           SERIAL PRIMARY KEY,
    collection   TEXT NOT NULL DEFAULT 'default',   -- lets you keep several document sets apart
    source       TEXT NOT NULL,                     -- file name
    page         INT  NOT NULL DEFAULT 1,
    chunk_index  INT  NOT NULL,
    content      TEXT NOT NULL,
    embedding    vector({EMBED_DIM}) NOT NULL       -- the embedding, stored as a vector
);

-- HNSW index = fast approximate nearest-neighbour search using cosine distance
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_collection_idx ON chunks (collection);

CREATE TABLE IF NOT EXISTS query_logs (
    id                SERIAL PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    collection        TEXT,
    question          TEXT,
    answer            TEXT,
    status            TEXT,        -- answered | not_found | unverified | blocked
    latency_ms        INT,
    retrieval_ms      INT,
    input_tokens      INT,
    output_tokens     INT,
    est_cost_usd      NUMERIC(12, 6),
    chunks_retrieved  INT,
    top_similarity    REAL,
    validation_passed BOOLEAN,     -- NULL when validation never ran
    retries           INT,
    llm_calls         INT,
    retry_log         JSONB        -- every REJECTED draft + why, so failures are inspectable later
);
"""


def init_db() -> None:
    """Create the pgvector extension and tables (safe to call repeatedly)."""
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(SCHEMA)


def get_conn() -> psycopg.Connection:
    """New connection that understands the `vector` type (needs the extension to exist)."""
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn
