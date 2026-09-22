"""Ingestion pipeline:  file -> pages -> chunks -> embeddings -> pgvector.

Run from the backend folder:
    python -m app.ingest ../sample_docs --collection default
    python -m app.ingest ../sample_docs --collection small --chunk-size 300
"""
import argparse
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from . import llm
from .config import CHUNK_SIZE
from .db import get_conn, init_db

SUPPORTED = {".pdf", ".txt", ".md"}


def load_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...]. Page numbers let us cite sources."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, (p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


def ingest_file(
    path: Path,
    collection: str = "default",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int | None = None,
    source_name: str | None = None,
) -> int:
    """Chunk, embed and store one file. Returns the number of chunks stored."""
    path = Path(path)
    source = source_name or path.name
    if chunk_overlap is None:
        chunk_overlap = int(chunk_size * 0.12)   # ~12% overlap keeps sentences from being cut off

    # Splits on paragraphs, then lines, then sentences/words - tries to keep meaning together.
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    rows: list[tuple[int, int, str]] = []   # (page, chunk_index, text)
    idx = 0
    for page_no, text in load_pages(path):
        for piece in splitter.split_text(text):
            if piece.strip():
                rows.append((page_no, idx, piece))
                idx += 1
    if not rows:
        return 0

    vectors = llm.embed([r[2] for r in rows], "RETRIEVAL_DOCUMENT")

    init_db()
    with get_conn() as conn:
        # re-ingesting the same file replaces it instead of duplicating it
        conn.execute("DELETE FROM chunks WHERE collection=%s AND source=%s", (collection, source))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (collection, source, page, chunk_index, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (collection, source, p, i, t, np.array(v, dtype=np.float32))
                    for (p, i, t), v in zip(rows, vectors)
                ],
            )
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest documents into pgvector")
    ap.add_argument("path", help="a file or a folder of .pdf/.txt/.md files")
    ap.add_argument("--collection", default="default")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=None)
    args = ap.parse_args()

    target = Path(args.path)
    files = [target] if target.is_file() else sorted(p for p in target.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        raise SystemExit(f"No supported files found in {target}")
    for f in files:
        n = ingest_file(f, args.collection, args.chunk_size, args.chunk_overlap)
        print(f"{f.name}: {n} chunks -> collection '{args.collection}' (chunk_size={args.chunk_size})")


if __name__ == "__main__":
    main()
