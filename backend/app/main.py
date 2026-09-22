"""FastAPI layer: exposes the graph over HTTP for the React UI."""
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import graph, metrics
from .config import CHUNK_SIZE
from .db import get_conn, init_db
from .ingest import SUPPORTED, ingest_file
from .schemas import AskRequest


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()          # create extension + tables on startup
    yield


app = FastAPI(title="Grounded Knowledge Assistant", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return graph.run(req.question, req.collection, req.top_k)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/ingest")
def ingest(file: UploadFile = File(...), collection: str = Form("default"), chunk_size: int = Form(CHUNK_SIZE)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(400, f"Unsupported file type. Use one of: {sorted(SUPPORTED)}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
    try:
        n = ingest_file(Path(tmp.name), collection, chunk_size, source_name=file.filename)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return {"file": file.filename, "collection": collection, "chunks": n}


@app.get("/collections")
def collections():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT collection, count(DISTINCT source), count(*) FROM chunks GROUP BY collection ORDER BY collection"
        ).fetchall()
    return [{"collection": r[0], "documents": r[1], "chunks": r[2]} for r in rows]


@app.get("/stats")
def stats():
    return metrics.get_stats()

@app.get("/retries")
def retries():
    return metrics.get_retries()
