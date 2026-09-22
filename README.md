# Grounded Knowledge Assistant (RAG + validation agent)

Ask questions about your own documents and get answers with citations back to the exact
passage they came from. If the documents don't support an answer, the system says so
instead of guessing — and every rejected draft the fact-checker catches is logged, not
just thrown away.

```
Question -> Guardrail -> Retrieve (pgvector) -> Generate -> Validate -> Answer + sources
                 |             |                     ^         |
              blocked      nothing relevant           |         +--(failed once)--> retry generate
             (0 LLM calls)  (0 LLM calls)              +---------+--(failed twice)--> refuse
```

Built with FastAPI, LangGraph, PostgreSQL + pgvector, Google Gemini, and a React front end.
Every query is logged (latency, tokens, estimated cost, retries) and every rejected draft
the validator catches is saved with the reason it was rejected — not silently discarded.

## Folder map
| Path | What it is |
|---|---|
| `docker-compose.yml` | Postgres + pgvector, and (via `--profile full`) the containerized API |
| `backend/app/config.py` | All settings, read from `.env` |
| `backend/app/db.py` | Tables + pgvector setup (`chunks`, `query_logs`) |
| `backend/app/llm.py` | The only file that talks to Gemini (embeddings + structured output) |
| `backend/app/ingest.py` | PDF/text -> chunks -> embeddings -> database |
| `backend/app/retrieval.py` | Vector similarity search |
| `backend/app/guardrails.py` | Blocks prompt-injection style questions before any LLM call |
| `backend/app/graph.py` | **The LangGraph workflow** (the core of the project) |
| `backend/app/metrics.py` | Logs latency/tokens/cost/retries, builds the stats + retry-history data |
| `backend/app/main.py` | FastAPI endpoints (`/ask`, `/ingest`, `/collections`, `/stats`, `/retries`) |
| `backend/eval/` | Evaluation sets + the evaluation script |
| `backend/tests/` | Unit tests for the graph logic (mocked LLM, no API key or database needed) |
| `backend/check_setup.py` | One-command self-test: API key, embeddings, LLM, database |
| `sample_docs/` | Source documents to ingest (swap in your own real ones here) |
| `ui/` | React front end |

## Setup, step by step

**Step 1 - Database.** From this folder:
```
docker compose up -d
docker compose ps        # STATUS should say "healthy"
```

**Step 2 - Python environment.**
```
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

**Step 3 - API key.** Get a free key at https://aistudio.google.com/apikey.
Copy `.env.example` to `backend/.env` and paste the key in. Never commit this file
(it's already in `.gitignore`).

**Step 4 - Self-test.** `python check_setup.py` -> all four lines must say `[ OK ]`.

**Step 5 - Run the unit tests.** `pytest -q` -> 7 passed. These use a mocked LLM, so
they run instantly and need no API key or running database.

**Step 6 - Ingest a document.**
```
python -m app.ingest ../sample_docs/your_document.pdf --collection your_collection_name
```
A `collection` is just a named bucket of chunks — questions asked against one collection
only ever search that collection, so different documents never mix unless you put them
in the same collection on purpose.

**Step 7 - Start the API.** `uvicorn app.main:app --reload`  (interactive docs at
http://localhost:8000/docs)

**Step 8 - Start the UI.** New terminal: `cd ui`, `npm install`, `npm run dev`, open
http://localhost:5173. Upload a document from the browser, pick its collection name, and ask it questions.

**Step 9 - Evaluate.** Write a `.json` eval set for your real document (see
`backend/eval/eval_set.json` for the format: `question`, `answerable`, `expected_keywords`).
Include direct-lookup questions, a couple of multi-hop questions that require combining
two separate rules/facts, and a few clearly unanswerable ones.
```
python -m eval.run_eval --collection your_collection_name --file eval/your_eval_set.json --tag v1
```
The summary reports retrieval hit@k, answer pass rate, correct-refusal rate,
hallucination rate, average latency, and estimated cost. Compare chunk sizes or
validation on/off by re-ingesting into a second collection and re-running with `--tag`
and `--no-validation`.

**Step 10 - Containerize everything.** `docker compose --profile full up -d --build`
runs the database *and* the API as containers, so the whole system starts with one
command and doesn't depend on your local Python setup.

**Step 11 - Inspect what the validator caught.** Open http://localhost:8000/retries
to see every question that needed at least one retry: the rejected draft, which claims
were flagged as unsupported, and why. The UI also shows this inline, as a collapsible
"What the fact-checker caught" section, whenever a question you ask triggers a retry.

## What each status means
| `status` | Meaning | LLM calls made |
|---|---|---|
| `answered` | A supported answer, cited to source passages | 2 (generate + validate), or 4 if one retry happened |
| `not_found` | No chunk was similar enough to the question, or the model itself said the answer isn't in the passages | 0 or 1 |
| `unverified` | The model claimed an answer was in the passages, but the validator rejected it twice in a row | 4 |
| `blocked` | The guardrail flagged the question (e.g. a prompt-injection attempt) before anything else ran | 0 |

## Known limitations
- **Scanned or image-only PDFs will silently produce 0 or very few chunks.** `ingest.py`
  extracts text via `pypdf`, which only works on digitally-created PDFs. A scanned
  document would need an OCR step first (image -> text) before chunking - not yet
  implemented here.
- **The validator checks support, not relevance.** It confirms a claim is stated
  *somewhere* in the retrieved passages, but not that the passages actually answer the
  question asked. On one real test (see `backend/eval/results/`), the system answered a
  question about leaving campus with a real but wrong rule about the general hostel
  curfew - a true sentence from the document, just not the one the question needed.
  This is a retrieval/generation mismatch, not a hallucination, and it passed validation
  because every word of it really was in the source text.
- **Rejected drafts are only logged from the point `retry_log` was added onward** -
  retries that happened earlier in development weren't captured.
- The evaluation sets are small (12-16 questions). Real deployment would need a larger,
  continuously updated set, ideally including questions sourced from actual user traffic.

