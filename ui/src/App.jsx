import { useEffect, useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function api(path, options) {
  const res = await fetch(API + path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

const STATUS = {
  answered: { label: "Answered from your documents", cls: "ok" },
  not_found: { label: "Not in the documents", cls: "muted" },
  unverified: { label: "Could not verify an answer", cls: "warn" },
  blocked: { label: "Question blocked", cls: "warn" },
};

export default function App() {
  const [tab, setTab] = useState("ask");
  return (
    <div className="page">
      <header className="top">
        <h1>Grounded Knowledge Assistant</h1>
        <nav>
          <button className={tab === "ask" ? "on" : ""} onClick={() => setTab("ask")}>Ask</button>
          <button className={tab === "stats" ? "on" : ""} onClick={() => setTab("stats")}>Usage</button>
        </nav>
      </header>
      {tab === "ask" ? <Ask /> : <Stats />}
    </div>
  );
}

function Ask() {
  const [collections, setCollections] = useState([]);
  const [collection, setCollection] = useState("default");
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [uploadMsg, setUploadMsg] = useState("");

  const loadCollections = () =>
    api("/collections").then(setCollections).catch((e) => setError(e.message));
  useEffect(() => { loadCollections(); }, []);

  async function ask() {
    if (!question.trim() || busy) return;
    setBusy(true); setError(""); setResult(null);
    try {
      setResult(await api("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, collection }),
      }));
    } catch (e) { setError(e.message); }
    setBusy(false);
  }

  async function upload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setUploadMsg(`Indexing ${file.name}…`);
    const form = new FormData();
    form.append("file", file);
    form.append("collection", collection);
    try {
      const r = await api("/ingest", { method: "POST", body: form });
      setUploadMsg(`${r.file} indexed: ${r.chunks} chunks in “${r.collection}”.`);
      loadCollections();
    } catch (err) { setUploadMsg(err.message); }
    e.target.value = "";
  }

  return (
    <main>
      <section className="panel">
        <div className="row">
          <label>Document set
            <input list="cols" value={collection} onChange={(e) => setCollection(e.target.value)} />
            <datalist id="cols">
              {collections.map((c) => <option key={c.collection} value={c.collection} />)}
            </datalist>
          </label>
          <label className="file">Add a document (.pdf, .txt, .md)
            <input type="file" accept=".pdf,.txt,.md" onChange={upload} />
          </label>
        </div>
        {uploadMsg && <p className="hint">{uploadMsg}</p>}
        <textarea
          rows={3}
          placeholder="Ask something your documents can answer…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask(); }}
        />
        <button className="primary" onClick={ask} disabled={busy || !question.trim()}>
          {busy ? "Checking the documents…" : "Ask"}
        </button>
      </section>

      {error && <p className="error">{error}</p>}
      {result && <Result r={result} />}
    </main>
  );
}

function Result({ r }) {
  const s = STATUS[r.status] || STATUS.not_found;
  return (
    <section className="result">
      <p className={`status ${s.cls}`}>{s.label}</p>
      <p className="answer">{r.answer}</p>

      {r.sources.length > 0 && (
        <ol className="sources">
          {r.sources.map((src) => (
            <li key={src.id}>
              <div className="src-head">
                <span className="cite">[{src.id}]</span> {src.source}, page {src.page}
                <span className="sim">match {Math.round(src.similarity * 100)}%</span>
              </div>
              <blockquote>{src.snippet}</blockquote>
            </li>
          ))}
        </ol>
      )}

      <dl className="metrics">
        <div><dt>Time</dt><dd>{(r.latency_ms / 1000).toFixed(1)} s</dd></div>
        <div><dt>Tokens</dt><dd>{r.input_tokens + r.output_tokens}</dd></div>
        <div><dt>Est. cost</dt><dd>${r.est_cost_usd.toFixed(5)}</dd></div>
        <div><dt>Passages used</dt><dd>{r.chunks_retrieved}</dd></div>
        <div><dt>Fact-check</dt><dd>{r.validation_passed === null ? "skipped" : r.validation_passed ? "passed" : "failed"}{r.retries ? ` (retried ${r.retries}×)` : ""}</dd></div>
      </dl>
    </section>
  );
}

function Stats() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => { api("/stats").then(setData).catch((e) => setError(e.message)); }, []);
  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="hint">Loading…</p>;
  const s = data.summary;
  return (
    <main>
      <section className="panel">
        <dl className="metrics big">
          <div><dt>Questions asked</dt><dd>{s.total_queries}</dd></div>
          <div><dt>Average time</dt><dd>{(s.avg_latency_ms / 1000).toFixed(1)} s</dd></div>
          <div><dt>Slowest 5% (p95)</dt><dd>{(s.p95_latency_ms / 1000).toFixed(1)} s</dd></div>
          <div><dt>Tokens used</dt><dd>{(s.input_tokens + s.output_tokens).toLocaleString()}</dd></div>
          <div><dt>Estimated cost</dt><dd>${s.est_cost_usd.toFixed(4)}</dd></div>
          <div><dt>Fact-check pass rate</dt><dd>{Math.round(s.validation_pass_rate * 100)}%</dd></div>
        </dl>
        <p className="hint">
          Answered {s.answered} · Not found {s.not_found} · Unverified {s.unverified} · Blocked {s.blocked}
        </p>
      </section>
      <section className="panel">
        <table>
          <thead><tr><th>Question</th><th>Result</th><th>Time</th><th>Tokens</th><th>Fact-check</th></tr></thead>
          <tbody>
            {data.recent.map((q, i) => (
              <tr key={i}>
                <td>{q.question}</td>
                <td>{STATUS[q.status]?.label || q.status}</td>
                <td>{(q.latency_ms / 1000).toFixed(1)} s</td>
                <td>{q.input_tokens + q.output_tokens}</td>
                <td>{q.validation_passed === null ? "-" : q.validation_passed ? "passed" : "failed"}</td>
              </tr>
            ))}
            {data.recent.length === 0 && <tr><td colSpan={5}>No questions yet. Ask one on the first tab.</td></tr>}
          </tbody>
        </table>
      </section>
    </main>
  );
}
