# PrivateSearch AI: your personal knowledge engine

A local-first AI search engine for your own documents. It finds passages by keyword **and** by meaning, answers questions with citations, and maps how ideas connect across files. Documents, indexes and search history never leave your computer.

## Features

| Feature | How it works |
|---|---|
| Ingestion | PDF (PyMuPDF), DOCX (python-docx), TXT, Markdown. Text is split into ~160-word, sentence-aware chunks that never cross a page, so page citations are exact. |
| Hybrid search | SQLite **FTS5/BM25** keyword ranking + **Sentence-Transformers** embeddings searched with **FAISS**, merged by Reciprocal Rank Fusion. |
| Question answering | Retrieves passages, then asks a **local** model through Ollama to answer using only those passages, with `[n]` citations. Invented citations are stripped, unsupported answers are flagged, and unanswerable questions return "not found". With no model installed, it quotes the best sentences verbatim instead. |
| Knowledge Discovery | Finds passages in *other* documents that are close in meaning to your search, links them, and states the evidence for every link (similarity score + literal shared terms). |
| History, management, privacy | Local history with clear button, add/remove documents, rebuild index, telemetry off, remote LLM addresses blocked. |

## Quick start (Windows 11, PowerShell, Python 3.13)

```powershell
cd privatesearch-ai
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Or run `.\run.ps1`, or press **F5** in VS Code and choose "PrivateSearch AI (Streamlit)".

On first launch the embedding model (about 90 MB) downloads once. Click **Try the sample documents** to see everything working.

### Optional: local language model (Ollama)

1. Install from https://ollama.com/download (or `winget install Ollama.Ollama`).
2. `ollama pull llama3.2:3b` (about 2 GB, runs on CPU with 16 GB RAM).
3. Leave Ollama running, then open the **Ask** page. Other good small models: `qwen2.5:3b`, `phi3:mini`.

## Testing

```powershell
pytest -v                      # 33 automated tests, no network or model needed
python scripts/smoke_ui.py     # headless click-through of every page
```

The tests use a small built-in hashing embedder so they run anywhere. Manual checklist:

1. Documents page: add all files in `sample_docs/`. Add one again and confirm "Already indexed".
2. Search "AI agents" in Hybrid, Keywords and Meaning modes and compare.
3. Ask "What does a supervisor agent do?" and open each citation. Then ask "What is the capital of Australia?" and confirm "not found".
4. Discover "AI agents": hover over lines, click circles, read "Why these passages connect".
5. Remove `study_habits.txt`; confirm it no longer appears. Restart the app; confirm everything persists.
6. Privacy page: clear history; delete all data.

## Architecture

```
app.py                     Streamlit entry point (routing only)
ui/                        Presentation: theme.py (CSS), components.py (escaped HTML), graph.py (Plotly map), pages.py
privatesearch/             All logic, no Streamlit imports
  config.py                Settings from env/.env, telemetry switches, localhost check
  extraction.py            PDF / DOCX / TXT / MD to text (parse only, never execute)
  chunking.py              Sentence-aware chunks with overlap, page-safe
  embeddings.py            SentenceTransformer embedder + offline hashing fallback
  vector_index.py          FAISS IndexIDMap2(IndexFlatIP), NumPy fallback
  database.py              SQLite: documents, chunks, FTS5, embeddings, history, prefs
  hybrid.py                Safe FTS5 query builder + Reciprocal Rank Fusion
  llm.py                   Ollama client (localhost only)
  qa.py                    Prompting, citation verification, grounding check, extractive fallback
  discovery.py             Knowledge graph construction
  engine.py                Facade used by the UI and the tests
sample_docs/  tests/  scripts/
data/                      Your database and stored files (git-ignored)
```

Data flow: file, extract, chunk, embed, then one SQLite transaction (chunks + FTS + vectors), then FAISS. Embeddings are stored in SQLite, so FAISS is rebuilt from disk at start-up and the index persists across restarts.

## Privacy

* No document, query or answer is sent to any external service. Ollama is called on `localhost` only; other hosts are refused unless you set `PSAI_ALLOW_REMOTE_LLM=1`.
* Telemetry is disabled (`gatherUsageStats=false`, Hugging Face and other analytics env vars). The UI uses system fonts and loads nothing from a CDN.
* After the first run, set `PSAI_OFFLINE=1` to forbid all model downloads.
* Original files are copied into `data/uploads/` under their SHA-256 name (so a file name can never escape that folder). Deleting a document removes the copy, its chunks, FTS entries and vectors.
* Local storage is not encrypted. Use Windows BitLocker for protection if the laptop is lost.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process Bypass`, then activate again. |
| `pip install faiss-cpu` fails | Run `python --version`; it must be 3.13 (or 3.10-3.12). Upgrade pip. If it still fails, delete the faiss line: the app switches to an exact NumPy index automatically. |
| Sidebar shows "offline fallback model" | The embedding model could not download. Connect to the internet once, restart, or run `pip install sentence-transformers`. Search still works but is word-based, not semantic. |
| "Ollama is not running" | Open the Ollama app, then `ollama list` in PowerShell. Pull the model named on the Privacy page. |
| Answers are slow | Use a 3B model, close other apps, and keep Results per search at 5-8. |
| "No readable text" for a PDF | The PDF is a scan of images. OCR is not included. |
| "Index was built with a different embedding model" | Documents page, **Rebuild index**. |
| Port already in use | `streamlit run app.py --server.port 8502` |

## Limits and honest notes

* Discovery links are suggestions backed by embedding similarity. The tool shows both passages so *you* decide; it never states a relationship the text does not support.
* The grounding score is a quick word-overlap heuristic, not proof of correctness.
* Scanned PDFs (no text layer) are not supported.
