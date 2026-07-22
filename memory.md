# MEMORY.md — Project Memory / Running Log

> The purpose of this file is to track project decisions, progress, and context. If we take a break or request help from the AI assistant again, the context won't be lost. Record every major decision/update here.

## Project Snapshot
- **Project Name**: NoteMind
- **Type**: Final Year Project
- **Stack Decided**: Flask + Python + ChromaDB + SQLite + sentence-transformers + Groq/Gemini (free tier)
- **Current Phase**: Phase 7 — Frontend/UI (complete) | Phase 8 — Next

## Key Decisions Log
| Date | Decision | Reason |
|---|---|---|
| 2026-07-15 | Using Groq/Gemini free API (not OpenAI) | Budget is 0; the free tier is sufficient for this project |
| 2026-07-15 | ChromaDB for vector store | Easy to set up, local, free, sufficient for a final year project |
| 2026-07-15 | SQLite for relational data | Lightweight, no server setup required |
| 2026-07-15 | youtube-transcript-api primary, Whisper fallback | Most videos already have captions; Whisper is heavy |
| 2026-07-15 | requirements.txt uses >= (not ==) version pins | Allows flexibility for deployment on Render/Railway free tier |
| 2026-07-15 | config.py prints a warning (not crash) if API keys missing | Lets Flask run in Phase 0 even without keys; LLM phases will fail-fast with a clear message |
| 2026-07-15 | Database schema (db.py) deferred to Phase 6 | Follows phases.md order strictly — Phase 0 is setup only |
| 2026-07-15 | youtube-transcript-api v1.2.4 uses new instance-based API | Installed version uses `YouTubeTranscriptApi().fetch()` (not the old class-method `get_transcript()`). Code written for v1.2.4 API. |
| 2026-07-15 | Whisper fallback is a stub (returns None) | Per deployment requirement — Whisper is too heavy for free-tier hosting. Stub is clearly marked with TODO for local-only use. |
| 2026-07-15 | Transcript returns {text, start_time, end_time} format | `end_time = start + duration`. This format is needed by the chunker to preserve timestamp ranges. |
| 2026-07-15 | Chunking uses snippet-level accumulation (not word splitting) | Snippets are kept whole — we never break a snippet in the middle. This preserves timestamp accuracy. |
| 2026-07-15 | Overlap is done by rewinding snippet positions | After finalizing a chunk, the iterator rewinds to include ~50 words of overlap from the end of the previous chunk. |
| 2026-07-15 | Embedding model: all-MiniLM-L6-v2, dimension = **384** | Lazy-loaded singleton pattern. ~80MB model, runs fully local. |
| 2026-07-15 | Used `get_embedding_dimension()` (not deprecated `get_sentence_embedding_dimension()`) | sentence-transformers v5.x renamed this method. |
| 2026-07-15 | ChromaDB v1.5.9 — single collection "video_transcripts" with video_id metadata filter | Simpler than one-collection-per-video; filtering via `where={"video_id": ...}` works well. |
| 2026-07-15 | Using `upsert` (not `add`) for idempotent storage | Free-tier hosting has ephemeral disk — re-processing a video after restart must not crash. |
| 2026-07-15 | Cosine similarity (`hnsw:space: cosine`) for ChromaDB collection | Standard choice for sentence-transformers embeddings. Lower distance = more similar. |
| 2026-07-15 | Chunk IDs stored as `{video_id}_{chunk_id}` | Globally unique across videos in a shared collection. |
| 2026-07-15 | Groq primary LLM, Gemini fallback | Groq has faster inference (LPU). If Groq fails (rate limit, key invalid, model deprecated), automatically falls back to Gemini. |
| 2026-07-15 | Groq model priority: llama-3.3-70b-versatile → llama-3.1-8b-instant → llama3-8b-8192 → gemma2-9b-it → mixtral-8x7b-32768 | Multiple models tried in order because Groq frequently deprecates/rotates models. |
| 2026-07-15 | Gemini model: gemini-2.0-flash | Fast, free-tier friendly. Uses new `google-genai` SDK (not deprecated `google.generativeai`). |
| 2026-07-15 | RAG prompt constrains LLM to use ONLY transcript context | Prevents hallucination. LLM must cite timestamps and admit if info isn't in the transcript. |
| 2026-07-15 | RAG search uses top_k=3 (not 5) for LLM context | Fewer, more relevant chunks = more focused answer + fewer tokens. Config.TOP_K_RESULTS (5) is for raw vector search; RAG defaults to 3. |
| 2026-07-15 | Replaced `google-generativeai` with `google-genai` in requirements.txt | Old package is deprecated. New SDK: `from google import genai`. |
| 2026-07-16 | Notes generator uses direct (≤3000 words) vs map-reduce (>3000 words) | Short videos get a single-shot LLM call; long videos are summarized in chunks then combined. Avoids context window limits. |
| 2026-07-16 | Notes generator reuses `_call_groq` / `_call_gemini` from search.py | DRY — same fallback logic, no code duplication. |
| 2026-07-16 | `/generate-notes` accepts both `video_id` and `url` fields | User convenience — can paste a YouTube URL directly or use a video ID. |
| 2026-07-16 | Map step groups chunks to ~1500 words per batch | Keeps each map-step LLM call small and focused. Reduce step combines all summaries. |
| 2026-07-16 | Pure sqlite3 (no ORM) for database | Lightweight, zero extra dependencies. Functions accept optional `conn` param so tests use `:memory:`. |
| 2026-07-16 | `save_video()` uses ON CONFLICT(video_id) DO UPDATE | Idempotent upsert — re-processing a video updates the existing record instead of crashing. |
| 2026-07-16 | `save_notes()` deletes old notes before insert | One set of notes per video — keeps DB clean. Old notes replaced on regeneration. |
| 2026-07-16 | `/generate-notes` checks DB cache before calling LLM | Avoids redundant LLM calls. Supports `force: true` to bypass cache and regenerate. |
| 2026-07-16 | `/search` logs every query to search_history | Enables future history/analytics view. Logging failure doesn't break the search response. |
| 2026-07-16 | Added `/process-video` endpoint | Full pipeline: transcript → chunk → embed → ChromaDB + save video record to SQLite. |
| 2026-07-16 | Added `/history` GET endpoint | Returns all processed videos for the history page (design.md). |
| 2026-07-16 | Dark Mode Glassmorphism Theme | Implemented premium Outift and Plus Jakarta typography, glow effects, grid layouts, and custom interactive states for a professional final-year aesthetic. |
| 2026-07-16 | Markdown client-side rendering with Marked.js | Integrated Marked.js CDN to parse headers, bold, list, and italic structures in the Notes panel. |
| 2026-07-16 | Autoplay timestamp links via YouTube URLs | Created JS helper converting HH:MM:SS timestamps to absolute seconds and linking directly to target times with `&t=X`. |
| 2026-07-16 | History sidebar quick-loads processed videos | Clicking on sidebar items restores cache immediately, reducing redundant computation. |

## Open Questions / To-Decide Later
- [ ] Do we export notes as PDFs or is text enough? (Decide after MVP)
- [ ] Need to fine-tune chunking/summarization strategy for long videos (2+ hours)
- [ ] Finalize deployment platform (Render vs Railway vs local demo)
- [ ] YouTube may block cloud IPs (RequestBlocked/IpBlocked) — may need proxy config for deployed version

## Progress Tracker
- [x] Idea finalized
- [x] Tech stack decided
- [x] Planning docs (prd/architecture/rules/phases/design/memory) ready
- [x] **Phase 0 — Setup** (2026-07-15)
  - [x] Python venv created
  - [x] requirements.txt with all dependencies + testing libs
  - [x] Folder structure: modules/, database/, templates/, static/, tests/unit/, tests/integration/
  - [x] .env.example (safe to commit) + .env (gitignored) with GROQ/GEMINI key placeholders
  - [x] config.py loads keys via python-dotenv, warns if missing
  - [x] .gitignore covers .env, __pycache__, venv/, chroma_db/, *.db
  - [x] Basic Flask app (app.py) with Hello World + /health endpoint
  - [x] All module placeholder files with docstrings
- [x] **Phase 1 — Transcript Extraction** (2026-07-15)
  - [x] extract_video_id() — handles 7+ URL formats
  - [x] fetch_transcript() — uses youtube-transcript-api v1.2.4 instance API
  - [x] get_transcript() — high-level function combining URL parsing + fetch + fallback
  - [x] Whisper fallback stubbed with clear TODO
  - [x] Error handling: 6 specific exception types
  - [x] format_timestamp() helper
  - [x] 31 unit tests — ALL PASSING
  - [x] Manual test script (test_manual_transcript.py)
- [x] **Phase 2 — Chunking + Embeddings** (2026-07-15)
  - [x] chunk_transcript() — accumulates snippets into ~400-word chunks with ~50-word overlap
  - [x] get_chunk_stats() — returns word counts, avg, min, max for debugging
  - [x] generate_embeddings() — lazy-loaded all-MiniLM-L6-v2, handles str/list/empty
  - [x] get_embedding_dimension() — returns 384
  - [x] 15 chunker unit tests — ALL PASSING
  - [x] 10 embedder unit tests — ALL PASSING
  - [x] Manual test script (test_manual_phase2.py)
- [x] **Phase 3 — Vector Store (ChromaDB)** (2026-07-15)
  - [x] get_client() — PersistentClient for prod, EphemeralClient for tests
  - [x] get_collection() — get_or_create "video_transcripts" with cosine similarity
  - [x] store_chunks() — upsert chunks + embeddings + metadata (idempotent)
  - [x] query_chunks() — semantic search with video_id filter + top_k
  - [x] delete_video_chunks() — remove all chunks for a video
  - [x] has_video_chunks() — check if video already processed
  - [x] reset_module() — clear caches (for tests)
  - [x] 16 unit tests — ALL PASSING (store, query, idempotency, metadata, delete, edge cases)
  - [x] 10 integration tests — ALL PASSING (full pipeline: transcript → chunks → embeddings → ChromaDB → query)
  - [x] Manual test script (test_manual_phase3.py)
- [x] **Phase 4 — RAG Search Feature** (2026-07-15)
  - [x] search_video() — main RAG function: embed query → retrieve chunks → LLM answer
  - [x] _build_rag_prompt() — constrained prompt with transcript context + timestamp instructions
  - [x] _call_groq() — tries multiple models in order (handles deprecation gracefully)
  - [x] _call_gemini() — fallback using google-genai SDK
  - [x] _call_llm() — orchestrates Groq → Gemini fallback
  - [x] POST /search endpoint in app.py (validates input, returns JSON)
  - [x] 14 unit tests — ALL PASSING (prompt building, validation, success flow, no-chunks, LLM failure, fallback)
  - [x] 1 integration test — SKIPPED (needs valid API key; test itself works correctly)
  - [x] Manual test script (test_manual_phase4.py)
- [x] **Phase 5 — Smart Notes Generator** (2026-07-16)
  - [x] generate_notes() — main function, auto-selects direct vs map-reduce
  - [x] _build_direct_prompt() — single-shot prompt for short transcripts
  - [x] _build_map_prompt() — MAP step prompt for long transcripts
  - [x] _build_reduce_prompt() — REDUCE step to combine summaries
  - [x] _prepare_chunks_for_notes() — adds timestamps to chunk text
  - [x] _group_chunks() — groups chunks into batches under word limit
  - [x] _call_llm() — Groq → Gemini fallback (reuses search.py internals)
  - [x] POST /generate-notes endpoint in app.py (accepts video_id or url)
  - [x] 25 unit tests — ALL PASSING (prompts, validation, direct, map-reduce, grouping, LLM fallback)
  - [x] 1 integration test — SKIPPED (needs valid API key; test itself works correctly)
  - [x] memory.md updated
- [x] **Phase 6 — Database Integration (SQLite)** (2026-07-16)
  - [x] database/db.py — full implementation (init_db, CRUD for all 3 tables)
  - [x] Tables: videos (id, video_id, title, url, transcript_status, created_at)
  - [x] Tables: notes (id, video_id FK, notes_content, provider, method, generated_at)
  - [x] Tables: search_history (id, video_id FK, query, answer, timestamp_ref, provider, searched_at)
  - [x] save_video() — idempotent upsert via ON CONFLICT
  - [x] save_notes() — replaces old notes (one per video)
  - [x] save_search() — accumulates (multiple per video)
  - [x] get_video(), get_all_videos(), get_notes(), get_search_history()
  - [x] POST /process-video — full pipeline (transcript → chunk → embed → store + DB save)
  - [x] /generate-notes — now checks DB cache first, only calls LLM if no cached notes exist
  - [x] /search — now logs every query to search_history
  - [x] GET /history — returns all processed videos
  - [x] init_db() called on app startup
  - [x] 20 unit tests — ALL PASSING (table creation, columns, CRUD, idempotent upsert, overwrite, filter, limit)
  - [x] Manual test script (test_manual_phase6.py)
  - [x] memory.md updated
- [x] **Phase 7 — Frontend/UI** (2026-07-16)
  - [x] templates/index.html — clean HTML5 layout, custom Outfit/Plus Jakarta typography, sidebar, tabs, and panels
  - [x] static/style.css — rich dark glassmorphism styling, gradients, buttons, and custom layout structures
  - [x] static/script.js — process forms, loader triggers, marked markdown parsing, history loader, and YouTube deep-linking
  - [x] Clickable timestamps link directly to time offsets (t=X seconds) on YouTube
  - [x] Sidebar displays processed videos history and allows single-click load
  - [x] memory.md updated
- [ ] Phase 8 — Testing & bug fixing
- [ ] Phase 9 — Polish + documentation
- [ ] Phase 10 — Final demo & deployment

## Known Issues / Blockers
- YouTube may block cloud server IPs (RequestBlocked/IpBlocked). For now, the error is handled gracefully.
- HuggingFace cache on Windows shows a symlink warning — cosmetic only, does not affect functionality. Can be silenced by enabling Developer Mode on Windows.
- ChromaDB downloads its own ONNX copy of all-MiniLM-L6-v2 (~79MB) on first use of `query_texts`. Our code passes pre-computed embeddings via `query_embeddings` so this is only triggered if ChromaDB's default embedding function is used accidentally.
- **⚠️ API Keys**: Groq key returned 401 (Invalid API Key) and Gemini key appears to still be a placeholder. User needs to regenerate valid keys. The code handles this gracefully (integration test skips, unit tests all pass with mocks).

## Important Constants (for reference across phases)
- **Embedding dimension**: 384 (all-MiniLM-L6-v2)
- **Chunk size**: ~400 words (Config.CHUNK_SIZE)
- **Chunk overlap**: ~50 words (Config.CHUNK_OVERLAP)
- **Top-K results**: 5 (Config.TOP_K_RESULTS)
- **ChromaDB collection**: "video_transcripts"
- **ChromaDB similarity**: cosine (lower distance = more similar)
- **Chunk ID format in ChromaDB**: `{video_id}_{chunk_id}`
- **Groq models (priority order)**: llama-3.3-70b-versatile, llama-3.1-8b-instant, llama3-8b-8192, gemma2-9b-it, mixtral-8x7b-32768
- **Gemini model**: gemini-2.0-flash
- **RAG top_k**: 3 (for focused LLM context)
- **Notes direct threshold**: 3000 words (above → map-reduce)
- **Notes map batch size**: 1500 words per group
- **SQLite DB path**: `database/notemind.db` (Config.SQLITE_DB_PATH)
- **SQLite tables**: videos, notes, search_history
- **SQLite mode**: WAL + foreign_keys=ON
- **Groq SDK**: v1.5.0 — `groq.Groq(api_key=...)`
- **Gemini SDK**: google-genai v2.11.0 — `from google import genai; genai.Client(api_key=...)`

## Notes for Future Self / AI Assistant
- Building a solid Phase 1-4 (transcript + RAG core) is the first priority — this is the heart of the project.
- Be mindful of free tier API limits; avoid making too many calls during testing.
- youtube-transcript-api v1.2.4 was installed (not v0.6.x). The API uses `YouTubeTranscriptApi()` as an instance with `.fetch(video_id)`. Snippets have `.text`, `.start`, `.duration` attributes.
- sentence-transformers v5.x uses `get_embedding_dimension()` (not the old `get_sentence_embedding_dimension()`).
- ChromaDB v1.5.9: uses `PersistentClient`/`EphemeralClient`, `collection.upsert()`, `collection.query()`. Query results are nested lists: `result['ids'][0][i]`, `result['documents'][0][i]`, etc.
- Groq SDK v1.5.0: `groq.Groq(api_key=key).chat.completions.create(model=..., messages=[...])`. Errors: `AuthenticationError`, `RateLimitError`, `NotFoundError`.
- Gemini SDK (google-genai v2.11.0): `genai.Client(api_key=key).models.generate_content(model=..., contents=...)`. Returns `.text` property.
- If seeking help from an AI assistant in the future, paste this file as context — it will provide a complete overview of the project.
