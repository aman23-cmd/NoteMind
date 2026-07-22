# PHASES.md — Development Roadmap

## Phase 0: Setup (Day 1-2)
- Create a Python environment (`venv`).
- Create `requirements.txt`: flask, youtube-transcript-api, sentence-transformers, chromadb, groq (or google-generativeai), python-dotenv, whisper.
- Obtain Groq/Gemini API keys (free signup).
- Run a basic Flask "Hello World" app to confirm the setup.

## Phase 1: Transcript Extraction (Day 3-5)
- Write a function to extract the video ID from a YouTube URL.
- Extract the transcript (with timestamps) using `youtube-transcript-api`.
- Fallback: If no transcript is found, download the audio and transcribe using Whisper.
- Test: Try on 3-4 different videos.

## Phase 2: Chunking + Embeddings (Day 6-8)
- Split the transcript into chunks (preserving timestamps).
- Generate embeddings using `sentence-transformers`.
- Test: Check chunk sizes and ensure embeddings are generated in the correct shape.

## Phase 3: Vector Store Setup (Day 9-10)
- Setup and connect ChromaDB.
- Store chunks + embeddings + metadata (timestamp, video_id).
- Create a query function — given a search text, it should return top-k similar chunks.

## Phase 4: RAG Search Feature (Day 11-13)
- User query -> embed -> ChromaDB search -> retrieve top chunks.
- Send retrieved chunks + query to the LLM (Groq/Gemini) -> get final answer + timestamp.
- Create Flask route: `/search` (POST: video_id, query).

## Phase 5: Smart Notes Generator (Day 14-16)
- Send the full transcript to the LLM (if it's too long, summarize in chunks and combine - map-reduce style).
- Design a structured notes format (headings, bullets, key points).
- Create Flask route: `/generate-notes` (POST: video_id).

## Phase 6: Database Integration (Day 17-18)
- SQLite setup — `videos`, `notes`, `search_history` tables.
- Save notes and search history to avoid recomputing for the same video.

## Phase 7: Frontend/UI (Day 19-21)
- Simple HTML/CSS/JS pages: video input page, notes display page, search page.
- Connect Flask templates with the backend.

## Phase 8: Testing & Bug Fixing (Day 22-24)
- End-to-end testing (multiple videos, edge cases).
- Improve error handling.
- Performance check (measure processing time for very long videos).

## Phase 9: Polish + Documentation (Day 25-27)
- Complete `README.md`.
- Improve the UI slightly (basic styling).
- Take screenshots for the project report/presentation.

## Phase 10: Final Demo Prep (Day 28-30)
- Deployment (Render/Railway free tier) or local demo setup.
- Create a presentation/PPT for the project.
- Prepare important questions/answers for Viva (e.g., What is RAG? What are embeddings?).

> Note: The timeline is flexible; adjust according to your availability. Phases 1-4 (transcript + RAG core) are the most critical — make sure they are solid first.
