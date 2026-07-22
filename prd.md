# PRD.md — AI-Based Smart Video Topic Search Agent & Smart Notes Generator

## 1. Problem Statement
It is time-consuming for students/users to find a specific topic within long videos (lectures, tutorials). Manually watching the entire video to take notes is also difficult. The goal of this project is to provide topic-based search within a video and automatically generate smart notes.

## 2. Goals
- The user provides a link to any YouTube video/lecture.
- The user searches for a specific topic (e.g., "explain recursion") — the system finds the exact part/timestamp in the video.
- The system summarizes the entire video to generate "smart notes" (headings, bullet points, key concepts).
- Everything will be built using FREE tools/APIs (student project budget = 0).

## 3. Target Users
- College students (exam revision, lecture notes)
- Self-learners (YouTube courses)
- Anyone who wants to quickly extract info from long-form video content

## 4. Core Features (MVP)
1. **Video Input** — Paste a YouTube URL.
2. **Transcript Extraction** — Extract transcripts using auto-captions or Whisper.
3. **Topic-Based Search (RAG)** — Enter a query, get the relevant transcript chunk + timestamp.
4. **Smart Notes Generator** — Generate structured notes (headings, bullets, summary) from the full transcript.
5. **Simple Web UI** — Flask-based UI where you enter a video link and view notes/search results.

## 5. Nice-to-Have (Future Scope, Post-MVP)
- Multiple video upload / playlist support
- PDF export for notes
- Local video file upload (not just YouTube)
- User login + history save
- Multi-language transcript support

## 6. Tech Constraints
- Use ONLY FREE tier APIs (Groq/Gemini free tier, HuggingFace local models).
- Use local/free databases only (SQLite + ChromaDB).
- Deployment will be on a free platform (Render/Railway free tier or local demo).

## 7. Success Metrics (For project evaluation)
- Correct transcript extraction accuracy.
- Topic search relevance (whether the correct timestamp/chunk is retrieved).
- Readability/quality of the notes.
- End-to-end working demo (from video link to generated notes).

## 8. Out of Scope (For now)
- Live video / real-time streaming support
- Mobile app (only web MVP for now)
- Paid API integration
