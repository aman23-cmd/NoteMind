# DESIGN.md — UI/UX & Data Design

## 1. UI Pages (MVP)

### Page 1: Home / Video Input
- Input box: "Paste YouTube video link here"
- Button: "Process Video"
- Loading state: "Extracting transcript..." / "Generating notes..."

### Page 2: Video Dashboard (after processing)
- Display Video title/thumbnail
- Two tabs: **"Smart Notes"** and **"Search Topic"**

### Tab A: Smart Notes
- Auto-generated structured notes (headings + bullet points)
- "Download as text" button (future: PDF)

### Tab B: Search Topic
- Search box: "What topic do you want to learn about?"
- Result: Relevant text snippet + timestamp link (clickable, jumps to that part of the video)

## 2. API Endpoints (Flask Routes)

| Method | Endpoint | Purpose | Input | Output |
|---|---|---|---|---|
| POST | `/process-video` | Process video (transcript + embed + store) | `{video_url}` | `{video_id, status}` |
| POST | `/generate-notes` | Generate smart notes | `{video_id}` | `{notes: "..."}` |
| POST | `/search` | Topic-based search (RAG) | `{video_id, query}` | `{answer, timestamp, source_chunk}` |
| GET | `/history` | View previous videos/notes | - | `{list of videos}` |

## 3. Database Schema (SQLite)

**Table: videos**
```
id (PK)
video_id (YouTube ID)
title
url
transcript_status  (pending/done/failed)
created_at
```

**Table: notes**
```
id (PK)
video_id (FK -> videos.video_id)
notes_content (TEXT)
generated_at
```

**Table: search_history**
```
id (PK)
video_id (FK)
query
answer
timestamp_ref
searched_at
```

## 4. ChromaDB Collection Design
```
Collection: "video_transcripts"
Each entry:
  - id: chunk_id (unique)
  - embedding: vector
  - document: chunk text
  - metadata: { video_id, start_time, end_time }
```

## 5. Notes Format (LLM Output Structure)
```
# Video Title

## Summary
(2-3 line overview)

## Key Topics Covered
- Topic 1
  - sub-point
- Topic 2
  - sub-point

## Important Points / Definitions
- ...

## Timestamps Reference (optional)
- 00:12:30 - Topic X explained
```

## 6. Visual Style (Basic, appropriate for a student project)
- Clean, minimal UI — use a primary color (like blue/teal) with a white background.
- Card-based layout for notes and search results.
- Mobile-responsiveness is optional (the MVP focus is on the backend/RAG pipeline).
