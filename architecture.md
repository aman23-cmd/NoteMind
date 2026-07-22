# ARCHITECTURE.md — System Design

## 1. High-Level Flow

```
User (Browser)
     |
     v
Flask Web App (Frontend + Routes)
     |
     v
[1] Video Input Handler --> Extracts video ID from YouTube URL
     |
     v
[2] Transcript Extractor --> youtube-transcript-api (primary)
                          --> Whisper (fallback, if captions are unavailable)
     |
     v
[3] Chunking Module --> Splits transcript into small timestamped chunks
     |
     v
[4] Embedding Module --> Generates vectors for each chunk using sentence-transformers (all-MiniLM-L6-v2)
     |
     v
[5] Vector Store (ChromaDB) --> Stores chunks + embeddings + timestamps
     |
     +--------------------------------------+
     |                                      |
     v                                      v
[6a] RAG Search Flow                [6b] Notes Generation Flow
User query -> embed query ->        Full transcript -> LLM (Groq/Gemini)
similarity search in ChromaDB ->    -> structured summary/notes
retrieve top-k relevant chunks
     |                                      |
     v                                      v
[7] LLM (Groq / Gemini free API) --> Generates an answer from query chunks
                                     or formats notes into a clean structure
     |
     v
[8] SQLite DB --> Saves video metadata, generated notes, and search history
     |
     v
Response back to User (Displayed in UI)
```

## 2. Components

| Component | Tool/Tech | Purpose |
|---|---|---|
| Backend Framework | Flask (Python) | Routes, API endpoints, server logic |
| Transcript Extraction | youtube-transcript-api / whisper | Extract text from video |
| Chunking | Custom Python (langchain text splitter, free) | Break text into manageable pieces |
| Embeddings | sentence-transformers (local, free) | Convert text into vectors |
| Vector DB | ChromaDB | Store chunks for semantic search |
| LLM | Groq API / Gemini API (free tier) | Notes generation + query answering |
| Relational DB | SQLite | User data, video metadata, history |
| Frontend | HTML/CSS/JS (or Streamlit) | User interface |

## 3. RAG Pipeline (Detail)
1. **Ingest**: Video transcript -> chunks (e.g., 300-500 words per chunk, with start-timestamp saved).
2. **Embed**: Each chunk generates an embedding via sentence-transformers.
3. **Store**: Store in ChromaDB collection — chunk text + embedding + metadata (timestamp, video_id).
4. **Query**: The user's search query is embedded.
5. **Retrieve**: Retrieve top 3-5 chunks from ChromaDB based on cosine similarity.
6. **Generate**: Provide retrieved chunks + user query to the LLM -> final answer with timestamp reference.

## 4. Folder Structure (suggested)
```
project/
├── app.py                  # Flask entry point
├── config.py                # API keys, settings
├── modules/
│   ├── transcript.py        # transcript extraction logic
│   ├── chunker.py           # text splitting logic
│   ├── embedder.py          # embedding generation
│   ├── vectorstore.py       # ChromaDB operations
│   ├── notes_generator.py   # LLM call for notes
│   └── search.py            # RAG search logic
├── database/
│   └── db.py                # SQLite models/queries
├── templates/                # HTML files (Flask)
├── static/                   # CSS/JS
├── requirements.txt
└── .env                       # API keys (never commit)
```

## 5. Data Flow Summary
YouTube URL → Transcript → Chunks → Embeddings → ChromaDB → (Search or Notes) → LLM → Output
