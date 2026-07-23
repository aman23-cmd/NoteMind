# 🧠 NoteMind

**AI-Powered Video Topic Search Agent & Smart Notes Generator**

NoteMind lets you paste any YouTube video link and instantly:
- 🔍 **Search** for a specific topic inside the video and jump straight to the right timestamp
- 📝 **Generate smart notes** — a structured summary with headings, key points, and definitions

Built as a final year project using a **Retrieval-Augmented Generation (RAG)** pipeline, entirely on **free-tier tools and APIs**.

---

## 🔗 Live Demo

| Part | URL |
|---|---|
| Frontend (UI) | https://notemind-aman.netlify.app |
| Backend (API) | https://notemind-g3rb.onrender.com |

> ⚠️ The backend is hosted on Render's free tier, which sleeps after 15 minutes of inactivity. The first request after a period of inactivity may take 30–60 seconds while the server wakes up.

---

## ✨ Features

- Paste any YouTube URL and extract its transcript automatically
- Semantic (meaning-based) search within a video — not just keyword matching
- Auto-generated structured notes: summary, key topics, definitions, and timestamp references
- Clickable timestamps that jump directly to that moment in the YouTube video
- Video processing history, so previously processed videos don't need to be reprocessed
- Fully free to run and deploy — no paid APIs or services required

---

## 🏗️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | Flask (Python) | Lightweight, simple REST API |
| Transcript Extraction | `youtube-transcript-api` | Free, no API key, timestamped captions |
| Proxy (for cloud hosting) | Webshare (free tier) | Bypasses YouTube's cloud-IP blocking |
| Embeddings | Google Gemini Embedding API (`gemini-embedding-001`) | Free tier, no local model needed (saves memory on free hosting) |
| Vector Database | ChromaDB | Free, local, simple semantic search |
| Relational Database | SQLite | Lightweight, zero-config, stores video/notes/search history |
| LLM (Notes + Search Answers) | Groq API (primary) → Gemini API (fallback) | Free, fast inference |
| Frontend | HTML, CSS, JavaScript (vanilla) | No framework overhead, deploys anywhere |
| Backend Hosting | Render (free tier) | Free web service hosting |
| Frontend Hosting | Netlify (free tier) | Free static site hosting |

---

## 🧩 How It Works (RAG Pipeline)

```
YouTube URL
   │
   ▼
Transcript Extraction (youtube-transcript-api + Webshare proxy)
   │
   ▼
Chunking (~400 words per chunk, 50-word overlap, timestamps preserved)
   │
   ▼
Embedding (Gemini Embedding API → 768-dim vectors)
   │
   ▼
Stored in ChromaDB (per video_id)
   │
   ├──► Topic Search: query → embed → similarity search → top chunks
   │         → sent to LLM (Groq/Gemini) → answer + timestamp
   │
   └──► Notes Generation: all chunks → LLM (Groq/Gemini)
             → structured markdown notes (summary, key topics, definitions)
```

---

## 📁 Project Structure

```
NoteMind/
├── app.py                     # Flask entry point / API routes
├── config.py                  # Loads API keys and settings from .env
├── requirements.txt
├── .env.example                # Template for required environment variables
├── modules/
│   ├── transcript.py          # YouTube transcript extraction
│   ├── chunker.py             # Splits transcript into overlapping chunks
│   ├── embedder.py            # Generates embeddings via Gemini API
│   ├── vectorstore.py         # ChromaDB storage and semantic search
│   ├── search.py              # RAG search (query → answer + timestamp)
│   └── notes_generator.py     # Structured notes generation
├── database/
│   └── db.py                  # SQLite operations (videos, notes, search history)
├── frontend/                   # Deployed separately on Netlify
│   ├── index.html
│   └── static/
│       ├── style.css
│       └── script.js
├── tests/
│   ├── unit/                   # Unit tests (mocked external calls)
│   └── integration/            # End-to-end pipeline tests
└── memory.md                   # Project decision log
```

---

## ⚙️ Setup & Local Installation

### 1. Clone the repository
```bash
git clone https://github.com/aman23-cmd/NoteMind.git
cd NoteMind
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Copy `.env.example` to `.env` and fill in your own free API keys:
```
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
WEBSHARE_PROXY_USERNAME=your_webshare_username
WEBSHARE_PROXY_PASSWORD=your_webshare_password
```

Where to get these (all free):
- **Groq API key** → https://console.groq.com/keys
- **Gemini API key** → https://aistudio.google.com/app/apikey
- **Webshare proxy** (needed mainly for cloud deployment, optional for local use) → https://webshare.io

### 5. Run the backend
```bash
python app.py
```
The API will be available at `http://127.0.0.1:5000`.

### 6. Open the frontend
Open `frontend/index.html` directly in a browser, or update `API_BASE_URL` in `frontend/static/script.js` to point to your local backend.

---

## 🧪 Running Tests

```bash
# Unit tests (mocked, no internet/API keys required)
pytest tests/unit/ -v

# Integration tests (uses real APIs — needs valid keys)
pytest tests/integration/ -v

# Full suite with coverage report
pytest --cov
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/process-video` | Extract transcript, chunk, embed, and store a video |
| POST | `/generate-notes` | Generate (or fetch cached) smart notes for a video |
| POST | `/search` | Semantic search for a topic within a processed video |
| GET | `/history` | List all previously processed videos |

---

## ☁️ Deployment Notes

- **Backend (Render):** Free tier has a 512MB RAM limit and ephemeral disk (data may reset on restart/redeploy). Local embedding models were avoided in favor of the Gemini Embedding API specifically to stay under this memory limit.
- **YouTube IP blocking:** YouTube blocks requests from most cloud-provider IP ranges (AWS, GCP, Azure, and by extension Render). This project routes transcript requests through a free Webshare rotating proxy to work around this.
- **Frontend (Netlify):** Deployed as a static site from the `frontend/` directory, calling the Render backend via its public URL.

---

## 🚧 Known Limitations

- Whisper-based transcription (for videos without captions) is stubbed for local development only — it is intentionally excluded from the deployed version due to its resource requirements.
- Render's free tier sleeps after inactivity, causing a delay on the first request.
- ChromaDB storage on Render's free tier is not guaranteed to persist across redeploys.

---

## 📌 Future Improvements

- PDF export for generated notes
- Support for uploading local video/audio files
- Multi-language transcript and notes support
- User accounts and saved history per user

---

## 👤 Author

Built by Aman as a final year project — **NoteMind**.
