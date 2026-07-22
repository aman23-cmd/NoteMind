# RULES.md — Project Rules & Conventions

## 1. General Coding Rules
- Follow Python PEP8 (readable code, proper indentation).
- Each module should have a single responsibility — e.g., `transcript.py` will only extract transcripts and nothing else.
- DO NOT hardcode any API keys or secrets — store them in the `.env` file and load them using `python-dotenv`.
- Add the `.env` file to `.gitignore` (never push it to GitHub).

## 2. API Usage Rules
- Be mindful of free tier limits (both Groq and Gemini have daily/per-minute request limits).
- If an API call fails, implement retry logic or display a proper error message. Do not let the app crash.
- Wrap every LLM call in a try-except block.
- If the rate limit is hit, show a friendly message to the user ("Please wait a moment, limit exceeded").

## 3. RAG / Data Rules
- Keep transcript chunk sizes consistent (approx 300-500 words, with a slight overlap of ~50 words to maintain context).
- Always save timestamp metadata with each chunk — otherwise, search will be useless.
- ChromaDB collection naming convention: `video_<video_id>` — use a separate collection for each video or apply proper metadata filtering.

## 4. Database Rules
- Keep the SQLite schema simple: `videos`, `notes`, `search_history` tables.
- Do not store any sensitive user data (this is a simple final year project).

## 5. Error Handling Rules
- Invalid YouTube URL -> Show a clear error message.
- If captions/transcripts are unavailable -> Try Whisper fallback. If that fails too, inform the user.
- If the LLM response is empty/garbage -> Retry or provide a default fallback message.

## 6. Git/Version Control Rules
- Make small, frequent commits with meaningful commit messages.
- Keep the `main` branch stable. Create new features on a `feature/xyz` branch.

## 7. Testing Rules
- Test each module individually before integration (test transcript extraction separately, test embeddings separately).
- Test on at least 2-3 different types of videos (short video, long lecture, video without captions).

## 8. Documentation Rules
- Write a short docstring for every function (explaining what it does, inputs, and outputs).
- Clearly list setup steps in `README.md` (how to run, which API keys are required).
