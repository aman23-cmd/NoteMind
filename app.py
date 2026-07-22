"""
app.py — Flask entry point for the NoteMind application.

This is the main file that starts the web server.
Run it with:  python app.py
Then open:    http://127.0.0.1:5000
"""

import os
from flask import Flask, render_template, request, jsonify
from config import Config

app = Flask(__name__)

# ── Initialize the database on startup ──
from database.db import init_db
init_db()

# ── Eager-load the embedding model at startup ──
# This moves the ~26s model load to server boot instead of first request.
import time as _startup_time
print("[BOOT] Loading embedding model (one-time cost)...")
_t0 = _startup_time.perf_counter()
from modules.embedder import generate_embeddings
generate_embeddings("warmup")  # triggers lazy model load
print(f"[BOOT] Embedding model ready in {_startup_time.perf_counter() - _t0:.1f}s")


@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')


@app.route('/health')
def health():
    """Simple health-check endpoint (useful for deployment platforms)."""
    return {'status': 'ok', 'project': 'NoteMind'}, 200


@app.route('/process-video', methods=['POST'])
def process_video():
    """
    Process a YouTube video: extract transcript, chunk, embed, store.

    Request body (JSON):
        {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        }
        OR
        {
            "video_id": "dQw4w9WgXcQ"
        }

    Response (JSON):
        {
            "success": true,
            "video_id": "dQw4w9WgXcQ",
            "status": "done",
            "chunks_stored": 5,
            "timing": { ... }
        }
    """
    import time as _time
    t_total_start = _time.perf_counter()
    timing = {}

    # ── Validate content type ──
    if not request.is_json:
        return jsonify({
            'success': False,
            'error': "Request must be JSON. Set Content-Type: application/json",
        }), 400

    data = request.get_json()

    # ── Extract video_id ──
    video_id = data.get('video_id', '').strip() if data.get('video_id') else ''
    url = data.get('url', '').strip() if data.get('url') else ''

    if not video_id and url:
        from modules.transcript import extract_video_id
        try:
            video_id = extract_video_id(url)
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f"Invalid URL: {e}",
            }), 400

    if not video_id:
        return jsonify({
            'success': False,
            'error': "Missing required field: 'video_id' or 'url'",
        }), 400

    if not url:
        url = f"https://www.youtube.com/watch?v={video_id}"

    # ── Save video record (status: pending) ──
    from database.db import save_video, get_video
    save_video(video_id=video_id, url=url, transcript_status='pending')

    # ── Step 1: Fetch transcript ──
    t0 = _time.perf_counter()
    from modules.transcript import get_transcript
    transcript_result = get_transcript(url)
    timing['1_transcript_fetch'] = round(_time.perf_counter() - t0, 3)

    if not transcript_result['success']:
        save_video(video_id=video_id, url=url, transcript_status='failed')
        return jsonify({
            'success': False,
            'error': f"Transcript extraction failed: {transcript_result['error']}",
            'timing': timing,
        }), 404

    timing['transcript_snippets'] = len(transcript_result['transcript'])

    # ── Step 2: Chunk the transcript ──
    t0 = _time.perf_counter()
    from modules.chunker import chunk_transcript
    chunks = chunk_transcript(transcript_result['transcript'])
    timing['2_chunking'] = round(_time.perf_counter() - t0, 3)

    if not chunks:
        save_video(video_id=video_id, url=url, transcript_status='failed')
        return jsonify({
            'success': False,
            'error': "Transcript produced no usable chunks.",
            'timing': timing,
        }), 404

    timing['chunks_created'] = len(chunks)

    # ── Step 3: Generate embeddings ──
    t0 = _time.perf_counter()
    from modules.embedder import generate_embeddings
    chunk_texts = [c['chunk_text'] for c in chunks]
    embeddings = generate_embeddings(chunk_texts)
    timing['3_embedding'] = round(_time.perf_counter() - t0, 3)

    # ── Step 4: Store in ChromaDB ──
    t0 = _time.perf_counter()
    from modules.vectorstore import store_chunks
    store_result = store_chunks(video_id, chunks, embeddings)
    timing['4_vectorstore'] = round(_time.perf_counter() - t0, 3)

    if not store_result['success']:
        save_video(video_id=video_id, url=url, transcript_status='failed')
        return jsonify({
            'success': False,
            'error': f"Vector store failed: {store_result['error']}",
            'timing': timing,
        }), 500

    # ── Update video record (status: done) ──
    save_video(video_id=video_id, url=url, transcript_status='done')

    timing['total'] = round(_time.perf_counter() - t_total_start, 3)

    return jsonify({
        'success': True,
        'video_id': video_id,
        'status': 'done',
        'chunks_stored': store_result['chunks_stored'],
        'timing': timing,
    }), 200


@app.route('/search', methods=['POST'])
def search():
    """
    RAG search endpoint.

    Request body (JSON):
        {
            "video_id": "dQw4w9WgXcQ",
            "query": "What is this video about?"
        }

    Response (JSON):
        {
            "success": true,
            "answer": "...",
            "provider": "groq",
            "timestamp": "00:01:23",
            "source_chunks": [...]
        }
    """
    # ── Validate content type ──
    if not request.is_json:
        return jsonify({
            'success': False,
            'error': "Request must be JSON. Set Content-Type: application/json",
        }), 400

    data = request.get_json()

    # ── Validate required fields ──
    video_id = data.get('video_id', '').strip() if data.get('video_id') else ''
    query = data.get('query', '').strip() if data.get('query') else ''

    if not video_id:
        return jsonify({
            'success': False,
            'error': "Missing required field: 'video_id'",
        }), 400

    if not query:
        return jsonify({
            'success': False,
            'error': "Missing required field: 'query'",
        }), 400

    # ── Perform the search (with timing) ──
    import time as _time
    t_search_start = _time.perf_counter()

    from modules.search import search_video
    result = search_video(query=query, video_id=video_id)

    search_elapsed = round(_time.perf_counter() - t_search_start, 3)

    # ── Log to search_history (regardless of success/failure) ──
    from database.db import save_search
    try:
        save_search(
            video_id=video_id,
            query=query,
            answer=result.get('answer', ''),
            timestamp_ref=result.get('timestamp', ''),
            provider=result.get('provider', ''),
        )
    except Exception:
        pass  # Don't let DB logging break the search response

    # Merge inner timing (from search_video) with outer total
    inner_timing = result.get('timing', {})
    inner_timing['search_total'] = search_elapsed
    result['timing'] = inner_timing

    if result['success']:
        return jsonify(result), 200
    else:
        return jsonify(result), 404


@app.route('/generate-notes', methods=['POST'])
def generate_notes():
    """
    Smart notes generation endpoint.

    Checks the database first — if notes already exist for this video_id,
    returns the cached version without calling the LLM again.

    Request body (JSON):
        {
            "video_id": "dQw4w9WgXcQ"
        }
        OR
        {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        }
        Optional:
        {
            "force": true   (bypass cache, regenerate notes)
        }

    Response (JSON):
        {
            "success": true,
            "notes": "# Video Title\\n...",
            "provider": "groq",
            "method": "direct",
            "chunks_processed": 5,
            "cached": false
        }
    """
    # ── Validate content type ──
    if not request.is_json:
        return jsonify({
            'success': False,
            'error': "Request must be JSON. Set Content-Type: application/json",
        }), 400

    data = request.get_json()
    force = data.get('force', False)

    # ── Extract video_id (from 'video_id' or 'url' field) ──
    video_id = data.get('video_id', '').strip() if data.get('video_id') else ''

    if not video_id and data.get('url'):
        # Try to extract video_id from a URL
        from modules.transcript import extract_video_id
        try:
            video_id = extract_video_id(data['url'])
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f"Invalid URL: {e}",
            }), 400

    if not video_id:
        return jsonify({
            'success': False,
            'error': "Missing required field: 'video_id' or 'url'",
        }), 400

    # ── Check DB cache (unless force=True) ──
    from database.db import get_notes, save_notes
    if not force:
        cached = get_notes(video_id)
        if cached:
            return jsonify({
                'success': True,
                'notes': cached['notes_content'],
                'provider': cached.get('provider', ''),
                'method': cached.get('method', ''),
                'chunks_processed': 0,
                'cached': True,
            }), 200

    # ── Timing setup ──
    import time as _time
    t_total_start = _time.perf_counter()
    timing = {}

    # ── Step 1: Fetch transcript ──
    t0 = _time.perf_counter()
    from modules.transcript import get_transcript
    transcript_result = get_transcript(f"https://www.youtube.com/watch?v={video_id}")
    timing['1_transcript_fetch'] = round(_time.perf_counter() - t0, 3)

    if not transcript_result['success']:
        return jsonify({
            'success': False,
            'error': f"Transcript extraction failed: {transcript_result['error']}",
            'timing': timing,
        }), 404

    # ── Step 2: Chunk the transcript ──
    t0 = _time.perf_counter()
    from modules.chunker import chunk_transcript
    chunks = chunk_transcript(transcript_result['transcript'])
    timing['2_chunking'] = round(_time.perf_counter() - t0, 3)

    if not chunks:
        return jsonify({
            'success': False,
            'error': "Transcript produced no usable chunks.",
            'timing': timing,
        }), 404

    # ── Step 3: Generate notes via LLM ──
    t0 = _time.perf_counter()
    from modules.notes_generator import generate_notes as gen_notes
    result = gen_notes(chunks=chunks, video_id=video_id)
    timing['3_llm_notes'] = round(_time.perf_counter() - t0, 3)
    timing['total'] = round(_time.perf_counter() - t_total_start, 3)

    if result['success']:
        # ── Save to DB for future cache hits ──
        try:
            save_notes(
                video_id=video_id,
                notes_content=result['notes'],
                provider=result.get('provider', ''),
                method=result.get('method', ''),
            )
        except Exception:
            pass  # Don't let DB save break the response

        result['cached'] = False
        result['timing'] = timing
        return jsonify(result), 200
    else:
        result['timing'] = timing
        return jsonify(result), 500


@app.route('/history', methods=['GET'])
def history():
    """
    View previously processed videos and their notes status.

    Response (JSON):
        {
            "success": true,
            "videos": [...]
        }
    """
    from database.db import get_all_videos
    videos = get_all_videos()
    return jsonify({
        'success': True,
        'videos': videos,
    }), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
