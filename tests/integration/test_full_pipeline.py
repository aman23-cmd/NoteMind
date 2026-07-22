"""
test_full_pipeline.py — Full end-to-end integration test for NoteMind.

Tests the complete user journey:
  1. Submit YouTube URL → process video (transcript → chunk → embed → store)
  2. Generate notes for the processed video
  3. Perform a topic search on the video
  4. Verify all outputs are correct
  5. Test edge cases: invalid URLs, empty queries, duplicate processing, etc.

Uses Flask's test client (no real HTTP server needed).
Uses mocked LLM calls (Groq/Gemini) to avoid API key dependency.
Uses the real embedding model (sentence-transformers) for true integration.

NOTE: Requires internet access for YouTube transcript fetching.
"""

import pytest
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ── Reset module caches before importing app ──
from modules.vectorstore import reset_module as reset_vectorstore
from database.db import reset_module as reset_db
reset_vectorstore()
reset_db()

from app import app
from database.db import init_db


# ── Short YouTube video with English captions ──
# Rick Astley - Never Gonna Give You Up (~3.5 min, definitely has captions)
TEST_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
TEST_VIDEO_ID = "dQw4w9WgXcQ"


@pytest.fixture(scope="module")
def client():
    """Create a Flask test client with a clean test database."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# ════════════════════════════════════════════════════════════
#  FULL PIPELINE: Process → Notes → Search
# ════════════════════════════════════════════════════════════

class TestFullPipeline:
    """End-to-end integration: complete user journey."""

    def test_01_health_check(self, client):
        """Health endpoint should return 200."""
        r = client.get('/health')
        assert r.status_code == 200
        data = r.get_json()
        assert data['status'] == 'ok'

    def test_02_process_video(self, client):
        """Process a real YouTube video end-to-end."""
        r = client.post('/process-video', json={"url": TEST_URL})

        assert r.status_code == 200, f"Failed: {r.get_json()}"
        data = r.get_json()
        assert data['success'] is True
        assert data['video_id'] == TEST_VIDEO_ID
        assert data['status'] == 'done'
        assert data['chunks_stored'] >= 1
        # Timing data should be present
        assert 'timing' in data
        assert data['timing']['total'] > 0

    def test_03_process_same_video_again(self, client):
        """Processing the same video again should succeed (idempotent upsert)."""
        r = client.post('/process-video', json={"video_id": TEST_VIDEO_ID})

        assert r.status_code == 200
        data = r.get_json()
        assert data['success'] is True
        assert data['video_id'] == TEST_VIDEO_ID

    def test_04_history_shows_video(self, client):
        """After processing, the video should appear in history."""
        r = client.get('/history')
        assert r.status_code == 200
        data = r.get_json()
        assert data['success'] is True
        assert len(data['videos']) >= 1

        video_ids = [v['video_id'] for v in data['videos']]
        assert TEST_VIDEO_ID in video_ids

    @pytest.mark.parametrize("query", [
        "What is this video about?",
        "never gonna give you up",
    ])
    def test_05_search_video(self, client, query):
        """Search a processed video — should return results with mocked LLM."""
        with pytest.importorskip("unittest.mock").patch(
            'modules.search._call_llm'
        ) as mock_llm:
            mock_llm.return_value = {
                'success': True,
                'answer': f'Mock answer for: {query}',
                'provider': 'groq',
            }
            r = client.post('/search', json={
                "video_id": TEST_VIDEO_ID,
                "query": query,
            })

        assert r.status_code == 200
        data = r.get_json()
        assert data['success'] is True
        assert 'answer' in data
        assert len(data['answer']) > 0
        assert data['provider'] == 'groq'
        assert 'timestamp' in data
        assert 'source_chunks' in data
        assert len(data['source_chunks']) >= 1

    def test_06_generate_notes(self, client):
        """Generate notes for a processed video (mocked LLM)."""
        with pytest.importorskip("unittest.mock").patch(
            'modules.notes_generator._call_llm'
        ) as mock_llm:
            mock_llm.return_value = {
                'success': True,
                'text': '# Video Notes\n- This is a test note\n- Key point 1',
                'provider': 'groq',
            }
            r = client.post('/generate-notes', json={
                "video_id": TEST_VIDEO_ID,
                "force": True,
            })

        assert r.status_code == 200
        data = r.get_json()
        assert data['success'] is True
        assert 'notes' in data
        assert len(data['notes']) > 0
        assert data['method'] in ('direct', 'map_reduce')
        assert data['cached'] is False

    def test_07_generate_notes_cached(self, client):
        """After generating, notes should be served from cache."""
        r = client.post('/generate-notes', json={
            "video_id": TEST_VIDEO_ID,
        })

        assert r.status_code == 200
        data = r.get_json()
        assert data['success'] is True
        assert data['cached'] is True
        assert len(data['notes']) > 0


# ════════════════════════════════════════════════════════════
#  EDGE CASES: Invalid Inputs
# ════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test error handling for invalid/edge-case inputs."""

    # ── Invalid URLs ──

    def test_invalid_url_returns_400(self, client):
        """A malformed URL should return 400."""
        r = client.post('/process-video', json={"url": "not-a-youtube-url"})
        assert r.status_code == 400
        data = r.get_json()
        assert data['success'] is False
        assert 'error' in data

    def test_empty_url_returns_400(self, client):
        """Empty URL with no video_id should return 400."""
        r = client.post('/process-video', json={"url": ""})
        assert r.status_code == 400
        data = r.get_json()
        assert data['success'] is False

    def test_missing_fields_returns_400(self, client):
        """No url or video_id should return 400."""
        r = client.post('/process-video', json={})
        assert r.status_code == 400
        data = r.get_json()
        assert data['success'] is False

    def test_non_json_returns_400(self, client):
        """Non-JSON request body should return 400."""
        r = client.post('/process-video', data="not json",
                        content_type='text/plain')
        assert r.status_code == 400
        data = r.get_json()
        assert data['success'] is False

    # ── Empty search query ──

    def test_empty_search_query_returns_400(self, client):
        """Empty search query should return 400."""
        r = client.post('/search', json={
            "video_id": TEST_VIDEO_ID,
            "query": "",
        })
        assert r.status_code == 400
        data = r.get_json()
        assert data['success'] is False

    def test_missing_search_video_id_returns_400(self, client):
        """Missing video_id in search should return 400."""
        r = client.post('/search', json={
            "query": "test query",
        })
        assert r.status_code == 400
        data = r.get_json()
        assert data['success'] is False

    # ── Search on unprocessed video ──

    def test_search_unprocessed_video_returns_404(self, client):
        """Searching a video that was never processed should return 404."""
        with pytest.importorskip("unittest.mock").patch(
            'modules.search._call_llm'
        ) as mock_llm:
            mock_llm.return_value = {
                'success': True,
                'answer': 'test',
                'provider': 'groq',
            }
            r = client.post('/search', json={
                "video_id": "XXXXXXXXXXX",  # never processed
                "query": "What is this?",
            })

        assert r.status_code == 404
        data = r.get_json()
        assert data['success'] is False
        assert 'no transcript chunks' in data['error'].lower() or 'not found' in data['error'].lower()

    # ── Generate notes edge cases ──

    def test_generate_notes_missing_video_id_returns_400(self, client):
        """Missing video_id in generate-notes should return 400."""
        r = client.post('/generate-notes', json={})
        assert r.status_code == 400
        data = r.get_json()
        assert data['success'] is False

    def test_generate_notes_non_json_returns_400(self, client):
        """Non-JSON body for generate-notes should return 400."""
        r = client.post('/generate-notes', data="not json",
                        content_type='text/plain')
        assert r.status_code == 400

    # ── LLM failure ──

    def test_search_llm_failure_returns_error(self, client):
        """When both LLMs fail, search should return a friendly error."""
        with pytest.importorskip("unittest.mock").patch(
            'modules.search._call_llm'
        ) as mock_llm:
            mock_llm.return_value = {
                'success': False,
                'error': 'Both LLM providers failed. Please check API keys.',
            }
            r = client.post('/search', json={
                "video_id": TEST_VIDEO_ID,
                "query": "test query",
            })

        # Should not crash — returns the error as JSON
        data = r.get_json()
        assert data['success'] is False
        assert 'error' in data
        assert 'failed' in data['error'].lower()

    def test_notes_llm_failure_returns_500(self, client):
        """When LLM fails during notes generation, return 500 with error."""
        with pytest.importorskip("unittest.mock").patch(
            'modules.notes_generator._call_llm'
        ) as mock_llm:
            mock_llm.return_value = {
                'success': False,
                'error': 'Both LLM providers failed.',
            }
            r = client.post('/generate-notes', json={
                "video_id": TEST_VIDEO_ID,
                "force": True,
            })

        assert r.status_code == 500
        data = r.get_json()
        assert data['success'] is False
        assert 'error' in data
