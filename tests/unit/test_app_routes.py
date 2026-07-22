"""
test_app_routes.py — Unit tests for Flask routes in app.py

Tests all Flask endpoints for correct HTTP status codes, JSON responses,
and error handling using Flask's test client with mocked backend modules.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app import app


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# ════════════════════════════════════════════════════════
#  HEALTH & INDEX
# ════════════════════════════════════════════════════════

class TestBasicRoutes:
    """Test basic Flask routes."""

    def test_health_returns_200(self, client):
        r = client.get('/health')
        assert r.status_code == 200
        data = r.get_json()
        assert data['status'] == 'ok'
        assert data['project'] == 'NoteMind'

    def test_index_returns_200(self, client):
        r = client.get('/')
        assert r.status_code == 200


# ════════════════════════════════════════════════════════
#  /process-video VALIDATION
# ════════════════════════════════════════════════════════

class TestProcessVideoValidation:
    """Test input validation for /process-video."""

    def test_non_json_returns_400(self, client):
        r = client.post('/process-video', data="not json",
                        content_type='text/plain')
        assert r.status_code == 400
        assert 'JSON' in r.get_json()['error']

    def test_empty_body_returns_400(self, client):
        r = client.post('/process-video', json={})
        assert r.status_code == 400

    def test_invalid_url_returns_400(self, client):
        r = client.post('/process-video', json={"url": "http://example.com"})
        assert r.status_code == 400
        assert r.get_json()['success'] is False

    @patch('modules.transcript.get_transcript')
    def test_transcript_failure_returns_404(self, mock_get, client):
        """When transcript fetch fails, return 404."""
        mock_get.return_value = {
            'success': False,
            'video_id': 'dQw4w9WgXcQ',
            'error': 'Video unavailable',
        }
        r = client.post('/process-video', json={"video_id": "dQw4w9WgXcQ"})
        assert r.status_code == 404
        assert 'Transcript extraction failed' in r.get_json()['error']


# ════════════════════════════════════════════════════════
#  /search VALIDATION
# ════════════════════════════════════════════════════════

class TestSearchValidation:
    """Test input validation for /search."""

    def test_non_json_returns_400(self, client):
        r = client.post('/search', data="not json",
                        content_type='text/plain')
        assert r.status_code == 400

    def test_missing_video_id_returns_400(self, client):
        r = client.post('/search', json={"query": "test"})
        assert r.status_code == 400
        assert 'video_id' in r.get_json()['error']

    def test_missing_query_returns_400(self, client):
        r = client.post('/search', json={"video_id": "test_vid"})
        assert r.status_code == 400
        assert 'query' in r.get_json()['error']

    def test_empty_query_returns_400(self, client):
        r = client.post('/search', json={"video_id": "test_vid", "query": "   "})
        assert r.status_code == 400


# ════════════════════════════════════════════════════════
#  /generate-notes VALIDATION
# ════════════════════════════════════════════════════════

class TestGenerateNotesValidation:
    """Test input validation for /generate-notes."""

    def test_non_json_returns_400(self, client):
        r = client.post('/generate-notes', data="not json",
                        content_type='text/plain')
        assert r.status_code == 400

    def test_missing_video_id_returns_400(self, client):
        r = client.post('/generate-notes', json={})
        assert r.status_code == 400

    def test_invalid_url_in_notes_returns_400(self, client):
        r = client.post('/generate-notes', json={"url": "http://example.com"})
        assert r.status_code == 400

    @patch('database.db.get_notes')
    def test_cached_notes_returned_immediately(self, mock_get_notes, client):
        """When cached notes exist, return them without calling LLM."""
        mock_get_notes.return_value = {
            'notes_content': '# Cached Notes',
            'provider': 'groq',
            'method': 'direct',
        }
        r = client.post('/generate-notes', json={"video_id": "cached_vid"})
        assert r.status_code == 200
        data = r.get_json()
        assert data['cached'] is True
        assert data['notes'] == '# Cached Notes'


# ════════════════════════════════════════════════════════
#  /history
# ════════════════════════════════════════════════════════

class TestHistory:
    """Test /history endpoint."""

    @patch('database.db.get_all_videos')
    def test_history_returns_videos(self, mock_get_all, client):
        mock_get_all.return_value = [
            {'video_id': 'test1', 'title': 'Test 1'},
        ]
        r = client.get('/history')
        assert r.status_code == 200
        data = r.get_json()
        assert data['success'] is True
        assert len(data['videos']) == 1
