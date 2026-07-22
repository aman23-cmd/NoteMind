"""
test_db.py — Unit tests for database/db.py

All tests use an in-memory SQLite database (":memory:") so no files
are created on disk and tests are fully isolated.

Tests cover:
  - Table creation (init_db)
  - Video CRUD: save, fetch, update, idempotent upsert
  - Notes CRUD: save, fetch, replacement on re-save
  - Search history CRUD: save, fetch, filtering, multiple entries
  - Edge cases: empty tables, missing records
"""

import pytest
import sqlite3
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from database.db import (
    init_db,
    save_video,
    get_video,
    get_all_videos,
    save_notes,
    get_notes,
    save_search,
    get_search_history,
)


@pytest.fixture
def db():
    """
    Create a fresh in-memory SQLite database for each test.
    Tables are initialized automatically.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn=conn)
    yield conn
    conn.close()


# ════════════════════════════════════════════════════
#  TESTS: init_db()
# ════════════════════════════════════════════════════

class TestInitDb:
    """Test database initialization."""

    def test_tables_created(self, db):
        """All three tables should exist after init_db."""
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "ORDER BY name"
        )
        tables = [row['name'] for row in cursor.fetchall()]
        assert 'videos' in tables
        assert 'notes' in tables
        assert 'search_history' in tables

    def test_init_idempotent(self, db):
        """Calling init_db twice should not crash."""
        init_db(conn=db)  # second call
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row['name'] for row in cursor.fetchall()]
        assert 'videos' in tables

    def test_videos_columns(self, db):
        """The videos table should have the expected columns."""
        cursor = db.execute("PRAGMA table_info(videos)")
        columns = {row['name'] for row in cursor.fetchall()}
        assert columns == {
            'id', 'video_id', 'title', 'url',
            'transcript_status', 'created_at'
        }

    def test_notes_columns(self, db):
        """The notes table should have the expected columns."""
        cursor = db.execute("PRAGMA table_info(notes)")
        columns = {row['name'] for row in cursor.fetchall()}
        assert columns == {
            'id', 'video_id', 'notes_content',
            'provider', 'method', 'generated_at'
        }

    def test_search_history_columns(self, db):
        """The search_history table should have the expected columns."""
        cursor = db.execute("PRAGMA table_info(search_history)")
        columns = {row['name'] for row in cursor.fetchall()}
        assert columns == {
            'id', 'video_id', 'query', 'answer',
            'timestamp_ref', 'provider', 'searched_at'
        }


# ════════════════════════════════════════════════════
#  TESTS: Video CRUD
# ════════════════════════════════════════════════════

class TestSaveVideo:
    """Test video record insertion and updates."""

    def test_save_and_fetch(self, db):
        """Save a video and retrieve it by video_id."""
        result = save_video(
            video_id='abc123',
            title='Test Video',
            url='https://youtube.com/watch?v=abc123',
            transcript_status='done',
            conn=db,
        )
        assert result['success'] is True

        video = get_video('abc123', conn=db)
        assert video is not None
        assert video['video_id'] == 'abc123'
        assert video['title'] == 'Test Video'
        assert video['transcript_status'] == 'done'

    def test_save_minimal(self, db):
        """Save with just video_id (other fields have defaults)."""
        save_video(video_id='min123', conn=db)
        video = get_video('min123', conn=db)
        assert video is not None
        assert video['title'] == ''
        assert video['transcript_status'] == 'pending'

    def test_idempotent_upsert(self, db):
        """Saving the same video_id twice should update, not crash."""
        save_video(video_id='dup123', title='First', conn=db)
        save_video(video_id='dup123', title='Updated', transcript_status='done', conn=db)

        video = get_video('dup123', conn=db)
        assert video['title'] == 'Updated'
        assert video['transcript_status'] == 'done'

        # Should still be one record, not two
        cursor = db.execute("SELECT COUNT(*) as cnt FROM videos")
        assert cursor.fetchone()['cnt'] == 1

    def test_fetch_nonexistent(self, db):
        """Fetching a video_id that doesn't exist returns None."""
        video = get_video('nonexistent', conn=db)
        assert video is None


class TestGetAllVideos:
    """Test fetching all video records."""

    def test_empty_table(self, db):
        """Empty videos table returns empty list."""
        videos = get_all_videos(conn=db)
        assert videos == []

    def test_multiple_videos(self, db):
        """Multiple videos are returned in order."""
        save_video(video_id='vid1', title='Video 1', conn=db)
        save_video(video_id='vid2', title='Video 2', conn=db)
        save_video(video_id='vid3', title='Video 3', conn=db)

        videos = get_all_videos(conn=db)
        assert len(videos) == 3


# ════════════════════════════════════════════════════
#  TESTS: Notes CRUD
# ════════════════════════════════════════════════════

class TestSaveNotes:
    """Test notes insertion and retrieval."""

    def test_save_and_fetch(self, db):
        """Save notes and retrieve them."""
        # Must save the video first (FK constraint)
        save_video(video_id='note_vid', conn=db)

        result = save_notes(
            video_id='note_vid',
            notes_content='# Test Notes\n- Point 1\n- Point 2',
            provider='groq',
            method='direct',
            conn=db,
        )
        assert result['success'] is True

        notes = get_notes('note_vid', conn=db)
        assert notes is not None
        assert '# Test Notes' in notes['notes_content']
        assert notes['provider'] == 'groq'
        assert notes['method'] == 'direct'

    def test_overwrite_on_resave(self, db):
        """Saving notes again for the same video replaces old notes."""
        save_video(video_id='resave_vid', conn=db)

        save_notes(video_id='resave_vid', notes_content='Old notes', conn=db)
        save_notes(video_id='resave_vid', notes_content='New notes', conn=db)

        notes = get_notes('resave_vid', conn=db)
        assert notes['notes_content'] == 'New notes'

        # Should be exactly one row (old was deleted)
        cursor = db.execute(
            "SELECT COUNT(*) as cnt FROM notes WHERE video_id = 'resave_vid'"
        )
        assert cursor.fetchone()['cnt'] == 1

    def test_fetch_nonexistent(self, db):
        """Fetching notes for a video with no notes returns None."""
        notes = get_notes('no_notes_vid', conn=db)
        assert notes is None


# ════════════════════════════════════════════════════
#  TESTS: Search History CRUD
# ════════════════════════════════════════════════════

class TestSaveSearch:
    """Test search history logging and retrieval."""

    def test_save_and_fetch(self, db):
        """Log a search and retrieve it."""
        save_video(video_id='search_vid', conn=db)

        result = save_search(
            video_id='search_vid',
            query='What is Python?',
            answer='Python is a programming language.',
            timestamp_ref='00:01:23',
            provider='groq',
            conn=db,
        )
        assert result['success'] is True

        history = get_search_history(video_id='search_vid', conn=db)
        assert len(history) == 1
        assert history[0]['query'] == 'What is Python?'
        assert history[0]['answer'] == 'Python is a programming language.'
        assert history[0]['timestamp_ref'] == '00:01:23'
        assert history[0]['provider'] == 'groq'

    def test_multiple_searches_accumulate(self, db):
        """Each search creates a new row (no replacement)."""
        save_video(video_id='multi_vid', conn=db)

        save_search(video_id='multi_vid', query='Q1', answer='A1', conn=db)
        save_search(video_id='multi_vid', query='Q2', answer='A2', conn=db)
        save_search(video_id='multi_vid', query='Q3', answer='A3', conn=db)

        history = get_search_history(video_id='multi_vid', conn=db)
        assert len(history) == 3

    def test_filter_by_video_id(self, db):
        """Filtering by video_id should only return that video's history."""
        save_video(video_id='vid_a', conn=db)
        save_video(video_id='vid_b', conn=db)

        save_search(video_id='vid_a', query='Q for A', conn=db)
        save_search(video_id='vid_b', query='Q for B', conn=db)

        history_a = get_search_history(video_id='vid_a', conn=db)
        history_b = get_search_history(video_id='vid_b', conn=db)

        assert len(history_a) == 1
        assert history_a[0]['query'] == 'Q for A'
        assert len(history_b) == 1
        assert history_b[0]['query'] == 'Q for B'

    def test_fetch_all_history(self, db):
        """Without video_id filter, returns all searches."""
        save_video(video_id='all_a', conn=db)
        save_video(video_id='all_b', conn=db)

        save_search(video_id='all_a', query='Q1', conn=db)
        save_search(video_id='all_b', query='Q2', conn=db)

        all_history = get_search_history(conn=db)
        assert len(all_history) == 2

    def test_empty_history(self, db):
        """Empty search history returns empty list."""
        history = get_search_history(conn=db)
        assert history == []

    def test_limit_parameter(self, db):
        """Limit parameter caps the number of results."""
        save_video(video_id='limit_vid', conn=db)

        for i in range(10):
            save_search(video_id='limit_vid', query=f'Q{i}', conn=db)

        history = get_search_history(video_id='limit_vid', limit=3, conn=db)
        assert len(history) == 3
