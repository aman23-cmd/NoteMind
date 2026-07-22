"""
db.py — SQLite database setup and helper functions.

Tables (per design.md):
  - videos:         Tracks processed YouTube videos and their status.
  - notes:          Stores LLM-generated notes (cached to avoid regeneration).
  - search_history: Logs every RAG search for analytics / history view.

Design decisions:
  - Pure sqlite3 (no ORM) — lightweight, no extra dependencies.
  - All functions accept an optional `conn` parameter so tests can pass
    an in-memory connection (:memory:) while production uses the real file.
  - `save_video()` uses INSERT OR REPLACE for idempotent upserts.
  - Timestamps use ISO 8601 format via SQLite's datetime('now').
"""

import sqlite3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import Config


# ── Module-level connection cache ──
_conn = None


def get_connection(db_path=None):
    """
    Get or create a SQLite connection.

    Uses WAL mode for better concurrent read performance.

    Args:
        db_path (str): Path to the SQLite database file.
                       Defaults to Config.SQLITE_DB_PATH.

    Returns:
        sqlite3.Connection: The database connection.
    """
    global _conn

    if _conn is not None:
        return _conn

    path = db_path or Config.SQLITE_DB_PATH

    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    _conn = sqlite3.connect(path, check_same_thread=False)
    _conn.row_factory = sqlite3.Row  # Return dict-like rows
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("PRAGMA foreign_keys=ON")

    return _conn


def init_db(conn=None):
    """
    Create all tables if they don't already exist.

    Safe to call multiple times (uses IF NOT EXISTS).

    Args:
        conn: Optional sqlite3.Connection (for testing with :memory:).
              If None, uses the default file-based connection.
    """
    if conn is None:
        conn = get_connection()

    cursor = conn.cursor()

    # ── Table: videos ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id        TEXT    NOT NULL UNIQUE,
            title           TEXT    DEFAULT '',
            url             TEXT    DEFAULT '',
            transcript_status TEXT  DEFAULT 'pending',
            created_at      TEXT    DEFAULT (datetime('now'))
        )
    """)

    # ── Table: notes ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id        TEXT    NOT NULL,
            notes_content   TEXT    NOT NULL,
            provider        TEXT    DEFAULT '',
            method          TEXT    DEFAULT '',
            generated_at    TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (video_id) REFERENCES videos(video_id)
                ON DELETE CASCADE
        )
    """)

    # ── Table: search_history ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id        TEXT    NOT NULL,
            query           TEXT    NOT NULL,
            answer          TEXT    DEFAULT '',
            timestamp_ref   TEXT    DEFAULT '',
            provider        TEXT    DEFAULT '',
            searched_at     TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (video_id) REFERENCES videos(video_id)
                ON DELETE CASCADE
        )
    """)

    conn.commit()


# ════════════════════════════════════════════════════
#  VIDEO CRUD
# ════════════════════════════════════════════════════

def save_video(video_id, title='', url='', transcript_status='pending', conn=None):
    """
    Insert or update a video record.

    Uses INSERT OR REPLACE so calling this with the same video_id
    is idempotent — it updates the existing record.

    Args:
        video_id (str):           YouTube video ID.
        title (str):              Video title (if known).
        url (str):                Original YouTube URL.
        transcript_status (str):  'pending', 'done', or 'failed'.
        conn:                     Optional connection (for testing).

    Returns:
        dict: {'success': True, 'video_id': str}
    """
    if conn is None:
        conn = get_connection()

    try:
        conn.execute("""
            INSERT INTO videos (video_id, title, url, transcript_status)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                title = excluded.title,
                url = excluded.url,
                transcript_status = excluded.transcript_status
        """, (video_id, title, url, transcript_status))
        conn.commit()
        return {'success': True, 'video_id': video_id}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_video(video_id, conn=None):
    """
    Fetch a single video record by video_id.

    Args:
        video_id (str): YouTube video ID.
        conn:           Optional connection (for testing).

    Returns:
        dict: The video row as a dict, or None if not found.
    """
    if conn is None:
        conn = get_connection()

    cursor = conn.execute(
        "SELECT * FROM videos WHERE video_id = ?", (video_id,)
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def get_all_videos(conn=None):
    """
    Fetch all video records, most recent first.

    Args:
        conn: Optional connection (for testing).

    Returns:
        list: List of video dicts.
    """
    if conn is None:
        conn = get_connection()

    cursor = conn.execute(
        "SELECT * FROM videos ORDER BY created_at DESC"
    )
    return [dict(row) for row in cursor.fetchall()]


# ════════════════════════════════════════════════════
#  NOTES CRUD
# ════════════════════════════════════════════════════

def save_notes(video_id, notes_content, provider='', method='', conn=None):
    """
    Save generated notes for a video. Replaces any existing notes for
    the same video_id (one set of notes per video).

    Args:
        video_id (str):       YouTube video ID.
        notes_content (str):  The generated markdown notes.
        provider (str):       Which LLM was used ('groq' or 'gemini').
        method (str):         Generation method ('direct' or 'map_reduce').
        conn:                 Optional connection (for testing).

    Returns:
        dict: {'success': True, 'video_id': str}
    """
    if conn is None:
        conn = get_connection()

    try:
        # Delete old notes for this video (keep only the latest)
        conn.execute(
            "DELETE FROM notes WHERE video_id = ?", (video_id,)
        )
        conn.execute("""
            INSERT INTO notes (video_id, notes_content, provider, method)
            VALUES (?, ?, ?, ?)
        """, (video_id, notes_content, provider, method))
        conn.commit()
        return {'success': True, 'video_id': video_id}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_notes(video_id, conn=None):
    """
    Fetch the saved notes for a video.

    Args:
        video_id (str): YouTube video ID.
        conn:           Optional connection (for testing).

    Returns:
        dict: The notes row as a dict, or None if no notes exist.
    """
    if conn is None:
        conn = get_connection()

    cursor = conn.execute(
        "SELECT * FROM notes WHERE video_id = ? ORDER BY generated_at DESC LIMIT 1",
        (video_id,)
    )
    row = cursor.fetchone()
    return dict(row) if row else None


# ════════════════════════════════════════════════════
#  SEARCH HISTORY CRUD
# ════════════════════════════════════════════════════

def save_search(video_id, query, answer='', timestamp_ref='', provider='',
                conn=None):
    """
    Log a search query and its result.

    Unlike notes (one per video), search history accumulates — every
    search is logged as a new row.

    Args:
        video_id (str):      YouTube video ID.
        query (str):         The user's search question.
        answer (str):        The LLM's answer.
        timestamp_ref (str): The best-match timestamp (e.g. "00:01:23").
        provider (str):      Which LLM was used.
        conn:                Optional connection (for testing).

    Returns:
        dict: {'success': True, 'video_id': str}
    """
    if conn is None:
        conn = get_connection()

    try:
        conn.execute("""
            INSERT INTO search_history
                (video_id, query, answer, timestamp_ref, provider)
            VALUES (?, ?, ?, ?, ?)
        """, (video_id, query, answer, timestamp_ref, provider))
        conn.commit()
        return {'success': True, 'video_id': video_id}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_search_history(video_id=None, limit=50, conn=None):
    """
    Fetch search history, optionally filtered by video_id.

    Args:
        video_id (str): If provided, filter to this video only.
        limit (int):    Max rows to return (default 50).
        conn:           Optional connection (for testing).

    Returns:
        list: List of search_history dicts, most recent first.
    """
    if conn is None:
        conn = get_connection()

    if video_id:
        cursor = conn.execute(
            "SELECT * FROM search_history WHERE video_id = ? "
            "ORDER BY searched_at DESC LIMIT ?",
            (video_id, limit)
        )
    else:
        cursor = conn.execute(
            "SELECT * FROM search_history ORDER BY searched_at DESC LIMIT ?",
            (limit,)
        )

    return [dict(row) for row in cursor.fetchall()]


# ════════════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════════════

def reset_module():
    """Reset module-level connection cache. Used in tests."""
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
    _conn = None
