"""
test_manual_phase6.py — Manual verification script for Phase 6 (Database).

Run this to verify the SQLite database works end-to-end:
  1. Initializes the DB
  2. Saves a video record
  3. Saves notes for the video
  4. Logs a couple of search queries
  5. Prints all rows from each table

Usage:
    python test_manual_phase6.py
"""

import sys
import os
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def main():
    print("=" * 60)
    print("  Phase 6 Manual Test — SQLite Database")
    print("=" * 60)

    # Use in-memory DB for this test (won't affect your real DB)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    # Step 1: Initialize tables
    print("\n📋 Step 1: Initializing database tables...")
    init_db(conn=conn)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row['name'] for row in cursor.fetchall()]
    print(f"   Tables created: {tables}")

    # Step 2: Save some videos
    print("\n🎬 Step 2: Saving video records...")
    save_video(
        video_id='dQw4w9WgXcQ',
        title='Rick Astley - Never Gonna Give You Up',
        url='https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        transcript_status='done',
        conn=conn,
    )
    save_video(
        video_id='9bZkp7q19f0',
        title='PSY - GANGNAM STYLE',
        url='https://www.youtube.com/watch?v=9bZkp7q19f0',
        transcript_status='pending',
        conn=conn,
    )
    print("   ✅ Saved 2 video records")

    # Test idempotent update
    save_video(
        video_id='9bZkp7q19f0',
        title='PSY - GANGNAM STYLE (M/V)',
        transcript_status='done',
        conn=conn,
    )
    print("   ✅ Updated video 9bZkp7q19f0 (idempotent upsert)")

    # Step 3: Save notes
    print("\n📝 Step 3: Saving notes...")
    save_notes(
        video_id='dQw4w9WgXcQ',
        notes_content=(
            "# Never Gonna Give You Up\n\n"
            "## Summary\n"
            "A classic pop song by Rick Astley from 1987.\n\n"
            "## Key Points\n"
            "- The song is about commitment and loyalty\n"
            "- It became an internet meme ('Rickrolling')\n\n"
            "## Key Takeaways\n"
            "- Never gonna give you up\n"
            "- Never gonna let you down\n"
        ),
        provider='groq',
        method='direct',
        conn=conn,
    )
    print("   ✅ Saved notes for dQw4w9WgXcQ")

    # Step 4: Log some searches
    print("\n🔍 Step 4: Logging search queries...")
    save_search(
        video_id='dQw4w9WgXcQ',
        query='What is this song about?',
        answer='This song is about commitment and loyalty in a relationship.',
        timestamp_ref='00:00:15',
        provider='groq',
        conn=conn,
    )
    save_search(
        video_id='dQw4w9WgXcQ',
        query='Who sings this song?',
        answer='Rick Astley sings this song.',
        timestamp_ref='00:00:00',
        provider='gemini',
        conn=conn,
    )
    print("   ✅ Logged 2 search queries")

    # ── Print all tables ──
    print("\n" + "=" * 60)
    print("  📊 DATABASE CONTENTS")
    print("=" * 60)

    # Videos table
    print("\n┌─── VIDEOS TABLE ───")
    videos = get_all_videos(conn=conn)
    for v in videos:
        print(f"│ id={v['id']}  video_id={v['video_id']}")
        print(f"│   title: {v['title']}")
        print(f"│   status: {v['transcript_status']}  created: {v['created_at']}")
        print("│")
    print(f"└─── Total: {len(videos)} records")

    # Notes table
    print("\n┌─── NOTES TABLE ───")
    for v in videos:
        notes = get_notes(v['video_id'], conn=conn)
        if notes:
            print(f"│ id={notes['id']}  video_id={notes['video_id']}")
            print(f"│   provider: {notes['provider']}  method: {notes['method']}")
            print(f"│   generated: {notes['generated_at']}")
            preview = notes['notes_content'][:100].replace('\n', ' ')
            print(f"│   content: {preview}...")
            print("│")
        else:
            print(f"│ video_id={v['video_id']}: (no notes)")
    print(f"└─── ")

    # Search History table
    print("\n┌─── SEARCH HISTORY TABLE ───")
    history = get_search_history(conn=conn)
    for h in history:
        print(f"│ id={h['id']}  video_id={h['video_id']}")
        print(f"│   query: {h['query']}")
        print(f"│   answer: {h['answer'][:80]}...")
        print(f"│   timestamp: {h['timestamp_ref']}  provider: {h['provider']}")
        print(f"│   searched: {h['searched_at']}")
        print("│")
    print(f"└─── Total: {len(history)} records")

    # ── Verify cache behavior ──
    print("\n" + "=" * 60)
    print("  🔄 CACHE VERIFICATION")
    print("=" * 60)
    cached_notes = get_notes('dQw4w9WgXcQ', conn=conn)
    if cached_notes:
        print(f"\n✅ Cache hit! Notes exist for dQw4w9WgXcQ")
        print(f"   → In production, /generate-notes would return these")
        print(f"     cached notes instead of calling the LLM again.")
    else:
        print("❌ Cache miss — notes not found")

    no_notes = get_notes('nonexistent_vid', conn=conn)
    print(f"\n{'✅' if no_notes is None else '❌'} Cache miss correctly "
          f"returned None for unknown video")

    print("\n" + "=" * 60)
    print("  ✅ Phase 6 manual test complete!")
    print("=" * 60)

    conn.close()


if __name__ == '__main__':
    main()
