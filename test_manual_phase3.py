"""
test_manual_phase3.py — Manual test for Phase 3 (ChromaDB Vector Store).

Runs the full pipeline:
  1. Fetches a real transcript from YouTube
  2. Chunks it
  3. Generates embeddings
  4. Stores in ChromaDB (persistent storage)
  5. Searches for a topic and displays results with timestamps

Run:  python test_manual_phase3.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.transcript import get_transcript, format_timestamp
from modules.chunker import chunk_transcript, get_chunk_stats
from modules.embedder import generate_embeddings
from modules.vectorstore import (
    store_chunks,
    query_chunks,
    has_video_chunks,
    get_client,
    get_collection,
    reset_module,
)


def main():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    search_query = "never gonna give you up"

    print("=" * 60)
    print("Phase 3 Manual Test: Full Pipeline + ChromaDB Search")
    print("=" * 60)

    # Reset module caches to use a fresh ephemeral client for testing
    reset_module()

    # ── Step 1: Fetch transcript ──
    print(f"\n1. Fetching transcript for: {url}")
    result = get_transcript(url)

    if not result['success']:
        print(f"   ❌ Failed: {result['error']}")
        return

    video_id = result['video_id']
    transcript = result['transcript']
    print(f"   ✅ Got {len(transcript)} snippets (video_id: {video_id})")

    # ── Step 2: Chunk ──
    print("\n2. Chunking transcript...")
    chunks = chunk_transcript(transcript)
    stats = get_chunk_stats(chunks)
    print(f"   ✅ Created {stats['total_chunks']} chunks "
          f"(avg {stats['avg_words']} words each)")

    # ── Step 3: Embed ──
    print("\n3. Generating embeddings...")
    chunk_texts = [c['chunk_text'] for c in chunks]
    embeddings = generate_embeddings(chunk_texts)
    print(f"   ✅ Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")

    # ── Step 4: Store in ChromaDB ──
    print("\n4. Storing in ChromaDB...")

    import chromadb
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="video_transcripts",
        metadata={"hnsw:space": "cosine"},
    )

    store_result = store_chunks(video_id, chunks, embeddings,
                                collection=collection)
    if store_result['success']:
        print(f"   ✅ Stored {store_result['chunks_stored']} chunks")
    else:
        print(f"   ❌ Failed: {store_result['error']}")
        return

    # Verify it's stored
    has_data = has_video_chunks(video_id, collection=collection)
    print(f"   Has data for {video_id}: {has_data}")

    # ── Step 5: Search! ──
    print(f"\n5. Searching for: \"{search_query}\"")
    query_embedding = generate_embeddings(search_query)[0]

    search_result = query_chunks(
        query_embedding=query_embedding,
        video_id=video_id,
        top_k=3,
        collection=collection,
    )

    if search_result['results']:
        print(f"   ✅ Found {len(search_result['results'])} results:\n")
        for i, r in enumerate(search_result['results']):
            ts_start = format_timestamp(r['start_time'])
            ts_end = format_timestamp(r['end_time'])
            print(f"   Result {i + 1} (distance: {r['distance']:.4f})")
            print(f"   Time:  [{ts_start}] → [{ts_end}]")
            preview = r['chunk_text'][:200]
            print(f"   Text:  {preview}...")
            print()
    else:
        print("   No results found.")

    # ── Step 6: Test idempotency ──
    print("6. Testing idempotency (re-storing same video)...")
    store_result2 = store_chunks(video_id, chunks, embeddings,
                                 collection=collection)
    count = collection.count()
    print(f"   ✅ Re-stored successfully. Collection count: {count} "
          f"(should still be {len(chunks)}, not doubled)")

    print("\n" + "=" * 60)
    print("Phase 3 test complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
