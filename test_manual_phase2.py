"""
test_manual_phase2.py — Manual test for Phase 2 (Chunking + Embeddings).

Fetches a real transcript from YouTube, chunks it, generates embeddings,
and prints a summary of the results.

Run:  python test_manual_phase2.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.transcript import get_transcript, format_timestamp
from modules.chunker import chunk_transcript, get_chunk_stats
from modules.embedder import generate_embeddings, get_embedding_dimension


def main():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    print("=" * 60)
    print("Phase 2 Manual Test: Chunking + Embeddings")
    print("=" * 60)

    # ── Step 1: Fetch transcript ──
    print(f"\n1. Fetching transcript for: {url}")
    result = get_transcript(url)

    if not result['success']:
        print(f"   ❌ Failed: {result['error']}")
        return

    transcript = result['transcript']
    total_words = sum(len(s['text'].split()) for s in transcript)
    print(f"   ✅ Got {len(transcript)} snippets ({total_words} total words)")

    # ── Step 2: Chunk the transcript ──
    print("\n2. Chunking transcript...")
    chunks = chunk_transcript(transcript)
    stats = get_chunk_stats(chunks)

    print(f"   ✅ Created {stats['total_chunks']} chunks")
    print(f"   Word counts per chunk: {stats['word_counts']}")
    print(f"   Average: {stats['avg_words']} words | "
          f"Min: {stats['min_words']} | Max: {stats['max_words']}")
    print(f"   Total video duration covered: "
          f"{format_timestamp(stats['total_duration_seconds'])}")

    # Show first chunk as a preview
    if chunks:
        print(f"\n   First chunk preview (chunk_0):")
        print(f"   Time: [{format_timestamp(chunks[0]['start_time'])}] → "
              f"[{format_timestamp(chunks[0]['end_time'])}]")
        preview = chunks[0]['chunk_text'][:200]
        print(f"   Text: {preview}...")

    # ── Step 3: Generate embeddings ──
    print("\n3. Generating embeddings...")
    chunk_texts = [c['chunk_text'] for c in chunks]
    embeddings = generate_embeddings(chunk_texts)

    dim = get_embedding_dimension()
    print(f"   ✅ Generated {len(embeddings)} embeddings")
    print(f"   Embedding dimension: {dim}")
    print(f"   Shape check: {len(embeddings)} × {len(embeddings[0]) if embeddings else 0}")

    # Verify all embeddings have the right dimension
    all_correct = all(len(e) == dim for e in embeddings)
    print(f"   All dimensions correct: {'✅ Yes' if all_correct else '❌ No'}")

    print("\n" + "=" * 60)
    print("Phase 2 test complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
