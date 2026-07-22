"""
test_manual_phase4.py — Manual test for Phase 4 (RAG Search).

Run the full pipeline: fetch transcript → chunk → embed → store → SEARCH.

This script:
  1. Processes a YouTube video (or uses existing data)
  2. Asks a question about the video
  3. Calls the real LLM (Groq or Gemini) and prints the answer

Run:  python test_manual_phase4.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chromadb
from modules.transcript import get_transcript, format_timestamp
from modules.chunker import chunk_transcript, get_chunk_stats
from modules.embedder import generate_embeddings
from modules.vectorstore import store_chunks, has_video_chunks, COLLECTION_NAME
from modules.search import search_video


def main():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    search_query = "What is this song about?"

    print("=" * 60)
    print("Phase 4 Manual Test: RAG Search with LLM")
    print("=" * 60)

    # Use in-memory ChromaDB for this test
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Step 1-3: Process the video ──
    print(f"\n1. Processing video: {url}")

    result = get_transcript(url)
    if not result['success']:
        print(f"   ❌ Transcript failed: {result['error']}")
        return

    video_id = result['video_id']
    transcript = result['transcript']
    print(f"   ✅ Transcript: {len(transcript)} snippets")

    chunks = chunk_transcript(transcript)
    stats = get_chunk_stats(chunks)
    print(f"   ✅ Chunks: {stats['total_chunks']} chunks")

    chunk_texts = [c['chunk_text'] for c in chunks]
    embeddings = generate_embeddings(chunk_texts)
    print(f"   ✅ Embeddings: {len(embeddings)} × {len(embeddings[0])}")

    store_result = store_chunks(video_id, chunks, embeddings,
                                collection=collection)
    print(f"   ✅ Stored in ChromaDB: {store_result['chunks_stored']} chunks")

    # ── Step 4: SEARCH! ──
    print(f"\n2. Searching: \"{search_query}\"")
    print("   Calling LLM (Groq → Gemini fallback)...")

    search_result = search_video(
        query=search_query,
        video_id=video_id,
        top_k=3,
        collection=collection,
    )

    if search_result['success']:
        print(f"\n   ✅ Answer (via {search_result['provider']}):")
        print(f"   ──────────────────────────────────────")
        print(f"   {search_result['answer']}")
        print(f"   ──────────────────────────────────────")
        print(f"   📍 Most relevant timestamp: {search_result['timestamp']}")
        print(f"\n   Source chunks used ({len(search_result['source_chunks'])}):")
        for i, sc in enumerate(search_result['source_chunks']):
            print(f"   [{sc['timestamp']}] (distance: {sc['distance']:.4f}) "
                  f"{sc['chunk_text'][:80]}...")
    else:
        print(f"\n   ❌ Search failed: {search_result['error']}")

    # ── Bonus: Try another query ──
    print(f"\n3. Trying another query: \"What are the lyrics?\"")
    search_result2 = search_video(
        query="What are the lyrics?",
        video_id=video_id,
        top_k=2,
        collection=collection,
    )

    if search_result2['success']:
        print(f"   ✅ Answer (via {search_result2['provider']}): "
              f"{search_result2['answer'][:200]}...")
    else:
        print(f"   ❌ {search_result2['error']}")

    print("\n" + "=" * 60)
    print("Phase 4 test complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
