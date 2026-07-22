"""
test_pipeline_phase4.py — Integration test for the full RAG pipeline.

Tests the COMPLETE Phase 1→4 flow with REAL API calls:
  1. Fetch transcript from YouTube (Phase 1)
  2. Chunk it (Phase 2)
  3. Generate embeddings (Phase 2)
  4. Store in ChromaDB (Phase 3)
  5. Search with a real LLM call (Phase 4)

This test uses ONE real LLM call (Groq or Gemini) to confirm
end-to-end works. Keep this as a single test to minimize API usage.

NOTE: Requires internet access and at least one valid API key
      (GROQ_API_KEY or GEMINI_API_KEY) in the .env file.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import chromadb
from modules.transcript import get_transcript
from modules.chunker import chunk_transcript
from modules.embedder import generate_embeddings
from modules.vectorstore import store_chunks, COLLECTION_NAME
from modules.search import search_video


@pytest.fixture(scope="module")
def prepared_video():
    """
    Prepare a video for searching: fetch, chunk, embed, store.
    Uses an in-memory ChromaDB (not persistent).
    Runs once for the entire test module.
    """
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    # Step 1: Fetch transcript
    result = get_transcript(url)
    if not result['success']:
        pytest.skip(f"Could not fetch transcript: {result['error']}")

    video_id = result['video_id']
    transcript = result['transcript']

    # Step 2: Chunk
    chunks = chunk_transcript(transcript)

    # Step 3: Embed
    chunk_texts = [c['chunk_text'] for c in chunks]
    embeddings = generate_embeddings(chunk_texts)

    # Step 4: Store in in-memory ChromaDB
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    store_chunks(video_id, chunks, embeddings, collection=collection)

    return {
        'video_id': video_id,
        'collection': collection,
    }


class TestFullRagPipeline:
    """Integration test for the complete RAG search pipeline."""

    def test_real_llm_search(self, prepared_video):
        """
        Full end-to-end: search a real processed video with a real LLM call.

        This is the ONLY test that makes a real API call.
        It confirms that the entire pipeline works together.
        """
        result = search_video(
            query="What is this song about?",
            video_id=prepared_video['video_id'],
            top_k=3,
            collection=prepared_video['collection'],
        )

        # If both LLM providers fail (e.g. keys invalid), skip gracefully
        if not result['success'] and "both llm providers failed" in result.get('error', '').lower():
            pytest.skip(
                f"LLM API call failed (likely invalid key): {result['error']}"
            )

        assert result['success'] is True, f"Search failed: {result.get('error')}"
        assert 'answer' in result
        assert len(result['answer']) > 10  # non-trivial answer
        assert 'provider' in result
        assert result['provider'] in ('groq', 'gemini')
        assert 'timestamp' in result
        assert 'source_chunks' in result
        assert len(result['source_chunks']) > 0

        # Print for manual verification
        print(f"\n  Provider: {result['provider']}")
        print(f"  Timestamp: {result['timestamp']}")
        print(f"  Answer: {result['answer'][:200]}...")
        print(f"  Source chunks: {len(result['source_chunks'])}")
