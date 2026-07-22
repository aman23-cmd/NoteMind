"""
test_pipeline_phase3.py — Integration test for the Phase 1→2→3 pipeline.

Tests the FULL flow:
  1. Fetch a real transcript from YouTube (Phase 1)
  2. Chunk it (Phase 2)
  3. Generate embeddings (Phase 2)
  4. Store in ChromaDB (Phase 3)
  5. Query back and verify results make sense

Uses an in-memory ChromaDB client (not persistent storage).
Uses the real sentence-transformers model (not mocked).

NOTE: This test hits the real YouTube API, so it requires internet access
      and may be slow on the first run (model download).
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import chromadb
from modules.transcript import get_transcript
from modules.chunker import chunk_transcript, get_chunk_stats
from modules.embedder import generate_embeddings
from modules.vectorstore import (
    store_chunks,
    query_chunks,
    has_video_chunks,
    COLLECTION_NAME,
)


@pytest.fixture(scope="module")
def pipeline_data():
    """
    Run the full pipeline once for the entire test module.

    Fetches a real YouTube transcript, chunks it, embeds it, and stores
    it in an in-memory ChromaDB collection.

    Returns a dict with all intermediate results for tests to verify.
    """
    # ── Step 1: Fetch real transcript ──
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    transcript_result = get_transcript(url)

    if not transcript_result['success']:
        pytest.skip(f"Could not fetch transcript: {transcript_result['error']}")

    video_id = transcript_result['video_id']
    transcript = transcript_result['transcript']

    # ── Step 2: Chunk ──
    chunks = chunk_transcript(transcript)

    # ── Step 3: Embed ──
    chunk_texts = [c['chunk_text'] for c in chunks]
    embeddings = generate_embeddings(chunk_texts)

    # ── Step 4: Store in in-memory ChromaDB ──
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    store_result = store_chunks(video_id, chunks, embeddings,
                                collection=collection)

    return {
        'video_id': video_id,
        'transcript': transcript,
        'chunks': chunks,
        'embeddings': embeddings,
        'store_result': store_result,
        'collection': collection,
    }


class TestFullPipeline:
    """Integration tests for the complete Phase 1-3 pipeline."""

    def test_transcript_was_fetched(self, pipeline_data):
        """The transcript should have been fetched successfully."""
        assert len(pipeline_data['transcript']) > 0

    def test_chunks_were_created(self, pipeline_data):
        """The chunker should have produced at least one chunk."""
        assert len(pipeline_data['chunks']) > 0

    def test_embeddings_match_chunks(self, pipeline_data):
        """Each chunk should have exactly one embedding."""
        assert len(pipeline_data['embeddings']) == len(pipeline_data['chunks'])

    def test_embeddings_correct_dimension(self, pipeline_data):
        """All embeddings should be 384-dimensional."""
        for emb in pipeline_data['embeddings']:
            assert len(emb) == 384

    def test_store_succeeded(self, pipeline_data):
        """Storing in ChromaDB should have succeeded."""
        assert pipeline_data['store_result']['success'] is True
        assert pipeline_data['store_result']['chunks_stored'] == \
            len(pipeline_data['chunks'])

    def test_has_video_chunks(self, pipeline_data):
        """has_video_chunks should return True after storing."""
        assert has_video_chunks(
            pipeline_data['video_id'],
            collection=pipeline_data['collection'],
        ) is True

    def test_query_returns_results(self, pipeline_data):
        """Querying with a chunk's own embedding should return results."""
        result = query_chunks(
            query_embedding=pipeline_data['embeddings'][0],
            video_id=pipeline_data['video_id'],
            top_k=3,
            collection=pipeline_data['collection'],
        )

        assert result['success'] is True
        assert len(result['results']) > 0

    def test_query_results_have_metadata(self, pipeline_data):
        """Query results should include timestamps and chunk IDs."""
        result = query_chunks(
            query_embedding=pipeline_data['embeddings'][0],
            video_id=pipeline_data['video_id'],
            top_k=3,
            collection=pipeline_data['collection'],
        )

        for r in result['results']:
            assert 'chunk_text' in r
            assert 'start_time' in r
            assert 'end_time' in r
            assert 'chunk_id' in r
            assert r['start_time'] < r['end_time']

    def test_semantic_query(self, pipeline_data):
        """
        A semantic search should return relevant results.

        We embed a query text and search for it — the results should
        contain text that is semantically related.
        """
        query_text = "never gonna give you up"
        query_embedding = generate_embeddings(query_text)[0]

        result = query_chunks(
            query_embedding=query_embedding,
            video_id=pipeline_data['video_id'],
            top_k=1,
            collection=pipeline_data['collection'],
        )

        assert result['success'] is True
        assert len(result['results']) > 0
        # The top result should contain some text (basic sanity check)
        assert len(result['results'][0]['chunk_text']) > 0

    def test_query_wrong_video_returns_empty(self, pipeline_data):
        """Querying a different video_id should return no results."""
        result = query_chunks(
            query_embedding=pipeline_data['embeddings'][0],
            video_id="completely_different_id",
            top_k=3,
            collection=pipeline_data['collection'],
        )

        assert result['success'] is True
        assert len(result['results']) == 0
