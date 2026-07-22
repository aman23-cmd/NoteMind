"""
test_vectorstore.py — Unit tests for modules/vectorstore.py

All tests use ChromaDB's EphemeralClient (in-memory) so nothing
touches the real persistent storage.

Tests cover:
  - Inserting chunks and querying them back
  - Metadata (timestamps, video_id) is preserved correctly
  - Querying a video_id with no data returns empty gracefully
  - Re-inserting same video_id doesn't crash (idempotency via upsert)
  - Deleting video chunks
  - has_video_chunks helper
  - Edge cases: empty inputs, mismatched lengths
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import chromadb
from modules.vectorstore import (
    store_chunks,
    query_chunks,
    delete_video_chunks,
    has_video_chunks,
    get_collection,
    COLLECTION_NAME,
)


# ── Fixture: fresh in-memory collection for each test ──
@pytest.fixture
def collection():
    """Create a fresh in-memory ChromaDB collection for each test."""
    client = chromadb.EphemeralClient()
    # Delete collection if it exists from a previous test in the same process
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return col


# ── Fixture: sample chunks + fake embeddings ──
@pytest.fixture
def sample_data():
    """Sample chunks and embeddings for testing."""
    chunks = [
        {
            'chunk_id': 'chunk_0',
            'chunk_text': 'Python is a programming language used for AI',
            'start_time': 0.0,
            'end_time': 30.0,
        },
        {
            'chunk_id': 'chunk_1',
            'chunk_text': 'Machine learning is a subset of artificial intelligence',
            'start_time': 30.0,
            'end_time': 60.0,
        },
        {
            'chunk_id': 'chunk_2',
            'chunk_text': 'Neural networks have layers of interconnected nodes',
            'start_time': 60.0,
            'end_time': 90.0,
        },
    ]

    # Fake 384-dim embeddings (just for testing — not real model output)
    embeddings = [
        [0.1] * 384,
        [0.2] * 384,
        [0.3] * 384,
    ]

    return chunks, embeddings


# ════════════════════════════════════════════════════
#  TESTS: store_chunks()
# ════════════════════════════════════════════════════

class TestStoreChunks:
    """Test inserting chunks into ChromaDB."""

    def test_store_success(self, collection, sample_data):
        """Storing chunks should succeed and return correct count."""
        chunks, embeddings = sample_data
        result = store_chunks("test_vid_01", chunks, embeddings,
                              collection=collection)

        assert result['success'] is True
        assert result['chunks_stored'] == 3
        assert result['video_id'] == "test_vid_01"

    def test_store_actually_persists(self, collection, sample_data):
        """After storing, the collection count should increase."""
        chunks, embeddings = sample_data
        assert collection.count() == 0

        store_chunks("test_vid_01", chunks, embeddings, collection=collection)
        assert collection.count() == 3

    def test_store_empty_chunks(self, collection):
        """Storing empty chunks returns an error, not a crash."""
        result = store_chunks("test_vid_01", [], [], collection=collection)
        assert result['success'] is False
        assert "no chunks" in result['error'].lower()

    def test_store_mismatched_lengths(self, collection, sample_data):
        """Mismatched chunk/embedding counts return an error."""
        chunks, embeddings = sample_data
        result = store_chunks("test_vid_01", chunks, embeddings[:2],
                              collection=collection)
        assert result['success'] is False
        assert "mismatch" in result['error'].lower()


# ════════════════════════════════════════════════════
#  TESTS: Idempotency (re-insert same video)
# ════════════════════════════════════════════════════

class TestIdempotency:
    """Test that re-inserting the same video doesn't crash or duplicate."""

    def test_upsert_same_video_twice(self, collection, sample_data):
        """Upserting the same video_id twice should NOT duplicate entries."""
        chunks, embeddings = sample_data

        # Insert first time
        store_chunks("test_vid_01", chunks, embeddings, collection=collection)
        assert collection.count() == 3

        # Insert same data again (should upsert, not duplicate)
        result = store_chunks("test_vid_01", chunks, embeddings,
                              collection=collection)
        assert result['success'] is True
        assert collection.count() == 3  # still 3, not 6

    def test_upsert_updates_text(self, collection, sample_data):
        """Upserting with modified text should update existing entries."""
        chunks, embeddings = sample_data

        store_chunks("test_vid_01", chunks, embeddings, collection=collection)

        # Modify chunk text and re-insert
        chunks[0]['chunk_text'] = "UPDATED: Python is awesome"
        store_chunks("test_vid_01", chunks, embeddings, collection=collection)

        # Query and verify the text was updated
        result = collection.get(ids=["test_vid_01_chunk_0"])
        assert "UPDATED" in result['documents'][0]


# ════════════════════════════════════════════════════
#  TESTS: query_chunks()
# ════════════════════════════════════════════════════

class TestQueryChunks:
    """Test querying ChromaDB for similar chunks."""

    def test_query_returns_results(self, collection, sample_data):
        """Querying after storing should return matching results."""
        chunks, embeddings = sample_data
        store_chunks("test_vid_01", chunks, embeddings, collection=collection)

        # Query with the first chunk's embedding (should match itself)
        result = query_chunks(
            query_embedding=embeddings[0],
            video_id="test_vid_01",
            top_k=3,
            collection=collection,
        )

        assert result['success'] is True
        assert len(result['results']) > 0
        assert result['video_id'] == "test_vid_01"

    def test_query_preserves_metadata(self, collection, sample_data):
        """Returned results should have correct timestamps and chunk_id."""
        chunks, embeddings = sample_data
        store_chunks("test_vid_01", chunks, embeddings, collection=collection)

        result = query_chunks(
            query_embedding=embeddings[0],
            video_id="test_vid_01",
            top_k=3,
            collection=collection,
        )

        # Check that each result has the required fields
        for r in result['results']:
            assert 'chunk_text' in r
            assert 'start_time' in r
            assert 'end_time' in r
            assert 'chunk_id' in r
            assert 'distance' in r
            assert isinstance(r['start_time'], float)
            assert isinstance(r['end_time'], float)

    def test_query_empty_collection(self, collection):
        """Querying an empty collection returns empty results, not an error."""
        result = query_chunks(
            query_embedding=[0.1] * 384,
            video_id="nonexistent_video",
            top_k=5,
            collection=collection,
        )

        assert result['success'] is True
        assert result['results'] == []

    def test_query_wrong_video_id(self, collection, sample_data):
        """Querying a video_id that has no data returns empty results."""
        chunks, embeddings = sample_data
        store_chunks("test_vid_01", chunks, embeddings, collection=collection)

        result = query_chunks(
            query_embedding=embeddings[0],
            video_id="different_video",
            top_k=3,
            collection=collection,
        )

        assert result['success'] is True
        assert result['results'] == []

    def test_query_top_k_limits(self, collection, sample_data):
        """top_k should limit the number of results."""
        chunks, embeddings = sample_data
        store_chunks("test_vid_01", chunks, embeddings, collection=collection)

        result = query_chunks(
            query_embedding=embeddings[0],
            video_id="test_vid_01",
            top_k=1,
            collection=collection,
        )

        assert len(result['results']) == 1


# ════════════════════════════════════════════════════
#  TESTS: delete_video_chunks()
# ════════════════════════════════════════════════════

class TestDeleteVideoChunks:
    """Test deleting a video's chunks from ChromaDB."""

    def test_delete_removes_data(self, collection, sample_data):
        """Deleting should remove all chunks for a video_id."""
        chunks, embeddings = sample_data
        store_chunks("test_vid_01", chunks, embeddings, collection=collection)
        assert collection.count() == 3

        result = delete_video_chunks("test_vid_01", collection=collection)
        assert result['success'] is True
        assert result['deleted'] == 3
        assert collection.count() == 0

    def test_delete_nonexistent_video(self, collection):
        """Deleting a nonexistent video should return deleted=0, not crash."""
        result = delete_video_chunks("nonexistent", collection=collection)
        assert result['success'] is True
        assert result['deleted'] == 0


# ════════════════════════════════════════════════════
#  TESTS: has_video_chunks()
# ════════════════════════════════════════════════════

class TestHasVideoChunks:
    """Test the has_video_chunks helper."""

    def test_returns_false_when_empty(self, collection):
        """Should return False for a video with no data."""
        assert has_video_chunks("nonexistent", collection=collection) is False

    def test_returns_true_after_storing(self, collection, sample_data):
        """Should return True after storing chunks."""
        chunks, embeddings = sample_data
        store_chunks("test_vid_01", chunks, embeddings, collection=collection)
        assert has_video_chunks("test_vid_01", collection=collection) is True

    def test_returns_false_after_deleting(self, collection, sample_data):
        """Should return False after deleting all chunks."""
        chunks, embeddings = sample_data
        store_chunks("test_vid_01", chunks, embeddings, collection=collection)
        delete_video_chunks("test_vid_01", collection=collection)
        assert has_video_chunks("test_vid_01", collection=collection) is False
