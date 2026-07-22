"""
test_chunker.py — Unit tests for modules/chunker.py

Tests cover:
  - Basic chunking with correct chunk sizes
  - Overlap between consecutive chunks
  - Timestamps are preserved correctly
  - Edge case: very short transcript (smaller than one chunk)
  - Edge case: empty transcript
  - Edge case: single snippet
  - Chunk IDs are sequential
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.chunker import chunk_transcript, get_chunk_stats


# ── Helper: generate a fake transcript with N snippets ──
def _make_transcript(num_snippets, words_per_snippet=10, start=0.0, gap=5.0):
    """
    Create a fake transcript with `num_snippets` snippets.
    Each snippet has `words_per_snippet` words.
    Snippets are spaced `gap` seconds apart.
    """
    transcript = []
    for i in range(num_snippets):
        words = ' '.join([f'word{i}_{w}' for w in range(words_per_snippet)])
        s_time = start + i * gap
        e_time = s_time + gap
        transcript.append({
            'text': words,
            'start_time': round(s_time, 2),
            'end_time': round(e_time, 2),
        })
    return transcript


# ════════════════════════════════════════════════════
#  TESTS: Basic chunking
# ════════════════════════════════════════════════════

class TestChunkTranscriptBasic:
    """Test basic chunking behavior."""

    def test_empty_transcript(self):
        """Empty input returns empty output."""
        result = chunk_transcript([])
        assert result == []

    def test_empty_text_snippets(self):
        """Snippets with empty text are filtered out."""
        transcript = [
            {'text': '', 'start_time': 0.0, 'end_time': 1.0},
            {'text': '   ', 'start_time': 1.0, 'end_time': 2.0},
        ]
        result = chunk_transcript(transcript)
        assert result == []

    def test_single_snippet(self):
        """A single snippet becomes a single chunk."""
        transcript = [
            {'text': 'Hello world this is a test', 'start_time': 0.0, 'end_time': 5.0}
        ]
        result = chunk_transcript(transcript, chunk_size=100)
        assert len(result) == 1
        assert result[0]['chunk_id'] == 'chunk_0'
        assert result[0]['start_time'] == 0.0
        assert result[0]['end_time'] == 5.0

    def test_short_transcript_single_chunk(self):
        """A transcript shorter than chunk_size produces exactly one chunk."""
        # 5 snippets × 10 words = 50 words total, chunk_size=400
        transcript = _make_transcript(5, words_per_snippet=10)
        result = chunk_transcript(transcript, chunk_size=400, chunk_overlap=50)

        assert len(result) == 1
        assert result[0]['chunk_id'] == 'chunk_0'
        # All 50 words should be in the single chunk
        word_count = len(result[0]['chunk_text'].split())
        assert word_count == 50


# ════════════════════════════════════════════════════
#  TESTS: Chunk sizes
# ════════════════════════════════════════════════════

class TestChunkSizes:
    """Test that chunks are within the expected size range."""

    def test_chunks_near_target_size(self):
        """Each chunk (except possibly the last) should be >= chunk_size words."""
        # 100 snippets × 10 words = 1000 words, chunk_size=100
        transcript = _make_transcript(100, words_per_snippet=10)
        result = chunk_transcript(transcript, chunk_size=100, chunk_overlap=20)

        assert len(result) > 1  # Should produce multiple chunks

        # Check all chunks except the last are at least chunk_size words
        for chunk in result[:-1]:
            word_count = len(chunk['chunk_text'].split())
            assert word_count >= 100, (
                f"Chunk {chunk['chunk_id']} has {word_count} words, "
                f"expected >= 100"
            )

    def test_last_chunk_can_be_smaller(self):
        """The last chunk may be smaller than chunk_size (remaining words)."""
        # 45 snippets × 10 words = 450 words, chunk_size=400
        transcript = _make_transcript(45, words_per_snippet=10)
        result = chunk_transcript(transcript, chunk_size=400, chunk_overlap=0)

        assert len(result) == 2
        last_chunk_words = len(result[-1]['chunk_text'].split())
        assert last_chunk_words <= 400


# ════════════════════════════════════════════════════
#  TESTS: Overlap
# ════════════════════════════════════════════════════

class TestChunkOverlap:
    """Test that overlap between consecutive chunks works correctly."""

    def test_overlap_exists(self):
        """Consecutive chunks should share some words (overlap)."""
        # 80 snippets × 10 words = 800 words, chunk_size=200, overlap=50
        transcript = _make_transcript(80, words_per_snippet=10)
        result = chunk_transcript(transcript, chunk_size=200, chunk_overlap=50)

        assert len(result) >= 2

        # Check that consecutive chunks share some text
        for i in range(len(result) - 1):
            current_words = set(result[i]['chunk_text'].split())
            next_words = set(result[i + 1]['chunk_text'].split())
            shared = current_words & next_words
            assert len(shared) > 0, (
                f"Chunks {i} and {i+1} share no words — overlap is broken"
            )

    def test_no_overlap_when_zero(self):
        """When chunk_overlap=0, consecutive chunks should not overlap (much)."""
        transcript = _make_transcript(80, words_per_snippet=10)
        result = chunk_transcript(transcript, chunk_size=200, chunk_overlap=0)

        assert len(result) >= 2

        # With 0 overlap, chunks should be mostly disjoint
        # (some words might naturally repeat, but not large sections)
        for i in range(len(result) - 1):
            words_current = result[i]['chunk_text'].split()
            words_next = result[i + 1]['chunk_text'].split()
            # The last N words of current should NOT be the first N words of next
            assert words_current[-10:] != words_next[:10]


# ════════════════════════════════════════════════════
#  TESTS: Timestamps
# ════════════════════════════════════════════════════

class TestChunkTimestamps:
    """Test that chunk timestamps are preserved correctly."""

    def test_first_chunk_starts_at_transcript_start(self):
        """The first chunk's start_time should match the transcript's start."""
        transcript = _make_transcript(50, words_per_snippet=10, start=10.0)
        result = chunk_transcript(transcript, chunk_size=100)

        assert result[0]['start_time'] == 10.0

    def test_last_chunk_ends_at_transcript_end(self):
        """The last chunk's end_time should match the transcript's end."""
        transcript = _make_transcript(50, words_per_snippet=10, start=0.0, gap=5.0)
        result = chunk_transcript(transcript, chunk_size=100, chunk_overlap=0)

        expected_end = transcript[-1]['end_time']
        assert result[-1]['end_time'] == expected_end

    def test_chunks_are_chronological(self):
        """Chunk start_times should be in ascending order."""
        transcript = _make_transcript(100, words_per_snippet=10)
        result = chunk_transcript(transcript, chunk_size=200, chunk_overlap=30)

        for i in range(len(result) - 1):
            assert result[i]['start_time'] <= result[i + 1]['start_time'], (
                f"Chunk {i} starts at {result[i]['start_time']} but "
                f"chunk {i+1} starts at {result[i+1]['start_time']}"
            )

    def test_start_before_end(self):
        """Each chunk's start_time should be < its end_time."""
        transcript = _make_transcript(100, words_per_snippet=10)
        result = chunk_transcript(transcript, chunk_size=200, chunk_overlap=30)

        for chunk in result:
            assert chunk['start_time'] < chunk['end_time'], (
                f"{chunk['chunk_id']}: start={chunk['start_time']} >= "
                f"end={chunk['end_time']}"
            )


# ════════════════════════════════════════════════════
#  TESTS: Chunk IDs
# ════════════════════════════════════════════════════

class TestChunkIds:
    """Test that chunk IDs are sequential and correctly formatted."""

    def test_sequential_ids(self):
        """Chunk IDs should be chunk_0, chunk_1, chunk_2, etc."""
        transcript = _make_transcript(100, words_per_snippet=10)
        result = chunk_transcript(transcript, chunk_size=200)

        for i, chunk in enumerate(result):
            assert chunk['chunk_id'] == f'chunk_{i}'


# ════════════════════════════════════════════════════
#  TESTS: get_chunk_stats()
# ════════════════════════════════════════════════════

class TestGetChunkStats:
    """Test the chunk statistics helper."""

    def test_stats_empty(self):
        """Empty chunks list returns zeroed stats."""
        stats = get_chunk_stats([])
        assert stats['total_chunks'] == 0
        assert stats['avg_words'] == 0

    def test_stats_correct(self):
        """Stats should accurately reflect the chunk data."""
        transcript = _make_transcript(100, words_per_snippet=10)
        chunks = chunk_transcript(transcript, chunk_size=200, chunk_overlap=30)
        stats = get_chunk_stats(chunks)

        assert stats['total_chunks'] == len(chunks)
        assert stats['total_chunks'] > 0
        assert stats['min_words'] > 0
        assert stats['avg_words'] > 0
        assert stats['min_words'] <= stats['avg_words'] <= stats['max_words']
