"""
test_notes_generator.py — Unit tests for modules/notes_generator.py

All LLM API calls are MOCKED. No real API keys or network calls are used.

Tests cover:
  - Input validation (empty chunks)
  - Direct generation (short transcripts ≤ 3000 words)
  - Map-reduce generation (long transcripts > 3000 words)
  - Prompt building (direct, map, reduce)
  - Chunk grouping logic
  - LLM failure handling
  - Timestamp formatting in prepared chunks
"""

import pytest
from unittest.mock import patch, MagicMock, call
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.notes_generator import (
    generate_notes,
    _build_direct_prompt,
    _build_map_prompt,
    _build_reduce_prompt,
    _prepare_chunks_for_notes,
    _group_chunks,
    _generate_direct,
    _generate_map_reduce,
    DIRECT_THRESHOLD,
    MAP_MAX_WORDS,
)


# ── Fake chunk data (short — fits in one shot) ──
FAKE_CHUNKS_SHORT = [
    {
        'chunk_id': 'chunk_0',
        'chunk_text': ' '.join(['word'] * 200),  # 200 words
        'start_time': 0.0,
        'end_time': 30.0,
    },
    {
        'chunk_id': 'chunk_1',
        'chunk_text': ' '.join(['word'] * 200),  # 200 words
        'start_time': 30.0,
        'end_time': 60.0,
    },
]  # Total: 400 words (well under DIRECT_THRESHOLD of 3000)


def _make_long_chunks(total_words=4000, words_per_chunk=400):
    """Helper: create a list of fake chunks exceeding DIRECT_THRESHOLD."""
    chunks = []
    num_chunks = total_words // words_per_chunk
    for i in range(num_chunks):
        chunks.append({
            'chunk_id': f'chunk_{i}',
            'chunk_text': ' '.join([f'word{i}'] * words_per_chunk),
            'start_time': float(i * 30),
            'end_time': float((i + 1) * 30),
        })
    return chunks


# ════════════════════════════════════════════════════
#  TESTS: _build_direct_prompt()
# ════════════════════════════════════════════════════

class TestBuildDirectPrompt:
    """Test the single-shot prompt builder."""

    def test_prompt_contains_transcript(self):
        """The prompt should contain the transcript text."""
        prompt = _build_direct_prompt("Hello world this is a test")
        assert "Hello world this is a test" in prompt

    def test_prompt_has_instructions(self):
        """The prompt should include formatting instructions."""
        prompt = _build_direct_prompt("Some transcript text")
        assert "TITLE" in prompt
        assert "SUMMARY" in prompt
        assert "KEY TAKEAWAYS" in prompt
        assert "markdown" in prompt.lower()

    def test_prompt_mentions_sections(self):
        """The prompt should ask for organized sections."""
        prompt = _build_direct_prompt("Some transcript text")
        assert "SECTIONS" in prompt
        assert "bullet" in prompt.lower()


# ════════════════════════════════════════════════════
#  TESTS: _build_map_prompt()
# ════════════════════════════════════════════════════

class TestBuildMapPrompt:
    """Test the MAP step prompt builder."""

    def test_prompt_contains_chunk_text(self):
        prompt = _build_map_prompt("ML is awesome", 0, 5)
        assert "ML is awesome" in prompt

    def test_prompt_includes_segment_info(self):
        prompt = _build_map_prompt("test text", 2, 5)
        assert "part 3" in prompt.lower()  # chunk_index=2 → part 3
        assert "5" in prompt

    def test_prompt_asks_for_bullet_points(self):
        prompt = _build_map_prompt("test text", 0, 1)
        assert "bullet" in prompt.lower()


# ════════════════════════════════════════════════════
#  TESTS: _build_reduce_prompt()
# ════════════════════════════════════════════════════

class TestBuildReducePrompt:
    """Test the REDUCE step prompt builder."""

    def test_prompt_contains_summaries(self):
        summaries = ["Summary A", "Summary B", "Summary C"]
        prompt = _build_reduce_prompt(summaries)
        assert "Summary A" in prompt
        assert "Summary B" in prompt
        assert "Summary C" in prompt

    def test_prompt_labels_segments(self):
        summaries = ["First", "Second"]
        prompt = _build_reduce_prompt(summaries)
        assert "Segment 1" in prompt
        assert "Segment 2" in prompt

    def test_prompt_asks_for_final_notes(self):
        prompt = _build_reduce_prompt(["test"])
        assert "TITLE" in prompt
        assert "KEY TAKEAWAYS" in prompt
        assert "markdown" in prompt.lower()


# ════════════════════════════════════════════════════
#  TESTS: _prepare_chunks_for_notes()
# ════════════════════════════════════════════════════

class TestPrepareChunks:
    """Test the chunk preparation function."""

    def test_adds_timestamps(self):
        chunks = [{
            'chunk_id': 'chunk_0',
            'chunk_text': 'Hello world',
            'start_time': 65.0,
            'end_time': 130.0,
        }]
        prepared = _prepare_chunks_for_notes(chunks)
        assert len(prepared) == 1
        assert "[00:01:05 - 00:02:10]" in prepared[0]['text']
        assert "Hello world" in prepared[0]['text']

    def test_word_count_correct(self):
        chunks = [{
            'chunk_id': 'chunk_0',
            'chunk_text': 'one two three four five',
            'start_time': 0.0,
            'end_time': 10.0,
        }]
        prepared = _prepare_chunks_for_notes(chunks)
        assert prepared[0]['word_count'] == 5


# ════════════════════════════════════════════════════
#  TESTS: _group_chunks()
# ════════════════════════════════════════════════════

class TestGroupChunks:
    """Test the chunk grouping logic for map-reduce."""

    def test_single_group_under_limit(self):
        chunks = [
            {'text': 'a', 'word_count': 100},
            {'text': 'b', 'word_count': 100},
        ]
        groups = _group_chunks(chunks, max_words=500)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_multiple_groups_over_limit(self):
        chunks = [
            {'text': 'a', 'word_count': 400},
            {'text': 'b', 'word_count': 400},
            {'text': 'c', 'word_count': 400},
        ]
        groups = _group_chunks(chunks, max_words=500)
        # Each chunk is 400 words; max is 500, so each chunk gets its own group
        assert len(groups) == 3

    def test_empty_input(self):
        groups = _group_chunks([], max_words=500)
        assert groups == []

    def test_exact_boundary(self):
        chunks = [
            {'text': 'a', 'word_count': 250},
            {'text': 'b', 'word_count': 250},
            {'text': 'c', 'word_count': 250},
        ]
        groups = _group_chunks(chunks, max_words=500)
        # 250+250=500 (exactly at limit, so they should be split because > check)
        # First two go into one group (250+250=500 ≤ 500), third starts a new one
        assert len(groups) == 2


# ════════════════════════════════════════════════════
#  TESTS: generate_notes() — Input validation
# ════════════════════════════════════════════════════

class TestGenerateNotesValidation:
    """Test input validation for generate_notes."""

    def test_empty_chunks(self):
        """Empty chunk list returns an error."""
        result = generate_notes(chunks=[])
        assert result['success'] is False
        assert "no transcript" in result['error'].lower()

    def test_none_chunks(self):
        """None chunk list returns an error."""
        result = generate_notes(chunks=None)
        assert result['success'] is False


# ════════════════════════════════════════════════════
#  TESTS: generate_notes() — Direct (short) strategy
# ════════════════════════════════════════════════════

class TestGenerateNotesDirect:
    """Test the single-shot notes generation for short transcripts."""

    @patch('modules.notes_generator._call_llm')
    def test_short_transcript_uses_direct(self, mock_llm):
        """Short transcripts (≤ 3000 words) should use the direct method."""
        mock_llm.return_value = {
            'success': True,
            'text': '# Test Notes\n- Point 1\n- Point 2',
            'provider': 'groq',
        }

        result = generate_notes(chunks=FAKE_CHUNKS_SHORT)

        assert result['success'] is True
        assert result['method'] == 'direct'
        assert result['provider'] == 'groq'
        assert '# Test Notes' in result['notes']
        assert result['chunks_processed'] == 2
        # Should be called exactly once (single shot)
        mock_llm.assert_called_once()

    @patch('modules.notes_generator._call_llm')
    def test_direct_llm_failure(self, mock_llm):
        """LLM failure during direct generation returns error."""
        mock_llm.return_value = {
            'success': False,
            'error': 'Both LLM providers failed.',
        }

        result = generate_notes(chunks=FAKE_CHUNKS_SHORT)

        assert result['success'] is False
        assert 'failed' in result['error'].lower()


# ════════════════════════════════════════════════════
#  TESTS: generate_notes() — Map-reduce (long) strategy
# ════════════════════════════════════════════════════

class TestGenerateNotesMapReduce:
    """Test the map-reduce strategy for long transcripts."""

    @patch('modules.notes_generator._call_llm')
    def test_long_transcript_uses_map_reduce(self, mock_llm):
        """Long transcripts (> 3000 words) should use map-reduce."""
        mock_llm.return_value = {
            'success': True,
            'text': '# Final Notes\n- Key point',
            'provider': 'gemini',
        }

        long_chunks = _make_long_chunks(total_words=4000, words_per_chunk=400)
        result = generate_notes(chunks=long_chunks)

        assert result['success'] is True
        assert result['method'] == 'map_reduce'
        assert result['provider'] == 'gemini'
        assert result['chunks_processed'] == 10
        # Should be called multiple times (map steps + 1 reduce step)
        assert mock_llm.call_count > 1

    @patch('modules.notes_generator._call_llm')
    def test_map_step_failure(self, mock_llm):
        """If a MAP step fails, the whole process returns an error."""
        mock_llm.return_value = {
            'success': False,
            'error': 'LLM call failed.',
        }

        long_chunks = _make_long_chunks(total_words=4000)
        result = generate_notes(chunks=long_chunks)

        assert result['success'] is False
        assert 'map step failed' in result['error'].lower()

    @patch('modules.notes_generator._call_llm')
    def test_reduce_step_failure(self, mock_llm):
        """If the REDUCE step fails, the process returns an error."""
        # MAP steps succeed, REDUCE fails
        long_chunks = _make_long_chunks(total_words=4000, words_per_chunk=400)
        num_chunks = len(long_chunks)

        # Calculate expected map calls
        prepared = _prepare_chunks_for_notes(long_chunks)
        groups = _group_chunks(prepared, MAP_MAX_WORDS)
        num_map_calls = len(groups)

        # First N calls succeed (map), last call fails (reduce)
        side_effects = [
            {'success': True, 'text': f'Summary {i}', 'provider': 'groq'}
            for i in range(num_map_calls)
        ]
        side_effects.append({'success': False, 'error': 'Reduce LLM failed.'})
        mock_llm.side_effect = side_effects

        result = generate_notes(chunks=long_chunks)

        assert result['success'] is False
        assert 'reduce step failed' in result['error'].lower()


# ════════════════════════════════════════════════════
#  TESTS: _call_llm() — LLM fallback (via notes_generator)
# ════════════════════════════════════════════════════

class TestNotesCallLlm:
    """Test the LLM fallback logic within notes_generator."""

    @patch('modules.search._call_groq')
    def test_groq_success(self, mock_groq):
        """When Groq succeeds, use its response."""
        mock_groq.return_value = "Generated notes here"
        from modules.notes_generator import _call_llm
        result = _call_llm("test prompt")
        assert result['success'] is True
        assert result['provider'] == 'groq'

    @patch('modules.search._call_gemini')
    @patch('modules.search._call_groq')
    def test_groq_fails_gemini_succeeds(self, mock_groq, mock_gemini):
        """When Groq fails, fall back to Gemini."""
        mock_groq.side_effect = RuntimeError("Groq failed")
        mock_gemini.return_value = "Gemini notes"
        from modules.notes_generator import _call_llm
        result = _call_llm("test prompt")
        assert result['success'] is True
        assert result['provider'] == 'gemini'

    @patch('modules.search._call_gemini')
    @patch('modules.search._call_groq')
    def test_both_fail(self, mock_groq, mock_gemini):
        """When both providers fail, return error."""
        mock_groq.side_effect = ValueError("Key invalid")
        mock_gemini.side_effect = ValueError("Key missing")
        from modules.notes_generator import _call_llm
        result = _call_llm("test prompt")
        assert result['success'] is False
        assert 'both' in result['error'].lower()
