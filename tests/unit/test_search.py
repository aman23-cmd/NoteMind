"""
test_search.py — Unit tests for modules/search.py

All LLM API calls and vectorstore queries are MOCKED.
No real API keys or network calls are used.

Tests cover:
  - Successful search flow (mocked LLM + mocked chunks)
  - No chunks found for video_id
  - LLM failure / fallback behavior
  - Empty query / video_id validation
  - Prompt building
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.search import (
    search_video,
    _build_rag_prompt,
    _call_llm,
)


# ── Fake chunk data for mocking ──
FAKE_CHUNKS = [
    {
        'chunk_text': 'Python is a great programming language for AI and ML',
        'start_time': 10.0,
        'end_time': 40.0,
        'chunk_id': 'chunk_0',
        'distance': 0.15,
    },
    {
        'chunk_text': 'Machine learning uses neural networks and data',
        'start_time': 40.0,
        'end_time': 70.0,
        'chunk_id': 'chunk_1',
        'distance': 0.25,
    },
    {
        'chunk_text': 'Deep learning is a subset of machine learning',
        'start_time': 70.0,
        'end_time': 100.0,
        'chunk_id': 'chunk_2',
        'distance': 0.35,
    },
]


# ════════════════════════════════════════════════════
#  TESTS: _build_rag_prompt()
# ════════════════════════════════════════════════════

class TestBuildRagPrompt:
    """Test the RAG prompt builder."""

    def test_prompt_contains_query(self):
        """The prompt should contain the user's query."""
        prompt = _build_rag_prompt("What is Python?", FAKE_CHUNKS)
        assert "What is Python?" in prompt

    def test_prompt_contains_chunk_text(self):
        """The prompt should contain all chunk texts."""
        prompt = _build_rag_prompt("test query", FAKE_CHUNKS)
        assert "Python is a great programming language" in prompt
        assert "Machine learning uses neural networks" in prompt

    def test_prompt_contains_timestamps(self):
        """The prompt should include formatted timestamps."""
        prompt = _build_rag_prompt("test query", FAKE_CHUNKS)
        assert "00:00:10" in prompt  # start of chunk_0
        assert "00:00:40" in prompt  # end of chunk_0

    def test_prompt_has_instructions(self):
        """The prompt should include instructions for the LLM."""
        prompt = _build_rag_prompt("test query", FAKE_CHUNKS)
        assert "ONLY" in prompt  # instruction to use only provided context
        assert "timestamp" in prompt.lower()


# ════════════════════════════════════════════════════
#  TESTS: search_video() — Input validation
# ════════════════════════════════════════════════════

class TestSearchVideoValidation:
    """Test input validation for search_video."""

    def test_empty_query(self):
        """Empty query returns an error."""
        result = search_video(query="", video_id="test_vid")
        assert result['success'] is False
        assert "empty" in result['error'].lower()

    def test_whitespace_query(self):
        """Whitespace-only query returns an error."""
        result = search_video(query="   ", video_id="test_vid")
        assert result['success'] is False

    def test_empty_video_id(self):
        """Empty video_id returns an error."""
        result = search_video(query="test query", video_id="")
        assert result['success'] is False
        assert "empty" in result['error'].lower()


# ════════════════════════════════════════════════════
#  TESTS: search_video() — Successful flow
# ════════════════════════════════════════════════════

class TestSearchVideoSuccess:
    """Test the full search flow with mocked dependencies."""

    @patch('modules.search._call_llm')
    @patch('modules.search.query_chunks')
    @patch('modules.search.generate_embeddings')
    def test_successful_search(self, mock_embed, mock_query, mock_llm):
        """A successful search returns answer, timestamp, and source chunks."""
        # Mock the embedder
        mock_embed.return_value = [[0.1] * 384]

        # Mock the vector search
        mock_query.return_value = {
            'success': True,
            'video_id': 'test_vid',
            'results': FAKE_CHUNKS,
        }

        # Mock the LLM response
        mock_llm.return_value = {
            'success': True,
            'answer': 'Python is great for AI, as mentioned at 00:00:10.',
            'provider': 'groq',
        }

        result = search_video(
            query="What is Python used for?",
            video_id="test_vid",
        )

        assert result['success'] is True
        assert 'answer' in result
        assert result['provider'] == 'groq'
        assert result['timestamp'] == '00:00:10'
        assert len(result['source_chunks']) == 3

    @patch('modules.search._call_llm')
    @patch('modules.search.query_chunks')
    @patch('modules.search.generate_embeddings')
    def test_source_chunks_have_timestamps(self, mock_embed, mock_query, mock_llm):
        """Source chunks in the response should have formatted timestamps."""
        mock_embed.return_value = [[0.1] * 384]
        mock_query.return_value = {
            'success': True,
            'video_id': 'test_vid',
            'results': FAKE_CHUNKS,
        }
        mock_llm.return_value = {
            'success': True,
            'answer': 'Test answer.',
            'provider': 'groq',
        }

        result = search_video(query="test", video_id="test_vid")

        for chunk in result['source_chunks']:
            assert 'timestamp' in chunk
            assert 'start_time' in chunk
            assert 'end_time' in chunk
            assert 'chunk_text' in chunk


# ════════════════════════════════════════════════════
#  TESTS: search_video() — No chunks found
# ════════════════════════════════════════════════════

class TestSearchVideoNoChunks:
    """Test behavior when no chunks are found."""

    @patch('modules.search.query_chunks')
    @patch('modules.search.generate_embeddings')
    def test_no_chunks_returns_error(self, mock_embed, mock_query):
        """When no chunks are found, return a clear error."""
        mock_embed.return_value = [[0.1] * 384]
        mock_query.return_value = {
            'success': True,
            'video_id': 'test_vid',
            'results': [],  # empty — no chunks found
        }

        result = search_video(query="test", video_id="test_vid")

        assert result['success'] is False
        assert "no transcript chunks" in result['error'].lower()


# ════════════════════════════════════════════════════
#  TESTS: search_video() — LLM failure / fallback
# ════════════════════════════════════════════════════

class TestSearchVideoLlmFailure:
    """Test LLM failure and fallback behavior."""

    @patch('modules.search._call_llm')
    @patch('modules.search.query_chunks')
    @patch('modules.search.generate_embeddings')
    def test_llm_failure_returns_error(self, mock_embed, mock_query, mock_llm):
        """When the LLM fails, return the error message."""
        mock_embed.return_value = [[0.1] * 384]
        mock_query.return_value = {
            'success': True,
            'video_id': 'test_vid',
            'results': FAKE_CHUNKS,
        }
        mock_llm.return_value = {
            'success': False,
            'error': 'Both LLM providers failed.',
        }

        result = search_video(query="test", video_id="test_vid")

        assert result['success'] is False
        assert "failed" in result['error'].lower()


# ════════════════════════════════════════════════════
#  TESTS: _call_llm() — Fallback logic
# ════════════════════════════════════════════════════

class TestCallLlmFallback:
    """Test the Groq → Gemini fallback logic."""

    @patch('modules.search._call_groq')
    def test_groq_success(self, mock_groq):
        """When Groq succeeds, use its response."""
        mock_groq.return_value = "Groq's answer"

        result = _call_llm("test prompt")

        assert result['success'] is True
        assert result['provider'] == 'groq'
        assert result['answer'] == "Groq's answer"

    @patch('modules.search._call_gemini')
    @patch('modules.search._call_groq')
    def test_groq_fails_gemini_succeeds(self, mock_groq, mock_gemini):
        """When Groq fails, fall back to Gemini."""
        mock_groq.side_effect = RuntimeError("Groq rate limit hit")
        mock_gemini.return_value = "Gemini's answer"

        result = _call_llm("test prompt")

        assert result['success'] is True
        assert result['provider'] == 'gemini'
        assert result['answer'] == "Gemini's answer"

    @patch('modules.search._call_gemini')
    @patch('modules.search._call_groq')
    def test_both_fail(self, mock_groq, mock_gemini):
        """When both providers fail, return a clear error."""
        mock_groq.side_effect = ValueError("Groq key invalid")
        mock_gemini.side_effect = ValueError("Gemini key missing")

        result = _call_llm("test prompt")

        assert result['success'] is False
        assert "both" in result['error'].lower()
