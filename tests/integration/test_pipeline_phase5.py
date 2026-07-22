"""
test_pipeline_phase5.py — Integration test for the notes generation pipeline.

Tests the COMPLETE Phase 1→5 flow with REAL API calls:
  1. Fetch transcript from YouTube (Phase 1)
  2. Chunk it (Phase 2)
  3. Generate notes via LLM (Phase 5)

This test uses ONE real LLM call to confirm end-to-end works.
Keep this as a single test to minimize API usage.

NOTE: Requires internet access and at least one valid API key
      (GROQ_API_KEY or GEMINI_API_KEY) in the .env file.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.transcript import get_transcript
from modules.chunker import chunk_transcript
from modules.notes_generator import generate_notes


@pytest.fixture(scope="module")
def prepared_chunks():
    """
    Fetch and chunk a real video transcript.
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
    if not chunks:
        pytest.skip("Chunking produced no results")

    return {
        'video_id': video_id,
        'chunks': chunks,
    }


class TestFullNotesPipeline:
    """Integration test for the complete notes generation pipeline."""

    def test_real_llm_notes_generation(self, prepared_chunks):
        """
        Full end-to-end: generate notes from a real processed video
        with a real LLM call.

        This is the ONLY test that makes a real API call.
        """
        result = generate_notes(
            chunks=prepared_chunks['chunks'],
            video_id=prepared_chunks['video_id'],
        )

        # If both LLM providers fail (e.g. keys invalid), skip gracefully
        if not result['success'] and "both llm providers failed" in result.get('error', '').lower():
            pytest.skip(
                f"LLM API call failed (likely invalid key): {result['error']}"
            )

        assert result['success'] is True, f"Notes generation failed: {result.get('error')}"
        assert 'notes' in result
        assert len(result['notes']) > 50  # non-trivial notes
        assert 'provider' in result
        assert result['provider'] in ('groq', 'gemini')
        assert 'method' in result
        assert result['method'] in ('direct', 'map_reduce')
        assert 'chunks_processed' in result
        assert result['chunks_processed'] > 0

        # Print for manual verification
        print(f"\n  Provider: {result['provider']}")
        print(f"  Method: {result['method']}")
        print(f"  Chunks processed: {result['chunks_processed']}")
        print(f"  Notes length: {len(result['notes'])} chars")
        print(f"  Notes preview:\n{result['notes'][:500]}...")
