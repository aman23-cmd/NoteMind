"""
test_transcript.py — Unit tests for modules/transcript.py

Tests cover:
  - Video ID extraction from various YouTube URL formats
  - Invalid URL handling
  - Successful transcript fetch (mocked — no real API calls)
  - Error cases: transcripts disabled, no transcript found, video unavailable
  - The high-level get_transcript() function
  - Timestamp formatting helper
"""

import pytest
from unittest.mock import patch, MagicMock

# We need to add the project root to the path so imports work
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.transcript import (
    extract_video_id,
    fetch_transcript,
    get_transcript,
    format_timestamp,
    _validate_video_id,
)
from youtube_transcript_api import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    InvalidVideoId,
)


# ════════════════════════════════════════════════════
#  TESTS: extract_video_id() — URL Parsing
# ════════════════════════════════════════════════════

class TestExtractVideoId:
    """Test video ID extraction from various YouTube URL formats."""

    def test_standard_url(self):
        """Standard youtube.com/watch?v= format."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_standard_url_without_www(self):
        """youtube.com without www prefix."""
        url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self):
        """URL with additional query parameters (list, index, t, etc.)."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLtest&index=5"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_url_with_timestamp(self):
        """URL with a timestamp parameter."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_short_url(self):
        """youtu.be short URL format."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_short_url_with_timestamp(self):
        """youtu.be short URL with a timestamp query param."""
        url = "https://youtu.be/dQw4w9WgXcQ?t=45"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_embed_url(self):
        """youtube.com/embed/ format."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_v_url(self):
        """youtube.com/v/ format (older embed style)."""
        url = "https://www.youtube.com/v/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_mobile_url(self):
        """Mobile YouTube URL (m.youtube.com)."""
        url = "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_http_url(self):
        """HTTP (not HTTPS) URL."""
        url = "http://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_video_id_with_hyphens_and_underscores(self):
        """Video IDs can contain hyphens and underscores."""
        url = "https://www.youtube.com/watch?v=a-B_c1D2e3f"
        assert extract_video_id(url) == "a-B_c1D2e3f"


class TestExtractVideoIdErrors:
    """Test that invalid URLs raise ValueError with clear messages."""

    def test_empty_string(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            extract_video_id("")

    def test_none_input(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            extract_video_id(None)

    def test_random_string(self):
        with pytest.raises(ValueError, match="Could not extract video ID"):
            extract_video_id("this is not a url")

    def test_non_youtube_url(self):
        with pytest.raises(ValueError, match="Could not extract video ID"):
            extract_video_id("https://www.google.com/search?q=test")

    def test_youtube_url_without_video_id(self):
        with pytest.raises(ValueError, match="Could not extract video ID"):
            extract_video_id("https://www.youtube.com/watch")

    def test_youtube_homepage(self):
        with pytest.raises(ValueError, match="Could not extract video ID"):
            extract_video_id("https://www.youtube.com")


# ════════════════════════════════════════════════════
#  TESTS: fetch_transcript() — Mocked API calls
# ════════════════════════════════════════════════════

def _make_mock_snippet(text, start, duration):
    """Helper: create a mock FetchedTranscriptSnippet."""
    snippet = MagicMock()
    snippet.text = text
    snippet.start = start
    snippet.duration = duration
    return snippet


class TestFetchTranscript:
    """Test transcript fetching with mocked youtube-transcript-api."""

    @patch('modules.transcript.YouTubeTranscriptApi')
    def test_successful_fetch(self, MockApi):
        """Successful transcript fetch returns structured data."""
        # Set up mock: YouTubeTranscriptApi() returns an instance,
        # and calling .fetch() on that instance returns our fake data.
        mock_instance = MagicMock()
        MockApi.return_value = mock_instance

        mock_snippets = [
            _make_mock_snippet("Hello everyone", 0.0, 2.5),
            _make_mock_snippet("welcome to the video", 2.5, 3.0),
            _make_mock_snippet("today we learn Python", 5.5, 4.0),
        ]
        # Make the mock iterable (FetchedTranscript acts like a list)
        mock_fetched = MagicMock()
        mock_fetched.__iter__ = MagicMock(return_value=iter(mock_snippets))
        mock_instance.fetch.return_value = mock_fetched

        result = fetch_transcript("dQw4w9WgXcQ")

        assert result['success'] is True
        assert result['video_id'] == "dQw4w9WgXcQ"
        assert len(result['transcript']) == 3

        # Check first snippet
        first = result['transcript'][0]
        assert first['text'] == "Hello everyone"
        assert first['start_time'] == 0.0
        assert first['end_time'] == 2.5

        # Check last snippet
        last = result['transcript'][2]
        assert last['text'] == "today we learn Python"
        assert last['start_time'] == 5.5
        assert last['end_time'] == 9.5

    @patch('modules.transcript.YouTubeTranscriptApi')
    def test_transcripts_disabled(self, MockApi):
        """Returns a clear error when transcripts are disabled."""
        mock_instance = MagicMock()
        MockApi.return_value = mock_instance
        mock_instance.fetch.side_effect = TranscriptsDisabled(
            "dQw4w9WgXcQ"
        )

        result = fetch_transcript("dQw4w9WgXcQ")

        assert result['success'] is False
        assert "disabled" in result['error'].lower()

    @patch('modules.transcript.YouTubeTranscriptApi')
    def test_no_transcript_found(self, MockApi):
        """Returns a clear error when no transcript is available."""
        mock_instance = MagicMock()
        MockApi.return_value = mock_instance
        mock_instance.fetch.side_effect = NoTranscriptFound(
            "dQw4w9WgXcQ", ['en'], MagicMock()
        )

        result = fetch_transcript("dQw4w9WgXcQ")

        assert result['success'] is False
        assert "no transcript found" in result['error'].lower()

    @patch('modules.transcript.YouTubeTranscriptApi')
    def test_video_unavailable(self, MockApi):
        """Returns a clear error when the video is unavailable."""
        mock_instance = MagicMock()
        MockApi.return_value = mock_instance
        mock_instance.fetch.side_effect = VideoUnavailable(
            "dQw4w9WgXcQ"
        )

        result = fetch_transcript("dQw4w9WgXcQ")

        assert result['success'] is False
        assert "unavailable" in result['error'].lower()

    @patch('modules.transcript.YouTubeTranscriptApi')
    def test_invalid_video_id(self, MockApi):
        """Returns a clear error for an invalid video ID."""
        mock_instance = MagicMock()
        MockApi.return_value = mock_instance
        mock_instance.fetch.side_effect = InvalidVideoId(
            "bad_id"
        )

        result = fetch_transcript("bad_id")

        assert result['success'] is False
        assert "not a valid" in result['error'].lower()

    @patch('modules.transcript.YouTubeTranscriptApi')
    def test_unexpected_exception(self, MockApi):
        """Unexpected errors are caught and returned gracefully."""
        mock_instance = MagicMock()
        MockApi.return_value = mock_instance
        mock_instance.fetch.side_effect = RuntimeError("something broke")

        result = fetch_transcript("dQw4w9WgXcQ")

        assert result['success'] is False
        assert "unexpected" in result['error'].lower()


# ════════════════════════════════════════════════════
#  TESTS: get_transcript() — High-level integration
# ════════════════════════════════════════════════════

class TestGetTranscript:
    """Test the high-level get_transcript() function."""

    def test_invalid_url_returns_error(self):
        """An invalid URL should return an error dict (not crash)."""
        result = get_transcript("not-a-url")

        assert result['success'] is False
        assert result['video_id'] is None
        assert "could not extract" in result['error'].lower()

    @patch('modules.transcript.fetch_transcript')
    def test_valid_url_calls_fetch(self, mock_fetch):
        """A valid URL should extract the ID and call fetch_transcript."""
        mock_fetch.return_value = {
            'success': True,
            'video_id': 'dQw4w9WgXcQ',
            'transcript': [],
        }

        result = get_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert result['success'] is True
        mock_fetch.assert_called_once_with('dQw4w9WgXcQ', languages=None)

    @patch('modules.transcript.fetch_transcript')
    def test_custom_languages_passed_through(self, mock_fetch):
        """Custom language preference should be forwarded."""
        mock_fetch.return_value = {
            'success': True,
            'video_id': 'dQw4w9WgXcQ',
            'transcript': [],
        }

        get_transcript(
            "https://youtu.be/dQw4w9WgXcQ",
            languages=['hi', 'en']
        )

        mock_fetch.assert_called_once_with('dQw4w9WgXcQ', languages=['hi', 'en'])


# ════════════════════════════════════════════════════
#  TESTS: format_timestamp() — Helper
# ════════════════════════════════════════════════════

class TestFormatTimestamp:
    """Test the timestamp formatting utility."""

    def test_zero(self):
        assert format_timestamp(0) == "00:00:00"

    def test_seconds_only(self):
        assert format_timestamp(45) == "00:00:45"

    def test_minutes_and_seconds(self):
        assert format_timestamp(125) == "00:02:05"

    def test_hours(self):
        assert format_timestamp(3661) == "01:01:01"

    def test_float_input(self):
        """Float seconds should be truncated (not rounded)."""
        assert format_timestamp(59.9) == "00:00:59"
