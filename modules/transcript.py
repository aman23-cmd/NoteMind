"""
transcript.py — Transcript extraction logic for NoteMind.

Primary method:  youtube-transcript-api (fast, no download needed)
Fallback:        OpenAI Whisper (local only, not for deployment — see TODO below)

This module is responsible for:
  1. Extracting the YouTube video ID from various URL formats.
  2. Fetching the transcript with timestamps.
  3. Returning a clean list of {text, start_time, end_time} dicts.
"""

import re
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    InvalidVideoId,
    RequestBlocked,
    IpBlocked,
)


def extract_video_id(url):
    """
    Extract the YouTube video ID from a URL string.

    Supports these formats:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtube.com/watch?v=VIDEO_ID&other_params
      - https://youtu.be/VIDEO_ID
      - https://youtu.be/VIDEO_ID?t=120
      - https://www.youtube.com/embed/VIDEO_ID
      - https://www.youtube.com/v/VIDEO_ID

    Args:
        url (str): A YouTube video URL.

    Returns:
        str: The video ID (typically 11 characters).

    Raises:
        ValueError: If the URL is empty, not a valid YouTube URL,
                    or the video ID cannot be extracted.
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL cannot be empty.")

    url = url.strip()

    # ── Try parsing as a proper URL first ──
    try:
        parsed = urlparse(url)
    except Exception:
        raise ValueError(f"Invalid URL format: {url}")

    hostname = (parsed.hostname or '').lower().replace('www.', '')

    # Format: youtube.com/watch?v=VIDEO_ID
    if hostname in ('youtube.com', 'm.youtube.com'):
        if parsed.path == '/watch':
            params = parse_qs(parsed.query)
            video_id = params.get('v', [None])[0]
            if video_id:
                return _validate_video_id(video_id)

        # Format: youtube.com/embed/VIDEO_ID or youtube.com/v/VIDEO_ID
        match = re.match(r'^/(embed|v)/([a-zA-Z0-9_-]{11})', parsed.path)
        if match:
            return match.group(2)

    # Format: youtu.be/VIDEO_ID
    if hostname == 'youtu.be':
        video_id = parsed.path.lstrip('/')
        if video_id:
            return _validate_video_id(video_id.split('/')[0])

    raise ValueError(
        f"Could not extract video ID from URL: {url}. "
        "Please use a standard YouTube link like: "
        "https://www.youtube.com/watch?v=VIDEO_ID"
    )


def _validate_video_id(video_id):
    """
    Validate that a video ID looks correct (11 chars, alphanumeric + _ -).

    Args:
        video_id (str): The extracted video ID.

    Returns:
        str: The validated video ID.

    Raises:
        ValueError: If the video ID format is invalid.
    """
    video_id = video_id.strip()
    if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        raise ValueError(
            f"Invalid video ID format: '{video_id}'. "
            "A YouTube video ID should be exactly 11 characters."
        )
    return video_id


def fetch_transcript(video_id, languages=None):
    """
    Fetch the transcript for a YouTube video using youtube-transcript-api.

    This is the PRIMARY transcript extraction method.  It works for videos
    that have captions (manual or auto-generated).

    Args:
        video_id (str):  The YouTube video ID (11 characters).
        languages (list): Optional list of language codes in priority order.
                          Defaults to ['en'] (English).

    Returns:
        dict: {
            'success': True,
            'video_id': str,
            'transcript': [
                {'text': str, 'start_time': float, 'end_time': float},
                ...
            ]
        }
        OR on failure:
        dict: {
            'success': False,
            'video_id': str,
            'error': str   (human-readable error message)
        }
    """
    if languages is None:
        languages = ['en']

    try:
        from youtube_transcript_api.proxies import WebshareProxyConfig
        from config import Config

        proxy_config = None
        if Config.WEBSHARE_PROXY_USERNAME and Config.WEBSHARE_PROXY_PASSWORD:
            proxy_config = WebshareProxyConfig(
                proxy_username=Config.WEBSHARE_PROXY_USERNAME,
                proxy_password=Config.WEBSHARE_PROXY_PASSWORD,
            )

        ytt_api = YouTubeTranscriptApi(proxy_config=proxy_config)
        fetched = ytt_api.fetch(video_id, languages=languages)

        # Convert FetchedTranscriptSnippet objects into plain dicts
        transcript = []
        for snippet in fetched:
            start = snippet.start
            duration = snippet.duration
            transcript.append({
                'text': snippet.text,
                'start_time': round(start, 2),
                'end_time': round(start + duration, 2),
            })

        return {
            'success': True,
            'video_id': video_id,
            'transcript': transcript,
        }

    except TranscriptsDisabled:
        return {
            'success': False,
            'video_id': video_id,
            'error': "Transcripts are disabled for this video. "
                     "The video owner has turned off captions.",
        }

    except NoTranscriptFound:
        # Fallback: if the requested languages are not available,
        # fetch the first available transcript (any language)
        try:
            transcript_list = ytt_api.list(video_id)
            for t in transcript_list:
                fetched = t.fetch()
                transcript = []
                for snippet in fetched:
                    start = snippet.start
                    duration = snippet.duration
                    transcript.append({
                        'text': snippet.text,
                        'start_time': round(start, 2),
                        'end_time': round(start + duration, 2),
                    })
                return {
                    'success': True,
                    'video_id': video_id,
                    'transcript': transcript,
                }
        except Exception:
            pass

        return {
            'success': False,
            'video_id': video_id,
            'error': f"No transcript found in the requested languages: "
                     f"{languages}. Try a different language or check if "
                     "the video has captions.",
        }

    except VideoUnavailable:
        return {
            'success': False,
            'video_id': video_id,
            'error': "This video is unavailable. It may be private, "
                     "deleted, or region-restricted.",
        }

    except InvalidVideoId:
        return {
            'success': False,
            'video_id': video_id,
            'error': f"'{video_id}' is not a valid YouTube video ID.",
        }

    except (RequestBlocked, IpBlocked):
        return {
            'success': False,
            'video_id': video_id,
            'error': "YouTube is blocking our request (IP-based block). "
                     "This can happen on cloud servers. Try again later "
                     "or use a different network.",
        }

    except Exception as e:
        return {
            'success': False,
            'video_id': video_id,
            'error': f"Unexpected error fetching transcript: {str(e)}",
        }


def get_transcript(url, languages=None):
    """
    High-level function: takes a full YouTube URL, returns the transcript.

    This is the main function other modules should call.  It handles:
      1. Extracting the video ID from the URL.
      2. Fetching the transcript via youtube-transcript-api.
      3. (Future) Falling back to Whisper if no transcript is available.

    Args:
        url (str):        A YouTube video URL.
        languages (list): Optional language codes in priority order.

    Returns:
        dict: Same format as fetch_transcript() — see its docstring.
    """
    # Step 1: Extract video ID
    try:
        video_id = extract_video_id(url)
    except ValueError as e:
        return {
            'success': False,
            'video_id': None,
            'error': str(e),
        }

    # Step 2: Try the primary method (youtube-transcript-api)
    result = fetch_transcript(video_id, languages=languages)

    # Step 3: If primary method failed, try Whisper fallback
    if not result['success']:
        whisper_result = _whisper_fallback(video_id)
        if whisper_result is not None:
            return whisper_result
        # If Whisper fallback also not available, return the original error

    return result


def _whisper_fallback(video_id):
    """
    Placeholder for Whisper-based transcript extraction.

    TODO: Implement this for LOCAL development only.
          Whisper requires downloading the video audio (via yt-dlp)
          and running the model locally.  This is too heavy for free-tier
          deployment (Render/Railway) and should NEVER be part of the
          production code path.

    Steps to implement later:
      1. pip install yt-dlp openai-whisper
      2. Download audio:  yt-dlp -x --audio-format wav <video_url>
      3. Transcribe:      whisper.load_model("base").transcribe(audio_path)
      4. Convert Whisper segments into our {text, start_time, end_time} format.

    Args:
        video_id (str): The YouTube video ID.

    Returns:
        None — Whisper is not implemented yet.
    """
    # Not implemented — return None to signal "no fallback available"
    return None


def format_timestamp(seconds):
    """
    Convert seconds (float) into HH:MM:SS format for display.

    Args:
        seconds (float): Time in seconds (e.g. 754.32).

    Returns:
        str: Formatted timestamp like "00:12:34".
    """
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
