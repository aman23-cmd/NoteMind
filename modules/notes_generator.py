"""
notes_generator.py — Smart notes generation using an LLM.

Takes the full transcript (or chunked summaries for long videos),
sends it to Groq/Gemini, and returns structured notes
(headings, bullet points, key takeaways).

For long transcripts that exceed the LLM context window, uses a
map-reduce strategy:
  1. MAP:    Summarize each chunk individually.
  2. REDUCE: Combine all chunk summaries into final structured notes.

Flow:
  1. Accept transcript chunks (from Phase 2 chunker).
  2. If total words ≤ DIRECT_THRESHOLD, send everything in one shot.
  3. If total words > DIRECT_THRESHOLD, use map-reduce.
  4. Return structured notes as a formatted string.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import Config

from modules.transcript import format_timestamp


# ── Constants ──
DIRECT_THRESHOLD = 3000   # Max words for a single-shot LLM call
MAP_MAX_WORDS = 1500      # Max words per map-step chunk group


def _build_direct_prompt(full_text, video_id=None):
    """
    Build a prompt for single-shot notes generation (short transcripts).

    Args:
        full_text (str): The entire transcript text.
        video_id (str):  Optional video ID for context.

    Returns:
        str: The LLM prompt.
    """
    prompt = f"""You are an expert note-taker. Generate comprehensive, well-structured study notes from the following video transcript.

TRANSCRIPT:
{full_text}

INSTRUCTIONS:
1. Create a clear TITLE for the video content.
2. Write a brief SUMMARY (2-3 sentences).
3. Organize the content into logical SECTIONS with descriptive headings.
4. Under each section, use bullet points for key points.
5. Include a KEY TAKEAWAYS section at the end with 3-5 main points.
6. If timestamps are mentioned in the context, reference them (e.g., "At 00:02:15...").
7. Use markdown formatting (# for headings, - for bullets, **bold** for emphasis).
8. Be thorough but concise — capture all important information without unnecessary repetition.

GENERATE THE NOTES NOW:"""

    return prompt


def _build_map_prompt(chunk_text, chunk_index, total_chunks):
    """
    Build a prompt for the MAP step — summarize a single chunk.

    Args:
        chunk_text (str):    Text of this transcript chunk.
        chunk_index (int):   Index of this chunk (0-based).
        total_chunks (int):  Total number of chunks being processed.

    Returns:
        str: The LLM prompt for summarizing this chunk.
    """
    prompt = f"""You are summarizing part {chunk_index + 1} of {total_chunks} of a video transcript.

TRANSCRIPT SEGMENT:
{chunk_text}

INSTRUCTIONS:
1. Summarize the key points from this segment in 3-6 bullet points.
2. Preserve any important details, definitions, or examples.
3. Note the main topic discussed in this segment.
4. Be concise but thorough — don't miss important information.

SUMMARY OF SEGMENT {chunk_index + 1}:"""

    return prompt


def _build_reduce_prompt(summaries):
    """
    Build a prompt for the REDUCE step — combine chunk summaries into
    final structured notes.

    Args:
        summaries (list): List of summary strings from the MAP step.

    Returns:
        str: The LLM prompt for combining summaries.
    """
    combined = "\n\n".join(
        f"--- Segment {i + 1} Summary ---\n{s}"
        for i, s in enumerate(summaries)
    )

    prompt = f"""You are an expert note-taker. Below are summaries of different segments of a video transcript. Combine them into comprehensive, well-structured study notes.

SEGMENT SUMMARIES:
{combined}

INSTRUCTIONS:
1. Create a clear TITLE for the video content.
2. Write a brief SUMMARY (2-3 sentences) covering the entire video.
3. Organize the content into logical SECTIONS with descriptive headings.
4. Under each section, use bullet points for key points.
5. Include a KEY TAKEAWAYS section at the end with 3-5 main points.
6. Use markdown formatting (# for headings, - for bullets, **bold** for emphasis).
7. Merge related points from different segments — avoid redundancy.
8. Ensure the notes flow logically from beginning to end.

GENERATE THE FINAL NOTES NOW:"""

    return prompt


def _call_llm(prompt):
    """
    Call the LLM with Groq → Gemini fallback (reuses search.py pattern).

    Args:
        prompt (str): The prompt to send.

    Returns:
        dict: {'success': True, 'text': str, 'provider': str}
              OR {'success': False, 'error': str}
    """
    from modules.search import _call_groq, _call_gemini

    errors = []

    # ── Try Groq first ──
    try:
        answer = _call_groq(prompt)
        return {'success': True, 'text': answer, 'provider': 'groq'}
    except ValueError as e:
        errors.append(f"Groq: {e}")
    except RuntimeError as e:
        errors.append(f"Groq: {e}")
    except Exception as e:
        errors.append(f"Groq: Unexpected error — {e}")

    # ── Try Gemini as fallback ──
    try:
        answer = _call_gemini(prompt)
        return {'success': True, 'text': answer, 'provider': 'gemini'}
    except ValueError as e:
        errors.append(f"Gemini: {e}")
    except Exception as e:
        errors.append(f"Gemini: {e}")

    # ── Both failed ──
    return {
        'success': False,
        'error': (
            "Both LLM providers failed. Errors:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n\nPlease check your API keys in the .env file."
        ),
    }


def _prepare_chunks_for_notes(chunks):
    """
    Convert chunk dicts into text strings with timestamps for the LLM.

    Args:
        chunks (list): List of chunk dicts from chunker.py.

    Returns:
        list: List of dicts with 'text' (str with timestamps) and
              'word_count' (int).
    """
    prepared = []
    for chunk in chunks:
        ts_start = format_timestamp(chunk['start_time'])
        ts_end = format_timestamp(chunk['end_time'])
        text = f"[{ts_start} - {ts_end}] {chunk['chunk_text']}"
        prepared.append({
            'text': text,
            'word_count': len(chunk['chunk_text'].split()),
        })
    return prepared


def generate_notes(chunks, video_id=None):
    """
    Generate structured study notes from transcript chunks.

    This is the main function that app.py and other modules should call.

    For short transcripts (≤ DIRECT_THRESHOLD words), generates notes in
    a single LLM call.  For longer ones, uses a map-reduce approach:
      1. MAP:    Summarize each group of chunks individually.
      2. REDUCE: Combine all summaries into final structured notes.

    Args:
        chunks (list):   List of chunk dicts from chunker.py, each with:
                         {'chunk_id', 'chunk_text', 'start_time', 'end_time'}
        video_id (str):  Optional YouTube video ID (for context).

    Returns:
        dict: {
            'success': True,
            'notes': str,          (markdown-formatted study notes)
            'provider': str,       (which LLM was used: 'groq' or 'gemini')
            'method': str,         ('direct' or 'map_reduce')
            'chunks_processed': int,
        }
        OR on failure:
        dict: {
            'success': False,
            'error': str,
        }
    """
    # ── Validate input ──
    if not chunks:
        return {
            'success': False,
            'error': "No transcript chunks provided. Process the video first.",
        }

    # ── Prepare chunk texts with timestamps ──
    prepared = _prepare_chunks_for_notes(chunks)
    total_words = sum(p['word_count'] for p in prepared)
    total_text = "\n\n".join(p['text'] for p in prepared)

    # ── Decide strategy ──
    if total_words <= DIRECT_THRESHOLD:
        return _generate_direct(total_text, video_id, len(chunks))
    else:
        return _generate_map_reduce(prepared, video_id, len(chunks))


def _generate_direct(full_text, video_id, chunk_count):
    """
    Generate notes in a single LLM call (for short transcripts).

    Args:
        full_text (str):    The combined transcript text.
        video_id (str):     Optional video ID.
        chunk_count (int):  Number of chunks processed.

    Returns:
        dict: Result dict with notes or error.
    """
    prompt = _build_direct_prompt(full_text, video_id)
    result = _call_llm(prompt)

    if not result['success']:
        return result

    return {
        'success': True,
        'notes': result['text'],
        'provider': result['provider'],
        'method': 'direct',
        'chunks_processed': chunk_count,
    }


def _generate_map_reduce(prepared_chunks, video_id, chunk_count):
    """
    Generate notes using a map-reduce strategy (for long transcripts).

    MAP:    Groups chunks into batches, summarizes each batch.
    REDUCE: Combines all summaries into final structured notes.

    Args:
        prepared_chunks (list): List of dicts with 'text' and 'word_count'.
        video_id (str):         Optional video ID.
        chunk_count (int):      Total number of chunks.

    Returns:
        dict: Result dict with notes or error.
    """
    # ── MAP phase: group chunks and summarize each group ──
    groups = _group_chunks(prepared_chunks, MAP_MAX_WORDS)
    summaries = []
    map_provider = None

    for i, group in enumerate(groups):
        group_text = "\n\n".join(c['text'] for c in group)
        prompt = _build_map_prompt(group_text, i, len(groups))

        result = _call_llm(prompt)
        if not result['success']:
            return {
                'success': False,
                'error': f"Map step failed on group {i + 1}/{len(groups)}: {result['error']}",
            }

        summaries.append(result['text'])
        if map_provider is None:
            map_provider = result['provider']

    # ── REDUCE phase: combine summaries into final notes ──
    reduce_prompt = _build_reduce_prompt(summaries)
    reduce_result = _call_llm(reduce_prompt)

    if not reduce_result['success']:
        return {
            'success': False,
            'error': f"Reduce step failed: {reduce_result['error']}",
        }

    return {
        'success': True,
        'notes': reduce_result['text'],
        'provider': reduce_result['provider'],
        'method': 'map_reduce',
        'chunks_processed': chunk_count,
    }


def _group_chunks(prepared_chunks, max_words):
    """
    Group prepared chunks into batches that stay under max_words each.

    Args:
        prepared_chunks (list): List of dicts with 'text' and 'word_count'.
        max_words (int):        Max words per group.

    Returns:
        list: List of groups, where each group is a list of chunk dicts.
    """
    groups = []
    current_group = []
    current_words = 0

    for chunk in prepared_chunks:
        if current_words + chunk['word_count'] > max_words and current_group:
            groups.append(current_group)
            current_group = []
            current_words = 0

        current_group.append(chunk)
        current_words += chunk['word_count']

    if current_group:
        groups.append(current_group)

    return groups
