"""
chunker.py — Splits transcript text into smaller, timestamped chunks.

Takes the transcript output from Phase 1 (list of {text, start_time, end_time})
and groups consecutive snippets into larger chunks of ~300-500 words, with ~50
word overlap between consecutive chunks so context is not lost.

Each chunk preserves the correct start_time (from its first snippet)
and end_time (from its last snippet).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import Config


def chunk_transcript(transcript, chunk_size=None, chunk_overlap=None):
    """
    Split a transcript into overlapping chunks of approximately `chunk_size` words.

    The function walks through transcript snippets sequentially, accumulating
    them until the word count reaches the target.  When a chunk is finalized,
    the next chunk starts by "rewinding" a few snippets to create overlap.

    Args:
        transcript (list): List of dicts from transcript.py, each with keys:
                           {'text': str, 'start_time': float, 'end_time': float}
        chunk_size (int):  Target number of words per chunk.
                           Defaults to Config.CHUNK_SIZE (400).
        chunk_overlap (int): Number of overlap words between consecutive chunks.
                             Defaults to Config.CHUNK_OVERLAP (50).

    Returns:
        list: List of chunk dicts:
              [
                  {
                      'chunk_id': 'chunk_0',
                      'chunk_text': str,
                      'start_time': float,
                      'end_time': float,
                  },
                  ...
              ]
              Returns an empty list if the transcript is empty.
    """
    if chunk_size is None:
        chunk_size = Config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = Config.CHUNK_OVERLAP

    if not transcript:
        return []

    # ── Step 1: Filter out empty snippets ──
    snippets = [
        s for s in transcript
        if s.get('text', '').strip()
    ]

    if not snippets:
        return []

    # ── Step 2: Walk through snippets, building chunks ──
    chunks = []
    chunk_index = 0
    i = 0  # current position in the snippets list

    while i < len(snippets):
        # Accumulate snippets for the current chunk
        current_texts = []
        current_word_count = 0
        chunk_start_idx = i

        while i < len(snippets) and current_word_count < chunk_size:
            text = snippets[i]['text'].strip()
            current_texts.append(text)
            current_word_count += len(text.split())
            i += 1

        # Build the chunk
        chunk_text = ' '.join(current_texts)
        start_time = snippets[chunk_start_idx]['start_time']
        end_time = snippets[i - 1]['end_time']

        chunks.append({
            'chunk_id': f'chunk_{chunk_index}',
            'chunk_text': chunk_text,
            'start_time': start_time,
            'end_time': end_time,
        })
        chunk_index += 1

        # ── Step 3: Rewind for overlap ──
        # Walk backwards from the end of the current chunk to find
        # the snippet position where ~chunk_overlap words remain.
        if i < len(snippets):
            overlap_words = 0
            rewind_to = i
            for j in range(i - 1, chunk_start_idx - 1, -1):
                words_in_snippet = len(snippets[j]['text'].split())
                if overlap_words + words_in_snippet > chunk_overlap:
                    break
                overlap_words += words_in_snippet
                rewind_to = j

            # Only rewind if we actually found overlap snippets
            if rewind_to < i:
                i = rewind_to

    return chunks


def get_chunk_stats(chunks):
    """
    Return statistics about the generated chunks (useful for debugging).

    Args:
        chunks (list): List of chunk dicts from chunk_transcript().

    Returns:
        dict: {
            'total_chunks': int,
            'word_counts': list of int (word count per chunk),
            'avg_words': float,
            'min_words': int,
            'max_words': int,
            'total_duration_seconds': float,
        }
    """
    if not chunks:
        return {
            'total_chunks': 0,
            'word_counts': [],
            'avg_words': 0,
            'min_words': 0,
            'max_words': 0,
            'total_duration_seconds': 0,
        }

    word_counts = [len(c['chunk_text'].split()) for c in chunks]

    return {
        'total_chunks': len(chunks),
        'word_counts': word_counts,
        'avg_words': round(sum(word_counts) / len(word_counts), 1),
        'min_words': min(word_counts),
        'max_words': max(word_counts),
        'total_duration_seconds': round(
            chunks[-1]['end_time'] - chunks[0]['start_time'], 2
        ),
    }
