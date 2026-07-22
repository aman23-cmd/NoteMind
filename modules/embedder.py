"""
embedder.py — Generates vector embeddings from text using Google's Gemini
embedding API (models/text-embedding-004).

Why an API-based model instead of a local one:
  - Loading sentence-transformers + torch locally requires 400MB+ of RAM,
    which exceeds free-tier hosting limits (e.g. Render's 512MB cap).
  - Using Gemini's free embedding API removes that memory burden entirely —
    no large ML library needs to be loaded into the process.

Model used: "models/text-embedding-004"
Embedding dimension: 768

The Gemini client is created lazily (on first use) and cached for the
lifetime of the process, matching the previous lazy-singleton pattern.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import Config

# ── Module-level cache for the client and dimension ──
_client = None
_EMBEDDING_DIMENSION = 768  # gemini-embedding-001, requested at 768 dims


def _get_client():
    """
    Create (or return cached) Gemini API client (lazy singleton).

    Returns:
        genai.Client: The Gemini API client.
    """
    global _client
    if _client is None:
        from google import genai
        api_key = getattr(Config, "GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Embeddings require a valid Gemini API key."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _embed_single(text, max_retries=3):
    """Call the Gemini embedding API for one piece of text, with retries."""
    from google.genai import types

    client = _get_client()
    last_error = None
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=768),
            )
            return list(result.embeddings[0].values)
        except Exception as e:
            last_error = e
            time.sleep(1.5 * (attempt + 1))  # simple backoff
    raise RuntimeError(
        f"Failed to generate embedding after {max_retries} attempts: {last_error}"
    )


def generate_embeddings(texts):
    """
    Generate embeddings for one or more text strings.

    Args:
        texts (str or list of str): A single text or a list of texts to embed.

    Returns:
        list: A list of embedding vectors (each is a list of floats).
              For text-embedding-004, each vector has 768 dimensions.

              If a single string is passed, still returns a list with one embedding.
              Empty or whitespace-only strings get a zero vector.
    """
    # Normalize input to a list
    single_input = isinstance(texts, str)
    if single_input:
        texts = [texts]

    if not texts:
        return []

    # Handle empty/whitespace strings: replace with a placeholder so the
    # API doesn't choke, then we'll zero out the result.
    empty_mask = []
    cleaned_texts = []
    for t in texts:
        if not t or not t.strip():
            empty_mask.append(True)
            cleaned_texts.append("empty")  # placeholder — will be zeroed
        else:
            empty_mask.append(False)
            cleaned_texts.append(t)

    # Generate embeddings one at a time (API call per text)
    result = []
    for i, text in enumerate(cleaned_texts):
        if empty_mask[i]:
            result.append([0.0] * _EMBEDDING_DIMENSION)
        else:
            result.append(_embed_single(text))

    return result


def get_embedding_dimension():
    """
    Return the dimension of the embedding vectors produced by the current model.

    For text-embedding-004 this is 768.

    Returns:
        int: The embedding dimension.
    """
    return _EMBEDDING_DIMENSION