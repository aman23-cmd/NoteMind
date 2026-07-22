"""
embedder.py — Generates vector embeddings from text using sentence-transformers.

Uses the all-MiniLM-L6-v2 model which:
  - Runs fully locally (no API key needed)
  - Produces 384-dimensional embeddings
  - Is fast and lightweight (~80MB model)

The model is loaded lazily (on first use) and cached for the lifetime of the
process, so it is NOT reloaded on every call.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import Config

# ── Module-level cache for the model ──
_model = None


def _get_model():
    """
    Load the sentence-transformers model (lazy singleton).

    The model is loaded on first call and cached in a module-level variable.
    Subsequent calls return the cached model instantly.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(Config.EMBEDDING_MODEL)
    return _model


def generate_embeddings(texts):
    """
    Generate embeddings for one or more text strings.

    Args:
        texts (str or list of str): A single text or a list of texts to embed.

    Returns:
        list: A list of embedding vectors (each is a list of floats).
              For all-MiniLM-L6-v2, each vector has 384 dimensions.

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
    # model doesn't choke, then we'll zero out the result.
    empty_mask = []
    cleaned_texts = []
    for t in texts:
        if not t or not t.strip():
            empty_mask.append(True)
            cleaned_texts.append("empty")  # placeholder — will be zeroed
        else:
            empty_mask.append(False)
            cleaned_texts.append(t)

    # Generate embeddings
    model = _get_model()
    embeddings = model.encode(cleaned_texts, show_progress_bar=False)

    # Convert numpy arrays to plain Python lists
    result = []
    for i, emb in enumerate(embeddings):
        if empty_mask[i]:
            # Zero vector for empty inputs
            result.append([0.0] * len(emb))
        else:
            result.append(emb.tolist())

    return result


def get_embedding_dimension():
    """
    Return the dimension of the embedding vectors produced by the current model.

    For all-MiniLM-L6-v2 this is 384.

    Returns:
        int: The embedding dimension.
    """
    model = _get_model()
    return model.get_embedding_dimension()
