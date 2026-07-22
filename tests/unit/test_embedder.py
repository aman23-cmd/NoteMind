"""
test_embedder.py — Unit tests for modules/embedder.py

Tests cover:
  - Embedding output shape/dimensions (should be 384 for all-MiniLM-L6-v2)
  - Single text input
  - Multiple text inputs (batch)
  - Empty string edge case (should return zero vector)
  - Consistency: same text produces same embedding

NOTE: These tests load the real sentence-transformers model (~80MB).
      The first run will download the model if it's not cached.
      Subsequent runs use the cached model and are fast.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.embedder import generate_embeddings, get_embedding_dimension


# Expected dimension for all-MiniLM-L6-v2
EXPECTED_DIM = 384


# ════════════════════════════════════════════════════
#  TESTS: Embedding dimensions
# ════════════════════════════════════════════════════

class TestEmbeddingDimensions:
    """Test that embeddings have the correct shape."""

    def test_dimension_constant(self):
        """get_embedding_dimension() should return 384."""
        dim = get_embedding_dimension()
        assert dim == EXPECTED_DIM

    def test_single_text_dimension(self):
        """A single text should produce one embedding of 384 dimensions."""
        result = generate_embeddings("Hello world")
        assert len(result) == 1
        assert len(result[0]) == EXPECTED_DIM

    def test_multiple_texts_dimensions(self):
        """Multiple texts should produce one embedding each, all 384-dim."""
        texts = ["Hello", "World", "Python is great"]
        result = generate_embeddings(texts)

        assert len(result) == 3
        for emb in result:
            assert len(emb) == EXPECTED_DIM


# ════════════════════════════════════════════════════
#  TESTS: Edge cases
# ════════════════════════════════════════════════════

class TestEmbeddingEdgeCases:
    """Test edge cases for the embedder."""

    def test_empty_string(self):
        """An empty string should return a zero vector."""
        result = generate_embeddings("")
        assert len(result) == 1
        assert len(result[0]) == EXPECTED_DIM
        assert all(v == 0.0 for v in result[0])

    def test_whitespace_string(self):
        """A whitespace-only string should return a zero vector."""
        result = generate_embeddings("   ")
        assert len(result) == 1
        assert all(v == 0.0 for v in result[0])

    def test_empty_list(self):
        """An empty list should return an empty list."""
        result = generate_embeddings([])
        assert result == []

    def test_mixed_empty_and_real(self):
        """A mix of real and empty texts handles both correctly."""
        result = generate_embeddings(["Hello world", "", "Python"])
        assert len(result) == 3

        # First and third should be non-zero
        assert any(v != 0.0 for v in result[0])
        assert any(v != 0.0 for v in result[2])

        # Second (empty) should be all zeros
        assert all(v == 0.0 for v in result[1])


# ════════════════════════════════════════════════════
#  TESTS: Embedding quality (basic sanity checks)
# ════════════════════════════════════════════════════

class TestEmbeddingQuality:
    """Basic sanity checks on embedding values."""

    def test_non_zero_for_real_text(self):
        """Real text should produce non-zero embeddings."""
        result = generate_embeddings("Machine learning is a type of AI")
        assert any(v != 0.0 for v in result[0])

    def test_same_text_same_embedding(self):
        """The same text should always produce the same embedding."""
        text = "Recursion is when a function calls itself"
        result1 = generate_embeddings(text)
        result2 = generate_embeddings(text)

        # Should be identical (deterministic model)
        for v1, v2 in zip(result1[0], result2[0]):
            assert abs(v1 - v2) < 1e-6

    def test_returns_python_lists(self):
        """Embeddings should be plain Python lists, not numpy arrays."""
        result = generate_embeddings("test")
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], float)
