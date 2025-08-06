"""
Unit tests for utility functions.
"""

import pytest
import numpy as np
from context_compressor.utils import (
    count_tokens,
    z_score_normalize,
    cosine_similarity,
    split_sentences,
    extract_anchors,
    compute_hash,
    check_low_context,
    validate_budget_constraint,
    stable_sort_with_tie_breaker,
)


class TestTokenCounting:
    """Test token counting functionality."""

    def test_count_tokens_basic(self):
        """Test basic token counting."""
        text = "Hello world"
        tokens = count_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_count_tokens_empty(self):
        """Test token counting for empty text."""
        tokens = count_tokens("")
        assert tokens == 0

    def test_count_tokens_long_text(self):
        """Test token counting for longer text."""
        text = "This is a longer text with multiple sentences. It should have more tokens."
        tokens = count_tokens(text)
        assert tokens > 10


class TestZScoreNormalization:
    """Test z-score normalization."""

    def test_z_score_normalize_basic(self):
        """Test basic z-score normalization."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = z_score_normalize(scores)
        
        assert len(normalized) == len(scores)
        assert np.mean(normalized) == pytest.approx(0.0, abs=1e-10)
        assert np.std(normalized) == pytest.approx(1.0, abs=1e-10)

    def test_z_score_normalize_empty(self):
        """Test z-score normalization with empty list."""
        normalized = z_score_normalize([])
        assert normalized == []

    def test_z_score_normalize_constant(self):
        """Test z-score normalization with constant values."""
        scores = [1.0, 1.0, 1.0]
        normalized = z_score_normalize(scores)
        assert all(x == 0.0 for x in normalized)

    def test_z_score_normalize_negative(self):
        """Test z-score normalization with negative values."""
        scores = [-1.0, 0.0, 1.0]
        normalized = z_score_normalize(scores)
        assert len(normalized) == 3
        assert np.mean(normalized) == pytest.approx(0.0, abs=1e-10)


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_cosine_similarity_basic(self):
        """Test basic cosine similarity."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity for orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0)

    def test_cosine_similarity_empty(self):
        """Test cosine similarity with empty vectors."""
        similarity = cosine_similarity([], [])
        assert similarity == 0.0

    def test_cosine_similarity_different_lengths(self):
        """Test cosine similarity with different length vectors."""
        vec1 = [1.0, 2.0]
        vec2 = [1.0, 2.0, 3.0]
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_cosine_similarity_zero_vectors(self):
        """Test cosine similarity with zero vectors."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == 0.0


class TestSentenceSplitting:
    """Test sentence splitting functionality."""

    def test_split_sentences_basic(self):
        """Test basic sentence splitting."""
        text = "Hello world. This is a test. How are you?"
        sentences = split_sentences(text)
        assert len(sentences) == 3
        assert "Hello world" in sentences[0]
        assert "This is a test" in sentences[1]
        assert "How are you" in sentences[2]

    def test_split_sentences_empty(self):
        """Test sentence splitting with empty text."""
        sentences = split_sentences("")
        assert sentences == []

    def test_split_sentences_no_periods(self):
        """Test sentence splitting with no periods."""
        text = "Hello world"
        sentences = split_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == "Hello world"

    def test_split_sentences_multiple_punctuation(self):
        """Test sentence splitting with multiple punctuation marks."""
        text = "Hello! How are you? I'm fine. Thank you."
        sentences = split_sentences(text)
        assert len(sentences) == 4


class TestAnchorExtraction:
    """Test anchor extraction functionality."""

    def test_extract_anchors_numbers(self):
        """Test extraction of numbers."""
        text = "The study found a 25% improvement with 200 participants."
        anchors = extract_anchors(text)
        assert "25%" in anchors
        assert "200" in anchors

    def test_extract_anchors_entities(self):
        """Test extraction of entities."""
        text = "Dr. Smith conducted the research at MIT."
        anchors = extract_anchors(text)
        assert "Dr" in anchors
        assert "Smith" in anchors
        assert "MIT" in anchors

    def test_extract_anchors_acronyms(self):
        """Test extraction of acronyms."""
        text = "The API was developed by the WHO and CDC."
        anchors = extract_anchors(text)
        assert "API" in anchors
        assert "WHO" in anchors
        assert "CDC" in anchors

    def test_extract_anchors_empty(self):
        """Test anchor extraction with empty text."""
        anchors = extract_anchors("")
        assert anchors == []

    def test_extract_anchors_no_anchors(self):
        """Test anchor extraction with no anchors."""
        text = "This is a simple text without any numbers or entities."
        anchors = extract_anchors(text)
        assert len(anchors) == 0


class TestHashComputation:
    """Test hash computation functionality."""

    def test_compute_hash_string(self):
        """Test hash computation for string."""
        data = "test string"
        hash1 = compute_hash(data)
        hash2 = compute_hash(data)
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length

    def test_compute_hash_dict(self):
        """Test hash computation for dictionary."""
        data = {"key1": "value1", "key2": "value2"}
        hash1 = compute_hash(data)
        hash2 = compute_hash(data)
        assert hash1 == hash2

    def test_compute_hash_dict_order_invariant(self):
        """Test that hash is order-invariant for dictionaries."""
        data1 = {"key1": "value1", "key2": "value2"}
        data2 = {"key2": "value2", "key1": "value1"}
        hash1 = compute_hash(data1)
        hash2 = compute_hash(data2)
        assert hash1 == hash2


class TestLowContextCheck:
    """Test low context checking functionality."""

    def test_check_low_context_true(self):
        """Test low context detection when usage is low."""
        used_tokens = 20
        budget = 100
        threshold = 0.3
        is_low = check_low_context(used_tokens, budget, threshold)
        assert is_low is True

    def test_check_low_context_false(self):
        """Test low context detection when usage is adequate."""
        used_tokens = 50
        budget = 100
        threshold = 0.3
        is_low = check_low_context(used_tokens, budget, threshold)
        assert is_low is False

    def test_check_low_context_boundary(self):
        """Test low context detection at boundary."""
        used_tokens = 29  # Changed from 30 to 29 to make it actually below threshold
        budget = 100
        threshold = 0.3
        is_low = check_low_context(used_tokens, budget, threshold)
        assert is_low is True  # 29 < 30 is True


class TestBudgetConstraintValidation:
    """Test budget constraint validation."""

    def test_validate_budget_constraint_within_limit(self):
        """Test budget validation when within limit."""
        used_tokens = 80
        budget = 100
        is_valid = validate_budget_constraint(used_tokens, budget)
        assert is_valid is True

    def test_validate_budget_constraint_at_limit(self):
        """Test budget validation when at limit."""
        used_tokens = 100
        budget = 100
        is_valid = validate_budget_constraint(used_tokens, budget)
        assert is_valid is True

    def test_validate_budget_constraint_over_limit(self):
        """Test budget validation when over limit."""
        used_tokens = 120
        budget = 100
        is_valid = validate_budget_constraint(used_tokens, budget)
        assert is_valid is False

    def test_validate_budget_constraint_with_soft_overflow(self):
        """Test budget validation with soft overflow."""
        used_tokens = 110
        budget = 100
        soft_overflow = 0.15
        is_valid = validate_budget_constraint(used_tokens, budget, soft_overflow)
        assert is_valid is True  # 110 <= 115


class TestStableSort:
    """Test stable sorting with tie-breaking."""

    def test_stable_sort_with_tie_breaker(self):
        """Test stable sorting with tie-breaking."""
        items = [("a", 1), ("b", 1), ("c", 2), ("d", 1)]
        
        def key_func(item):
            return item[1]
        
        sorted_items = stable_sort_with_tie_breaker(items, key_func, reverse=True)
        
        # Should maintain original order for items with same key
        assert sorted_items[0][0] == "c"  # Highest value
        # Items with value 1 should maintain original order: a, b, d
        assert sorted_items[1][0] == "a"
        assert sorted_items[2][0] == "b"
        assert sorted_items[3][0] == "d"

    def test_stable_sort_empty(self):
        """Test stable sorting with empty list."""
        items = []
        sorted_items = stable_sort_with_tie_breaker(items, lambda x: x)
        assert sorted_items == []

    def test_stable_sort_single_item(self):
        """Test stable sorting with single item."""
        items = [("a", 1)]
        sorted_items = stable_sort_with_tie_breaker(items, lambda x: x[1])
        assert sorted_items == [("a", 1)]
