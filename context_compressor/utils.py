"""
Utility functions for Context Compressor.
"""

import hashlib
import time
import re
from typing import List, Dict, Any, Optional
import tiktoken
import numpy as np
from functools import wraps


# Default tokenizer model - using cl100k_base which is used by GPT-4
DEFAULT_TOKENIZER = "cl100k_base"


def get_tokenizer(model: str = DEFAULT_TOKENIZER) -> tiktoken.Encoding:
    """Get a tiktoken tokenizer for the specified model."""
    try:
        return tiktoken.get_encoding(model)
    except KeyError:
        # Fallback to cl100k_base if model not found
        return tiktoken.get_encoding(DEFAULT_TOKENIZER)


def count_tokens(text: str, model: str = DEFAULT_TOKENIZER) -> int:
    """Count tokens in text using tiktoken."""
    tokenizer = get_tokenizer(model)
    return len(tokenizer.encode(text))


def z_score_normalize(scores: List[float]) -> List[float]:
    """Normalize scores using z-score normalization."""
    if not scores:
        return []
    
    scores_array = np.array(scores)
    mean = np.mean(scores_array)
    std = np.std(scores_array)
    
    if std == 0:
        return [0.0] * len(scores)
    
    return ((scores_array - mean) / std).tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    vec1_array = np.array(vec1)
    vec2_array = np.array(vec2)
    
    dot_product = np.dot(vec1_array, vec2_array)
    norm1 = np.linalg.norm(vec1_array)
    norm2 = np.linalg.norm(vec2_array)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving sentence boundaries."""
    # Simple sentence splitting - can be enhanced with more sophisticated NLP
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def extract_anchors(text: str) -> List[str]:
    """Extract anchor elements (numbers, entities) from text."""
    anchors = []
    
    # Extract numbers (including decimals, percentages, etc.)
    numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|st|nd|rd|th)?\b', text)
    # Also extract percentages that might not have word boundaries
    percentages = re.findall(r'\d+(?:\.\d+)?%', text)
    numbers.extend(percentages)
    anchors.extend(numbers)
    
    # Extract potential entities (capitalized words, acronyms)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    anchors.extend(entities)
    
    # Extract acronyms
    acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
    anchors.extend(acronyms)
    
    # Filter out common words that aren't really anchors
    common_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By'}
    filtered_anchors = [anchor for anchor in anchors if anchor not in common_words]
    
    return list(set(filtered_anchors))  # Remove duplicates


def compute_hash(data: Any) -> str:
    """Compute a hash for caching purposes."""
    if isinstance(data, dict):
        # Sort dictionary items for deterministic hashing
        sorted_items = sorted(data.items())
        data_str = str(sorted_items)
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode()).hexdigest()


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        return result, execution_time_ms
    return wrapper


def stable_sort_with_tie_breaker(items: List[Any], key_func, reverse: bool = False) -> List[Any]:
    """Sort items with stable tie-breaking using original indices."""
    # Add original index as secondary sort key for stability
    indexed_items = [(item, i) for i, item in enumerate(items)]
    
    def sort_key(pair):
        item, original_index = pair
        # For reverse=True, we need to negate the original_index to maintain order
        if reverse:
            return (key_func(item), -original_index)
        else:
            return (key_func(item), original_index)
    
    sorted_pairs = sorted(indexed_items, key=sort_key, reverse=reverse)
    return [item for item, _ in sorted_pairs]


def check_low_context(used_tokens: int, budget: int, threshold: float = 0.3) -> bool:
    """Check if context usage is below threshold."""
    return used_tokens < (budget * threshold)


def validate_budget_constraint(used_tokens: int, budget: int, soft_overflow: float = 0.15) -> bool:
    """Validate that token usage is within budget constraints."""
    hard_limit = budget
    soft_limit = int(budget * (1 + soft_overflow))
    
    if used_tokens > soft_limit:
        return False
    return True
