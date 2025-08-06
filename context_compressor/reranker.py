"""
Reranker module for Context Compressor.
"""

import time
from typing import List, Tuple, Optional, Dict, Any
from sentence_transformers import CrossEncoder
import numpy as np
from .utils import compute_hash


class Reranker:
    """Implements cross-encoder reranking using BAAI/bge-reranker-large."""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", cache_size: int = 500):
        """
        Initialize reranker.
        
        Args:
            model_name: Name of the cross-encoder model
            cache_size: Maximum number of cached scores
        """
        self.model_name = model_name
        self.cache_size = cache_size
        self.cache = {}
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            self.model = CrossEncoder(self.model_name)
            print(f"Loaded reranker model: {self.model_name}")
        except Exception as e:
            print(f"Warning: Could not load {self.model_name}, falling back to no reranking: {e}")
            self.model = None
    
    def _get_cache_key(self, query: str, text: str) -> str:
        """Generate cache key for query-text pair."""
        return compute_hash(f"{query}|||{text}")
    
    def _add_to_cache(self, query: str, text: str, score: float):
        """Add score to cache with LRU eviction."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        cache_key = self._get_cache_key(query, text)
        self.cache[cache_key] = {
            'score': score,
            'timestamp': time.time()
        }
    
    def _get_from_cache(self, query: str, text: str) -> Optional[float]:
        """Get score from cache if available."""
        cache_key = self._get_cache_key(query, text)
        if cache_key in self.cache:
            return self.cache[cache_key]['score']
        return None
    
    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        """
        Score query-text pairs using cross-encoder.
        
        Args:
            query: Query text
            texts: List of texts to score
            
        Returns:
            List of relevance scores
        """
        if not self.model:
            # Fallback: return uniform scores
            return [0.5] * len(texts)
        
        scores = []
        pairs_to_score = []
        pair_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cached_score = self._get_from_cache(query, text)
            if cached_score is not None:
                scores.append(cached_score)
            else:
                pairs_to_score.append([query, text])
                pair_indices.append(i)
        
        # Score pairs not in cache
        if pairs_to_score:
            try:
                new_scores = self.model.predict(pairs_to_score)
                
                # Add to cache and results
                for i, (text, score) in enumerate(zip(texts, new_scores)):
                    if i in pair_indices:
                        self._add_to_cache(query, text, float(score))
                        scores.insert(pair_indices[pair_indices.index(i)], float(score))
                        
            except Exception as e:
                print(f"Warning: Reranker scoring failed: {e}")
                # Fill with default scores for failed predictions
                for idx in pair_indices:
                    scores.insert(idx, 0.5)
        
        return scores
    
    def score_single(self, query: str, text: str) -> float:
        """
        Score a single query-text pair.
        
        Args:
            query: Query text
            text: Text to score
            
        Returns:
            Relevance score
        """
        scores = self.score_pairs(query, [text])
        return scores[0] if scores else 0.5
    
    def rerank_sentences(
        self, 
        query: str, 
        sentences: List[str], 
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Rerank sentences by relevance to query.
        
        Args:
            query: Query text
            sentences: List of sentences to rerank
            top_k: Return only top-k sentences (None for all)
            
        Returns:
            List of (sentence, score) tuples sorted by score
        """
        if not sentences:
            return []
        
        # Score all sentences
        scores = self.score_pairs(query, sentences)
        
        # Create sentence-score pairs
        sentence_scores = list(zip(sentences, scores))
        
        # Sort by score (descending)
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            sentence_scores = sentence_scores[:top_k]
        
        return sentence_scores
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics."""
        return {
            'model_name': self.model_name,
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'model_loaded': self.model is not None
        }
    
    def clear_cache(self):
        """Clear the reranker cache."""
        self.cache.clear()


# Global reranker instance
_reranker = None


def get_reranker(model_name: str = "BAAI/bge-reranker-large") -> Reranker:
    """Get or create global reranker instance."""
    global _reranker
    if _reranker is None or _reranker.model_name != model_name:
        _reranker = Reranker(model_name)
    return _reranker
