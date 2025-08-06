"""
Rank fusion module for Context Compressor.
"""

from typing import List, Dict, Tuple
import numpy as np
from .schemas import Candidate
from .utils import z_score_normalize, stable_sort_with_tie_breaker


class RankFusion:
    """Implements rank fusion using z-score normalization."""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize rank fusion with weights.
        
        Args:
            weights: Dictionary with 'dense' and 'bm25' weights. 
                    Defaults to {'dense': 0.7, 'bm25': 0.3}
        """
        self.weights = weights or {'dense': 0.7, 'bm25': 0.3}
        
        # Validate weights
        if not all(key in self.weights for key in ['dense', 'bm25']):
            raise ValueError("weights must contain 'dense' and 'bm25' keys")
        if not all(0.0 <= w <= 1.0 for w in self.weights.values()):
            raise ValueError("weights must be between 0.0 and 1.0")
    
    def compute_fusion_scores(self, candidates: List[Candidate]) -> List[Tuple[Candidate, float]]:
        """
        Compute fusion scores for candidates using z-score normalization.
        
        Args:
            candidates: List of candidate passages
            
        Returns:
            List of (candidate, fusion_score) tuples
        """
        if not candidates:
            return []
        
        # Extract scores
        dense_scores = [c.dense_sim for c in candidates]
        bm25_scores = [c.bm25 for c in candidates]
        
        # Z-score normalization
        dense_z_scores = z_score_normalize(dense_scores)
        bm25_z_scores = z_score_normalize(bm25_scores)
        
        # Compute fusion scores
        fusion_scores = []
        for i, candidate in enumerate(candidates):
            fusion_score = (
                self.weights['dense'] * dense_z_scores[i] +
                self.weights['bm25'] * bm25_z_scores[i]
            )
            fusion_scores.append((candidate, fusion_score))
        
        return fusion_scores
    
    def select_top_m(self, candidates: List[Candidate], m: int = 200) -> List[Candidate]:
        """
        Select top-M candidates using rank fusion.
        
        Args:
            candidates: List of candidate passages
            m: Number of top candidates to select (default: 200)
            
        Returns:
            List of top-M candidates sorted by fusion score
        """
        if not candidates:
            return []
        
        # Compute fusion scores
        scored_candidates = self.compute_fusion_scores(candidates)
        
        # Sort by fusion score (descending) with stable tie-breaking
        def sort_key(item):
            candidate, score = item
            return score
        
        sorted_candidates = stable_sort_with_tie_breaker(
            scored_candidates, 
            sort_key, 
            reverse=True
        )
        
        # Select top-M
        top_m = sorted_candidates[:m]
        
        # Return just the candidates (without scores)
        return [candidate for candidate, _ in top_m]
    
    def get_fusion_stats(self, candidates: List[Candidate]) -> Dict:
        """
        Get statistics about the fusion process.
        
        Args:
            candidates: List of candidate passages
            
        Returns:
            Dictionary with fusion statistics
        """
        if not candidates:
            return {
                'num_candidates': 0,
                'dense_scores_range': (0, 0),
                'bm25_scores_range': (0, 0),
                'weights_used': self.weights
            }
        
        dense_scores = [c.dense_sim for c in candidates]
        bm25_scores = [c.bm25 for c in candidates]
        
        return {
            'num_candidates': len(candidates),
            'dense_scores_range': (min(dense_scores), max(dense_scores)),
            'bm25_scores_range': (min(bm25_scores), max(bm25_scores)),
            'weights_used': self.weights
        }
