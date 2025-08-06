"""
Mode router for Cross-Document Compression (CDC) vs Single-Document Mode (SDM).
"""

import math
from collections import Counter
from typing import List, Tuple, Dict, Optional
from .schemas import Candidate, RouterStats


class ModeRouter:
    """Routes between CDC and SDM modes based on document concentration."""
    
    def __init__(self, threshold: float = 0.8):
        """
        Initialize mode router.
        
        Args:
            threshold: Fraction threshold for single-document routing (default: 0.8)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = threshold
    
    def route_mode(
        self, 
        candidates: List[Candidate], 
        top_k: int = 50
    ) -> Tuple[str, Optional[str], RouterStats]:
        """
        Route between CDC and SDM modes.
        
        Args:
            candidates: List of candidates (should be sorted by relevance)
            top_k: Number of top candidates to consider for routing
            
        Returns:
            Tuple of (mode, doc_id, router_stats)
            - mode: 'cross_doc' or 'single_doc'
            - doc_id: Target document ID for SDM (None for CDC)
            - router_stats: Routing statistics
        """
        if not candidates:
            return "cross_doc", None, RouterStats(top1_doc_frac=0.0, entropy=0.0)
        
        # Take top-k candidates for routing decision
        top_candidates = candidates[:min(top_k, len(candidates))]
        
        # Count document frequencies
        doc_freq = Counter([c.doc_id for c in top_candidates])
        
        # Calculate top1 document fraction
        if doc_freq:
            top1_doc, count = max(doc_freq.items(), key=lambda kv: kv[1])
            top1_doc_frac = count / len(top_candidates)
        else:
            top1_doc_frac = 0.0
            top1_doc = None
        
        # Calculate entropy for tie-breaking
        entropy = self._compute_entropy(doc_freq, len(top_candidates))
        
        # Make routing decision
        if top1_doc_frac >= self.threshold:
            mode = "single_doc"
            doc_id = top1_doc
        else:
            mode = "cross_doc"
            doc_id = None
        
        router_stats = RouterStats(
            top1_doc_frac=top1_doc_frac,
            entropy=entropy
        )
        
        return mode, doc_id, router_stats
    
    def _compute_entropy(self, doc_freq: Counter, total: int) -> float:
        """
        Compute entropy of document distribution.
        
        Args:
            doc_freq: Document frequency counter
            total: Total number of candidates
            
        Returns:
            Entropy value
        """
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in doc_freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)
        
        return entropy
    
    def get_routing_stats(self, candidates: List[Candidate]) -> Dict:
        """
        Get detailed routing statistics.
        
        Args:
            candidates: List of candidates
            
        Returns:
            Dictionary with routing statistics
        """
        if not candidates:
            return {
                'num_candidates': 0,
                'num_documents': 0,
                'threshold': self.threshold,
                'document_distribution': {}
            }
        
        doc_freq = Counter([c.doc_id for c in candidates])
        
        return {
            'num_candidates': len(candidates),
            'num_documents': len(doc_freq),
            'threshold': self.threshold,
            'document_distribution': dict(doc_freq)
        }
