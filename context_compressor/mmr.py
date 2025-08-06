"""
MMR (Maximal Marginal Relevance) diversity selection module.
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from .schemas import Candidate
from .utils import cosine_similarity, stable_sort_with_tie_breaker


class MMRSelector:
    """Implements MMR diversity selection with budget and section constraints."""
    
    def __init__(self, lambda_param: float = 0.7, section_cap: int = 2, doc_cap: int = 6):
        """
        Initialize MMR selector.
        
        Args:
            lambda_param: MMR diversity parameter (0.0 = pure diversity, 1.0 = pure relevance)
            section_cap: Maximum number of chunks per section
            doc_cap: Maximum number of chunks per document (for CDC mode)
        """
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError("lambda_param must be between 0.0 and 1.0")
        if section_cap < 1:
            raise ValueError("section_cap must be at least 1")
        if doc_cap < 1:
            raise ValueError("doc_cap must be at least 1")
        
        self.lambda_param = lambda_param
        self.section_cap = section_cap
        self.doc_cap = doc_cap
    
    def compute_mmr_score(
        self, 
        candidate: Candidate, 
        selected_candidates: List[Candidate],
        query_embedding: Optional[List[float]] = None
    ) -> float:
        """
        Compute MMR score for a candidate given already selected candidates.
        
        Args:
            candidate: Candidate to score
            selected_candidates: Already selected candidates
            query_embedding: Query embedding (if available)
            
        Returns:
            MMR score (higher is better)
        """
        if not candidate.embedding:
            # Fallback to dense_sim if no embedding available
            relevance_score = candidate.dense_sim
            diversity_score = 0.0
        else:
            # Compute relevance score (similarity to query)
            if query_embedding:
                relevance_score = cosine_similarity(candidate.embedding, query_embedding)
            else:
                relevance_score = candidate.dense_sim
            
            # Compute diversity score (max similarity to selected candidates)
            if selected_candidates:
                max_similarity = 0.0
                for selected in selected_candidates:
                    if selected.embedding:
                        similarity = cosine_similarity(candidate.embedding, selected.embedding)
                        max_similarity = max(max_similarity, similarity)
                diversity_score = max_similarity
            else:
                diversity_score = 0.0
        
        # MMR formula: lambda * relevance - (1 - lambda) * diversity
        mmr_score = self.lambda_param * relevance_score - (1 - self.lambda_param) * diversity_score
        return mmr_score
    
    def check_section_constraint(
        self, 
        candidate: Candidate, 
        selected_candidates: List[Candidate]
    ) -> bool:
        """
        Check if adding candidate would violate section cap constraint.
        
        Args:
            candidate: Candidate to check
            selected_candidates: Already selected candidates
            
        Returns:
            True if section constraint is satisfied
        """
        section_count = 0
        for selected in selected_candidates:
            if selected.section == candidate.section:
                section_count += 1
        
        return section_count < self.section_cap
    
    def check_doc_constraint(
        self, 
        candidate: Candidate, 
        selected_candidates: List[Candidate]
    ) -> bool:
        """
        Check if adding candidate would violate document cap constraint.
        
        Args:
            candidate: Candidate to check
            selected_candidates: Already selected candidates
            
        Returns:
            True if document constraint is satisfied
        """
        doc_count = 0
        for selected in selected_candidates:
            if selected.doc_id == candidate.doc_id:
                doc_count += 1
        
        return doc_count < self.doc_cap
    
    def select_diverse_candidates(
        self, 
        candidates: List[Candidate], 
        budget: int,
        query_embedding: Optional[List[float]] = None,
        single_doc_mode: bool = False
    ) -> List[Candidate]:
        """
        Select diverse candidates using MMR within budget and constraints.
        
        Args:
            candidates: List of candidate passages
            budget: Token budget
            query_embedding: Query embedding (optional)
            single_doc_mode: Whether in single-document mode (relaxes doc_cap)
            
        Returns:
            List of selected candidates
        """
        if not candidates:
            return []
        
        selected_candidates = []
        remaining_candidates = candidates.copy()
        used_tokens = 0
        
        # Use very high doc_cap for single-document mode
        effective_doc_cap = 10**6 if single_doc_mode else self.doc_cap
        
        while remaining_candidates and used_tokens < budget:
            # Find best candidate that fits constraints
            best_candidate = None
            best_score = float('-inf')
            
            for candidate in remaining_candidates:
                # Check budget constraint (with soft overflow allowance)
                if used_tokens + candidate.tokens > budget * 1.15:
                    continue
                
                # Check section constraint
                if not self.check_section_constraint(candidate, selected_candidates):
                    continue
                
                # Check document constraint (only in CDC mode)
                if not single_doc_mode:
                    doc_count = sum(1 for s in selected_candidates if s.doc_id == candidate.doc_id)
                    if doc_count >= effective_doc_cap:
                        continue
                
                # Compute MMR score
                mmr_score = self.compute_mmr_score(
                    candidate, selected_candidates, query_embedding
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate
            
            # If no candidate fits constraints, break
            if best_candidate is None:
                break
            
            # Add best candidate
            selected_candidates.append(best_candidate)
            used_tokens += best_candidate.tokens
            remaining_candidates.remove(best_candidate)
        
        return selected_candidates
    
    def get_mmr_stats(
        self, 
        candidates: List[Candidate], 
        selected_candidates: List[Candidate],
        single_doc_mode: bool = False
    ) -> Dict:
        """
        Get statistics about the MMR selection process.
        
        Args:
            candidates: Original candidate pool
            selected_candidates: Selected candidates
            single_doc_mode: Whether in single-document mode
            
        Returns:
            Dictionary with MMR statistics
        """
        if not candidates:
            return {
                'num_candidates': 0,
                'num_selected': 0,
                'lambda_used': self.lambda_param,
                'section_cap_used': self.section_cap,
                'doc_cap_used': self.doc_cap if not single_doc_mode else 10**6,
                'section_distribution': {},
                'doc_distribution': {},
                'token_usage': 0
            }
        
        # Section distribution
        section_counts = {}
        for candidate in selected_candidates:
            section = candidate.section
            section_counts[section] = section_counts.get(section, 0) + 1
        
        # Document distribution
        doc_counts = {}
        for candidate in selected_candidates:
            doc_id = candidate.doc_id
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        
        # Token usage
        total_tokens = sum(c.tokens for c in selected_candidates)
        
        return {
            'num_candidates': len(candidates),
            'num_selected': len(selected_candidates),
            'lambda_used': self.lambda_param,
            'section_cap_used': self.section_cap,
            'doc_cap_used': self.doc_cap if not single_doc_mode else 10**6,
            'section_distribution': section_counts,
            'doc_distribution': doc_counts,
            'token_usage': total_tokens
        }
