"""
Oracle creation using greedy set-cover algorithm for optimal compression selection.
"""

import numpy as np
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class OracleCreator:
    """Creates oracle selections using greedy set-cover algorithm."""
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize oracle creator.
        
        Args:
            alpha: Weight for redundancy penalty (0.0 = no penalty, 1.0 = full penalty)
        """
        self.alpha = alpha
    
    def create_oracle(
        self,
        candidates: List[Dict[str, Any]],
        anchors: Dict[str, List[str]],
        budget: int,
        embeddings: List[List[float]] = None
    ) -> List[int]:
        """
        Create oracle selection using greedy set-cover with redundancy penalty.
        
        Args:
            candidates: List of candidate passages
            anchors: Dictionary of anchor types and values
            budget: Token budget
            embeddings: Optional embeddings for redundancy calculation
            
        Returns:
            List of selected candidate indices
        """
        logger.info(f"Creating oracle selection for {len(candidates)} candidates with budget {budget}")
        
        # Extract anchor sets for each candidate
        anchor_sets = self._extract_anchor_sets(candidates, anchors)
        
        # Calculate redundancy matrix if embeddings provided
        redundancy_matrix = None
        if embeddings:
            redundancy_matrix = self._calculate_redundancy_matrix(embeddings)
        
        # Run greedy set-cover
        selected_indices = self._greedy_set_cover(
            candidates, anchor_sets, budget, redundancy_matrix
        )
        
        logger.info(f"Oracle selection complete: {len(selected_indices)} candidates selected")
        return selected_indices
    
    def _extract_anchor_sets(
        self, 
        candidates: List[Dict[str, Any]], 
        anchors: Dict[str, List[str]]
    ) -> List[Set[str]]:
        """
        Extract anchor sets for each candidate.
        
        Args:
            candidates: List of candidate passages
            anchors: Dictionary of anchor types and values
            
        Returns:
            List of anchor sets for each candidate
        """
        anchor_sets = []
        
        for candidate in candidates:
            text = candidate.get('text', '').lower()
            candidate_anchors = set()
            
            # Check which anchors are present in this candidate
            for anchor_type, anchor_values in anchors.items():
                for anchor in anchor_values:
                    if anchor.lower() in text:
                        candidate_anchors.add(anchor.lower())
            
            anchor_sets.append(candidate_anchors)
        
        return anchor_sets
    
    def _calculate_redundancy_matrix(self, embeddings: List[List[float]]) -> np.ndarray:
        """
        Calculate redundancy matrix using cosine similarity.
        
        Args:
            embeddings: List of embeddings for candidates
            
        Returns:
            Redundancy matrix
        """
        n = len(embeddings)
        redundancy_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                redundancy_matrix[i, j] = similarity
                redundancy_matrix[j, i] = similarity
        
        return redundancy_matrix
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _greedy_set_cover(
        self,
        candidates: List[Dict[str, Any]],
        anchor_sets: List[Set[str]],
        budget: int,
        redundancy_matrix: np.ndarray = None
    ) -> List[int]:
        """
        Greedy set-cover algorithm with redundancy penalty.
        
        Args:
            candidates: List of candidate passages
            anchor_sets: List of anchor sets for each candidate
            budget: Token budget
            redundancy_matrix: Optional redundancy matrix
            
        Returns:
            List of selected candidate indices
        """
        selected_indices = []
        covered_anchors = set()
        used_tokens = 0
        
        while used_tokens < budget:
            best_candidate = None
            best_score = -1
            
            for i, candidate in enumerate(candidates):
                if i in selected_indices:
                    continue
                
                # Check budget constraint
                candidate_tokens = candidate.get('tokens', 0)
                if used_tokens + candidate_tokens > budget:
                    continue
                
                # Calculate gain (new anchors covered)
                candidate_anchors = anchor_sets[i]
                new_anchors = candidate_anchors - covered_anchors
                gain = len(new_anchors)
                
                if gain == 0:
                    continue  # No new anchors covered
                
                # Calculate redundancy penalty
                penalty = 0.0
                if redundancy_matrix is not None and selected_indices:
                    max_similarity = max(redundancy_matrix[i, j] for j in selected_indices)
                    penalty = max_similarity
                
                # Calculate score: gain - alpha * penalty
                score = gain - self.alpha * penalty
                
                if score > best_score:
                    best_score = score
                    best_candidate = i
            
            if best_candidate is None:
                break  # No more candidates can be added
            
            # Add best candidate
            selected_indices.append(best_candidate)
            covered_anchors.update(anchor_sets[best_candidate])
            used_tokens += candidates[best_candidate].get('tokens', 0)
            
            logger.debug(f"Selected candidate {best_candidate}: gain={best_score:.2f}, tokens={candidates[best_candidate].get('tokens', 0)}")
        
        return selected_indices
    
    def evaluate_oracle(
        self,
        oracle_indices: List[int],
        candidates: List[Dict[str, Any]],
        anchors: Dict[str, List[str]],
        budget: int
    ) -> Dict[str, float]:
        """
        Evaluate oracle selection.
        
        Args:
            oracle_indices: List of selected candidate indices
            candidates: List of all candidates
            anchors: Dictionary of anchor types and values
            budget: Token budget
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Calculate metrics
        total_tokens = sum(candidates[i].get('tokens', 0) for i in oracle_indices)
        token_usage = total_tokens / budget if budget > 0 else 0.0
        
        # Calculate anchor coverage
        all_anchors = set()
        for anchor_type, anchor_values in anchors.items():
            all_anchors.update(anchor_values)
        
        covered_anchors = set()
        for i in oracle_indices:
            text = candidates[i].get('text', '').lower()
            for anchor in all_anchors:
                if anchor.lower() in text:
                    covered_anchors.add(anchor.lower())
        
        anchor_coverage = len(covered_anchors) / len(all_anchors) if all_anchors else 0.0
        
        # Calculate redundancy
        redundancy = 0.0
        if len(oracle_indices) > 1:
            # Simple redundancy calculation based on text overlap
            texts = [candidates[i].get('text', '') for i in oracle_indices]
            total_overlap = 0
            total_pairs = 0
            
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    words1 = set(texts[i].lower().split())
                    words2 = set(texts[j].lower().split())
                    if words1 and words2:
                        overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                        total_overlap += overlap
                        total_pairs += 1
            
            redundancy = total_overlap / total_pairs if total_pairs > 0 else 0.0
        
        metrics = {
            'num_selected': len(oracle_indices),
            'token_usage': token_usage,
            'anchor_coverage': anchor_coverage,
            'redundancy': redundancy,
            'efficiency': anchor_coverage / (1 + redundancy)  # Coverage per unit redundancy
        }
        
        return metrics
    
    def compare_with_baseline(
        self,
        oracle_indices: List[int],
        baseline_indices: List[int],
        candidates: List[Dict[str, Any]],
        anchors: Dict[str, List[str]],
        budget: int
    ) -> Dict[str, Any]:
        """
        Compare oracle selection with baseline selection.
        
        Args:
            oracle_indices: Oracle selected indices
            baseline_indices: Baseline selected indices
            candidates: List of all candidates
            anchors: Dictionary of anchor types and values
            budget: Token budget
            
        Returns:
            Dictionary with comparison metrics
        """
        oracle_metrics = self.evaluate_oracle(oracle_indices, candidates, anchors, budget)
        baseline_metrics = self.evaluate_oracle(baseline_indices, candidates, anchors, budget)
        
        comparison = {
            'oracle': oracle_metrics,
            'baseline': baseline_metrics,
            'improvements': {}
        }
        
        # Calculate improvements
        for metric in oracle_metrics:
            if metric in baseline_metrics:
                oracle_val = oracle_metrics[metric]
                baseline_val = baseline_metrics[metric]
                
                if baseline_val > 0:
                    improvement = (oracle_val - baseline_val) / baseline_val
                    comparison['improvements'][metric] = improvement
        
        return comparison
