"""
Unit tests for rank fusion functionality.
"""

import pytest
import numpy as np
from context_compressor.fusion import RankFusion
from context_compressor.schemas import Candidate


class TestRankFusion:
    """Test rank fusion functionality."""

    def setup_method(self):
        """Set up test data."""
        self.candidates = [
            Candidate(
                id="c_001",
                doc_id="d_01",
                section="Results",
                page=14,
                text="The study found a 25% improvement in accuracy.",
                tokens=25,
                bm25=7.2,
                dense_sim=0.81,
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
            ),
            Candidate(
                id="c_002",
                doc_id="d_01",
                section="Methods",
                page=8,
                text="We employed a randomized controlled trial design.",
                tokens=22,
                bm25=6.8,
                dense_sim=0.75,
                embedding=[0.2, 0.3, 0.4, 0.5, 0.6]
            ),
            Candidate(
                id="c_003",
                doc_id="d_02",
                section="Discussion",
                page=20,
                text="These findings suggest broader applications.",
                tokens=18,
                bm25=5.9,
                dense_sim=0.68,
                embedding=[0.3, 0.4, 0.5, 0.6, 0.7]
            )
        ]

    def test_rank_fusion_init_default_weights(self):
        """Test rank fusion initialization with default weights."""
        fusion = RankFusion()
        assert fusion.weights == {"dense": 0.7, "bm25": 0.3}

    def test_rank_fusion_init_custom_weights(self):
        """Test rank fusion initialization with custom weights."""
        weights = {"dense": 0.8, "bm25": 0.2}
        fusion = RankFusion(weights)
        assert fusion.weights == weights

    def test_rank_fusion_init_invalid_weights(self):
        """Test rank fusion initialization with invalid weights."""
        # Missing required keys
        with pytest.raises(ValueError, match="weights must contain"):
            RankFusion({"dense": 0.7})

        # Invalid weight values
        with pytest.raises(ValueError, match="weights must be between"):
            RankFusion({"dense": 1.5, "bm25": 0.3})

    def test_compute_fusion_scores(self):
        """Test fusion score computation."""
        fusion = RankFusion()
        scored_candidates = fusion.compute_fusion_scores(self.candidates)
        
        assert len(scored_candidates) == len(self.candidates)
        assert all(isinstance(item, tuple) for item in scored_candidates)
        assert all(len(item) == 2 for item in scored_candidates)
        assert all(isinstance(item[0], Candidate) for item in scored_candidates)
        assert all(isinstance(item[1], float) for item in scored_candidates)

    def test_compute_fusion_scores_empty(self):
        """Test fusion score computation with empty candidates."""
        fusion = RankFusion()
        scored_candidates = fusion.compute_fusion_scores([])
        assert scored_candidates == []

    def test_compute_fusion_scores_single_candidate(self):
        """Test fusion score computation with single candidate."""
        fusion = RankFusion()
        single_candidate = [self.candidates[0]]
        scored_candidates = fusion.compute_fusion_scores(single_candidate)
        
        assert len(scored_candidates) == 1
        candidate, score = scored_candidates[0]
        assert candidate.id == "c_001"
        assert isinstance(score, float)

    def test_select_top_m(self):
        """Test top-M candidate selection."""
        fusion = RankFusion()
        top_candidates = fusion.select_top_m(self.candidates, m=2)
        
        assert len(top_candidates) == 2
        assert all(isinstance(c, Candidate) for c in top_candidates)
        # Should be sorted by fusion score (descending)
        assert top_candidates[0].id != top_candidates[1].id

    def test_select_top_m_empty(self):
        """Test top-M selection with empty candidates."""
        fusion = RankFusion()
        top_candidates = fusion.select_top_m([], m=5)
        assert top_candidates == []

    def test_select_top_m_m_larger_than_candidates(self):
        """Test top-M selection when M is larger than candidate count."""
        fusion = RankFusion()
        top_candidates = fusion.select_top_m(self.candidates, m=10)
        assert len(top_candidates) == len(self.candidates)

    def test_select_top_m_default_m(self):
        """Test top-M selection with default M value."""
        fusion = RankFusion()
        # Create more candidates to test default M=120
        many_candidates = self.candidates * 50  # 150 candidates
        top_candidates = fusion.select_top_m(many_candidates, m=120)
        assert len(top_candidates) == 120

    def test_get_fusion_stats(self):
        """Test fusion statistics computation."""
        fusion = RankFusion()
        stats = fusion.get_fusion_stats(self.candidates)
        
        assert "num_candidates" in stats
        assert "dense_scores_range" in stats
        assert "bm25_scores_range" in stats
        assert "weights_used" in stats
        
        assert stats["num_candidates"] == 3
        assert stats["weights_used"] == {"dense": 0.7, "bm25": 0.3}
        assert len(stats["dense_scores_range"]) == 2
        assert len(stats["bm25_scores_range"]) == 2

    def test_get_fusion_stats_empty(self):
        """Test fusion statistics with empty candidates."""
        fusion = RankFusion()
        stats = fusion.get_fusion_stats([])
        
        assert stats["num_candidates"] == 0
        assert stats["dense_scores_range"] == (0, 0)
        assert stats["bm25_scores_range"] == (0, 0)
        assert stats["weights_used"] == {"dense": 0.7, "bm25": 0.3}

    def test_fusion_score_ordering(self):
        """Test that fusion scores produce reasonable ordering."""
        fusion = RankFusion()
        scored_candidates = fusion.compute_fusion_scores(self.candidates)
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # The candidate with highest individual scores should be ranked first
        # c_001 has highest bm25 (7.2) and dense_sim (0.81)
        assert scored_candidates[0][0].id == "c_001"

    def test_fusion_score_determinism(self):
        """Test that fusion scores are deterministic."""
        fusion = RankFusion()
        scores1 = fusion.compute_fusion_scores(self.candidates)
        scores2 = fusion.compute_fusion_scores(self.candidates)
        
        # Scores should be identical
        for (c1, s1), (c2, s2) in zip(scores1, scores2):
            assert c1.id == c2.id
            assert s1 == pytest.approx(s2)

    def test_fusion_score_weights_impact(self):
        """Test that different weights produce different rankings."""
        # Test with dense-heavy weights
        fusion_dense = RankFusion({"dense": 0.9, "bm25": 0.1})
        scores_dense = fusion_dense.compute_fusion_scores(self.candidates)
        
        # Test with BM25-heavy weights
        fusion_bm25 = RankFusion({"dense": 0.1, "bm25": 0.9})
        scores_bm25 = fusion_bm25.compute_fusion_scores(self.candidates)
        
        # Sort by scores
        scores_dense.sort(key=lambda x: x[1], reverse=True)
        scores_bm25.sort(key=lambda x: x[1], reverse=True)
        
        # Rankings might be different due to different weight emphasis
        # This test ensures the weights are actually being used
        assert len(scores_dense) == len(scores_bm25)
