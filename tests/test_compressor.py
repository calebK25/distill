"""
Integration tests for ContextCompressor.
"""

import pytest
import time
from context_compressor.compressor import ContextCompressor
from context_compressor.schemas import (
    Candidate,
    CompressionRequest,
    CompressionParams
)


class TestContextCompressor:
    """Test the main ContextCompressor class."""

    def setup_method(self):
        """Set up test data."""
        self.candidates = [
            Candidate(
                id="c_001",
                doc_id="d_01",
                section="Results",
                page=14,
                text="The study found a 25% improvement in accuracy. The results were consistent across all test conditions.",
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
                text="We employed a randomized controlled trial design with 200 participants. The intervention group received the new treatment protocol.",
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
                text="These findings suggest that the proposed approach may have broader applications. Future research should explore scalability.",
                tokens=18,
                bm25=5.9,
                dense_sim=0.68,
                embedding=[0.3, 0.4, 0.5, 0.6, 0.7]
            )
        ]
        
        self.compressor = ContextCompressor()

    def test_compressor_init(self):
        """Test compressor initialization."""
        compressor = ContextCompressor()
        assert compressor.cache == {}
        assert compressor.cache_ttl == 3600

    def test_basic_compression(self):
        """Test basic compression functionality."""
        request = CompressionRequest(
            q="What were the main findings?",
            B=50,
            candidates=self.candidates
        )
        
        response = self.compressor.compress(request)
        
        # Basic response validation
        assert response.context is not None
        assert len(response.context) > 0
        assert len(response.mapping) > 0
        assert response.stats.budget == 50
        assert response.stats.used <= 50  # Should not exceed budget
        assert response.stats.saved_vs_pool >= 0

    def test_compression_with_custom_params(self):
        """Test compression with custom parameters."""
        params = CompressionParams(
            fusion_weights={"dense": 0.8, "bm25": 0.2},
            lambda_=0.8,
            section_cap=3,
            use_reranker=False,
            embedding_model="intfloat/e5-large-v2"
        )
        
        request = CompressionRequest(
            q="Analyze the methodology",
            B=75,
            candidates=self.candidates,
            params=params
        )
        
        response = self.compressor.compress(request)
        
        assert response.stats.lambda_ == 0.8
        assert response.stats.fusion_weights == {"dense": 0.8, "bm25": 0.2}
        assert response.stats.section_cap == 3

    def test_compression_budget_constraint(self):
        """Test that compression respects budget constraints."""
        request = CompressionRequest(
            q="Summarize the study",
            B=30,
            candidates=self.candidates
        )
        
        response = self.compressor.compress(request)
        
        # Should not exceed budget
        assert response.stats.used <= 30
        assert response.stats.budget == 30

    def test_compression_with_reranker(self):
        """Test compression with reranker enabled."""
        params = CompressionParams(use_reranker=True)
        
        request = CompressionRequest(
            q="What were the results?",
            B=60,
            candidates=self.candidates,
            params=params
        )
        
        response = self.compressor.compress(request)
        
        # Should complete successfully (even if reranker fails)
        assert response.context is not None
        assert response.stats.used <= 60

    def test_compression_caching(self):
        """Test compression caching functionality."""
        request = CompressionRequest(
            q="Test query for caching",
            B=40,
            candidates=self.candidates
        )
        
        # First request
        response1 = self.compressor.compress(request)
        
        # Second request (should use cache)
        response2 = self.compressor.compress(request)
        
        # Responses should be identical
        assert response1.context == response2.context
        assert response1.stats.used == response2.stats.used

    def test_compression_no_cache(self):
        """Test compression with cache bypassed."""
        request = CompressionRequest(
            q="Test query no cache",
            B=40,
            candidates=self.candidates,
            no_cache=True
        )
        
        # First request
        response1 = self.compressor.compress(request)
        
        # Second request (should not use cache)
        response2 = self.compressor.compress(request)
        
        # Responses should be identical (deterministic)
        assert response1.context == response2.context

    def test_compression_empty_candidates(self):
        """Test compression with empty candidates list."""
        with pytest.raises(ValueError, match="candidates list cannot be empty"):
            request = CompressionRequest(
                q="Test query",
                B=50,
                candidates=[]
            )

    def test_compression_low_context_detection(self):
        """Test low context detection."""
        # Use a very small budget to trigger low context
        request = CompressionRequest(
            q="Test query",
            B=5,  # Very small budget
            candidates=self.candidates
        )
        
        response = self.compressor.compress(request)
        
        # Should detect low context usage
        assert response.stats.low_context is True

    def test_compression_timing_stats(self):
        """Test that timing statistics are included."""
        request = CompressionRequest(
            q="Test timing",
            B=50,
            candidates=self.candidates
        )
        
        response = self.compressor.compress(request)
        
        # Timing stats should be present
        assert response.stats.total_ms is not None
        assert response.stats.total_ms >= 0
        assert response.stats.fusion_ms is not None
        assert response.stats.mmr_ms is not None
        assert response.stats.trim_ms is not None

    def test_compressor_stats(self):
        """Test compressor statistics."""
        stats = self.compressor.get_stats()
        
        assert "cache_size" in stats
        assert "cache_ttl" in stats
        assert stats["cache_size"] == 0  # Initially empty
        assert stats["cache_ttl"] == 3600

    def test_compressor_clear_cache(self):
        """Test cache clearing functionality."""
        # Make a request to populate cache
        request = CompressionRequest(
            q="Test cache clear",
            B=40,
            candidates=self.candidates
        )
        
        self.compressor.compress(request)
        
        # Cache should have entries
        assert self.compressor.get_stats()["cache_size"] > 0
        
        # Clear cache
        self.compressor.clear_cache()
        
        # Cache should be empty
        assert self.compressor.get_stats()["cache_size"] == 0

    def test_compression_determinism(self):
        """Test that compression is deterministic."""
        request = CompressionRequest(
            q="Determinism test",
            B=50,
            candidates=self.candidates
        )
        
        # Multiple runs should produce identical results
        response1 = self.compressor.compress(request)
        response2 = self.compressor.compress(request)
        response3 = self.compressor.compress(request)
        
        assert response1.context == response2.context
        assert response2.context == response3.context
        assert response1.stats.used == response2.stats.used
        assert response2.stats.used == response3.stats.used

    def test_compression_mapping_integrity(self):
        """Test that mapping preserves candidate information."""
        request = CompressionRequest(
            q="Test mapping",
            B=50,
            candidates=self.candidates
        )
        
        response = self.compressor.compress(request)
        
        # Each mapping item should correspond to a candidate
        candidate_ids = {c.id for c in self.candidates}
        mapping_ids = {m.id for m in response.mapping}
        
        # All mapping IDs should be from original candidates
        assert mapping_ids.issubset(candidate_ids)
        
        # Check mapping item structure
        for mapping_item in response.mapping:
            assert mapping_item.id in candidate_ids
            assert mapping_item.tokens > 0
            assert isinstance(mapping_item.trimmed, bool)

    @pytest.mark.slow
    def test_compression_performance(self):
        """Test compression performance (marked as slow)."""
        # Create larger candidate set
        large_candidates = self.candidates * 20  # 60 candidates
        
        request = CompressionRequest(
            q="Performance test query",
            B=100,
            candidates=large_candidates
        )
        
        start_time = time.time()
        response = self.compressor.compress(request)
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 1000  # 1 second
        assert response.stats.used <= 100
        assert response.stats.total_ms < 1000

    def test_compression_section_diversity(self):
        """Test that compression maintains section diversity."""
        # Create candidates from different sections
        diverse_candidates = [
            Candidate(
                id="c_001",
                doc_id="d_01",
                section="Introduction",
                page=1,
                text="This study investigates...",
                tokens=20,
                bm25=7.0,
                dense_sim=0.8
            ),
            Candidate(
                id="c_002",
                doc_id="d_01",
                section="Methods",
                page=5,
                text="We used a randomized design...",
                tokens=25,
                bm25=6.5,
                dense_sim=0.75
            ),
            Candidate(
                id="c_003",
                doc_id="d_01",
                section="Results",
                page=10,
                text="The results showed...",
                tokens=22,
                bm25=7.2,
                dense_sim=0.85
            ),
            Candidate(
                id="c_004",
                doc_id="d_01",
                section="Discussion",
                page=15,
                text="These findings suggest...",
                tokens=18,
                bm25=6.0,
                dense_sim=0.7
            )
        ]
        
        request = CompressionRequest(
            q="Summarize the study",
            B=80,
            candidates=diverse_candidates,
            params=CompressionParams(section_cap=2)
        )
        
        response = self.compressor.compress(request)
        
        # Should have candidates from multiple sections
        sections_in_output = {m.section for m in response.mapping}
        assert len(sections_in_output) > 1
