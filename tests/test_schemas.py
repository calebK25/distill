"""
Unit tests for Pydantic schemas.
"""

import pytest
import numpy as np
from context_compressor.schemas import (
    Candidate,
    CompressionParams,
    CompressionRequest,
    MappingItem,
    CompressionStats,
    CompressionResponse,
)


class TestCandidate:
    """Test Candidate schema validation."""

    def test_valid_candidate(self):
        """Test valid candidate creation."""
        candidate = Candidate(
            id="c_001",
            doc_id="d_01",
            section="Results",
            page=14,
            text="The study found significant improvements.",
            tokens=25,
            bm25=7.2,
            dense_sim=0.81,
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        assert candidate.id == "c_001"
        assert candidate.tokens == 25
        assert candidate.bm25 == 7.2
        assert candidate.dense_sim == 0.81

    def test_invalid_tokens(self):
        """Test validation of token count."""
        with pytest.raises(ValueError, match="tokens must be positive"):
            Candidate(
                id="c_001",
                doc_id="d_01",
                section="Results",
                page=14,
                text="Test text",
                tokens=0,  # Invalid
                bm25=7.2,
                dense_sim=0.81
            )

    def test_invalid_scores(self):
        """Test validation of BM25 and dense similarity scores."""
        with pytest.raises(ValueError, match="scores must be valid numbers"):
            Candidate(
                id="c_001",
                doc_id="d_01",
                section="Results",
                page=14,
                text="Test text",
                tokens=25,
                bm25=np.nan,  # Invalid
                dense_sim=0.81
            )

    def test_invalid_embedding(self):
        """Test validation of embedding vector."""
        with pytest.raises(ValueError, match="embedding must be a non-empty list"):
            Candidate(
                id="c_001",
                doc_id="d_01",
                section="Results",
                page=14,
                text="Test text",
                tokens=25,
                bm25=7.2,
                dense_sim=0.81,
                embedding=[]  # Invalid empty list
            )


class TestCompressionParams:
    """Test CompressionParams schema validation."""

    def test_valid_params(self):
        """Test valid parameters creation."""
        params = CompressionParams(
            fusion_weights={"dense": 0.7, "bm25": 0.3},
            lambda_=0.7,
            section_cap=2,
            use_reranker=False,
            embedding_model="intfloat/e5-large-v2",
            reranker_model="BAAI/bge-reranker-large"
        )
        assert params.lambda_ == 0.7
        assert params.section_cap == 2
        assert params.use_reranker is False

    def test_default_params(self):
        """Test default parameter values."""
        params = CompressionParams()
        assert params.fusion_weights == {"dense": 0.7, "bm25": 0.3}
        assert params.lambda_ == 0.7
        assert params.section_cap == 2
        assert params.use_reranker is False
        assert params.embedding_model == "intfloat/e5-large-v2"
        assert params.reranker_model == "BAAI/bge-reranker-large"

    def test_invalid_lambda(self):
        """Test lambda parameter validation."""
        with pytest.raises(ValueError):
            CompressionParams(lambda_=1.5)  # > 1.0

    def test_invalid_section_cap(self):
        """Test section cap validation."""
        with pytest.raises(ValueError):
            CompressionParams(section_cap=0)  # < 1

    def test_invalid_fusion_weights(self):
        """Test fusion weights validation."""
        with pytest.raises(ValueError, match="fusion_weights must contain keys"):
            CompressionParams(fusion_weights={"dense": 0.7})  # Missing bm25

        with pytest.raises(ValueError, match="fusion weights must be between"):
            CompressionParams(fusion_weights={"dense": 1.5, "bm25": 0.3})  # > 1.0


class TestCompressionRequest:
    """Test CompressionRequest schema validation."""

    def test_valid_request(self):
        """Test valid request creation."""
        candidates = [
            Candidate(
                id="c_001",
                doc_id="d_01",
                section="Results",
                page=14,
                text="Test text",
                tokens=25,
                bm25=7.2,
                dense_sim=0.81
            )
        ]
        
        request = CompressionRequest(
            q="What were the findings?",
            B=100,
            candidates=candidates
        )
        assert request.q == "What were the findings?"
        assert request.B == 100
        assert len(request.candidates) == 1

    def test_empty_candidates(self):
        """Test validation of empty candidates list."""
        with pytest.raises(ValueError, match="candidates list cannot be empty"):
            CompressionRequest(
                q="Test query",
                B=100,
                candidates=[]
            )

    def test_invalid_budget(self):
        """Test budget validation."""
        candidates = [
            Candidate(
                id="c_001",
                doc_id="d_01",
                section="Results",
                page=14,
                text="Test text",
                tokens=25,
                bm25=7.2,
                dense_sim=0.81
            )
        ]
        
        with pytest.raises(ValueError):
            CompressionRequest(
                q="Test query",
                B=0,  # Invalid
                candidates=candidates
            )


class TestMappingItem:
    """Test MappingItem schema."""

    def test_valid_mapping_item(self):
        """Test valid mapping item creation."""
        mapping = MappingItem(
            id="c_001",
            doc_id="d_01",
            section="Results",
            page=14,
            tokens=20,
            trimmed=True
        )
        assert mapping.id == "c_001"
        assert mapping.tokens == 20
        assert mapping.trimmed is True


class TestCompressionStats:
    """Test CompressionStats schema."""

    def test_valid_stats(self):
        """Test valid stats creation."""
        stats = CompressionStats(
            budget=100,
            used=80,
            saved_vs_pool=200,
            lambda_=0.7,
            fusion_weights={"dense": 0.7, "bm25": 0.3},
            section_cap=2,
            low_context=False,
            fusion_ms=5.0,
            mmr_ms=10.0,
            trim_ms=15.0,
            total_ms=30.0
        )
        assert stats.budget == 100
        assert stats.used == 80
        assert stats.saved_vs_pool == 200
        assert stats.low_context is False
        assert stats.total_ms == 30.0


class TestCompressionResponse:
    """Test CompressionResponse schema."""

    def test_valid_response(self):
        """Test valid response creation."""
        mapping = [
            MappingItem(
                id="c_001",
                doc_id="d_01",
                section="Results",
                page=14,
                tokens=20,
                trimmed=True
            )
        ]
        
        stats = CompressionStats(
            budget=100,
            used=80,
            saved_vs_pool=200,
            lambda_=0.7,
            fusion_weights={"dense": 0.7, "bm25": 0.3},
            section_cap=2,
            low_context=False
        )
        
        response = CompressionResponse(
            context="Compressed context text",
            mapping=mapping,
            stats=stats
        )
        assert response.context == "Compressed context text"
        assert len(response.mapping) == 1
        assert response.stats.budget == 100
