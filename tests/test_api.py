"""
Tests for FastAPI endpoints.
"""

import pytest
import json
from fastapi.testclient import TestClient
from context_compressor.api import app
from context_compressor.schemas import Candidate


class TestAPI:
    """Test FastAPI endpoints."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
        self.sample_candidates = [
            {
                "id": "c_001",
                "doc_id": "d_01",
                "section": "Results",
                "page": 14,
                "text": "The study found a 25% improvement in accuracy.",
                "tokens": 25,
                "bm25": 7.2,
                "dense_sim": 0.81,
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
            },
            {
                "id": "c_002",
                "doc_id": "d_01",
                "section": "Methods",
                "page": 8,
                "text": "We employed a randomized controlled trial design.",
                "tokens": 22,
                "bm25": 6.8,
                "dense_sim": 0.75,
                "embedding": [0.2, 0.3, 0.4, 0.5, 0.6]
            }
        ]

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["message"] == "Context Compressor API"

    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["compressor_ready"] is True

    def test_stats_endpoint(self):
        """Test stats endpoint."""
        response = self.client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "cache_size" in data
        assert "cache_ttl" in data

    def test_compress_endpoint_basic(self):
        """Test basic compression endpoint."""
        request_data = {
            "q": "What were the main findings?",
            "B": 50,
            "candidates": self.sample_candidates
        }
        
        response = self.client.post("/compress", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "context" in data
        assert "mapping" in data
        assert "stats" in data
        assert data["stats"]["budget"] == 50
        assert data["stats"]["used"] <= 50

    def test_compress_endpoint_with_params(self):
        """Test compression endpoint with custom parameters."""
        request_data = {
            "q": "Analyze the methodology",
            "B": 75,
            "candidates": self.sample_candidates,
            "params": {
                "fusion_weights": {"dense": 0.8, "bm25": 0.2},
                "lambda": 0.8,
                "section_cap": 3,
                "use_reranker": False,
                "embedding_model": "intfloat/e5-large-v2"
            }
        }
        
        response = self.client.post("/compress", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["stats"]["lambda"] == 0.8
        assert data["stats"]["fusion_weights"] == {"dense": 0.8, "bm25": 0.2}

    def test_compress_endpoint_empty_candidates(self):
        """Test compression endpoint with empty candidates."""
        request_data = {
            "q": "Test query",
            "B": 50,
            "candidates": []
        }
        
        response = self.client.post("/compress", json=request_data)
        assert response.status_code == 400
        assert "At least one candidate is required" in response.json()["detail"]

    def test_compress_endpoint_invalid_budget(self):
        """Test compression endpoint with invalid budget."""
        request_data = {
            "q": "Test query",
            "B": 0,  # Invalid budget
            "candidates": self.sample_candidates
        }
        
        response = self.client.post("/compress", json=request_data)
        assert response.status_code == 400
        assert "Budget must be positive" in response.json()["detail"]

    def test_compress_endpoint_missing_required_fields(self):
        """Test compression endpoint with missing required fields."""
        # Missing query
        request_data = {
            "B": 50,
            "candidates": self.sample_candidates
        }
        
        response = self.client.post("/compress", json=request_data)
        assert response.status_code == 422  # Validation error

        # Missing budget
        request_data = {
            "q": "Test query",
            "candidates": self.sample_candidates
        }
        
        response = self.client.post("/compress", json=request_data)
        assert response.status_code == 422

        # Missing candidates
        request_data = {
            "q": "Test query",
            "B": 50
        }
        
        response = self.client.post("/compress", json=request_data)
        assert response.status_code == 422

    def test_compress_batch_endpoint(self):
        """Test batch compression endpoint."""
        batch_requests = [
            {
                "q": "Query 1",
                "B": 50,
                "candidates": self.sample_candidates
            },
            {
                "q": "Query 2",
                "B": 60,
                "candidates": self.sample_candidates
            }
        ]
        
        response = self.client.post("/compress/batch", json=batch_requests)
        assert response.status_code == 200
        
        data = response.json()
        assert "responses" in data
        assert len(data["responses"]) == 2
        
        # Check each response
        for resp in data["responses"]:
            assert "context" in resp
            assert "mapping" in resp
            assert "stats" in resp

    def test_compress_batch_endpoint_with_errors(self):
        """Test batch compression endpoint with some failing requests."""
        batch_requests = [
            {
                "q": "Valid query",
                "B": 50,
                "candidates": self.sample_candidates
            },
            {
                "q": "Invalid query",
                "B": 0,  # Invalid budget
                "candidates": self.sample_candidates
            }
        ]
        
        response = self.client.post("/compress/batch", json=batch_requests)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["responses"]) == 2
        
        # First response should be valid
        assert "context" in data["responses"][0]
        
        # Second response should have error
        assert "error" in data["responses"][1]

    def test_clear_cache_endpoint(self):
        """Test cache clearing endpoint."""
        response = self.client.delete("/cache")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Cache cleared successfully"

    def test_compress_endpoint_caching(self):
        """Test that compression endpoint uses caching."""
        request_data = {
            "q": "Caching test query",
            "B": 50,
            "candidates": self.sample_candidates
        }
        
        # First request
        response1 = self.client.post("/compress", json=request_data)
        assert response1.status_code == 200
        
        # Second request (should use cache)
        response2 = self.client.post("/compress", json=request_data)
        assert response2.status_code == 200
        
        # Responses should be identical
        data1 = response1.json()
        data2 = response2.json()
        assert data1["context"] == data2["context"]
        assert data1["stats"]["used"] == data2["stats"]["used"]

    def test_compress_endpoint_no_cache(self):
        """Test compression endpoint with cache bypassed."""
        request_data = {
            "q": "No cache test query",
            "B": 50,
            "candidates": self.sample_candidates,
            "no_cache": True
        }
        
        # First request
        response1 = self.client.post("/compress", json=request_data)
        assert response1.status_code == 200
        
        # Second request (should not use cache)
        response2 = self.client.post("/compress", json=request_data)
        assert response2.status_code == 200
        
        # Responses should be identical (deterministic)
        data1 = response1.json()
        data2 = response2.json()
        assert data1["context"] == data2["context"]

    def test_compress_endpoint_large_request(self):
        """Test compression endpoint with large request."""
        # Create many candidates
        many_candidates = self.sample_candidates * 10  # 20 candidates
        
        request_data = {
            "q": "Large request test",
            "B": 100,
            "candidates": many_candidates
        }
        
        response = self.client.post("/compress", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["stats"]["used"] <= 100

    def test_compress_endpoint_timing_stats(self):
        """Test that timing statistics are included in response."""
        request_data = {
            "q": "Timing test query",
            "B": 50,
            "candidates": self.sample_candidates
        }
        
        response = self.client.post("/compress", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        stats = data["stats"]
        
        # Timing stats should be present
        assert "total_ms" in stats
        assert "fusion_ms" in stats
        assert "mmr_ms" in stats
        assert "trim_ms" in stats
        
        # Timing should be reasonable
        assert stats["total_ms"] >= 0
        assert stats["total_ms"] < 10000  # Less than 10 seconds

    def test_compress_endpoint_determinism(self):
        """Test that compression endpoint is deterministic."""
        request_data = {
            "q": "Determinism test query",
            "B": 50,
            "candidates": self.sample_candidates
        }
        
        # Multiple requests should produce identical results
        response1 = self.client.post("/compress", json=request_data)
        response2 = self.client.post("/compress", json=request_data)
        response3 = self.client.post("/compress", json=request_data)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response3.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        data3 = response3.json()
        
        assert data1["context"] == data2["context"]
        assert data2["context"] == data3["context"]
        assert data1["stats"]["used"] == data2["stats"]["used"]
        assert data2["stats"]["used"] == data3["stats"]["used"]

    def test_compress_endpoint_mapping_integrity(self):
        """Test that mapping preserves candidate information."""
        request_data = {
            "q": "Mapping integrity test",
            "B": 50,
            "candidates": self.sample_candidates
        }
        
        response = self.client.post("/compress", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        mapping = data["mapping"]
        
        # Each mapping item should have required fields
        for item in mapping:
            assert "id" in item
            assert "doc_id" in item
            assert "section" in item
            assert "page" in item
            assert "tokens" in item
            assert "trimmed" in item
            
            # Values should be valid
            assert item["tokens"] > 0
            assert isinstance(item["trimmed"], bool)
            
            # ID should correspond to original candidate
            candidate_ids = {c["id"] for c in self.sample_candidates}
            assert item["id"] in candidate_ids
