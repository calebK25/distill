"""
Pytest configuration and fixtures for Context Compressor tests.
"""

import pytest
import tempfile
import json
from context_compressor.schemas import Candidate


@pytest.fixture
def sample_candidates():
    """Provide sample candidates for testing."""
    return [
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


@pytest.fixture
def sample_candidates_file(sample_candidates):
    """Provide a temporary file with sample candidates."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        candidates_data = [c.dict() for c in sample_candidates]
        json.dump(candidates_data, f)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    import os
    os.unlink(temp_file)


@pytest.fixture
def large_candidate_set():
    """Provide a larger set of candidates for performance testing."""
    candidates = []
    for i in range(100):
        candidate = Candidate(
            id=f"c_{i:03d}",
            doc_id=f"d_{i // 10:02d}",
            section=f"Section_{i % 5}",
            page=i + 1,
            text=f"This is test candidate {i} with detailed information about the study findings and methodology.",
            tokens=20 + (i % 10),
            bm25=5.0 + (i % 5) * 0.5,
            dense_sim=0.5 + (i % 10) * 0.05,
            embedding=[0.1 + i * 0.01] * 5
        )
        candidates.append(candidate)
    return candidates


@pytest.fixture
def diverse_candidate_set():
    """Provide candidates from diverse sections for diversity testing."""
    return [
        Candidate(
            id="c_001",
            doc_id="d_01",
            section="Introduction",
            page=1,
            text="This study investigates the impact of machine learning on healthcare outcomes.",
            tokens=25,
            bm25=7.0,
            dense_sim=0.8,
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
        ),
        Candidate(
            id="c_002",
            doc_id="d_01",
            section="Methods",
            page=5,
            text="We used a randomized controlled trial design with 500 participants.",
            tokens=22,
            bm25=6.5,
            dense_sim=0.75,
            embedding=[0.2, 0.3, 0.4, 0.5, 0.6]
        ),
        Candidate(
            id="c_003",
            doc_id="d_01",
            section="Results",
            page=10,
            text="The results showed a 30% improvement in diagnostic accuracy.",
            tokens=20,
            bm25=7.2,
            dense_sim=0.85,
            embedding=[0.3, 0.4, 0.5, 0.6, 0.7]
        ),
        Candidate(
            id="c_004",
            doc_id="d_01",
            section="Discussion",
            page=15,
            text="These findings suggest broader applications in clinical practice.",
            tokens=18,
            bm25=6.0,
            dense_sim=0.7,
            embedding=[0.4, 0.5, 0.6, 0.7, 0.8]
        ),
        Candidate(
            id="c_005",
            doc_id="d_01",
            section="Conclusion",
            page=20,
            text="The study demonstrates the potential of AI in healthcare.",
            tokens=15,
            bm25=6.8,
            dense_sim=0.78,
            embedding=[0.5, 0.6, 0.7, 0.8, 0.9]
        )
    ]


@pytest.fixture
def test_queries():
    """Provide test queries for different scenarios."""
    return {
        "general": "What were the main findings?",
        "methodology": "How was the study conducted?",
        "results": "What results were obtained?",
        "implications": "What are the implications of this research?",
        "technical": "What technical methods were used?",
        "long": "Please provide a comprehensive analysis of the methodology, results, and implications of this research study"
    }


@pytest.fixture
def test_budgets():
    """Provide test budgets for different scenarios."""
    return {
        "small": 25,
        "medium": 50,
        "large": 100,
        "very_large": 200
    }


@pytest.fixture
def test_params():
    """Provide test parameter configurations."""
    return {
        "default": {},
        "dense_heavy": {"fusion_weights": {"dense": 0.9, "bm25": 0.1}},
        "bm25_heavy": {"fusion_weights": {"dense": 0.1, "bm25": 0.9}},
        "high_diversity": {"lambda_": 0.3, "section_cap": 3},
        "low_diversity": {"lambda_": 0.9, "section_cap": 1},
        "with_reranker": {"use_reranker": True},
        "custom_models": {
            "embedding_model": "gte-large",
            "reranker_model": "BAAI/bge-reranker-large"
        }
    }


# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Skip certain tests if dependencies are not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip tests based on dependencies."""
    skip_benchmark = pytest.mark.skip(reason="Benchmark tests require psutil")
    skip_reranker = pytest.mark.skip(reason="Reranker tests require sentence-transformers")
    
    for item in items:
        # Skip benchmark tests if psutil is not available
        if "benchmark" in item.keywords and "memory" in item.name:
            try:
                import psutil
            except ImportError:
                item.add_marker(skip_benchmark)
        
        # Skip reranker tests if sentence-transformers is not available
        if "reranker" in item.name:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                item.add_marker(skip_reranker)
