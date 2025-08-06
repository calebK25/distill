"""
Context Compressor - Intelligent context compression for LLM applications.
"""

__version__ = "0.1.0"
__author__ = "Context Compressor Team"

# Core modules
from .schemas import (
    Candidate, CompressionRequest, CompressionParams, 
    CompressionResponse, MappingItem, CompressionStats
)
from .compressor import ContextCompressor
from .fusion import RankFusion
from .mmr import MMRSelector
from .trimming import SentenceTrimmer
from .embeddings import EmbeddingManager
from .reranker import Reranker
from .utils import (
    count_tokens, z_score_normalize, cosine_similarity,
    split_sentences, extract_anchors, compute_hash,
    timing_decorator, check_low_context, validate_budget_constraint,
    stable_sort_with_tie_breaker
)

# Multimodal modules
from .multimodal_schemas import (
    ModalityType, BoundingBox, ImageReference, MultimodalCandidate,
    MultimodalCompressionRequest, MultimodalCompressionResponse, QueryClassifier
)
from .multimodal_extractor import MultimodalExtractor
from .multimodal_compressor import MultimodalCompressor

# Enhanced multimodal modules
from .image_encoder import ImageEncoder
from .table_extractor import TableExtractor
from .image_captioner import ImageCaptioner
from .vector_store import VectorStore, VectorType, VectorRecord
from .enhanced_multimodal_compressor import EnhancedMultimodalCompressor

# API and CLI
try:
    from .api import app
except ImportError:
    app = None

try:
    from .cli import main as cli_main
except ImportError:
    cli_main = None

__all__ = [
    # Core
    "ContextCompressor", "CompressionRequest", "CompressionParams",
    "CompressionResponse", "Candidate", "MappingItem", "CompressionStats",
    "RankFusion", "MMRSelector", "SentenceTrimmer", "EmbeddingManager", "Reranker",
    
    # Utils
    "count_tokens", "z_score_normalize", "cosine_similarity",
    "split_sentences", "extract_anchors", "compute_hash",
    "timing_decorator", "check_low_context", "validate_budget_constraint",
    "stable_sort_with_tie_breaker",
    
    # Multimodal
    "ModalityType", "BoundingBox", "ImageReference", "MultimodalCandidate",
    "MultimodalCompressionRequest", "MultimodalCompressionResponse", "QueryClassifier",
    "MultimodalExtractor", "MultimodalCompressor",
    
    # Enhanced Multimodal
    "ImageEncoder", "TableExtractor", "ImageCaptioner", "VectorStore", "VectorType", "VectorRecord", "EnhancedMultimodalCompressor",
    
    # API/CLI
    "app", "cli_main"
]
