"""
Pydantic schemas for Context Compressor data models.
"""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
import numpy as np


class Candidate(BaseModel):
    """A candidate passage for compression."""
    id: str = Field(..., description="Unique candidate identifier")
    doc_id: str = Field(..., description="Document identifier")
    section: str = Field(..., description="Section name (e.g., 'Results', 'Methods')")
    page: int = Field(..., description="Page number")
    text: str = Field(..., description="Passage text content")
    tokens: int = Field(..., description="Token count of the passage")
    bm25: float = Field(..., description="BM25 relevance score")
    dense_sim: float = Field(..., description="Dense similarity score")
    embedding: Optional[List[float]] = Field(None, description="Embedding vector")

    @validator('tokens')
    def tokens_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('tokens must be positive')
        return v

    @validator('bm25', 'dense_sim')
    def scores_must_be_valid(cls, v):
        if not isinstance(v, (int, float)) or np.isnan(v):
            raise ValueError('scores must be valid numbers')
        return float(v)

    @validator('embedding')
    def embedding_must_be_valid(cls, v):
        if v is not None:
            if not isinstance(v, list) or len(v) == 0:
                raise ValueError('embedding must be a non-empty list')
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError('embedding values must be numbers')
        return v


class CompressionParams(BaseModel):
    """Parameters for context compression."""
    fusion_weights: Dict[str, float] = Field(
        default={"dense": 0.7, "bm25": 0.3},
        description="Weights for rank fusion (dense and BM25)"
    )
    lambda_: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        alias="lambda",
        description="MMR diversity parameter"
    )
    section_cap: int = Field(
        default=2,
        ge=1,
        description="Maximum chunks per section"
    )
    doc_cap: int = Field(
        default=6,
        ge=1,
        description="Maximum chunks per document (CDC mode)"
    )
    topM: int = Field(
        default=200,
        ge=1,
        description="Top-M candidates for MMR pool after fusion"
    )
    auto_router: bool = Field(
        default=True,
        description="Automatically route between CDC and SDM modes"
    )
    use_reranker: bool = Field(
        default=False,
        description="Whether to use cross-encoder reranker"
    )
    embedding_model: str = Field(
        default="intfloat/e5-large-v2",
        description="Embedding model to use (e.g., 'intfloat/e5-large-v2', 'gte-large')"
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-large",
        description="Reranker model to use"
    )

    @validator('fusion_weights')
    def validate_fusion_weights(cls, v):
        required_keys = {'dense', 'bm25'}
        if not all(key in v for key in required_keys):
            raise ValueError(f'fusion_weights must contain keys: {required_keys}')
        if not all(0.0 <= weight <= 1.0 for weight in v.values()):
            raise ValueError('fusion weights must be between 0.0 and 1.0')
        return v
    
    class Config:
        populate_by_name = True


class CompressionRequest(BaseModel):
    """Request for context compression."""
    q: str = Field(..., description="Query string")
    B: int = Field(..., gt=0, description="Token budget")
    candidates: List[Candidate] = Field(..., description="Candidate passages")
    params: CompressionParams = Field(
        default_factory=CompressionParams,
        description="Compression parameters"
    )
    no_cache: bool = Field(
        default=False,
        description="Bypass caching"
    )

    @validator('candidates')
    def candidates_must_not_be_empty(cls, v):
        if len(v) == 0:
            raise ValueError('candidates list cannot be empty')
        return v


class MappingItem(BaseModel):
    """Mapping information for a compressed passage."""
    id: str = Field(..., description="Original candidate ID")
    doc_id: str = Field(..., description="Document ID")
    section: str = Field(..., description="Section name")
    page: int = Field(..., description="Page number")
    tokens: int = Field(..., description="Final token count")
    trimmed: bool = Field(..., description="Whether passage was trimmed")


class RouterStats(BaseModel):
    """Statistics about mode routing."""
    top1_doc_frac: float = Field(..., description="Fraction of top candidates from single document")
    entropy: float = Field(..., description="Entropy of document distribution")


class CompressionStats(BaseModel):
    """Statistics about the compression process."""
    mode: str = Field(..., description="Compression mode: 'cross_doc' or 'single_doc'")
    budget: int = Field(..., description="Token budget")
    used: int = Field(..., description="Tokens actually used")
    saved_vs_pool: int = Field(..., description="Tokens saved vs original pool")
    lambda_: float = Field(..., alias="lambda", description="MMR lambda parameter used")
    fusion_weights: Dict[str, float] = Field(..., description="Fusion weights used")
    section_cap: int = Field(..., description="Section cap used")
    doc_cap: int = Field(..., description="Document cap used")
    router_score: Optional[RouterStats] = Field(None, description="Router statistics")
    low_context: bool = Field(..., description="Whether context usage is low (<30%)")
    
    # Timing information
    fusion_ms: Optional[float] = Field(None, description="Fusion step time in ms")
    mmr_ms: Optional[float] = Field(None, description="MMR step time in ms")
    trim_ms: Optional[float] = Field(None, description="Trimming step time in ms")
    total_ms: Optional[float] = Field(None, description="Total processing time in ms")
    
    class Config:
        populate_by_name = True


class CompressionResponse(BaseModel):
    """Response from context compression."""
    context: str = Field(..., description="Compressed context")
    mapping: List[MappingItem] = Field(..., description="Mapping information")
    stats: CompressionStats = Field(..., description="Compression statistics")
