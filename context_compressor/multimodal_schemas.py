#!/usr/bin/env python3
"""
Multimodal schemas for Context Compressor.
Supports text, images, and tables as first-class modalities.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from enum import Enum


class ModalityType(str, Enum):
    """Supported modalities."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x0: float = Field(..., description="Left coordinate")
    y0: float = Field(..., description="Top coordinate")
    x1: float = Field(..., description="Right coordinate")
    y1: float = Field(..., description="Bottom coordinate")


class ImageReference(BaseModel):
    """Reference to an image in the document."""
    doc_id: str = Field(..., description="Document ID")
    page: int = Field(..., description="Page number")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    image_path: Optional[str] = Field(None, description="Path to cropped image file")


class MultimodalCandidate(BaseModel):
    """Enhanced candidate with multimodal support."""
    id: str = Field(..., description="Unique candidate ID")
    doc_id: str = Field(..., description="Document ID")
    modality: ModalityType = Field(..., description="Content modality")
    section: str = Field(..., description="Document section")
    page: int = Field(..., description="Page number")
    
    # Content fields
    text: str = Field(..., description="Text content (for text/table) or caption (for image)")
    tokens: int = Field(..., description="Token count")
    
    # Modality-specific fields
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box for images/tables")
    image_ref: Optional[ImageReference] = Field(None, description="Image reference for images")
    table_md: Optional[str] = Field(None, description="Markdown table representation")
    caption: Optional[str] = Field(None, description="Figure caption")
    caption_generated: bool = Field(False, description="Whether caption was AI-generated")
    
    # Scoring fields
    bm25: float = Field(..., description="BM25 score")
    dense_sim: float = Field(..., description="Dense similarity score")
    image_sim: Optional[float] = Field(None, description="Image similarity score (for images)")
    fusion_score: Optional[float] = Field(None, description="Combined fusion score")
    
    # Embeddings
    text_embedding: Optional[List[float]] = Field(None, description="Text embedding vector")
    image_embedding: Optional[List[float]] = Field(None, description="Image embedding vector")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MultimodalCompressionRequest(BaseModel):
    """Request for multimodal compression."""
    q: str = Field(..., description="Query string")
    B: int = Field(..., description="Token budget")
    candidates: List[MultimodalCandidate] = Field(..., description="Multimodal candidates")
    
    # Multimodal parameters
    enable_image_search: bool = Field(True, description="Enable image retrieval")
    enable_table_search: bool = Field(True, description="Enable table retrieval")
    max_images: int = Field(2, description="Maximum images in output")
    max_tables: int = Field(3, description="Maximum tables in output")
    
    # Fusion weights
    text_weight: float = Field(0.6, description="Weight for text similarity")
    bm25_weight: float = Field(0.2, description="Weight for BM25")
    image_weight: float = Field(0.2, description="Weight for image similarity")
    
    # Other parameters
    lambda_: float = Field(0.7, description="MMR diversity parameter", alias="lambda")
    section_cap: int = Field(2, description="Maximum chunks per section")
    use_reranker: bool = Field(False, description="Use cross-encoder reranker")
    no_cache: bool = Field(False, description="Bypass caching")
    
    class Config:
        populate_by_name = True


class MultimodalCompressionResponse(BaseModel):
    """Response from multimodal compression."""
    context: str = Field(..., description="Compressed multimodal context")
    mapping: List[MultimodalCandidate] = Field(..., description="Selected candidates")
    
    # Statistics
    used_tokens: int = Field(..., description="Tokens used")
    budget: int = Field(..., description="Token budget")
    saved_vs_pool: int = Field(..., description="Tokens saved vs original pool")
    total_ms: float = Field(..., description="Processing time in milliseconds")
    low_context: bool = Field(..., description="Low context flag")
    
    # Modality breakdown
    text_chunks: int = Field(..., description="Number of text chunks")
    image_chunks: int = Field(..., description="Number of image chunks")
    table_chunks: int = Field(..., description="Number of table chunks")
    
    # Parameters used
    lambda_: float = Field(..., description="MMR diversity parameter", alias="lambda")
    fusion_weights: Dict[str, float] = Field(..., description="Fusion weights used")
    section_cap: int = Field(..., description="Section cap used")
    
    class Config:
        populate_by_name = True


class QueryClassifier(BaseModel):
    """Query classification for modality routing."""
    has_image_terms: bool = Field(False, description="Query mentions images/figures")
    has_table_terms: bool = Field(False, description="Query mentions tables/data")
    has_numeric_terms: bool = Field(False, description="Query mentions numbers/values")
    primary_modality: ModalityType = Field(ModalityType.TEXT, description="Primary modality")
    confidence: float = Field(1.0, description="Classification confidence")
