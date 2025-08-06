#!/usr/bin/env python3
"""
Multimodal Context Compressor.
Handles text, images, and tables with proper fusion and selection.
"""

import time
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

from .multimodal_schemas import (
    MultimodalCandidate, ModalityType, MultimodalCompressionRequest,
    MultimodalCompressionResponse, QueryClassifier
)
from .utils import z_score_normalize, check_low_context, count_tokens


class MultimodalCompressor:
    """Multimodal context compressor with support for text, images, and tables."""
    
    def __init__(self, 
                 text_model: str = "BAAI/bge-large-en-v1.5",
                 image_model: str = "openai/clip-vit-large-patch14",
                 device: str = "auto"):
        """
        Initialize the multimodal compressor.
        
        Args:
            text_model: Text embedding model
            image_model: Image embedding model
            device: Device to use for models
        """
        self.text_model = text_model
        self.image_model = image_model
        self.device = device
        
        # Initialize models
        self.text_encoder = None
        self.image_encoder = None
        self._load_models()
    
    def _load_models(self):
        """Load text and image embedding models."""
        try:
            print(f"Loading text model: {self.text_model}")
            # Fix device parameter - use "cuda" or "cpu" instead of "auto"
            device = "cuda" if self.device == "auto" else self.device
            self.text_encoder = SentenceTransformer(self.text_model, device=device)
            print(f"✓ Loaded text model")
        except Exception as e:
            print(f"✗ Failed to load text model: {e}")
            self.text_encoder = None
        
        # Note: Image encoder would be loaded here if needed
        # For now, we'll use text embeddings for image captions
        self.image_encoder = None
    
    def compress(self, request: MultimodalCompressionRequest) -> MultimodalCompressionResponse:
        """
        Compress multimodal context.
        
        Args:
            request: Multimodal compression request
            
        Returns:
            Multimodal compression response
        """
        start_time = time.time()
        
        # Step 1: Classify query
        query_classifier = self._classify_query(request.q)
        
        # Step 2: Score candidates
        scored_candidates = self._score_candidates(request.q, request.candidates, query_classifier)
        
        # Step 3: Apply modality-specific filtering
        filtered_candidates = self._filter_by_modality(
            scored_candidates, 
            request, 
            query_classifier
        )
        
        # Step 4: Apply fusion and ranking
        ranked_candidates = self._fuse_and_rank(
            filtered_candidates, 
            request.text_weight,
            request.bm25_weight,
            request.image_weight
        )
        
        # Step 5: Select diverse candidates
        selected_candidates = self._select_diverse_candidates(
            ranked_candidates,
            request.B,
            request.lambda_,
            request.section_cap,
            request.max_images,
            request.max_tables
        )
        
        # Step 6: Build context
        context = self._build_context(selected_candidates)
        
        # Step 7: Calculate statistics
        used_tokens = sum(c.tokens for c in selected_candidates)
        total_tokens = sum(c.tokens for c in request.candidates)
        saved_tokens = total_tokens - used_tokens
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Count modality breakdown
        text_chunks = sum(1 for c in selected_candidates if c.modality == ModalityType.TEXT)
        image_chunks = sum(1 for c in selected_candidates if c.modality == ModalityType.IMAGE)
        table_chunks = sum(1 for c in selected_candidates if c.modality == ModalityType.TABLE)
        
        return MultimodalCompressionResponse(
            context=context,
            mapping=selected_candidates,
            used_tokens=used_tokens,
            budget=request.B,
            saved_vs_pool=saved_tokens,
            total_ms=processing_time,
            low_context=check_low_context(used_tokens, request.B),
            text_chunks=text_chunks,
            image_chunks=image_chunks,
            table_chunks=table_chunks,
            lambda_=request.lambda_,
            fusion_weights={
                "text": request.text_weight,
                "bm25": request.bm25_weight,
                "image": request.image_weight
            },
            section_cap=request.section_cap
        )
    
    def _classify_query(self, query: str) -> QueryClassifier:
        """Classify query to determine modality preferences."""
        query_lower = query.lower()
        
        # Check for image terms
        image_terms = ['figure', 'image', 'diagram', 'chart', 'plot', 'see figure', 'visual']
        has_image_terms = any(term in query_lower for term in image_terms)
        
        # Check for table terms
        table_terms = ['table', 'data', 'value', '%', '$', 'increase', 'decrease', 'statistics']
        has_table_terms = any(term in query_lower for term in table_terms)
        
        # Check for numeric terms
        has_numeric_terms = bool(re.search(r'\d+', query))
        
        # Determine primary modality
        if has_image_terms:
            primary_modality = ModalityType.IMAGE
        elif has_table_terms:
            primary_modality = ModalityType.TABLE
        else:
            primary_modality = ModalityType.TEXT
        
        return QueryClassifier(
            has_image_terms=has_image_terms,
            has_table_terms=has_table_terms,
            has_numeric_terms=has_numeric_terms,
            primary_modality=primary_modality,
            confidence=0.8 if (has_image_terms or has_table_terms) else 1.0
        )
    
    def _score_candidates(self, 
                         query: str, 
                         candidates: List[MultimodalCandidate],
                         query_classifier: QueryClassifier) -> List[MultimodalCandidate]:
        """Score candidates using BM25 and dense similarity."""
        if not candidates:
            return []
        
        # Prepare texts for scoring
        texts = [c.text for c in candidates]
        
        # BM25 scoring
        tokenized_texts = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized_texts)
        bm25_scores = bm25.get_scores(query.lower().split())
        
        # Dense similarity scoring
        dense_scores = []
        if self.text_encoder:
            try:
                # Encode query and texts
                query_embedding = self.text_encoder.encode([query])
                text_embeddings = self.text_encoder.encode(texts)
                
                # Compute similarities
                similarities = util.cos_sim(query_embedding, text_embeddings)[0]
                dense_scores = similarities.cpu().numpy()
                
                # Store embeddings
                for i, candidate in enumerate(candidates):
                    candidate.text_embedding = text_embeddings[i].tolist()
                    candidate.dense_sim = float(dense_scores[i])
                    candidate.bm25 = float(bm25_scores[i])
                    
            except Exception as e:
                print(f"Error computing dense similarities: {e}")
                # Fallback to BM25 only
                for i, candidate in enumerate(candidates):
                    candidate.dense_sim = 0.0
                    candidate.bm25 = float(bm25_scores[i])
        else:
            # No text encoder available
            for i, candidate in enumerate(candidates):
                candidate.dense_sim = 0.0
                candidate.bm25 = float(bm25_scores[i])
        
        return candidates
    
    def _filter_by_modality(self, 
                           candidates: List[MultimodalCandidate],
                           request: MultimodalCompressionRequest,
                           query_classifier: QueryClassifier) -> List[MultimodalCandidate]:
        """Filter candidates based on modality preferences."""
        filtered = []
        
        for candidate in candidates:
            # Apply modality-specific filtering
            if candidate.modality == ModalityType.IMAGE:
                if not request.enable_image_search:
                    continue
                if query_classifier.has_image_terms:
                    # Boost image candidates when query mentions images
                    candidate.dense_sim *= 1.5
                    
            elif candidate.modality == ModalityType.TABLE:
                if not request.enable_table_search:
                    continue
                if query_classifier.has_table_terms or query_classifier.has_numeric_terms:
                    # Boost table candidates when query mentions tables/numbers
                    candidate.dense_sim *= 1.3
            
            filtered.append(candidate)
        
        return filtered
    
    def _fuse_and_rank(self, 
                       candidates: List[MultimodalCandidate],
                       text_weight: float,
                       bm25_weight: float,
                       image_weight: float) -> List[MultimodalCandidate]:
        """Fuse scores and rank candidates."""
        if not candidates:
            return []
        
        # Extract scores
        dense_scores = [c.dense_sim for c in candidates]
        bm25_scores = [c.bm25 for c in candidates]
        
        # Normalize scores
        dense_scores_norm = z_score_normalize(dense_scores)
        bm25_scores_norm = z_score_normalize(bm25_scores)
        
        # Compute fusion scores
        for i, candidate in enumerate(candidates):
            fusion_score = (
                text_weight * dense_scores_norm[i] +
                bm25_weight * bm25_scores_norm[i]
            )
            
            # Add image weight if applicable
            if candidate.modality == ModalityType.IMAGE and candidate.image_sim is not None:
                fusion_score += image_weight * candidate.image_sim
            
            candidate.fusion_score = fusion_score
        
        # Sort by fusion score
        candidates.sort(key=lambda x: x.fusion_score or 0.0, reverse=True)
        
        return candidates
    
    def _select_diverse_candidates(self,
                                 candidates: List[MultimodalCandidate],
                                 budget: int,
                                 lambda_: float,
                                 section_cap: int,
                                 max_images: int,
                                 max_tables: int) -> List[MultimodalCandidate]:
        """Select diverse candidates using MMR-like algorithm with modality constraints."""
        if not candidates:
            return []
        
        selected = []
        used_tokens = 0
        section_counts = {}
        modality_counts = {ModalityType.TEXT: 0, ModalityType.IMAGE: 0, ModalityType.TABLE: 0}
        
        # Group candidates by section
        section_groups = {}
        for candidate in candidates:
            section = candidate.section
            if section not in section_groups:
                section_groups[section] = []
            section_groups[section].append(candidate)
        
        # Sort sections by relevance (highest scoring candidate in each section)
        section_scores = {}
        for section, group in section_groups.items():
            if group:
                section_scores[section] = max(c.fusion_score or 0.0 for c in group)
        
        sorted_sections = sorted(section_scores.keys(), 
                               key=lambda s: section_scores[s], reverse=True)
        
        # Select candidates from each section
        for section in sorted_sections:
            if used_tokens >= budget:
                break
            
            section_candidates = section_groups[section]
            section_used = 0
            
            for candidate in section_candidates:
                if used_tokens + candidate.tokens > budget:
                    break
                
                # Check modality limits
                if candidate.modality == ModalityType.IMAGE:
                    if modality_counts[ModalityType.IMAGE] >= max_images:
                        continue
                elif candidate.modality == ModalityType.TABLE:
                    if modality_counts[ModalityType.TABLE] >= max_tables:
                        continue
                
                # Check section cap
                if section_used >= section_cap:
                    break
                
                # Add candidate
                selected.append(candidate)
                used_tokens += candidate.tokens
                section_used += 1
                modality_counts[candidate.modality] += 1
        
        return selected
    
    def _build_context(self, candidates: List[MultimodalCandidate]) -> str:
        """Build context string from selected candidates."""
        if not candidates:
            return ""
        
        context_parts = []
        
        for candidate in candidates:
            if candidate.modality == ModalityType.TEXT:
                context_parts.append(f"[{candidate.id}] {candidate.text}")
                
            elif candidate.modality == ModalityType.IMAGE:
                caption = candidate.caption or candidate.text
                context_parts.append(f"[img {candidate.id}] {caption}")
                
            elif candidate.modality == ModalityType.TABLE:
                if candidate.table_md:
                    context_parts.append(f"[{candidate.id}]\n{candidate.table_md}")
                else:
                    context_parts.append(f"[{candidate.id}] {candidate.text}")
        
        return "\n\n".join(context_parts)
