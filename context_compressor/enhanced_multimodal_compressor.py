#!/usr/bin/env python3
"""
Enhanced Multimodal Context Compressor.
Integrates image embedding, table extraction, image captioning, and vector database.
Implements the advanced features from test.md blueprint.
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
from .image_encoder import ImageEncoder
from .table_extractor import TableExtractor
from .image_captioner import ImageCaptioner
from .vector_store import VectorStore, VectorType
from .performance_optimizer import performance_optimizer


class EnhancedMultimodalCompressor:
    """Enhanced multimodal context compressor with all advanced features from test.md."""
    
    def __init__(self, 
                 text_model: str = "BAAI/bge-large-en-v1.5",
                 image_model: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",  # Updated to recommended model
                 caption_model: str = "Salesforce/blip-image-captioning-large",  # Updated to large model
                 device: str = "auto",
                 vector_store_path: str = "vector_store"):
        """
        Initialize the enhanced multimodal compressor.
        
        Args:
            text_model: Text embedding model (BGE large as recommended)
            image_model: Image embedding model (OpenCLIP H-14 as recommended)
            caption_model: Image captioning model (BLIP large as recommended)
            device: Device to use for models
            vector_store_path: Path to vector store
        """
        self.text_model = text_model
        self.image_model = image_model
        self.caption_model = caption_model
        self.device = device
        
        # Initialize components
        self.text_encoder = None
        self.image_encoder = None
        self.table_extractor = None
        self.image_captioner = None
        self.vector_store = None
        
        self._load_components(vector_store_path)
    
    def _load_components(self, vector_store_path: str):
        """Load all components with recommended models from test.md."""
        # Load text encoder (BGE large)
        try:
            print(f"Loading text model: {self.text_model}")
            device = "cuda" if self.device == "auto" else self.device
            self.text_encoder = SentenceTransformer(self.text_model, device=device)
            print(f"✓ Loaded text model: {self.text_model}")
        except Exception as e:
            print(f"✗ Failed to load text model: {e}")
            self.text_encoder = None
        
        # Load image encoder (OpenCLIP H-14)
        try:
            self.image_encoder = ImageEncoder(
                model_name="ViT-H-14",  # Use H-14 variant
                pretrained="laion2b_s32b_b79k",  # Use recommended pretrained weights
                device=self.device
            )
            print(f"✓ Loaded image encoder: {self.image_model}")
        except Exception as e:
            print(f"✗ Failed to load image encoder: {e}")
            self.image_encoder = None
        
        # Load table extractor (Camelot)
        try:
            self.table_extractor = TableExtractor(
                flavor="lattice",  # Better for structured tables
                edge_tol=500,
                row_tol=10
            )
            print(f"✓ Loaded table extractor (Camelot)")
        except Exception as e:
            print(f"✗ Failed to load table extractor: {e}")
            self.table_extractor = None
        
        # Load image captioner (BLIP large)
        try:
            self.image_captioner = ImageCaptioner(
                model_name=self.caption_model,
                device=self.device
            )
            print(f"✓ Loaded image captioner: {self.caption_model}")
        except Exception as e:
            print(f"✗ Failed to load image captioner: {e}")
            self.image_captioner = None
        
        # Initialize vector store with named vectors support
        try:
            # Determine dimension based on text model
            dimension = 1024  # Default
            if self.text_encoder:
                # Get dimension from a sample encoding
                sample_embedding = self.text_encoder.encode(["sample"])
                dimension = len(sample_embedding[0])
            
            self.vector_store = VectorStore(
                store_path=vector_store_path,
                dimension=dimension
            )
            print(f"✓ Initialized vector store (dimension: {dimension})")
        except Exception as e:
            print(f"✗ Failed to initialize vector store: {e}")
            self.vector_store = None
    
    def compress(self, request: MultimodalCompressionRequest) -> MultimodalCompressionResponse:
        """
        Compress multimodal context with enhanced features from test.md.
        
        Args:
            request: Multimodal compression request
            
        Returns:
            Multimodal compression response
        """
        start_time = time.time()
        
        # Performance optimization: Check memory before starting
        if performance_optimizer.memory_manager:
            performance_optimizer.memory_manager.optimize_memory()
        
        # Step 1: Classify query (as per test.md section 3A)
        query_classifier = self._classify_query(request.q)
        
        # Step 2: Enhance candidates with advanced extraction (optimized)
        enhanced_candidates = self._enhance_candidates_optimized(request.candidates)
        
        # Step 3: Score candidates with multiple modalities (as per test.md section 3B)
        scored_candidates = self._score_candidates_enhanced(
            request.q, enhanced_candidates, query_classifier
        )
        
        # Step 4: Store vectors in database with named vectors (as per test.md section 2C)
        if self.vector_store:
            self._store_vectors(scored_candidates)
        
        # Step 5: Apply modality-specific filtering with boosts
        filtered_candidates = self._filter_by_modality_enhanced(
            scored_candidates, request, query_classifier
        )
        
        # Step 6: Apply fusion and ranking (as per test.md section 3B)
        ranked_candidates = self._fuse_and_rank_enhanced(
            filtered_candidates, 
            request.text_weight,
            request.bm25_weight,
            request.image_weight
        )
        
        # Step 7: Select diverse candidates (as per test.md section 4)
        selected_candidates = self._select_diverse_candidates_enhanced(
            ranked_candidates,
            request.B,
            request.lambda_,
            request.section_cap,
            request.max_images,
            request.max_tables
        )
        
        # Step 8: Build enhanced context (as per test.md section 5)
        context = self._build_enhanced_context(selected_candidates)
        
        # Performance optimization: Cleanup after processing
        if performance_optimizer.memory_manager:
            performance_optimizer.memory_manager.clear_gpu_cache()
        
        # Step 9: Calculate statistics
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
    
    def _enhance_candidates(self, candidates: List[MultimodalCandidate]) -> List[MultimodalCandidate]:
        """Enhance candidates with advanced extraction and captioning (as per test.md section 2B)."""
        enhanced = []
        
        for candidate in candidates:
            # Enhance image candidates with captioning (prefer native captions, BLIP as fallback)
            if candidate.modality == ModalityType.IMAGE and self.image_captioner:
                if candidate.image_ref and candidate.image_ref.image_path:
                    # Only generate AI caption if no native caption exists
                    if not candidate.caption:
                        ai_caption = self.image_captioner.generate_figure_caption(
                            candidate.image_ref.image_path
                        )
                        if ai_caption:
                            candidate.caption = ai_caption
                            candidate.caption_generated = True
                            candidate.text = ai_caption
                            candidate.tokens = len(ai_caption.split())
                    
                    # Generate image embedding for similarity search
                    if self.image_encoder:
                        image_embedding = self.image_encoder.encode_image(
                            candidate.image_ref.image_path
                        )
                        if image_embedding:
                            candidate.image_embedding = image_embedding
            
            # Enhance table candidates with better extraction
            if candidate.modality == ModalityType.TABLE and self.table_extractor:
                # For existing table candidates, we could re-extract with better quality
                # but that would be expensive. For now, use existing data.
                # In a full implementation, this would re-extract tables from the PDF
                pass
            
            enhanced.append(candidate)
        
        return enhanced
    
    def _enhance_candidates_optimized(self, candidates: List[MultimodalCandidate]) -> List[MultimodalCandidate]:
        """Optimized enhancement with batch processing and caching."""
        enhanced_candidates = []
        
        # Separate candidates by modality for batch processing
        text_candidates = [c for c in candidates if c.modality in [ModalityType.TEXT, ModalityType.TABLE]]
        image_candidates = [c for c in candidates if c.modality == ModalityType.IMAGE]
        
        # Batch process text embeddings
        if text_candidates and self.text_encoder:
            texts = [c.text for c in text_candidates]
            
            # Use performance optimizer for text embedding
            if performance_optimizer.enable_caching:
                embeddings = performance_optimizer.optimize_embedding_generation(
                    texts, 
                    lambda x: self.text_encoder.encode(x).tolist()
                )
            else:
                embeddings = self.text_encoder.encode(texts).tolist()
            
            # Assign embeddings back to candidates
            for candidate, embedding in zip(text_candidates, embeddings):
                candidate.text_embedding = embedding
        
        # Batch process image embeddings and captions
        if image_candidates and self.image_encoder:
            image_paths = []
            valid_image_candidates = []
            
            for candidate in image_candidates:
                if candidate.image_ref and candidate.image_ref.image_path:
                    image_paths.append(candidate.image_ref.image_path)
                    valid_image_candidates.append(candidate)
            
            if image_paths:
                # Use performance optimizer for image processing
                image_embeddings = performance_optimizer.optimize_image_processing(
                    image_paths,
                    lambda x: [self.image_encoder.encode_image(path) for path in x]
                )
                
                # Generate captions if needed
                if self.image_captioner:
                    captions = performance_optimizer.optimize_image_processing(
                        image_paths,
                        lambda x: [self.image_captioner.generate_caption(path) for path in x]
                    )
                else:
                    captions = [None] * len(image_paths)
                
                # Assign embeddings and captions back to candidates
                for candidate, embedding, caption in zip(valid_image_candidates, image_embeddings, captions):
                    if embedding is not None:
                        # Handle both tensor and list types
                        if hasattr(embedding, 'tolist'):
                            candidate.image_embedding = embedding.tolist()
                        else:
                            candidate.image_embedding = embedding
                    if caption is not None and not candidate.caption:
                        candidate.caption = caption
                        candidate.caption_generated = True
        
        # Process table content
        table_candidates = [c for c in candidates if c.modality == ModalityType.TABLE]
        for candidate in table_candidates:
            if self.table_extractor:
                try:
                    candidate.table_md = candidate.text
                except Exception as e:
                    print(f"Error extracting table for {candidate.id}: {e}")
        
        # Return all candidates (enhanced in place)
        return candidates
    
    def _score_candidates_enhanced(self, 
                                  query: str, 
                                  candidates: List[MultimodalCandidate],
                                  query_classifier: QueryClassifier) -> List[MultimodalCandidate]:
        """Score candidates using multiple modalities (as per test.md section 3B)."""
        if not candidates:
            return []
        
        # Prepare texts for scoring
        texts = [c.text for c in candidates]
        
        # BM25 scoring
        tokenized_texts = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized_texts)
        bm25_scores = bm25.get_scores(query.lower().split())
        
        # Text similarity scoring (BGE)
        text_scores = []
        if self.text_encoder:
            try:
                query_embedding = self.text_encoder.encode([query])
                text_embeddings = self.text_encoder.encode(texts)
                similarities = util.cos_sim(query_embedding, text_embeddings)[0]
                text_scores = similarities.cpu().numpy()
                
                # Store text embeddings
                for i, candidate in enumerate(candidates):
                    candidate.text_embedding = text_embeddings[i].tolist()
                    candidate.dense_sim = float(text_scores[i])
                    candidate.bm25 = float(bm25_scores[i])
                    
            except Exception as e:
                print(f"Error computing text similarities: {e}")
                for i, candidate in enumerate(candidates):
                    candidate.dense_sim = 0.0
                    candidate.bm25 = float(bm25_scores[i])
        else:
            for i, candidate in enumerate(candidates):
                candidate.dense_sim = 0.0
                candidate.bm25 = float(bm25_scores[i])
        
        # Image similarity scoring (OpenCLIP text→image)
        if self.image_encoder and query_classifier.has_image_terms:
            for candidate in candidates:
                if candidate.modality == ModalityType.IMAGE and candidate.image_ref:
                    if candidate.image_ref.image_path:
                        image_sim = self.image_encoder.compute_similarity(
                            candidate.image_ref.image_path, query
                        )
                        candidate.image_sim = image_sim if image_sim else 0.0
        
        return candidates
    
    def _store_vectors(self, candidates: List[MultimodalCandidate]):
        """Store candidate vectors in the vector database with named vectors (as per test.md section 2C)."""
        if not self.vector_store:
            return
        
        vectors_to_add = []
        
        for candidate in candidates:
            # Store text embeddings for all modalities
            if candidate.text_embedding:
                vectors_to_add.append((
                    f"{candidate.id}_text",
                    candidate.doc_id,
                    VectorType.TEXT,
                    candidate.text_embedding,
                    {
                        "modality": candidate.modality.value,
                        "section": candidate.section,
                        "page": candidate.page,
                        "tokens": candidate.tokens,
                        "text": candidate.text,
                        "caption": candidate.caption
                    }
                ))
            
            # Store image embeddings for image modality
            if candidate.image_embedding:
                vectors_to_add.append((
                    f"{candidate.id}_image",
                    candidate.doc_id,
                    VectorType.IMAGE,
                    candidate.image_embedding,
                    {
                        "modality": candidate.modality.value,
                        "section": candidate.section,
                        "page": candidate.page,
                        "caption": candidate.caption,
                        "bbox": candidate.bbox.dict() if candidate.bbox else None
                    }
                ))
        
        # Add vectors in batch
        if vectors_to_add:
            added_count = self.vector_store.add_vectors_batch(vectors_to_add)
            print(f"✓ Stored {added_count} vectors in database")
    
    def _filter_by_modality_enhanced(self, 
                                    candidates: List[MultimodalCandidate],
                                    request: MultimodalCompressionRequest,
                                    query_classifier: QueryClassifier) -> List[MultimodalCandidate]:
        """Enhanced modality filtering with boosts (as per test.md section 3A)."""
        filtered = []
        
        for candidate in candidates:
            # Apply modality-specific filtering and boosts
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
            
            # Add section boosts (as per test.md section 3B)
            if candidate.section in ["title", "abstract", "figure_caption"]:
                if any(term in request.q.lower() for term in [candidate.section, "title", "abstract", "figure"]):
                    candidate.dense_sim *= 1.2
            
            filtered.append(candidate)
        
        # Optional: Use vector database for additional candidates
        if self.vector_store and query_classifier.has_image_terms:
            # Search for similar images in the database
            if self.text_encoder:
                query_embedding = self.text_encoder.encode([request.q])[0].tolist()
                similar_vectors = self.vector_store.search_similar(
                    query_embedding,
                    vector_type=VectorType.IMAGE,
                    top_k=5,
                    threshold=0.3
                )
                
                # Add high-similarity vectors as additional candidates
                for vector_record, similarity in similar_vectors:
                    if similarity > 0.5:  # High similarity threshold
                        # Create additional candidate from vector record
                        # This would require reconstructing the candidate from metadata
                        pass
        
        return filtered
    
    def _fuse_and_rank_enhanced(self, 
                               candidates: List[MultimodalCandidate],
                               text_weight: float,
                               bm25_weight: float,
                               image_weight: float) -> List[MultimodalCandidate]:
        """Enhanced fusion with multiple similarity scores (as per test.md section 3B)."""
        if not candidates:
            return []
        
        # Extract scores
        dense_scores = [c.dense_sim for c in candidates]
        bm25_scores = [c.bm25 for c in candidates]
        
        # Normalize scores
        dense_scores_norm = z_score_normalize(dense_scores)
        bm25_scores_norm = z_score_normalize(bm25_scores)
        
        # Compute fusion scores (as per test.md: 0.6*z(dense_text) + 0.2*z(bm25) + 0.2*z(clip_text→image))
        for i, candidate in enumerate(candidates):
            fusion_score = (
                text_weight * dense_scores_norm[i] +
                bm25_weight * bm25_scores_norm[i]
            )
            
            # Add image weight if applicable
            if candidate.modality == ModalityType.IMAGE and candidate.image_sim is not None:
                fusion_score += image_weight * candidate.image_sim
            
            # Add quality bonuses
            if candidate.modality == ModalityType.TABLE and candidate.table_md:
                # Bonus for well-structured tables
                fusion_score += 0.1
            
            if candidate.modality == ModalityType.IMAGE and candidate.caption:
                # Bonus for images with captions
                fusion_score += 0.1
            
            candidate.fusion_score = fusion_score
        
        # Sort by fusion score
        candidates.sort(key=lambda x: x.fusion_score or 0.0, reverse=True)
        
        return candidates
    
    def _select_diverse_candidates_enhanced(self,
                                          candidates: List[MultimodalCandidate],
                                          budget: int,
                                          lambda_: float,
                                          section_cap: int,
                                          max_images: int,
                                          max_tables: int) -> List[MultimodalCandidate]:
        """Select diverse candidates using MMR-like algorithm with modality constraints (as per test.md section 4)."""
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
    
    def _build_enhanced_context(self, candidates: List[MultimodalCandidate]) -> str:
        """Build enhanced context string (as per test.md section 5)."""
        if not candidates:
            return ""
        
        context_parts = []
        
        for candidate in candidates:
            if candidate.modality == ModalityType.TEXT:
                context_parts.append(f"[{candidate.id}] {candidate.text}")
            
            elif candidate.modality == ModalityType.IMAGE:
                # Include image tag with caption (not pixels)
                caption = candidate.caption or candidate.text
                bbox_info = ""
                if candidate.bbox:
                    bbox_info = f" (page {candidate.page}, bbox {candidate.bbox.x0:.1f},{candidate.bbox.y0:.1f},{candidate.bbox.x1:.1f},{candidate.bbox.y1:.1f})"
                context_parts.append(f"[img {candidate.id}] {caption}{bbox_info}")
            
            elif candidate.modality == ModalityType.TABLE:
                if candidate.table_md:
                    # Limit table size to avoid token bloat (as per test.md section 8)
                    table_lines = candidate.table_md.split('\n')
                    if len(table_lines) > 10:  # Limit to 10 rows
                        table_lines = table_lines[:10]
                        table_lines.append("... (table truncated)")
                    context_parts.append(f"[{candidate.id}]\n" + '\n'.join(table_lines))
                else:
                    context_parts.append(f"[{candidate.id}] {candidate.text}")
        
        return "\n\n".join(context_parts)
    
    def _classify_query(self, query: str) -> QueryClassifier:
        """Classify query to determine modality preferences (as per test.md section 3A)."""
        query_lower = query.lower()
        
        # Check for image terms (as per test.md)
        image_terms = ['figure', 'image', 'diagram', 'chart', 'plot', 'see figure', 'what does the image', 'visual']
        has_image_terms = any(term in query_lower for term in image_terms)
        
        # Check for table terms (as per test.md)
        table_terms = ['table', 'tabulated', 'value', '%', '$', 'increase', 'decrease', 'statistics', 'data']
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
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if self.vector_store:
            return self.vector_store.get_stats()
        return {"error": "Vector store not available"}
