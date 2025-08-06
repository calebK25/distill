"""
Main Context Compressor implementation.
"""

import time
from typing import List, Optional, Dict, Any
from .schemas import (
    Candidate, 
    CompressionRequest, 
    CompressionResponse, 
    CompressionStats,
    MappingItem
)
from .fusion import RankFusion
from .mmr import MMRSelector
from .trimming import SentenceTrimmer
from .router import ModeRouter
from .utils import (
    timing_decorator, 
    check_low_context, 
    compute_hash,
    count_tokens
)


class ContextCompressor:
    """Main context compressor that orchestrates fusion, MMR, and trimming."""
    
    def __init__(self):
        """Initialize the context compressor."""
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 3600  # 1 hour TTL
    
    @timing_decorator
    def _fusion_step(self, candidates: List[Candidate], weights: Dict[str, float], top_m: int) -> List[Candidate]:
        """Execute rank fusion step."""
        fusion = RankFusion(weights)
        return fusion.select_top_m(candidates, m=top_m)
    
    @timing_decorator
    def _mmr_step(
        self, 
        candidates: List[Candidate], 
        budget: int,
        lambda_param: float,
        section_cap: int,
        doc_cap: int,
        query_embedding: Optional[List[float]] = None,
        single_doc_mode: bool = False
    ) -> List[Candidate]:
        """Execute MMR diversity selection step."""
        mmr_selector = MMRSelector(lambda_param, section_cap, doc_cap)
        return mmr_selector.select_diverse_candidates(
            candidates, budget, query_embedding, single_doc_mode
        )
    
    @timing_decorator
    def _trimming_step(
        self,
        candidates: List[Candidate],
        budget: int,
        query: str,
        use_reranker: bool = False,
        query_embedding: Optional[List[float]] = None
    ) -> tuple:
        """Execute sentence trimming step."""
        trimmer = SentenceTrimmer(use_reranker)
        return trimmer.trim_candidates(
            candidates, budget, query, query_embedding
        )
    
    def _compute_query_embedding(self, query: str, model_name: str = "intfloat/e5-large-v2") -> Optional[List[float]]:
        """
        Compute query embedding using the configured embedding model.
        """
        try:
            from .embeddings import get_embedding_manager
            embedding_manager = get_embedding_manager(model_name)
            return embedding_manager.encode_single(query)
        except Exception as e:
            print(f"Warning: Query embedding computation failed: {e}")
            return None
    
    def _get_cache_key(self, request: CompressionRequest) -> str:
        """Generate cache key for the request."""
        # Create a hashable representation of the request
        cache_data = {
            'q': request.q,
            'B': request.B,
            'candidates_hash': compute_hash([c.dict() for c in request.candidates]),
            'params_hash': compute_hash(request.params.dict())
        }
        return compute_hash(cache_data)
    
    def _check_cache(self, cache_key: str) -> Optional[CompressionResponse]:
        """Check if result is cached."""
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            if time.time() - cached_item['timestamp'] < self.cache_ttl:
                return cached_item['response']
            else:
                # Expired cache entry
                del self.cache[cache_key]
        return None
    
    def _store_cache(self, cache_key: str, response: CompressionResponse):
        """Store result in cache."""
        self.cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
    
    def compress(self, request: CompressionRequest) -> CompressionResponse:
        """
        Compress context according to the request.
        
        Args:
            request: Compression request
            
        Returns:
            Compression response
        """
        start_time = time.time()
        
        # Check cache if not bypassed
        if not request.no_cache:
            cache_key = self._get_cache_key(request)
            cached_response = self._check_cache(cache_key)
            if cached_response:
                return cached_response
        
        # Step 1: Rank Fusion
        fusion_candidates, fusion_time = self._fusion_step(
            request.candidates, 
            request.params.fusion_weights,
            request.params.topM
        )
        
        # Step 2: Mode Routing (if auto_router enabled)
        mode = "cross_doc"
        doc_id = None
        router_stats = None
        
        if request.params.auto_router:
            router = ModeRouter(threshold=0.8)
            mode, doc_id, router_stats = router.route_mode(fusion_candidates)
        
        # Step 3: MMR Diversity Selection
        query_embedding = self._compute_query_embedding(request.q, request.params.embedding_model)
        single_doc_mode = (mode == "single_doc")
        
        # Filter candidates for single-document mode
        if single_doc_mode and doc_id:
            mmr_candidates = [c for c in fusion_candidates if c.doc_id == doc_id]
        else:
            mmr_candidates = fusion_candidates
        
        mmr_candidates, mmr_time = self._mmr_step(
            mmr_candidates,
            request.B,
            request.params.lambda_,
            request.params.section_cap,
            request.params.doc_cap,
            query_embedding,
            single_doc_mode
        )
        
        # Step 4: Sentence Trimming
        (trimmed_texts, mapping_items, total_tokens), trim_time = self._trimming_step(
            mmr_candidates,
            request.B,
            request.q,
            request.params.use_reranker,
            query_embedding
        )
        
        # Combine trimmed texts
        context = "\n\n".join(trimmed_texts)
        
        # Calculate statistics
        original_tokens = sum(c.tokens for c in request.candidates)
        saved_tokens = original_tokens - total_tokens
        low_context = check_low_context(total_tokens, request.B)
        
        # Create stats
        total_time = (time.time() - start_time) * 1000
        stats = CompressionStats(
            mode=mode,
            budget=request.B,
            used=total_tokens,
            saved_vs_pool=saved_tokens,
            lambda_=request.params.lambda_,
            fusion_weights=request.params.fusion_weights,
            section_cap=request.params.section_cap,
            doc_cap=request.params.doc_cap,
            router_score=router_stats,
            low_context=low_context,
            fusion_ms=fusion_time,
            mmr_ms=mmr_time,
            trim_ms=trim_time,
            total_ms=total_time
        )
        
        # Create response
        response = CompressionResponse(
            context=context,
            mapping=mapping_items,
            stats=stats
        )
        
        # Store in cache if not bypassed
        if not request.no_cache:
            cache_key = self._get_cache_key(request)
            self._store_cache(cache_key, response)
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compressor statistics."""
        return {
            'cache_size': len(self.cache),
            'cache_ttl': self.cache_ttl
        }
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
