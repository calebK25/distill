"""
Embeddings module for Context Compressor.
"""

import time
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from .utils import compute_hash


class EmbeddingManager:
    """Manages embedding computation with caching and model selection."""
    
    def __init__(self, model_name: str = "intfloat/e5-large-v2", cache_size: int = 1000):
        """
        Initialize embedding manager.
        
        Args:
            model_name: Name of the sentence transformer model
            cache_size: Maximum number of cached embeddings
        """
        self.model_name = model_name
        self.cache_size = cache_size
        self.cache = {}
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            print(f"Warning: Could not load {self.model_name}, falling back to no embeddings: {e}")
            self.model = None
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return compute_hash(text)
    
    def _add_to_cache(self, text: str, embedding: List[float]):
        """Add embedding to cache with LRU eviction."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[text] = {
            'embedding': embedding,
            'timestamp': time.time()
        }
    
    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache if available."""
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]['embedding']
        return None
    
    def encode(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            normalize: Whether to normalize embeddings
            
        Returns:
            List of embedding vectors
        """
        if not self.model:
            # Fallback: return None embeddings
            return [None] * len(texts)
        
        embeddings = []
        texts_to_encode = []
        text_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cached_embedding = self._get_from_cache(text)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                texts_to_encode.append(text)
                text_indices.append(i)
        
        # Encode texts not in cache
        if texts_to_encode:
            try:
                new_embeddings = self.model.encode(
                    texts_to_encode,
                    normalize_embeddings=normalize,
                    convert_to_numpy=False
                )
                
                # Add to cache and results
                for i, (text, embedding) in enumerate(zip(texts_to_encode, new_embeddings)):
                    embedding_list = embedding.tolist()
                    self._add_to_cache(text, embedding_list)
                    embeddings.insert(text_indices[i], embedding_list)
                    
            except Exception as e:
                print(f"Warning: Embedding encoding failed: {e}")
                # Fill with None for failed encodings
                for idx in text_indices:
                    embeddings.insert(idx, None)
        
        return embeddings
    
    def encode_single(self, text: str, normalize: bool = True) -> Optional[List[float]]:
        """
        Encode a single text to embedding.
        
        Args:
            text: Text to encode
            normalize: Whether to normalize embedding
            
        Returns:
            Embedding vector or None if failed
        """
        embeddings = self.encode([text], normalize)
        return embeddings[0] if embeddings else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding manager statistics."""
        return {
            'model_name': self.model_name,
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'model_loaded': self.model is not None
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()


# Global embedding manager instance
_embedding_manager = None


def get_embedding_manager(model_name: str = "intfloat/e5-large-v2") -> EmbeddingManager:
    """Get or create global embedding manager instance."""
    global _embedding_manager
    if _embedding_manager is None or _embedding_manager.model_name != model_name:
        _embedding_manager = EmbeddingManager(model_name)
    return _embedding_manager
