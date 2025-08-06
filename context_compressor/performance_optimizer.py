#!/usr/bin/env python3
"""
Performance optimization module for Context Compressor.
Provides batch processing, memory management, and caching optimizations.
"""

import time
import gc
import psutil
import torch
from typing import List, Dict, Any, Optional, Callable
from functools import lru_cache
from pathlib import Path
import pickle
import hashlib
from context_compressor.logging_config import production_logger


class MemoryManager:
    """GPU and system memory management."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.memory_threshold = 0.8  # 80% memory usage threshold
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = {
            "system_ram_percent": psutil.virtual_memory().percent,
            "system_ram_available_gb": psutil.virtual_memory().available / (1024**3)
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            stats.update({
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })
        
        return stats
    
    def is_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        stats = self.get_memory_usage()
        
        # Check system RAM
        if stats["system_ram_percent"] > 90:
            return True
        
        # Check GPU memory
        if self.device == "cuda" and torch.cuda.is_available():
            allocated = stats["gpu_memory_allocated_gb"]
            total = stats["gpu_memory_total_gb"]
            if allocated / total > self.memory_threshold:
                return True
        
        return False
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            production_logger.info("GPU cache cleared")
    
    def optimize_memory(self):
        """Perform memory optimization."""
        if self.is_memory_pressure():
            production_logger.warning("Memory pressure detected, performing optimization")
            self.clear_gpu_cache()
            gc.collect()
            
            # Log memory usage after optimization
            stats = self.get_memory_usage()
            production_logger.info("Memory optimization completed", memory_stats=stats)


class BatchProcessor:
    """Batch processing for large document sets."""
    
    def __init__(self, batch_size: int = 5, max_workers: int = 2):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.memory_manager = MemoryManager()
    
    def process_batch(self, 
                     items: List[Any], 
                     processor_func: Callable,
                     **kwargs) -> List[Any]:
        """Process items in batches with memory management."""
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        production_logger.info("Starting batch processing", 
                             total_items=len(items),
                             batch_size=self.batch_size,
                             total_batches=total_batches)
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            with production_logger.operation_timer("batch_processing", 
                                                 batch_num=batch_num,
                                                 batch_size=len(batch)):
                try:
                    # Check memory before processing
                    if self.memory_manager.is_memory_pressure():
                        self.memory_manager.optimize_memory()
                    
                    # Process batch
                    batch_results = processor_func(batch, **kwargs)
                    results.extend(batch_results)
                    
                    # Clear cache after each batch
                    self.memory_manager.clear_gpu_cache()
                    
                    production_logger.info("Batch completed", 
                                         batch_num=batch_num,
                                         results_count=len(batch_results))
                    
                except Exception as e:
                    production_logger.error("Batch processing failed", 
                                          error=e,
                                          batch_num=batch_num)
                    # Continue with next batch
                    continue
        
        production_logger.info("Batch processing completed", 
                             total_results=len(results))
        return results


class CacheManager:
    """Caching system for embeddings and results."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_stats = {"hits": 0, "misses": 0}
    
    def _get_cache_key(self, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = str(sorted(data.items()))
        else:
            content = str(data)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, key: str, prefix: str = "") -> Path:
        """Get cache file path."""
        filename = f"{prefix}_{key}.pkl" if prefix else f"{key}.pkl"
        return self.cache_dir / filename
    
    def get(self, key: str, prefix: str = "") -> Optional[Any]:
        """Get item from cache."""
        cache_path = self._get_cache_path(key, prefix)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                self.cache_stats["hits"] += 1
                production_logger.debug("Cache hit", key=key, prefix=prefix)
                return data
            except Exception as e:
                production_logger.warning("Cache read failed", error=e, key=key)
        
        self.cache_stats["misses"] += 1
        production_logger.debug("Cache miss", key=key, prefix=prefix)
        return None
    
    def set(self, key: str, data: Any, prefix: str = ""):
        """Set item in cache."""
        cache_path = self._get_cache_path(key, prefix)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            production_logger.debug("Cache set", key=key, prefix=prefix)
        except Exception as e:
            production_logger.warning("Cache write failed", error=e, key=key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0
        if self.cache_stats["hits"] + self.cache_stats["misses"] > 0:
            hit_rate = self.cache_stats["hits"] / (self.cache_stats["hits"] + self.cache_stats["misses"])
        
        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "cache_size_mb": self._get_cache_size()
        }
    
    def _get_cache_size(self) -> float:
        """Get cache directory size in MB."""
        total_size = 0
        for file_path in self.cache_dir.rglob("*.pkl"):
            total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)
    
    def clear_cache(self, prefix: str = ""):
        """Clear cache files."""
        if prefix:
            pattern = f"{prefix}_*.pkl"
        else:
            pattern = "*.pkl"
        
        files_removed = 0
        for file_path in self.cache_dir.glob(pattern):
            file_path.unlink()
            files_removed += 1
        
        production_logger.info("Cache cleared", 
                             prefix=prefix,
                             files_removed=files_removed)


class PerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self, 
                 enable_caching: bool = True,
                 enable_batching: bool = True,
                 enable_memory_management: bool = True):
        self.enable_caching = enable_caching
        self.enable_batching = enable_batching
        self.enable_memory_management = enable_memory_management
        
        # Initialize components
        self.cache_manager = CacheManager() if enable_caching else None
        self.batch_processor = BatchProcessor() if enable_batching else None
        self.memory_manager = MemoryManager() if enable_memory_management else None
        
        production_logger.info("Performance optimizer initialized",
                             caching=enable_caching,
                             batching=enable_batching,
                             memory_management=enable_memory_management)
    
    def optimize_embedding_generation(self, 
                                    texts: List[str],
                                    embedding_func: Callable) -> List[List[float]]:
        """Optimized embedding generation with caching and batching."""
        if not texts:
            return []
        
        embeddings = []
        
        # Use caching if enabled
        if self.enable_caching and self.cache_manager:
            for i, text in enumerate(texts):
                cache_key = self.cache_manager._get_cache_key(text)
                cached_embedding = self.cache_manager.get(cache_key, "embedding")
                
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                else:
                    # Generate embedding
                    embedding = embedding_func([text])[0]
                    self.cache_manager.set(cache_key, embedding, "embedding")
                    embeddings.append(embedding)
        else:
            # Use batching if enabled
            if self.enable_batching and self.batch_processor:
                embeddings = self.batch_processor.process_batch(texts, embedding_func)
            else:
                # Direct processing
                embeddings = embedding_func(texts)
        
        return embeddings
    
    def optimize_image_processing(self, 
                                image_paths: List[str],
                                processor_func: Callable) -> List[Any]:
        """Optimized image processing with memory management."""
        if not image_paths:
            return []
        
        results = []
        
        # Use batching for image processing
        if self.enable_batching and self.batch_processor:
            results = self.batch_processor.process_batch(image_paths, processor_func)
        else:
            # Direct processing with memory management
            for i, image_path in enumerate(image_paths):
                with production_logger.operation_timer("image_processing", 
                                                     image_index=i,
                                                     image_path=image_path):
                    try:
                        # Check memory before processing
                        if self.enable_memory_management and self.memory_manager:
                            if self.memory_manager.is_memory_pressure():
                                self.memory_manager.optimize_memory()
                        
                        result = processor_func([image_path])[0]
                        results.append(result)
                        
                        # Clear GPU cache after each image
                        if self.enable_memory_management and self.memory_manager:
                            self.memory_manager.clear_gpu_cache()
                    
                    except Exception as e:
                        production_logger.error("Image processing failed", 
                                              error=e,
                                              image_path=image_path)
                        results.append(None)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "optimization_enabled": {
                "caching": self.enable_caching,
                "batching": self.enable_batching,
                "memory_management": self.enable_memory_management
            }
        }
        
        # Add cache stats
        if self.cache_manager:
            stats["cache"] = self.cache_manager.get_stats()
        
        # Add memory stats
        if self.memory_manager:
            stats["memory"] = self.memory_manager.get_memory_usage()
        
        return stats
    
    def cleanup(self):
        """Cleanup resources."""
        if self.memory_manager:
            self.memory_manager.clear_gpu_cache()
        
        production_logger.info("Performance optimizer cleanup completed")


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()
