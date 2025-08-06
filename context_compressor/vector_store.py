#!/usr/bin/env python3
"""
Vector database for Context Compressor.
Provides efficient storage and retrieval of embeddings.
Implements named vectors support as recommended in test.md blueprint.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import pickle
import json
from pathlib import Path
import time
from dataclasses import dataclass, asdict
from enum import Enum


class VectorType(str, Enum):
    """Types of vectors that can be stored (as per test.md section 2C)."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class VectorRecord:
    """Record for storing vector data with named vectors support."""
    id: str
    doc_id: str
    vector_type: VectorType
    embedding: List[float]
    metadata: Dict[str, Any]
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorRecord':
        """Create from dictionary."""
        return cls(**data)


class VectorStore:
    """Simple vector database for storing and retrieving embeddings with named vectors support (as per test.md section 2C)."""

    def __init__(self,
                 store_path: str = "vector_store",
                 dimension: int = 1024):
        """
        Initialize the vector store.
        
        Args:
            store_path: Path to store vectors
            dimension: Expected embedding dimension
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(exist_ok=True)
        self.dimension = dimension

        # In-memory storage for fast access
        self.vectors: Dict[str, VectorRecord] = {}
        self.doc_vectors: Dict[str, List[str]] = {}  # doc_id -> vector_ids
        self.type_vectors: Dict[VectorType, List[str]] = {vt: [] for vt in VectorType}  # type -> vector_ids

        # Load existing vectors
        self._load_vectors()

    def _load_vectors(self):
        """Load existing vectors from disk."""
        try:
            vectors_file = self.store_path / "vectors.pkl"
            if vectors_file.exists():
                with open(vectors_file, 'rb') as f:
                    self.vectors = pickle.load(f)

                # Rebuild indices
                self.doc_vectors = {}
                self.type_vectors = {vt: [] for vt in VectorType}
                
                for vector_id, record in self.vectors.items():
                    # Rebuild doc index
                    if record.doc_id not in self.doc_vectors:
                        self.doc_vectors[record.doc_id] = []
                    self.doc_vectors[record.doc_id].append(vector_id)
                    
                    # Rebuild type index
                    self.type_vectors[record.vector_type].append(vector_id)

                print(f"✓ Loaded {len(self.vectors)} vectors from {self.store_path}")
                print(f"  Text vectors: {len(self.type_vectors[VectorType.TEXT])}")
                print(f"  Image vectors: {len(self.type_vectors[VectorType.IMAGE])}")
                print(f"  Table vectors: {len(self.type_vectors[VectorType.TABLE])}")
                
        except Exception as e:
            print(f"Warning: Could not load vectors: {e}")

    def _save_vectors(self):
        """Save vectors to disk."""
        try:
            vectors_file = self.store_path / "vectors.pkl"
            with open(vectors_file, 'wb') as f:
                pickle.dump(self.vectors, f)
        except Exception as e:
            print(f"Warning: Could not save vectors: {e}")

    def add_vector(self,
                   vector_id: str,
                   doc_id: str,
                   vector_type: VectorType,
                   embedding: List[float],
                   metadata: Dict[str, Any]) -> bool:
        """
        Add a vector to the store (as per test.md section 2C).
        
        Args:
            vector_id: Unique identifier for the vector
            doc_id: Document identifier
            vector_type: Type of vector (text/image/table)
            embedding: Embedding vector
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate embedding dimension
            if len(embedding) != self.dimension:
                print(f"Warning: Expected dimension {self.dimension}, got {len(embedding)}")

            # Create vector record
            record = VectorRecord(
                id=vector_id,
                doc_id=doc_id,
                vector_type=vector_type,
                embedding=embedding,
                metadata=metadata,
                created_at=time.time()
            )

            # Store in memory
            self.vectors[vector_id] = record

            # Update indices
            if doc_id not in self.doc_vectors:
                self.doc_vectors[doc_id] = []
            self.doc_vectors[doc_id].append(vector_id)
            
            self.type_vectors[vector_type].append(vector_id)

            # Save to disk
            self._save_vectors()

            return True

        except Exception as e:
            print(f"Error adding vector {vector_id}: {e}")
            return False

    def add_vectors_batch(self,
                          vectors: List[Tuple[str, str, VectorType, List[float], Dict[str, Any]]]) -> int:
        """
        Add multiple vectors in batch.
        
        Args:
            vectors: List of (vector_id, doc_id, vector_type, embedding, metadata) tuples
            
        Returns:
            Number of successfully added vectors
        """
        success_count = 0

        for vector_data in vectors:
            if self.add_vector(*vector_data):
                success_count += 1

        return success_count

    def get_vector(self, vector_id: str) -> Optional[VectorRecord]:
        """
        Get a vector by ID.
        
        Args:
            vector_id: Vector identifier
            
        Returns:
            Vector record or None if not found
        """
        return self.vectors.get(vector_id)

    def get_document_vectors(self, doc_id: str) -> List[VectorRecord]:
        """
        Get all vectors for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            List of vector records
        """
        vector_ids = self.doc_vectors.get(doc_id, [])
        return [self.vectors[vid] for vid in vector_ids if vid in self.vectors]

    def search_similar(self,
                       query_embedding: List[float],
                       vector_type: Optional[VectorType] = None,
                       top_k: int = 10,
                       threshold: float = 0.0) -> List[Tuple[VectorRecord, float]]:
        """
        Search for similar vectors (as per test.md section 3B).
        
        Args:
            query_embedding: Query embedding vector
            vector_type: Filter by vector type (None for all types)
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (vector_record, similarity_score) tuples
        """
        if not self.vectors:
            return []

        # Filter vectors by type if specified
        if vector_type:
            vector_ids = self.type_vectors.get(vector_type, [])
            candidates = [self.vectors[vid] for vid in vector_ids if vid in self.vectors]
        else:
            candidates = list(self.vectors.values())

        if not candidates:
            return []

        # Compute similarities
        similarities = []
        query_array = np.array(query_embedding)
        
        for record in candidates:
            candidate_array = np.array(record.embedding)
            similarity = self._cosine_similarity(query_array, candidate_array)
            if similarity >= threshold:
                similarities.append((record, similarity))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def search_by_metadata(self,
                          metadata_filters: Dict[str, Any],
                          vector_type: Optional[VectorType] = None) -> List[VectorRecord]:
        """
        Search vectors by metadata filters.
        
        Args:
            metadata_filters: Dictionary of metadata key-value pairs to filter by
            vector_type: Filter by vector type (None for all types)
            
        Returns:
            List of matching vector records
        """
        if not self.vectors:
            return []

        # Filter vectors by type if specified
        if vector_type:
            vector_ids = self.type_vectors.get(vector_type, [])
            candidates = [self.vectors[vid] for vid in vector_ids if vid in self.vectors]
        else:
            candidates = list(self.vectors.values())

        # Apply metadata filters
        results = []
        for record in candidates:
            if self._matches_metadata(record.metadata, metadata_filters):
                results.append(record)

        return results

    def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector from the store.
        
        Args:
            vector_id: Vector identifier
            
        Returns:
            True if successful, False otherwise
        """
        if vector_id not in self.vectors:
            return False

        record = self.vectors[vector_id]
        
        # Remove from indices
        if record.doc_id in self.doc_vectors:
            self.doc_vectors[record.doc_id] = [vid for vid in self.doc_vectors[record.doc_id] if vid != vector_id]
            if not self.doc_vectors[record.doc_id]:
                del self.doc_vectors[record.doc_id]
        
        self.type_vectors[record.vector_type] = [vid for vid in self.type_vectors[record.vector_type] if vid != vector_id]
        
        # Remove from main storage
        del self.vectors[vector_id]
        
        # Save to disk
        self._save_vectors()
        
        return True

    def delete_document_vectors(self, doc_id: str) -> int:
        """
        Delete all vectors for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Number of deleted vectors
        """
        vector_ids = self.doc_vectors.get(doc_id, [])
        deleted_count = 0
        
        for vector_id in vector_ids:
            if self.delete_vector(vector_id):
                deleted_count += 1
        
        return deleted_count

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_vectors": len(self.vectors),
            "documents": len(self.doc_vectors),
            "vector_types": {
                vt.value: len(self.type_vectors[vt]) for vt in VectorType
            },
            "dimension": self.dimension,
            "store_path": str(self.store_path)
        }

    def export_metadata(self, output_file: str):
        """
        Export metadata to JSON file.
        
        Args:
            output_file: Output file path
        """
        metadata = {}
        for vector_id, record in self.vectors.items():
            metadata[vector_id] = {
                "doc_id": record.doc_id,
                "vector_type": record.vector_type.value,
                "metadata": record.metadata,
                "created_at": record.created_at
            }
        
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _matches_metadata(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if metadata matches the given filters.
        
        Args:
            metadata: Vector metadata
            filters: Filter criteria
            
        Returns:
            True if metadata matches filters
        """
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
