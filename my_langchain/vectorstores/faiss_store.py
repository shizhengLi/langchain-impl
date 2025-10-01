# -*- coding: utf-8 -*-
"""
FAISS vector store implementation

This module provides a FAISS-based vector store implementation.
FAISS is optional - if not available, this will fall back to in-memory implementation.
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np

from my_langchain.vectorstores.base import BaseVectorStore
from my_langchain.vectorstores.types import (
    VectorStoreConfig, VectorStoreResult, VectorStoreQuery, Vector, Document,
    VectorStoreValidationError, VectorStoreRetrievalError, VectorStoreIndexError,
    IndexStatus, DistanceMetric
)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store implementation

    This implementation uses FAISS for efficient vector similarity search.
    Falls back to in-memory implementation if FAISS is not available.
    """

    def __init__(self, config: VectorStoreConfig, **kwargs):
        """
        Initialize FAISS vector store

        Args:
            config: Vector store configuration
            **kwargs: Additional parameters
        """
        super().__init__(config=config, **kwargs)

        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is not installed. Install with: pip install faiss-cpu or faiss-gpu"
            )

        # Internal storage
        self._vectors: Dict[str, Vector] = {}
        self._documents: Dict[str, Document] = {}
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}

        # FAISS index
        self._index = None
        self._initialize_index()

    def _initialize_index(self) -> None:
        """Initialize FAISS index based on configuration"""
        if not FAISS_AVAILABLE:
            return

        dimension = self.config.dimension

        if self.config.index_type == "flat":
            self._index = faiss.IndexFlat(dimension)
        elif self.config.index_type == "ivf":
            # IVF index
            nlist = self.config.nlist or min(100, max(1, dimension // 4))
            quantizer = faiss.IndexFlat(dimension)
            self._index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif self.config.index_type == "hnsw":
            # HNSW index
            ef_construction = self.config.ef_construction or 200
            self._index = faiss.IndexHNSWFlat(dimension, 32)  # M=32 is default
            self._index.hnsw.efConstruction = ef_construction
        else:
            # Default to flat index
            self._index = faiss.IndexFlat(dimension)

        # Set distance metric
        if self.config.metric == DistanceMetric.EUCLIDEAN:
            # FAISS uses L2 (Euclidean) by default
            pass
        elif self.config.metric == DistanceMetric.COSINE:
            # For cosine similarity, we need to normalize vectors
            pass  # We'll normalize vectors during search
        elif self.config.metric == DistanceMetric.DOT_PRODUCT:
            # Use inner product index
            if hasattr(faiss, 'IndexFlatIP'):
                self._index = faiss.IndexFlatIP(dimension)
            else:
                # Fallback to flat index
                pass

        self.index_status = IndexStatus.EMPTY

    def add_vectors(self, vectors: List[Vector]) -> List[str]:
        """
        Add vectors to the store

        Args:
            vectors: List of vectors to add

        Returns:
            List of vector IDs that were added
        """
        if not FAISS_AVAILABLE:
            return self._fallback_add_vectors(vectors)

        self._validate_vectors(vectors)

        added_ids = []
        embeddings = []

        for vector in vectors:
            # Generate ID if not provided
            if not vector.id:
                import uuid
                vector.id = str(uuid.uuid4())

            # Store vector
            self._vectors[vector.id] = vector
            added_ids.append(vector.id)

            # Prepare embedding for FAISS
            embedding = np.array([vector.embedding], dtype=np.float32)

            # Normalize for cosine similarity
            if self.config.metric == DistanceMetric.COSINE:
                embedding = self._normalize_vector(embedding)

            embeddings.append(embedding[0])

        if embeddings:
            # Add to FAISS index
            embeddings_array = np.array(embeddings, dtype=np.float32)
            start_idx = self._index.ntotal

            if self.config.index_type == "ivf" and not self._index.is_trained:
                # Train IVF index if not trained
                self._index.train(embeddings_array)

            self._index.add(embeddings_array)

            # Update ID mappings
            for i, vector_id in enumerate(added_ids):
                index = start_idx + i
                self._id_to_index[vector_id] = index
                self._index_to_id[index] = vector_id

        self.vector_count = len(self._vectors)
        self.index_status = IndexStatus.READY

        return added_ids

    def _fallback_add_vectors(self, vectors: List[Vector]) -> List[str]:
        """Fallback method when FAISS is not available"""
        # Simple fallback implementation
        added_ids = []
        for vector in vectors:
            if not vector.id:
                import uuid
                vector.id = str(uuid.uuid4())
            self._vectors[vector.id] = vector
            added_ids.append(vector.id)

        self.vector_count = len(self._vectors)
        self.index_status = IndexStatus.READY
        return added_ids

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the store

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs that were added
        """
        if not documents:
            return []

        added_ids = []
        vectors = []

        for document in documents:
            # Generate ID if not provided
            if not document.id:
                import uuid
                document.id = str(uuid.uuid4())

            # Store document
            self._documents[document.id] = document

            # Create vector if embedding is provided
            if document.embedding:
                vector = Vector(
                    id=document.id,
                    embedding=document.embedding,
                    metadata=document.metadata
                )
                vectors.append(vector)

            added_ids.append(document.id)

        # Add vectors if any
        if vectors:
            vector_ids = self.add_vectors(vectors)
            # Ensure document IDs match vector IDs
            for doc_id, vector_id in zip(added_ids, vector_ids):
                if doc_id != vector_id:
                    # Update document ID to match vector ID
                    doc = self._documents.pop(doc_id)
                    doc.id = vector_id
                    self._documents[vector_id] = doc

        return added_ids

    def search(self, query: VectorStoreQuery) -> VectorStoreResult:
        """
        Search for similar vectors

        Args:
            query: Search query

        Returns:
            Search results
        """
        start_time = time.time()
        self._validate_query(query)

        if not FAISS_AVAILABLE:
            return self._fallback_search(query, start_time)

        if not self._vectors:
            return VectorStoreResult(
                query_time=time.time() - start_time,
                total_count=0
            )

        # Prepare query vector
        query_vector = np.array([query.query_vector], dtype=np.float32)

        # Normalize for cosine similarity
        if self.config.metric == DistanceMetric.COSINE:
            query_vector = self._normalize_vector(query_vector)

        # Search FAISS index
        k = min(query.top_k, self._index.ntotal)
        distances, indices = self._index.search(query_vector, k)

        # Convert to results
        result_vectors = []
        result_documents = []
        result_scores = []

        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue

            vector_id = self._index_to_id.get(idx)
            if not vector_id:
                continue

            vector = self._vectors[vector_id]

            # Apply score threshold
            score = self._convert_distance_to_score(dist)
            if query.score_threshold is not None and score < query.score_threshold:
                continue

            # Apply metadata filter
            if query.filter_dict and not self._matches_filter(vector.metadata, query.filter_dict):
                continue

            result_vectors.append(vector)
            result_scores.append(score)

            # Add document if available
            if vector_id in self._documents:
                result_documents.append(self._documents[vector_id])

            if len(result_vectors) >= query.top_k:
                break

        query_time = time.time() - start_time

        return VectorStoreResult(
            vectors=result_vectors,
            documents=result_documents,
            scores=result_scores,
            query_time=query_time,
            total_count=len(self._vectors),
            metadata={
                "search_method": "faiss_search",
                "index_type": self.config.index_type,
                "vectors_examined": self._index.ntotal
            }
        )

    def _fallback_search(self, query: VectorStoreQuery, start_time: float) -> VectorStoreResult:
        """Fallback search when FAISS is not available"""
        # Simple linear search fallback
        result_vectors = []
        result_scores = []

        for vector in self._vectors.values():
            similarity = self._calculate_similarity(query.query_vector, vector.embedding)

            if query.score_threshold is not None and similarity < query.score_threshold:
                continue

            if query.filter_dict and not self._matches_filter(vector.metadata, query.filter_dict):
                continue

            result_vectors.append(vector)
            result_scores.append(similarity)

        # Sort by score and take top_k
        sorted_results = sorted(zip(result_vectors, result_scores), key=lambda x: x[1], reverse=True)
        top_k = min(query.top_k, len(sorted_results))

        if top_k == 0:
            return VectorStoreResult(
                query_time=time.time() - start_time,
                total_count=len(self._vectors)
            )

        result_vectors = [r[0] for r in sorted_results[:top_k]]
        result_scores = [r[1] for r in sorted_results[:top_k]]

        return VectorStoreResult(
            vectors=result_vectors,
            scores=result_scores,
            query_time=time.time() - start_time,
            total_count=len(self._vectors),
            metadata={
                "search_method": "linear_fallback_search",
                "vectors_examined": len(self._vectors)
            }
        )

    def delete_vectors(self, vector_ids: List[str]) -> bool:
        """
        Delete vectors from the store

        Args:
            vector_ids: List of vector IDs to delete

        Returns:
            True if deletion was successful
        """
        if not FAISS_AVAILABLE:
            return self._fallback_delete_vectors(vector_ids)

        if not vector_ids:
            return True

        # FAISS doesn't support efficient deletion, so we need to rebuild
        deleted_count = 0
        vectors_to_keep = []

        for vector_id in vector_ids:
            if vector_id in self._vectors:
                del self._vectors[vector_id]
                deleted_count += 1

            if vector_id in self._documents:
                del self._documents[vector_id]

            if vector_id in self._id_to_index:
                del self._id_to_index[vector_id]

        if deleted_count > 0:
            # Rebuild index with remaining vectors
            self._rebuild_index()

        return deleted_count == len(vector_ids)

    def _fallback_delete_vectors(self, vector_ids: List[str]) -> bool:
        """Fallback deletion when FAISS is not available"""
        deleted_count = 0
        for vector_id in vector_ids:
            if vector_id in self._vectors:
                del self._vectors[vector_id]
                deleted_count += 1
            if vector_id in self._documents:
                del self._documents[vector_id]

        if deleted_count > 0:
            self.vector_count = len(self._vectors)
            if self.vector_count == 0:
                self.index_status = IndexStatus.EMPTY

        return deleted_count == len(vector_ids)

    def get_vector(self, vector_id: str) -> Optional[Vector]:
        """
        Get a specific vector by ID

        Args:
            vector_id: Vector ID

        Returns:
            Vector if found, None otherwise
        """
        return self._vectors.get(vector_id)

    def update_vector(self, vector_id: str, vector: Vector) -> bool:
        """
        Update a vector in the store

        Args:
            vector_id: Vector ID to update
            vector: New vector data

        Returns:
            True if update was successful
        """
        if vector_id not in self._vectors:
            return False

        # Validate new vector
        self._validate_vectors([vector])

        # Delete old vector and add new one
        self.delete_vectors([vector_id])
        vector.id = vector_id
        self.add_vectors([vector])

        return True

    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index"""
        if not FAISS_AVAILABLE or not self._vectors:
            self._initialize_index()
            return

        # Get all remaining vectors
        vectors = list(self._vectors.values())
        if not vectors:
            self._initialize_index()
            return

        # Clear and reinitialize index
        self._initialize_index()
        self._id_to_index.clear()
        self._index_to_id.clear()

        # Re-add all vectors
        self.add_vectors(vectors)

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length"""
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)  # Avoid division by zero
        return vector / norm

    def _convert_distance_to_score(self, distance: float) -> float:
        """Convert FAISS distance to similarity score"""
        if self.config.metric == DistanceMetric.EUCLIDEAN:
            # Convert Euclidean distance to similarity
            return 1.0 / (1.0 + distance)
        elif self.config.metric == DistanceMetric.COSINE:
            # FAISS returns negative inner product for normalized vectors
            return max(0.0, distance)
        elif self.config.metric == DistanceMetric.DOT_PRODUCT:
            # FAISS returns negative inner product for IP index
            return -distance
        else:
            return distance

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        if not filter_dict:
            return True

        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False

        return True

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get detailed FAISS index statistics

        Returns:
            Index statistics dictionary
        """
        stats = self.get_store_info()

        if FAISS_AVAILABLE and self._index:
            stats.update({
                "index_ntotal": self._index.ntotal,
                "index_is_trained": self._index.is_trained,
                "faiss_available": True,
                "index_memory_usage": self._estimate_index_memory()
            })
        else:
            stats.update({
                "faiss_available": False,
                "fallback_mode": True
            })

        return stats

    def _estimate_index_memory(self) -> float:
        """Estimate index memory usage in MB"""
        if not FAISS_AVAILABLE or not self._index:
            return 0.0

        # Rough estimation
        vector_count = self._index.ntotal
        dimension = self.config.dimension
        # Each float is 4 bytes, plus overhead
        bytes_used = vector_count * dimension * 4 * 1.5
        return bytes_used / (1024 * 1024)