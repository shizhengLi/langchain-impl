# -*- coding: utf-8 -*-
"""
In-memory vector store implementation
"""

import time
from typing import Any, Dict, List, Optional, Set
import numpy as np

from my_langchain.vectorstores.base import BaseVectorStore
from my_langchain.vectorstores.types import (
    VectorStoreConfig, VectorStoreResult, VectorStoreQuery, Vector, Document,
    VectorStoreValidationError, VectorStoreRetrievalError, VectorStoreIndexError,
    IndexStatus
)
import uuid


class InMemoryVectorStore(BaseVectorStore):
    """
    In-memory vector store implementation

    This is a simple vector store that keeps all vectors in memory.
    Suitable for testing, development, and small datasets.
    """

    def __init__(self, config: VectorStoreConfig, **kwargs):
        """
        Initialize in-memory vector store

        Args:
            config: Vector store configuration
            **kwargs: Additional parameters
        """
        super().__init__(config=config, **kwargs)

        # Internal storage
        self._vectors: Dict[str, Vector] = {}
        self._documents: Dict[str, Document] = {}
        self._vector_embeddings: Dict[str, List[float]] = {}

        # Build index for efficient search
        self._embedding_matrix: Optional[np.ndarray] = None
        self._vector_ids: List[str] = []
        self._index_needs_rebuild = True

    def add_vectors(self, vectors: List[Vector]) -> List[str]:
        """
        Add vectors to the store

        Args:
            vectors: List of vectors to add

        Returns:
            List of vector IDs that were added
        """
        self._validate_vectors(vectors)

        added_ids = []
        for vector in vectors:
            # Generate ID if not provided
            if not vector.id:
                vector.id = str(uuid.uuid4())

            # Store vector
            self._vectors[vector.id] = vector
            self._vector_embeddings[vector.id] = vector.embedding

            added_ids.append(vector.id)

        # Mark index for rebuild
        self._index_needs_rebuild = True
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

        if not self._vectors:
            return VectorStoreResult(
                query_time=time.time() - start_time,
                total_count=0
            )

        # Rebuild index if needed
        if self._index_needs_rebuild:
            self._rebuild_index()

        # Calculate similarities and apply filters
        query_vector = np.array(query.query_vector)
        similarities = []

        # Get all valid indices after filtering
        valid_indices = []

        for i, vector_id in enumerate(self._vector_ids):
            vector = self._vectors[vector_id]

            # Apply metadata filter first
            if query.filter_dict and not self._matches_filter(vector.metadata, query.filter_dict):
                continue

            # Calculate similarity
            vector_embedding = self._vector_embeddings[vector_id]
            similarity = self._calculate_similarity(
                query.query_vector,
                vector_embedding
            )

            # Apply score threshold
            if query.score_threshold is not None and similarity < query.score_threshold:
                continue

            valid_indices.append(i)
            similarities.append(similarity)

        # Get indices of top-k results from filtered similarities
        top_k = min(query.top_k, len(valid_indices))
        if top_k == 0:
            top_indices = []
        else:
            # Get top_k indices from valid_indices based on similarity scores
            sorted_pairs = sorted(zip(valid_indices, similarities), key=lambda x: x[1], reverse=True)
            top_indices = [pair[0] for pair in sorted_pairs[:top_k]]
            similarities = [pair[1] for pair in sorted_pairs[:top_k]]

        # Collect results
        result_vectors = []
        result_documents = []
        result_scores = []

        for i, idx in enumerate(top_indices):
            vector_id = self._vector_ids[idx]
            vector = self._vectors[vector_id]
            similarity = similarities[i]

            result_vectors.append(vector)
            result_scores.append(similarity)

            # Add document if available
            if vector_id in self._documents:
                result_documents.append(self._documents[vector_id])

        query_time = time.time() - start_time

        return VectorStoreResult(
            vectors=result_vectors,
            documents=result_documents,
            scores=result_scores,
            query_time=query_time,
            total_count=len(self._vectors),
            metadata={
                "search_method": "linear_search",
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
        if not vector_ids:
            return True

        deleted_count = 0
        for vector_id in vector_ids:
            if vector_id in self._vectors:
                del self._vectors[vector_id]
                del self._vector_embeddings[vector_id]
                deleted_count += 1

            if vector_id in self._documents:
                del self._documents[vector_id]

        if deleted_count > 0:
            self._index_needs_rebuild = True
            self.vector_count = len(self._vectors)

            if self.vector_count == 0:
                self.index_status = IndexStatus.EMPTY
            else:
                self.index_status = IndexStatus.READY

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

        # Update vector
        vector.id = vector_id  # Ensure ID matches
        self._vectors[vector_id] = vector
        self._vector_embeddings[vector_id] = vector.embedding

        # Mark index for rebuild
        self._index_needs_rebuild = True

        return True

    def _rebuild_index(self) -> None:
        """Rebuild the internal index for efficient search"""
        if not self._vectors:
            self._embedding_matrix = None
            self._vector_ids = []
            return

        # Create embedding matrix
        embeddings = []
        self._vector_ids = []

        for vector_id, vector in self._vectors.items():
            embeddings.append(vector.embedding)
            self._vector_ids.append(vector_id)

        self._embedding_matrix = np.array(embeddings)
        self._index_needs_rebuild = False

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """
        Check if metadata matches filter criteria

        Args:
            metadata: Vector metadata
            filter_dict: Filter criteria

        Returns:
            True if metadata matches filter
        """
        if not filter_dict:
            return True

        for key, value in filter_dict.items():
            if key not in metadata:
                return False

            if isinstance(value, dict):
                # Handle nested filters
                if not self._matches_filter(metadata[key], value):
                    return False
            elif metadata[key] != value:
                return False

        return True

    def clear(self) -> None:
        """Clear all vectors and documents from the store"""
        self._vectors.clear()
        self._documents.clear()
        self._vector_embeddings.clear()
        self._embedding_matrix = None
        self._vector_ids = []
        self._index_needs_rebuild = True
        self.vector_count = 0
        self.index_status = IndexStatus.EMPTY

    def get_all_vectors(self) -> List[Vector]:
        """
        Get all vectors in the store

        Returns:
            List of all vectors
        """
        return list(self._vectors.values())

    def get_all_documents(self) -> List[Document]:
        """
        Get all documents in the store

        Returns:
            List of all documents
        """
        return list(self._documents.values())

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the store

        Returns:
            Statistics dictionary
        """
        stats = self.get_store_info()
        stats.update({
            "memory_usage_mb": self._estimate_memory_usage(),
            "index_built": not self._index_needs_rebuild,
            "has_documents": len(self._documents) > 0,
            "avg_vector_dimension": np.mean([
                len(v.embedding) for v in self._vectors.values()
            ]) if self._vectors else 0
        })
        return stats

    def _estimate_memory_usage(self) -> float:
        """
        Estimate memory usage in MB

        Returns:
            Memory usage in megabytes
        """
        total_elements = sum(
            len(vector.embedding) for vector in self._vectors.values()
        )
        # Each float is 8 bytes
        bytes_used = total_elements * 8
        return bytes_used / (1024 * 1024)  # Convert to MB