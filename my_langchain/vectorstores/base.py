# -*- coding: utf-8 -*-
"""
Base vector store implementation
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Sequence
from my_langchain.vectorstores.types import (
    VectorStoreConfig, VectorStoreResult, VectorStoreQuery, Vector, Document,
    VectorStoreError, VectorStoreValidationError, VectorStoreRetrievalError,
    DistanceMetric, IndexStatus
)
from pydantic import ConfigDict, Field
import numpy as np


class BaseVectorStore(ABC):
    """
    Base vector store implementation providing common functionality

    This class defines the interface that all vector store implementations must follow
    and provides common utility methods for vector storage, retrieval, and management.
    """

    def __init__(self, config: VectorStoreConfig, **kwargs):
        """
        Initialize vector store

        Args:
            config: Vector store configuration
            **kwargs: Additional parameters
        """
        self.config = config
        self.index_status = IndexStatus.EMPTY
        self.vector_count = 0

    @abstractmethod
    def add_vectors(self, vectors: List[Vector]) -> List[str]:
        """
        Add vectors to the store

        Args:
            vectors: List of vectors to add

        Returns:
            List of vector IDs that were added

        Raises:
            VectorStoreValidationError: If vectors are invalid
            VectorStoreIndexError: If indexing fails
        """
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the store

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs that were added

        Raises:
            VectorStoreValidationError: If documents are invalid
            VectorStoreIndexError: If indexing fails
        """
        pass

    @abstractmethod
    def search(self, query: VectorStoreQuery) -> VectorStoreResult:
        """
        Search for similar vectors

        Args:
            query: Search query

        Returns:
            Search results

        Raises:
            VectorStoreRetrievalError: If search fails
        """
        pass

    @abstractmethod
    def delete_vectors(self, vector_ids: List[str]) -> bool:
        """
        Delete vectors from the store

        Args:
            vector_ids: List of vector IDs to delete

        Returns:
            True if deletion was successful

        Raises:
            VectorStoreError: If deletion fails
        """
        pass

    @abstractmethod
    def get_vector(self, vector_id: str) -> Optional[Vector]:
        """
        Get a specific vector by ID

        Args:
            vector_id: Vector ID

        Returns:
            Vector if found, None otherwise

        Raises:
            VectorStoreError: If retrieval fails
        """
        pass

    @abstractmethod
    def update_vector(self, vector_id: str, vector: Vector) -> bool:
        """
        Update a vector in the store

        Args:
            vector_id: Vector ID to update
            vector: New vector data

        Returns:
            True if update was successful

        Raises:
            VectorStoreError: If update fails
        """
        pass

    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> VectorStoreResult:
        """
        Convenience method for similarity search

        Args:
            query_vector: Query embedding
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filter_dict: Metadata filter

        Returns:
            Search results
        """
        query = VectorStoreQuery(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_dict=filter_dict
        )
        return self.search(query)

    def max_marginal_relevance_search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        lambda_mult: float = 0.5,
        fetch_k: int = 20
    ) -> VectorStoreResult:
        """
        Maximal marginal relevance search

        Args:
            query_vector: Query embedding
            top_k: Number of results to return
            lambda_mult: Diversity parameter (0-1)
            fetch_k: Number of candidates to fetch initially

        Returns:
            Diverse search results
        """
        # Fetch more candidates initially
        query = VectorStoreQuery(query_vector=query_vector, top_k=fetch_k)
        candidates = self.search(query)

        if not candidates.vectors or len(candidates.vectors) <= top_k:
            return candidates

        # Apply maximal marginal relevance
        selected_vectors = []
        selected_scores = []
        remaining_candidates = list(zip(candidates.vectors, candidates.scores))

        # Select first vector (most similar)
        selected_vectors.append(remaining_candidates[0][0])
        selected_scores.append(remaining_candidates[0][1])
        remaining_candidates.pop(0)

        # Select remaining vectors based on MMR
        while len(selected_vectors) < top_k and remaining_candidates:
            best_score = -float('inf')
            best_idx = 0

            for i, (candidate, relevance) in enumerate(remaining_candidates):
                # Calculate maximum similarity to already selected vectors
                max_similarity = 0.0
                for selected in selected_vectors:
                    similarity = self._calculate_similarity(
                        candidate.embedding,
                        selected.embedding
                    )
                    max_similarity = max(max_similarity, similarity)

                # Calculate MMR score
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected_vectors.append(remaining_candidates[best_idx][0])
            selected_scores.append(best_score)
            remaining_candidates.pop(best_idx)

        # Create result with selected vectors
        result = VectorStoreResult(
            vectors=selected_vectors,
            scores=selected_scores,
            query_time=candidates.query_time,
            total_count=candidates.total_count
        )

        return result

    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate similarity between two vectors based on configured metric

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score
        """
        if self.config.metric == DistanceMetric.COSINE:
            return self._cosine_similarity(vec1, vec2)
        elif self.config.metric == DistanceMetric.EUCLIDEAN:
            # Convert distance to similarity (higher is better)
            distance = self._euclidean_distance(vec1, vec2)
            return 1.0 / (1.0 + distance)
        elif self.config.metric == DistanceMetric.MANHATTAN:
            # Convert distance to similarity
            distance = self._manhattan_distance(vec1, vec2)
            return 1.0 / (1.0 + distance)
        elif self.config.metric == DistanceMetric.DOT_PRODUCT:
            return self._dot_product(vec1, vec2)
        else:
            raise VectorStoreError(
                f"Unsupported distance metric: {self.config.metric}",
                self.__class__.__name__
            )

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        if not vec1 or not vec2:
            return 0.0

        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)

        norm1 = np.linalg.norm(vec1_array)
        norm2 = np.linalg.norm(vec2_array)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1_array, vec2_array) / (norm1 * norm2)

    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate Euclidean distance"""
        if not vec1 or not vec2:
            return float('inf')

        return np.linalg.norm(np.array(vec1) - np.array(vec2))

    def _manhattan_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate Manhattan distance"""
        if not vec1 or not vec2:
            return float('inf')

        return np.sum(np.abs(np.array(vec1) - np.array(vec2)))

    def _dot_product(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate dot product"""
        if not vec1 or not vec2:
            return 0.0

        return np.dot(np.array(vec1), np.array(vec2))

    def _validate_vectors(self, vectors: List[Vector]) -> None:
        """
        Validate vectors before adding to store

        Args:
            vectors: Vectors to validate

        Raises:
            VectorStoreValidationError: If vectors are invalid
        """
        if not vectors:
            raise VectorStoreValidationError(
                "No vectors provided",
                self.__class__.__name__
            )

        for vector in vectors:
            if not vector.embedding:
                raise VectorStoreValidationError(
                    f"Vector {vector.id} has empty embedding",
                    self.__class__.__name__,
                    vector
                )

            if len(vector.embedding) != self.config.dimension:
                raise VectorStoreValidationError(
                    f"Vector {vector.id} has dimension {len(vector.embedding)}, "
                    f"expected {self.config.dimension}",
                    self.__class__.__name__,
                    vector
                )

    def _validate_query(self, query: VectorStoreQuery) -> None:
        """
        Validate search query

        Args:
            query: Query to validate

        Raises:
            VectorStoreValidationError: If query is invalid
        """
        if not query.query_vector:
            raise VectorStoreValidationError(
                "Query vector is empty",
                self.__class__.__name__,
                query
            )

        if len(query.query_vector) != self.config.dimension:
            raise VectorStoreValidationError(
                f"Query vector has dimension {len(query.query_vector)}, "
                f"expected {self.config.dimension}",
                self.__class__.__name__,
                query
            )

        if query.top_k <= 0:
            raise VectorStoreValidationError(
                f"top_k must be positive, got {query.top_k}",
                self.__class__.__name__,
                query
            )

    def get_store_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store

        Returns:
            Store information dictionary
        """
        return {
            "store_type": self.__class__.__name__,
            "config": self.config.model_dump(),
            "index_status": self.index_status,
            "vector_count": self.vector_count,
            "dimension": self.config.dimension,
            "metric": self.config.metric,
            "index_type": self.config.index_type
        }

    def __len__(self) -> int:
        """Get number of vectors in store"""
        return self.vector_count

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vectors={self.vector_count}, dimension={self.config.dimension})"