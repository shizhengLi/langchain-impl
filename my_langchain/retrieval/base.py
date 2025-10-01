# -*- coding: utf-8 -*-
"""
Base retriever implementation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import time
import logging
from collections import defaultdict

from .types import (
    RetrievalConfig, RetrievalResult, RetrievalError,
    RetrievalValidationError, RetrievalProcessingError,
    Document, RetrievedDocument, RetrievalQuery,
    calculate_retrieval_metrics
)


class BaseRetriever(ABC):
    """
    Base retriever class providing common functionality

    This class defines the interface that all retriever implementations must follow
    and provides common utility methods for document retrieval and processing.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None, **kwargs):
        """
        Initialize retriever

        Args:
            config: Retrieval configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or RetrievalConfig(**kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_retriever()

    def _setup_retriever(self):
        """Setup retriever-specific configuration"""
        # Override in subclasses if needed
        pass

    @abstractmethod
    def _retrieve_documents(self, query: str, config: RetrievalConfig) -> List[RetrievedDocument]:
        """
        Retrieve documents for a query

        Args:
            query: Query text
            config: Retrieval configuration

        Returns:
            List of retrieved documents

        Raises:
            RetrievalProcessingError: If retrieval fails
        """
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the retriever

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs

        Raises:
            RetrievalProcessingError: If adding documents fails
        """
        pass

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Retrieve documents for a query

        Args:
            query: Query text

        Returns:
            RetrievalResult object

        Raises:
            RetrievalValidationError: If query is invalid
            RetrievalProcessingError: If retrieval fails
        """
        self._validate_query(query)

        start_time = time.time()
        try:
            # Apply caching if enabled
            if self.config.enable_caching:
                cached_result = self._get_cached_result(query)
                if cached_result is not None:
                    return cached_result

            # Perform retrieval
            documents = self._retrieve_documents(query, self.config)

            # Apply post-processing
            documents = self._post_process_results(documents)

            # Create result
            search_time = time.time() - start_time
            result = RetrievalResult(
                documents=documents,
                query=query,
                total_results=len(documents),
                search_time=search_time,
                retrieval_method=self.__class__.__name__,
                metadata=self._get_result_metadata()
            )

            # Cache result if enabled
            if self.config.enable_caching:
                self._cache_result(query, result)

            return result

        except Exception as e:
            if isinstance(e, RetrievalError):
                raise
            raise RetrievalProcessingError(
                f"Failed to retrieve documents: {str(e)}",
                self.__class__.__name__,
                {"query": query[:100]}  # Truncate for logging
            ) from e

    def retrieve_with_config(self, query: str, config: RetrievalConfig) -> RetrievalResult:
        """
        Retrieve documents with custom configuration

        Args:
            query: Query text
            config: Custom retrieval configuration

        Returns:
            RetrievalResult object
        """
        # Store original config
        original_config = self.config

        try:
            # Use custom config temporarily
            self.config = config
            result = self.retrieve(query)
            return result
        finally:
            # Restore original config
            self.config = original_config

    def batch_retrieve(self, queries: List[str]) -> List[RetrievalResult]:
        """
        Retrieve documents for multiple queries

        Args:
            queries: List of query texts

        Returns:
            List of RetrievalResult objects
        """
        self._validate_queries(queries)

        results = []
        for query in queries:
            try:
                result = self.retrieve(query)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to retrieve for query '{query}': {str(e)}")
                # Create empty result for failed query
                empty_result = RetrievalResult(
                    documents=[],
                    query=query,
                    total_results=0,
                    search_time=0.0,
                    retrieval_method=self.__class__.__name__ + "_failed",
                    metadata={"error": str(e)}
                )
                results.append(empty_result)

        return results

    def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Get relevant documents as plain Document objects

        Args:
            query: Query text
            k: Number of documents to return

        Returns:
            List of Document objects
        """
        result = self.retrieve(query)

        if k is not None:
            docs = result.documents[:k]
        else:
            docs = result.documents

        # Convert to plain Document objects
        return [
            Document(
                content=doc.content,
                metadata=doc.metadata,
                id=doc.id
            )
            for doc in docs
        ]

    def async_retrieve(self, query: str) -> RetrievalResult:
        """
        Asynchronously retrieve documents

        Args:
            query: Query text

        Returns:
            RetrievalResult object
        """
        # Default synchronous implementation
        # Override in subclasses for true async support
        return self.retrieve(query)

    def _post_process_results(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        Post-process retrieval results

        Args:
            documents: List of retrieved documents

        Returns:
            Processed list of documents
        """
        # Apply score threshold
        if self.config.score_threshold is not None:
            documents = [
                doc for doc in documents
                if doc.relevance_score >= self.config.score_threshold
            ]

        # Apply top_k limit
        if self.config.top_k is not None:
            documents = documents[:self.config.top_k]

        # Normalize scores if required
        if self.config.normalize_scores and documents:
            max_score = max(doc.relevance_score for doc in documents)
            min_score = min(doc.relevance_score for doc in documents)

            if max_score > min_score:
                for doc in documents:
                    doc.relevance_score = (doc.relevance_score - min_score) / (max_score - min_score)

        # Rerank if required
        if self.config.rerank and len(documents) > 1:
            documents = self._rerank_documents(documents)

        return documents

    def _rerank_documents(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        Rerank documents (default implementation by relevance score)

        Args:
            documents: List of documents to rerank

        Returns:
            Reranked list of documents
        """
        # Default reranking by relevance score (already sorted)
        # Override in subclasses for custom reranking
        reranked_docs = documents[:self.config.rerank_top_k]

        # Update ranks
        for i, doc in enumerate(reranked_docs):
            doc.rank = i

        return reranked_docs

    def _validate_query(self, query: str):
        """
        Validate input query

        Args:
            query: Query to validate

        Raises:
            RetrievalValidationError: If query is invalid
        """
        if not isinstance(query, str):
            raise RetrievalValidationError("Query must be a string")

        if not query.strip():
            raise RetrievalValidationError("Query cannot be empty")

    def _validate_queries(self, queries: List[str]):
        """
        Validate input queries

        Args:
            queries: List of queries to validate

        Raises:
            RetrievalValidationError: If queries are invalid
        """
        if not isinstance(queries, list):
            raise RetrievalValidationError("Queries must be a list")

        if not queries:
            raise RetrievalValidationError("Queries list cannot be empty")

        for i, query in enumerate(queries):
            try:
                self._validate_query(query)
            except RetrievalValidationError as e:
                raise RetrievalValidationError(f"Query at index {i} is invalid: {str(e)}") from e

    def _get_result_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for retrieval result

        Returns:
            Dictionary with metadata
        """
        return {
            "retriever_type": self.__class__.__name__,
            "config": self.config.model_dump(),
            "timestamp": time.time()
        }

    def _get_cached_result(self, query: str) -> Optional[RetrievalResult]:
        """
        Get cached result for query (placeholder implementation)

        Args:
            query: Query text

        Returns:
            Cached result or None
        """
        # Override in subclasses for actual caching
        return None

    def _cache_result(self, query: str, result: RetrievalResult):
        """
        Cache retrieval result (placeholder implementation)

        Args:
            query: Query text
            result: Result to cache
        """
        # Override in subclasses for actual caching
        pass

    def evaluate_retrieval(
        self,
        queries: List[str],
        relevant_docs: Dict[str, List[str]],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance

        Args:
            queries: List of test queries
            relevant_docs: Dictionary mapping query to relevant document IDs
            k: Cut-off for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        if len(queries) != len(relevant_docs):
            raise RetrievalValidationError("Queries and relevant_docs must have same length")

        all_metrics = []

        for query in queries:
            if query not in relevant_docs:
                continue

            try:
                result = self.retrieve(query)
                metrics = calculate_retrieval_metrics(
                    result.documents,
                    relevant_docs[query],
                    k
                )
                all_metrics.append(metrics)
            except Exception as e:
                self.logger.error(f"Failed to evaluate query '{query}': {str(e)}")

        if not all_metrics:
            return {"error": "No successful retrievals for evaluation"}

        # Average metrics across all queries
        avg_metrics = sum(all_metrics, type(all_metrics[0])()) / len(all_metrics)

        return {
            "precision_at_k": avg_metrics.precision,
            "recall_at_k": avg_metrics.recall,
            "f1_score_at_k": avg_metrics.f1_score,
            "hit_rate": avg_metrics.hit_rate,
            "mean_reciprocal_rank": avg_metrics.mean_reciprocal_rank,
            "mean_average_precision": avg_metrics.mean_average_precision,
            "num_queries": len(all_metrics)
        }

    def get_document_count(self) -> int:
        """
        Get total number of documents in retriever

        Returns:
            Number of documents
        """
        # Override in subclasses
        return 0

    def get_retriever_info(self) -> Dict[str, Any]:
        """
        Get information about the retriever

        Returns:
            Dictionary with retriever information
        """
        return {
            "retriever_type": self.__class__.__name__,
            "config": self.config.model_dump(),
            "document_count": self.get_document_count()
        }

    def clear_cache(self):
        """Clear retrieval cache (placeholder implementation)"""
        # Override in subclasses for actual cache clearing
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"search_type={self.config.search_type}, "
            f"top_k={self.config.top_k}"
            f")"
        )