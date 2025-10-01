# -*- coding: utf-8 -*-
"""
Vector retriever implementation
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .base import BaseRetriever
from .types import (
    RetrievalConfig, RetrievalResult, RetrievalProcessingError,
    Document, RetrievedDocument
)
from ..embeddings import BaseEmbedding
from ..vectorstores import BaseVectorStore


class VectorRetriever(BaseRetriever):
    """
    Vector-based retriever using embedding similarity

    This retriever performs document retrieval based on vector similarity
    between query embeddings and document embeddings.
    """

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        vector_store: BaseVectorStore,
        config: Optional[RetrievalConfig] = None,
        **kwargs
    ):
        """
        Initialize vector retriever

        Args:
            embedding_model: Embedding model for converting text to vectors
            vector_store: Vector store for storing and searching embeddings
            config: Retrieval configuration
            **kwargs: Additional configuration parameters
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        super().__init__(config, **kwargs)
        self._last_retrieval_method = "vector_similarity"

    def _setup_retriever(self):
        """Setup vector-specific configuration"""
        self.query_cache = {}  # Simple query cache
        self.embedding_cache = {}  # Embedding cache

    def retrieve(self, query: str) -> 'RetrievalResult':
        """
        Override retrieve method to provide custom retrieval method name
        """
        from .types import RetrievalResult

        # Use parent's retrieve method but override the retrieval_method
        result = super().retrieve(query)

        # Update retrieval method if we have a custom one
        if hasattr(self, '_last_retrieval_method'):
            # Create new result with updated retrieval method
            result = RetrievalResult(
                documents=result.documents,
                query=result.query,
                total_results=result.total_results,
                search_time=result.search_time,
                retrieval_method=self._last_retrieval_method,
                metadata=result.metadata.copy()
            )

        return result

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the retriever

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        # Generate embeddings for all documents
        texts = [doc.content for doc in documents]
        try:
            embedding_result = self.embedding_model.embed_texts(texts)
        except Exception as e:
            raise RetrievalProcessingError(
                f"Failed to generate embeddings: {str(e)}",
                "VectorRetriever"
            ) from e

        # Create vector objects and add them to vector store
        vectors = []
        for i, (doc, embedding) in enumerate(zip(documents, embedding_result.embeddings)):
            from ..vectorstores.types import Vector
            vector = Vector(
                id=doc.id,
                embedding=embedding.vector,
                metadata={
                    "content": doc.content,
                    **doc.metadata
                }
            )
            vectors.append(vector)

        # Add all vectors at once
        doc_ids = self.vector_store.add_vectors(vectors)

        return doc_ids

    def _retrieve_documents(self, query: str, config: RetrievalConfig) -> List[RetrievedDocument]:
        """
        Retrieve documents for a query

        Args:
            query: Query text
            config: Retrieval configuration

        Returns:
            List of retrieved documents
        """
        # Generate query embedding
        query_embedding = self._get_query_embedding(query)

        # Search in vector store
        if config.search_type == "similarity":
            search_results = self._similarity_search(query_embedding, config)
        elif config.search_type == "mmr":
            search_results = self._mmr_search(query_embedding, config)
        else:
            search_results = self._similarity_search(query_embedding, config)

        # Convert to RetrievedDocument objects
        retrieved_docs = []
        retrieval_method = f"vector_{config.search_type}"

        for i, result in enumerate(search_results):
            # Extract document content from metadata
            content = result.metadata.get("content", "")
            if not content:
                continue

            # Remove content from metadata to avoid duplication
            metadata = result.metadata.copy()
            metadata.pop("content", None)

            # Normalize similarity score to [0, 1] range
            # Cosine similarity ranges from -1 to 1, so we normalize it
            raw_score = result.score
            normalized_score = (raw_score + 1.0) / 2.0  # Maps [-1, 1] to [0, 1]
            normalized_score = max(0.0, min(1.0, normalized_score))  # Clamp to [0, 1]

            retrieved_doc = RetrievedDocument(
                content=content,
                metadata=metadata,
                id=result.id,
                relevance_score=normalized_score,
                retrieval_method=retrieval_method,
                query=query,
                rank=i,
                additional_info={
                    "vector_score": raw_score,
                    "normalized_score": normalized_score,
                    "embedding_distance": getattr(result, 'distance', None)
                }
            )
            retrieved_docs.append(retrieved_doc)

        # Store retrieval method for result metadata
        self._last_retrieval_method = retrieval_method
        return retrieved_docs

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for query (with caching)

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        # Check cache first
        if query in self.embedding_cache:
            return self.embedding_cache[query]

        # Generate embedding
        try:
            embedding_result = self.embedding_model.embed_text(query)
            embedding = embedding_result.vector
        except Exception as e:
            raise RetrievalProcessingError(
                f"Failed to generate query embedding: {str(e)}",
                "VectorRetriever"
            ) from e

        # Cache the embedding
        self.embedding_cache[query] = embedding

        return embedding

    def _similarity_search(self, query_embedding: List[float], config: RetrievalConfig) -> List[Any]:
        """
        Perform similarity search

        Args:
            query_embedding: Query embedding vector
            config: Retrieval configuration

        Returns:
            List of search results with scores
        """
        # Prepare search parameters
        top_k = config.fetch_k if config.search_type == "mmr" else config.top_k

        # Call similarity_search with correct parameter name
        try:
            result = self.vector_store.similarity_search(
                query_vector=query_embedding,
                top_k=top_k,
                score_threshold=config.score_threshold,
                filter_dict=config.filter_dict if config.filter_dict else None
            )

            # Combine vectors with their scores using simple wrapper objects
            search_results = []
            if hasattr(result, 'vectors') and hasattr(result, 'scores'):
                for i, vector in enumerate(result.vectors):
                    score = result.scores[i] if i < len(result.scores) else 0.0
                    # Create a simple wrapper object that has both vector data and score
                    vector_with_score = type('VectorWithScore', (), {
                        'id': vector.id,
                        'embedding': vector.embedding,
                        'metadata': vector.metadata,
                        'score': score,
                        'vector': vector.embedding  # for MMR calculations
                    })()
                    search_results.append(vector_with_score)

            return search_results
        except Exception as e:
            raise RetrievalProcessingError(
                f"Vector store search failed: {str(e)}",
                "VectorRetriever"
            ) from e

    def _mmr_search(self, query_embedding: List[float], config: RetrievalConfig) -> List[Any]:
        """
        Perform Maximal Marginal Relevance (MMR) search

        Args:
            query_embedding: Query embedding vector
            config: Retrieval configuration

        Returns:
            List of search results
        """
        # First, fetch more documents
        fetch_config = RetrievalConfig(
            top_k=config.fetch_k,
            search_type="similarity",
            filter_dict=config.filter_dict
        )
        candidates = self._similarity_search(query_embedding, fetch_config)

        if not candidates:
            return []

        # Apply MMR algorithm
        selected_results = []
        remaining_candidates = candidates.copy()

        # Select the most relevant document first
        if remaining_candidates:
            selected_results.append(remaining_candidates.pop(0))

        # Select remaining documents using MMR
        while len(selected_results) < config.top_k and remaining_candidates:
            best_candidate = None
            best_score = -1

            for candidate in remaining_candidates:
                # Calculate MMR score
                relevance_score = candidate.score
                diversity_penalty = self._calculate_max_similarity(
                    candidate,
                    selected_results
                )

                mmr_score = (
                    config.mmr_lambda * relevance_score -
                    (1 - config.mmr_lambda) * diversity_penalty
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate

            if best_candidate:
                selected_results.append(best_candidate)
                remaining_candidates.remove(best_candidate)
            else:
                break

        return selected_results

    def _calculate_max_similarity(self, candidate: Any, selected_results: List[Any]) -> float:
        """
        Calculate maximum similarity between candidate and selected results

        Args:
            candidate: Candidate result
            selected_results: Already selected results

        Returns:
            Maximum similarity score
        """
        if not selected_results:
            return 0.0

        max_similarity = 0.0
        try:
            # Get candidate embedding if available
            if hasattr(candidate, 'embedding') and candidate.embedding:
                candidate_embedding = candidate.embedding
            else:
                # Skip if we can't get embedding
                return 0.0

            for selected in selected_results:
                if hasattr(selected, 'embedding') and selected.embedding:
                    selected_embedding = selected.embedding
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(candidate_embedding, selected_embedding)
                    max_similarity = max(max_similarity, similarity)
        except Exception:
            # If similarity calculation fails, return 0
            pass

        return max_similarity

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _rerank_documents(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        Rerank documents using additional vector-based features

        Args:
            documents: List of documents to rerank

        Returns:
            Reranked list of documents
        """
        if len(documents) <= 1:
            return documents

        # Get query embedding for additional calculations
        query_embedding = self._get_query_embedding(documents[0].query)

        # Calculate additional scores
        for doc in documents:
            # Diversity score (distance from other top results)
            diversity_score = self._calculate_diversity_score(doc, documents)

            # Length normalization
            length_score = self._calculate_length_score(doc)

            # Embedding quality score (based on vector magnitude)
            vector_score = doc.additional_info.get("vector_score", 0.0)

            # Combine scores
            combined_score = (
                0.6 * doc.relevance_score +
                0.2 * diversity_score +
                0.1 * length_score +
                0.1 * vector_score
            )

            doc.additional_info["rerank_score"] = combined_score

        # Sort by combined score
        reranked_docs = sorted(
            documents,
            key=lambda doc: doc.additional_info.get("rerank_score", doc.relevance_score),
            reverse=True
        )

        # Update ranks and final scores
        for i, doc in enumerate(reranked_docs[:self.config.rerank_top_k]):
            doc.rank = i
            doc.relevance_score = doc.additional_info.get("rerank_score", doc.relevance_score)

        return reranked_docs[:self.config.rerank_top_k]

    def _calculate_diversity_score(self, doc: RetrievedDocument, all_docs: List[RetrievedDocument]) -> float:
        """
        Calculate diversity score for a document

        Args:
            doc: Document to calculate diversity for
            all_docs: All retrieved documents

        Returns:
            Diversity score (0-1)
        """
        if len(all_docs) <= 1:
            return 1.0

        # Calculate average distance to other top documents
        total_distance = 0.0
        count = 0

        top_k = min(5, len(all_docs))  # Consider top 5 documents

        for other_doc in all_docs[:top_k]:
            if other_doc.id != doc.id:
                # Simple diversity based on content difference
                content_similarity = self._content_similarity(doc.content, other_doc.content)
                distance = 1.0 - content_similarity
                total_distance += distance
                count += 1

        if count == 0:
            return 1.0

        avg_distance = total_distance / count
        return min(avg_distance, 1.0)

    def _calculate_length_score(self, doc: RetrievedDocument) -> float:
        """
        Calculate length-based score

        Args:
            doc: Document to score

        Returns:
            Length score (0-1)
        """
        content_length = len(doc.content)

        # Prefer moderate length documents
        if content_length < 50:
            return 0.3  # Too short
        elif content_length > 2000:
            return 0.7  # Too long but acceptable
        elif 100 <= content_length <= 800:
            return 1.0  # Ideal length
        else:
            return 0.8  # Acceptable length

    def _content_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate simple content similarity

        Args:
            content1: First content
            content2: Second content

        Returns:
            Similarity score (0-1)
        """
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def get_document_count(self) -> int:
        """Get total number of documents in vector store"""
        try:
            if hasattr(self.vector_store, 'get_vector_count'):
                return self.vector_store.get_vector_count()
            elif hasattr(self.vector_store, 'vector_count'):
                return self.vector_store.vector_count
            else:
                return 0
        except Exception:
            return 0

    def clear_cache(self):
        """Clear embedding and query cache"""
        self.embedding_cache.clear()
        self.query_cache.clear()

    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about embeddings

        Returns:
            Dictionary with embedding statistics
        """
        try:
            if hasattr(self.vector_store, 'get_vector_count'):
                vector_count = self.vector_store.get_vector_count()
            elif hasattr(self.vector_store, 'vector_count'):
                vector_count = self.vector_store.vector_count
            else:
                vector_count = 0
            index_status = getattr(self.vector_store, 'index_status', 'unknown')
        except Exception:
            vector_count = 0
            index_status = 'unknown'

        return {
            "vector_count": vector_count,
            "index_status": index_status,
            "embedding_model": self.embedding_model.get_model_info(),
            "cache_size": len(self.embedding_cache),
            "embedding_dimension": self.embedding_model.get_embedding_dimension()
        }

    def search_by_embedding(self, embedding: List[float], k: int = 5) -> List[RetrievedDocument]:
        """
        Search documents using a pre-computed embedding

        Args:
            embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of retrieved documents
        """
        config = RetrievalConfig(top_k=k, search_type="similarity")
        raw_results = self._similarity_search(embedding, config)

        # Convert to RetrievedDocument objects
        retrieved_docs = []
        for i, result in enumerate(raw_results):
            content = result.metadata.get("content", "")
            if not content:
                continue

            metadata = result.metadata.copy()
            metadata.pop("content", None)

            retrieved_doc = RetrievedDocument(
                content=content,
                metadata=metadata,
                id=result.id,
                relevance_score=result.score,
                retrieval_method="vector_embedding_search",
                query="<embedding>",
                rank=i
            )
            retrieved_docs.append(retrieved_doc)

        return retrieved_docs

    def get_retriever_info(self) -> Dict[str, Any]:
        """Get information about the retriever"""
        base_info = super().get_retriever_info()
        base_info.update({
            "embedding_model": self.embedding_model.get_model_info(),
            "vector_store": {
                "type": self.vector_store.__class__.__name__,
                "vector_count": self.get_document_count()
            },
            "embedding_stats": self.get_embedding_stats()
        })
        return base_info