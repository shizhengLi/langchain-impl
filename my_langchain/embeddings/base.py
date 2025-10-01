# -*- coding: utf-8 -*-
"""
Base embedding implementation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable
import time
import logging

from .types import (
    EmbeddingConfig, EmbeddingResult, Embedding, EmbeddingError,
    EmbeddingValidationError, EmbeddingProcessingError,
    estimate_token_count, validate_embedding_vector, create_embedding
)


class BaseEmbedding(ABC):
    """
    Base embedding class providing common functionality

    This class defines the interface that all embedding implementations must follow
    and provides common utility methods for text processing and validation.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None, **kwargs):
        """
        Initialize embedding model

        Args:
            config: Embedding configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or EmbeddingConfig(**kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_model()

    def _setup_model(self):
        """Setup model-specific configuration"""
        # Override in subclasses if needed
        pass

    @abstractmethod
    def _embed_single_text(self, text: str) -> List[float]:
        """
        Embed a single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            EmbeddingProcessingError: If embedding fails
        """
        pass

    @abstractmethod
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingProcessingError: If embedding fails
        """
        pass

    def embed_text(self, text: str) -> Embedding:
        """
        Embed a single text

        Args:
            text: Text to embed

        Returns:
            Embedding object

        Raises:
            EmbeddingValidationError: If input is invalid
            EmbeddingProcessingError: If embedding fails
        """
        self._validate_input_text(text)

        start_time = time.time()
        try:
            vector = self._embed_single_text(text)
            processing_time = time.time() - start_time

            validate_embedding_vector(vector, self.config.embedding_dimension)

            if self.config.normalize_embeddings:
                vector = self._normalize_vector(vector)

            token_count = estimate_token_count(text, self.config.model_name)

            return create_embedding(
                vector=vector,
                text=text,
                model_name=self.config.model_name,
                processing_time=processing_time,
                token_count=token_count
            )

        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingProcessingError(
                f"Failed to embed text: {str(e)}",
                self.__class__.__name__,
                {"text": text[:100]}  # Truncate for logging
            ) from e

    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Embed multiple texts

        Args:
            texts: List of texts to embed

        Returns:
            EmbeddingResult object

        Raises:
            EmbeddingValidationError: If input is invalid
            EmbeddingProcessingError: If embedding fails
        """
        self._validate_input_texts(texts)

        if not texts:
            raise EmbeddingValidationError("No texts provided for embedding")

        start_time = time.time()
        all_embeddings = []
        total_tokens = 0
        batch_count = 0

        try:
            # Process in batches
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                batch_count += 1

                if self.config.show_progress:
                    self.logger.info(f"Processing batch {batch_count} ({len(batch_texts)} texts)")

                # Get embeddings for this batch
                batch_vectors = self._embed_batch_with_retry(batch_texts)

                # Create embedding objects
                for j, (text, vector) in enumerate(zip(batch_texts, batch_vectors)):
                    validate_embedding_vector(vector, self.config.embedding_dimension)

                    if self.config.normalize_embeddings:
                        vector = self._normalize_vector(vector)

                    token_count = estimate_token_count(text, self.config.model_name)
                    total_tokens += token_count

                    embedding = create_embedding(
                        vector=vector,
                        text=text,
                        model_name=self.config.model_name,
                        token_count=token_count
                    )
                    all_embeddings.append(embedding)

            total_time = time.time() - start_time

            return EmbeddingResult(
                embeddings=all_embeddings,
                model_name=self.config.model_name,
                total_tokens=total_tokens,
                total_time=total_time,
                batch_count=batch_count,
                metadata={
                    "average_tokens_per_text": total_tokens / len(texts) if texts else 0,
                    "texts_processed": len(texts),
                    "batch_size_used": self.config.batch_size
                }
            )

        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingProcessingError(
                f"Failed to embed texts: {str(e)}",
                self.__class__.__name__,
                {"text_count": len(texts)}
            ) from e

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query text (convenience method)

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        embedding = self.embed_text(query)
        return embedding.vector

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed document texts (convenience method)

        Args:
            documents: List of document texts

        Returns:
            List of embedding vectors
        """
        result = self.embed_texts(documents)
        return [embedding.vector for embedding in result.embeddings]

    async def aembed_text(self, text: str) -> Embedding:
        """
        Asynchronously embed a single text

        Args:
            text: Text to embed

        Returns:
            Embedding object
        """
        # Default synchronous implementation
        # Override in subclasses for true async support
        return self.embed_text(text)

    async def aembed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Asynchronously embed multiple texts

        Args:
            texts: List of texts to embed

        Returns:
            EmbeddingResult object
        """
        # Default synchronous implementation
        # Override in subclasses for true async support
        return self.embed_texts(texts)

    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Embed batch with retry logic

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return self._embed_batch(texts)

            except Exception as e:
                last_exception = e

                if attempt < self.config.max_retries:
                    wait_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Embedding attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}"
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Embedding failed after {attempt + 1} attempts")

        # If we get here, all retries failed
        raise EmbeddingProcessingError(
            f"Embedding failed after {self.config.max_retries + 1} attempts: {str(last_exception)}",
            self.__class__.__name__
        ) from last_exception

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize a vector to unit length

        Args:
            vector: Vector to normalize

        Returns:
            Normalized vector
        """
        import math

        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            return vector

        return [x / norm for x in vector]

    def _validate_input_text(self, text: str):
        """
        Validate input text

        Args:
            text: Text to validate

        Raises:
            EmbeddingValidationError: If text is invalid
        """
        if not isinstance(text, str):
            raise EmbeddingValidationError("Text must be a string")

        if not text.strip():
            raise EmbeddingValidationError("Text cannot be empty")

        if len(text) > self.config.max_tokens * 4:  # Rough estimation
            raise EmbeddingValidationError(
                f"Text too long (approximate token limit: {self.config.max_tokens})"
            )

    def _validate_input_texts(self, texts: List[str]):
        """
        Validate input texts

        Args:
            texts: List of texts to validate

        Raises:
            EmbeddingValidationError: If texts are invalid
        """
        if not isinstance(texts, list):
            raise EmbeddingValidationError("Texts must be a list")

        if not texts:
            raise EmbeddingValidationError("Texts list cannot be empty")

        for i, text in enumerate(texts):
            try:
                self._validate_input_text(text)
            except EmbeddingValidationError as e:
                raise EmbeddingValidationError(f"Text at index {i} is invalid: {str(e)}") from e

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings

        Returns:
            Embedding dimension
        """
        return self.config.embedding_dimension

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.config.model_name,
            "embedding_dimension": self.config.embedding_dimension,
            "batch_size": self.config.batch_size,
            "max_tokens": self.config.max_tokens,
            "normalize_embeddings": self.config.normalize_embeddings,
            "model_kwargs": self.config.model_kwargs
        }

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity (0-1)
        """
        embedding1 = self.embed_text(text1)
        embedding2 = self.embed_text(text2)

        return embedding1.cosine_similarity(embedding2)

    def find_most_similar(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find most similar texts to a query

        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            top_k: Number of top results to return

        Returns:
            List of dictionaries with text and similarity scores
        """
        if not candidate_texts:
            return []

        # Embed query and candidates
        query_embedding = self.embed_text(query_text)
        candidate_embeddings = self.embed_texts(candidate_texts)

        # Calculate similarities
        similarities = []
        for i, candidate_embedding in enumerate(candidate_embeddings.embeddings):
            similarity = query_embedding.cosine_similarity(candidate_embedding)
            similarities.append({
                "text": candidate_texts[i],
                "similarity": similarity,
                "index": i
            })

        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.config.model_name}, "
            f"dimension={self.config.embedding_dimension}"
            f")"
        )