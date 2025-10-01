# -*- coding: utf-8 -*-
"""
Types and data structures for embeddings module
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
import time
import uuid


class EmbeddingError(Exception):
    """Base exception for embedding operations"""

    def __init__(self, message: str, embedding_type: str = None, context: Any = None):
        super().__init__(message)
        self.embedding_type = embedding_type
        self.context = context


class EmbeddingValidationError(EmbeddingError):
    """Exception raised for embedding validation errors"""
    pass


class EmbeddingProcessingError(EmbeddingError):
    """Exception raised for embedding processing errors"""
    pass


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models"""

    model_name: str = Field(default="text-embedding-ada-002", description="Name of the embedding model")
    embedding_dimension: int = Field(default=1536, description="Dimension of the embedding vectors")
    batch_size: int = Field(default=100, description="Batch size for processing multiple texts")
    max_tokens: int = Field(default=8192, description="Maximum tokens per text")
    normalize_embeddings: bool = Field(default=True, description="Whether to normalize embeddings")
    show_progress: bool = Field(default=False, description="Whether to show progress")
    timeout: Optional[float] = Field(default=30.0, description="Timeout for embedding requests")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")

    # Model-specific settings
    model_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional model parameters")

    @validator('embedding_dimension')
    def embedding_dimension_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('embedding_dimension must be positive')
        return v

    @validator('batch_size')
    def batch_size_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('batch_size must be positive')
        return v

    @validator('max_tokens')
    def max_tokens_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('max_tokens must be positive')
        return v

    @validator('timeout')
    def timeout_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError('timeout must be positive')
        return v

    @validator('max_retries')
    def max_retries_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('max_retries must be non-negative')
        return v


class Embedding(BaseModel):
    """Single embedding vector with metadata"""

    vector: List[float] = Field(..., description="Embedding vector")
    text: str = Field(..., description="Original text that was embedded")
    model_name: str = Field(..., description="Model used for embedding")
    embedding_dimension: int = Field(..., description="Dimension of the embedding")
    token_count: Optional[int] = Field(None, description="Number of tokens in the text")
    processing_time: Optional[float] = Field(None, description="Time taken to process in seconds")

    class Config:
        arbitrary_types_allowed = True

    @validator('vector')
    def vector_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('vector cannot be empty')
        return v

    @validator('embedding_dimension')
    def dimension_must_match_vector_length(cls, v, values):
        if 'vector' in values and len(values['vector']) != v:
            raise ValueError('embedding_dimension must match vector length')
        return v

    def normalize(self) -> 'Embedding':
        """Return a normalized copy of this embedding"""
        import math

        norm = math.sqrt(sum(x * x for x in self.vector))
        if norm == 0:
            return self

        normalized_vector = [x / norm for x in self.vector]
        return Embedding(
            vector=normalized_vector,
            text=self.text,
            model_name=self.model_name,
            embedding_dimension=self.embedding_dimension,
            token_count=self.token_count,
            processing_time=self.processing_time
        )

    def cosine_similarity(self, other: 'Embedding') -> float:
        """Calculate cosine similarity with another embedding"""
        if self.embedding_dimension != other.embedding_dimension:
            raise ValueError("Embeddings must have same dimension")

        dot_product = sum(a * b for a, b in zip(self.vector, other.vector))
        norm_a = sum(a * a for a in self.vector) ** 0.5
        norm_b = sum(b * b for b in other.vector) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def euclidean_distance(self, other: 'Embedding') -> float:
        """Calculate Euclidean distance to another embedding"""
        if self.embedding_dimension != other.embedding_dimension:
            raise ValueError("Embeddings must have same dimension")

        return sum((a - b) ** 2 for a, b in zip(self.vector, other.vector)) ** 0.5

    def manhattan_distance(self, other: 'Embedding') -> float:
        """Calculate Manhattan distance to another embedding"""
        if self.embedding_dimension != other.embedding_dimension:
            raise ValueError("Embeddings must have same dimension")

        return sum(abs(a - b) for a, b in zip(self.vector, other.vector))


class EmbeddingResult(BaseModel):
    """Result of embedding operation"""

    embeddings: List[Embedding] = Field(..., description="List of embeddings")
    model_name: str = Field(..., description="Model used for embedding")
    total_tokens: Optional[int] = Field(None, description="Total number of tokens processed")
    total_time: Optional[float] = Field(None, description="Total time taken in seconds")
    batch_count: int = Field(..., description="Number of batches processed")

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator('embeddings')
    def embeddings_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('embeddings cannot be empty')
        return v

    @validator('batch_count')
    def batch_count_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('batch_count must be positive')
        return v

    def __len__(self) -> int:
        """Return number of embeddings"""
        return len(self.embeddings)

    def __getitem__(self, index: int) -> Embedding:
        """Get embedding by index"""
        return self.embeddings[index]

    def __iter__(self):
        """Iterate over embeddings"""
        return iter(self.embeddings)

    def get_average_embedding(self) -> List[float]:
        """Get average embedding vector"""
        if not self.embeddings:
            return []

        dimension = self.embeddings[0].embedding_dimension
        avg_vector = [0.0] * dimension

        for embedding in self.embeddings:
            for i, value in enumerate(embedding.vector):
                avg_vector[i] += value

        # Normalize by number of embeddings
        avg_vector = [value / len(self.embeddings) for value in avg_vector]
        return avg_vector

    def get_embedding_by_text(self, text: str) -> Optional[Embedding]:
        """Find embedding by original text"""
        for embedding in self.embeddings:
            if embedding.text == text:
                return embedding
        return None

    def filter_by_token_count(self, min_tokens: int = None, max_tokens: int = None) -> 'EmbeddingResult':
        """Filter embeddings by token count"""
        filtered = []
        for embedding in self.embeddings:
            if embedding.token_count is None:
                continue

            if min_tokens is not None and embedding.token_count < min_tokens:
                continue
            if max_tokens is not None and embedding.token_count > max_tokens:
                continue

            filtered.append(embedding)

        return EmbeddingResult(
            embeddings=filtered,
            model_name=self.model_name,
            total_tokens=sum(e.token_count for e in filtered if e.token_count),
            total_time=self.total_time,
            batch_count=len(filtered),
            metadata=self.metadata.copy()
        )


@dataclass
class EmbeddingUsage:
    """Usage information for embedding operation"""

    prompt_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: 'EmbeddingUsage') -> 'EmbeddingUsage':
        return EmbeddingUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )


# Utility functions for embeddings
def estimate_token_count(text: str, model_name: str = "text-embedding-ada-002") -> int:
    """
    Estimate token count for text

    Args:
        text: Text to estimate tokens for
        model_name: Name of the embedding model

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Simple estimation: roughly 4 characters per token for English
    # This is a rough approximation and should be replaced with proper tokenization
    estimated_tokens = len(text) // 4
    return max(1, estimated_tokens)


def validate_embedding_vector(vector: List[float], expected_dimension: int = None) -> bool:
    """
    Validate an embedding vector

    Args:
        vector: Vector to validate
        expected_dimension: Expected dimension (optional)

    Returns:
        True if valid

    Raises:
        EmbeddingValidationError: If vector is invalid
    """
    if not isinstance(vector, list):
        raise EmbeddingValidationError("Vector must be a list")

    if not vector:
        raise EmbeddingValidationError("Vector cannot be empty")

    if not all(isinstance(x, (int, float)) for x in vector):
        raise EmbeddingValidationError("Vector must contain only numbers")

    if any(x != x for x in vector):  # Check for NaN
        raise EmbeddingValidationError("Vector cannot contain NaN values")

    if any(abs(x) == float('inf') for x in vector):  # Check for infinity
        raise EmbeddingValidationError("Vector cannot contain infinite values")

    if expected_dimension is not None and len(vector) != expected_dimension:
        raise EmbeddingValidationError(
            f"Vector dimension {len(vector)} does not match expected {expected_dimension}"
        )

    return True


def normalize_vector(vector: List[float]) -> List[float]:
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


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (0-1)
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same dimension")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def create_embedding(
    vector: List[float],
    text: str,
    model_name: str = "unknown",
    processing_time: float = None,
    token_count: int = None
) -> Embedding:
    """
    Create an Embedding object with validation

    Args:
        vector: Embedding vector
        text: Original text
        model_name: Model name
        processing_time: Processing time
        token_count: Token count

    Returns:
        Embedding object
    """
    validate_embedding_vector(vector, len(vector))

    return Embedding(
        vector=vector,
        text=text,
        model_name=model_name,
        embedding_dimension=len(vector),
        processing_time=processing_time,
        token_count=token_count
    )


def merge_embedding_results(results: List[EmbeddingResult]) -> EmbeddingResult:
    """
    Merge multiple embedding results

    Args:
        results: List of embedding results

    Returns:
        Merged embedding result
    """
    if not results:
        raise EmbeddingValidationError("Cannot merge empty results list")

    all_embeddings = []
    total_tokens = 0
    total_time = 0
    batch_count = 0

    for result in results:
        all_embeddings.extend(result.embeddings)
        if result.total_tokens:
            total_tokens += result.total_tokens
        if result.total_time:
            total_time += result.total_time
        batch_count += result.batch_count

    # Merge metadata
    merged_metadata = {}
    for result in results:
        merged_metadata.update(result.metadata)

    return EmbeddingResult(
        embeddings=all_embeddings,
        model_name=results[0].model_name,  # Use first model name
        total_tokens=total_tokens,
        total_time=total_time,
        batch_count=batch_count,
        metadata=merged_metadata
    )