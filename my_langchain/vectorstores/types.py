# -*- coding: utf-8 -*-
"""
Vector store types and data structures
"""

from typing import Any, Dict, List, Optional, Union, Sequence
from enum import Enum
from pydantic import BaseModel, Field
import numpy as np


class VectorStoreConfig(BaseModel):
    """
    Vector store configuration
    """
    dimension: int = Field(..., description="Dimension of vectors")
    metric: str = Field(default="cosine", description="Distance metric: cosine, euclidean, manhattan")
    index_type: str = Field(default="flat", description="Index type: flat, ivf, hnsw")
    ef_construction: Optional[int] = Field(default=None, description="HNSW ef_construction parameter")
    ef_search: Optional[int] = Field(default=None, description="HNSW ef_search parameter")
    nlist: Optional[int] = Field(default=None, description="IVF nlist parameter")
    nprobe: Optional[int] = Field(default=None, description="IVF nprobe parameter")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Vector(BaseModel):
    """
    Vector representation
    """
    id: str = Field(..., description="Unique identifier")
    embedding: List[float] = Field(..., description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        return f"Vector(id={self.id}, dimension={len(self.embedding)})"


class Document(BaseModel):
    """
    Document representation
    """
    id: str = Field(..., description="Unique identifier")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Document embedding")

    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(id={self.id}, content='{content_preview}')"


class Embedding(BaseModel):
    """
    Embedding representation
    """
    vector: List[float] = Field(..., description="Embedding vector")
    model: str = Field(..., description="Embedding model name")
    dimension: int = Field(..., description="Embedding dimension")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def __repr__(self) -> str:
        return f"Embedding(model={self.model}, dimension={self.dimension})"


class VectorStoreQuery(BaseModel):
    """
    Vector store query
    """
    query_vector: List[float] = Field(..., description="Query embedding")
    top_k: int = Field(default=10, description="Number of results to return")
    include_metadata: bool = Field(default=True, description="Include metadata in results")
    filter_dict: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")
    score_threshold: Optional[float] = Field(default=None, description="Minimum similarity score")


class VectorStoreResult(BaseModel):
    """
    Vector store query result
    """
    vectors: List[Vector] = Field(default_factory=list, description="Retrieved vectors")
    documents: List[Document] = Field(default_factory=list, description="Retrieved documents")
    scores: List[float] = Field(default_factory=list, description="Similarity scores")
    query_time: float = Field(default=0.0, description="Query execution time in seconds")
    total_count: int = Field(default=0, description="Total number of vectors in store")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")

    def __repr__(self) -> str:
        return f"VectorStoreResult(vectors={len(self.vectors)}, time={self.query_time:.3f}s)"


# Error types
class VectorStoreError(Exception):
    """
    Base vector store error class
    """
    message: str
    store_name: str
    details: Dict[str, Any] = {}

    def __init__(self, message: str, store_name: str, details: Dict[str, Any] = None):
        self.message = message
        self.store_name = store_name
        self.details = details or {}
        super().__init__(message)


class VectorStoreValidationError(VectorStoreError):
    """
    Vector store validation error
    """
    input_data: Any

    def __init__(self, message: str, store_name: str, input_data: Any = None):
        self.input_data = input_data
        validation_details = {
            "input_data": input_data,
            "validation_error": True
        }
        super().__init__(message, store_name, validation_details)


class VectorStoreRetrievalError(VectorStoreError):
    """
    Vector store retrieval error
    """
    query: Optional[VectorStoreQuery] = None

    def __init__(self, message: str, store_name: str, query: VectorStoreQuery = None):
        self.query = query
        retrieval_details = {
            "query": query.dict() if query else None,
            "retrieval_error": True
        }
        super().__init__(message, store_name, retrieval_details)


class VectorStoreIndexError(VectorStoreError):
    """
    Vector store indexing error
    """
    vectors: Optional[List[Vector]] = None

    def __init__(self, message: str, store_name: str, vectors: List[Vector] = None):
        self.vectors = vectors
        index_details = {
            "vector_count": len(vectors) if vectors else 0,
            "index_error": True
        }
        super().__init__(message, store_name, index_details)


# Enums
class DistanceMetric(str, Enum):
    """Distance metric enumeration"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    DOT_PRODUCT = "dot_product"


class IndexType(str, Enum):
    """Index type enumeration"""
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"
    LSH = "lsh"


class IndexStatus(str, Enum):
    """Index status enumeration"""
    BUILDING = "building"
    READY = "ready"
    ERROR = "error"
    EMPTY = "empty"


# Helper functions
def normalize_vector(vector: List[float]) -> List[float]:
    """
    Normalize vector to unit length

    Args:
        vector: Input vector

    Returns:
        Normalized vector
    """
    if not vector:
        return vector

    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector

    return (np.array(vector) / norm).tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score
    """
    if not vec1 or not vec2:
        return 0.0

    vec1_norm = normalize_vector(vec1)
    vec2_norm = normalize_vector(vec2)

    return np.dot(vec1_norm, vec2_norm)


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate Euclidean distance between two vectors

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Euclidean distance
    """
    if not vec1 or not vec2:
        return float('inf')

    return np.linalg.norm(np.array(vec1) - np.array(vec2))


def manhattan_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate Manhattan distance between two vectors

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Manhattan distance
    """
    if not vec1 or not vec2:
        return float('inf')

    return np.sum(np.abs(np.array(vec1) - np.array(vec2)))