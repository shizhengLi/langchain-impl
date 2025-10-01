# -*- coding: utf-8 -*-
"""
Vector store module for implementing vector storage and retrieval functionality
"""

from .base import BaseVectorStore
from .types import (
    VectorStoreConfig, VectorStoreResult, VectorStoreQuery,
    Vector, Document, Embedding,
    VectorStoreError, VectorStoreValidationError, VectorStoreRetrievalError
)
from .in_memory_store import InMemoryVectorStore
from .faiss_store import FAISSVectorStore

__all__ = [
    # Base classes
    "BaseVectorStore",

    # Vector store implementations
    "InMemoryVectorStore",
    "FAISSVectorStore",

    # Types
    "VectorStoreConfig",
    "VectorStoreResult",
    "VectorStoreQuery",
    "Vector",
    "Document",
    "Embedding",
    "VectorStoreError",
    "VectorStoreValidationError",
    "VectorStoreRetrievalError"
]