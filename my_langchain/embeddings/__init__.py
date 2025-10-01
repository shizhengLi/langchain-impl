# -*- coding: utf-8 -*-
"""
Embedding module for text vectorization
"""

from .base import BaseEmbedding
from .types import (
    EmbeddingConfig, EmbeddingResult, EmbeddingError,
    EmbeddingValidationError, EmbeddingProcessingError,
    Embedding
)
from .mock_embedding import MockEmbedding

__all__ = [
    "BaseEmbedding",
    "EmbeddingConfig",
    "EmbeddingResult",
    "EmbeddingError",
    "EmbeddingValidationError",
    "EmbeddingProcessingError",
    "Embedding",
    "MockEmbedding"
]