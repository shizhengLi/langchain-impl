# -*- coding: utf-8 -*-
"""
Retrieval module for RAG (Retrieval-Augmented Generation) functionality
"""

from .base import BaseRetriever
from .types import (
    RetrievalConfig, RetrievalResult, RetrievalError,
    RetrievalValidationError, RetrievalProcessingError,
    Document, RetrievedDocument, RetrievalQuery
)
from .document_retriever import DocumentRetriever
from .vector_retriever import VectorRetriever
from .ensemble_retriever import EnsembleRetriever

__all__ = [
    "BaseRetriever",
    "RetrievalConfig",
    "RetrievalResult",
    "RetrievalError",
    "RetrievalValidationError",
    "RetrievalProcessingError",
    "Document",
    "RetrievedDocument",
    "RetrievalQuery",
    "DocumentRetriever",
    "VectorRetriever",
    "EnsembleRetriever"
]