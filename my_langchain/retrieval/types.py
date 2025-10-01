# -*- coding: utf-8 -*-
"""
Types and data structures for retrieval module
"""

from typing import List, Dict, Any, Optional, Union, Callable
from pydantic import BaseModel, Field, validator, root_validator
from dataclasses import dataclass
import time
import uuid


class RetrievalError(Exception):
    """Base exception for retrieval operations"""

    def __init__(self, message: str, retriever_type: str = None, context: Any = None):
        super().__init__(message)
        self.retriever_type = retriever_type
        self.context = context


class RetrievalValidationError(RetrievalError):
    """Exception raised for retrieval validation errors"""
    pass


class RetrievalProcessingError(RetrievalError):
    """Exception raised for retrieval processing errors"""
    pass


class RetrievalConfig(BaseModel):
    """Configuration for retrieval systems"""

    # Basic retrieval parameters
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    score_threshold: Optional[float] = Field(default=None, description="Minimum similarity score threshold")
    search_type: str = Field(default="similarity", description="Type of search: similarity, mmr, etc.")

    # MMR (Maximal Marginal Relevance) parameters
    mmr_lambda: float = Field(default=0.5, description="Lambda parameter for MMR (0-1)")
    fetch_k: int = Field(default=20, description="Number of documents to fetch for MMR")

    # Filtering parameters
    filter_dict: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters")
    search_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional search parameters")

    # Relevance scoring
    relevance_score_func: Optional[str] = Field(default="cosine", description="Relevance scoring function")
    normalize_scores: bool = Field(default=True, description="Whether to normalize relevance scores")

    # Performance settings
    enable_caching: bool = Field(default=True, description="Whether to enable result caching")
    cache_ttl: Optional[float] = Field(default=300.0, description="Cache TTL in seconds")

    # Result processing
    rerank: bool = Field(default=False, description="Whether to rerank results")
    rerank_top_k: int = Field(default=10, description="Number of results to rerank")

    @validator('top_k')
    def top_k_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('top_k must be positive')
        return v

    @validator('mmr_lambda')
    def mmr_lambda_must_be_between_0_and_1(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('mmr_lambda must be between 0 and 1')
        return v

    @validator('fetch_k')
    def fetch_k_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('fetch_k must be positive')
        return v

    @validator('score_threshold')
    def score_threshold_must_be_between_0_and_1(cls, v):
        if v is not None and not 0 <= v <= 1:
            raise ValueError('score_threshold must be between 0 and 1')
        return v

    @validator('search_type')
    def search_type_must_be_valid(cls, v):
        valid_types = ['similarity', 'mmr', 'hybrid', 'tfidf', 'bm25']
        if v not in valid_types:
            raise ValueError(f'search_type must be one of {valid_types}')
        return v


class Document(BaseModel):
    """Document representation for retrieval"""

    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Document ID")

    class Config:
        arbitrary_types_allowed = True

    @validator('content')
    def content_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('content cannot be empty')
        return v.strip()

    def __str__(self) -> str:
        return f"Document(id={self.id[:8]}..., content_length={len(self.content)})"

    def get_text_snippet(self, max_length: int = 100) -> str:
        """Get a snippet of the document content"""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."

    def matches_filter(self, filter_dict: Dict[str, Any]) -> bool:
        """Check if document matches metadata filters"""
        if not filter_dict:
            return True

        for key, value in filter_dict.items():
            if key not in self.metadata:
                return False
            if self.metadata[key] != value:
                return False
        return True


class RetrievedDocument(Document):
    """Document that has been retrieved with relevance information"""

    relevance_score: float = Field(..., description="Relevance score (0-1)")
    retrieval_method: str = Field(..., description="Method used for retrieval")
    query: str = Field(..., description="Query that retrieved this document")
    rank: int = Field(..., description="Rank in retrieval results")
    additional_info: Dict[str, Any] = Field(default_factory=dict, description="Additional retrieval information")

    @validator('relevance_score')
    def relevance_score_must_be_between_0_and_1(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('relevance_score must be between 0 and 1')
        return v

    @validator('rank')
    def rank_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('rank must be non-negative')
        return v

    def __str__(self) -> str:
        return (
            f"RetrievedDocument(id={self.id[:8]}..., "
            f"score={self.relevance_score:.3f}, rank={self.rank})"
        )


class RetrievalQuery(BaseModel):
    """Query for retrieval operations"""

    query: str = Field(..., description="Query text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Query metadata")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Document filters")
    max_results: Optional[int] = Field(None, description="Maximum results to return")
    min_score: Optional[float] = Field(None, description="Minimum relevance score")

    @validator('query')
    def query_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('query cannot be empty')
        return v.strip()

    def __str__(self) -> str:
        query_preview = self.query[:50] + "..." if len(self.query) > 50 else self.query
        return f"RetrievalQuery(query='{query_preview}', max_results={self.max_results})"


class RetrievalResult(BaseModel):
    """Result of a retrieval operation"""

    documents: List[RetrievedDocument] = Field(..., description="Retrieved documents")
    query: str = Field(..., description="Original query")
    total_results: int = Field(..., description="Total number of results found")
    search_time: float = Field(..., description="Time taken for search in seconds")
    retrieval_method: str = Field(..., description="Method used for retrieval")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator('documents')
    def documents_must_not_be_empty(cls, v):
        # Allow empty documents for cases where no relevant documents are found
        if v is None:
            raise ValueError('documents cannot be None')
        return v

    @validator('total_results')
    def total_results_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('total_results must be non-negative')
        return v

    @validator('search_time')
    def search_time_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('search_time must be positive')
        return v

    @root_validator(skip_on_failure=True)
    def validate_result_consistency(cls, values):
        """Validate consistency between documents and total_results"""
        documents = values.get('documents', [])
        total_results = values.get('total_results', 0)

        # If we have documents, total_results should match or be greater
        if documents and total_results < len(documents):
            raise ValueError('total_results must be >= number of documents')

        # If we have no documents, total_results should be 0
        if not documents and total_results != 0:
            raise ValueError('total_results must be 0 when no documents')

        return values

    def __len__(self) -> int:
        """Return number of retrieved documents"""
        return len(self.documents)

    def __getitem__(self, index: int) -> RetrievedDocument:
        """Get document by index"""
        return self.documents[index]

    def __iter__(self):
        """Iterate over documents"""
        return iter(self.documents)

    def get_top_k(self, k: int) -> List[RetrievedDocument]:
        """Get top k documents"""
        return self.documents[:k]

    def get_documents_above_threshold(self, threshold: float) -> List[RetrievedDocument]:
        """Get documents with relevance score above threshold"""
        return [doc for doc in self.documents if doc.relevance_score >= threshold]

    def get_average_score(self) -> float:
        """Get average relevance score"""
        if not self.documents:
            return 0.0
        return sum(doc.relevance_score for doc in self.documents) / len(self.documents)

    def get_score_distribution(self) -> Dict[str, float]:
        """Get score distribution statistics"""
        if not self.documents:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "median": 0.0}

        scores = [doc.relevance_score for doc in self.documents]
        scores.sort()

        return {
            "min": min(scores),
            "max": max(scores),
            "avg": sum(scores) / len(scores),
            "median": scores[len(scores) // 2] if scores else 0.0
        }

    def filter_by_metadata(self, key: str, value: Any) -> 'RetrievalResult':
        """Filter results by metadata"""
        filtered_docs = [
            doc for doc in self.documents
            if doc.metadata.get(key) == value
        ]

        return RetrievalResult(
            documents=filtered_docs,
            query=self.query,
            total_results=len(filtered_docs),
            search_time=self.search_time,
            retrieval_method=self.retrieval_method,
            metadata=self.metadata.copy()
        )

    def rerank_by_length(self, ascending: bool = False) -> 'RetrievalResult':
        """Rerank documents by content length"""
        sorted_docs = sorted(
            self.documents,
            key=lambda doc: len(doc.content),
            reverse=not ascending
        )

        # Update ranks
        for i, doc in enumerate(sorted_docs):
            doc.rank = i

        return RetrievalResult(
            documents=sorted_docs,
            query=self.query,
            total_results=len(sorted_docs),
            search_time=self.search_time,
            retrieval_method=self.retrieval_method + "_reranked_by_length",
            metadata=self.metadata.copy()
        )


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval performance"""

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    hit_rate: float = 0.0
    mean_reciprocal_rank: float = 0.0
    mean_average_precision: float = 0.0

    def __add__(self, other: 'RetrievalMetrics') -> 'RetrievalMetrics':
        return RetrievalMetrics(
            precision=self.precision + other.precision,
            recall=self.recall + other.recall,
            f1_score=self.f1_score + other.f1_score,
            hit_rate=self.hit_rate + other.hit_rate,
            mean_reciprocal_rank=self.mean_reciprocal_rank + other.mean_reciprocal_rank,
            mean_average_precision=self.mean_average_precision + other.mean_average_precision
        )

    def __truediv__(self, divisor: float) -> 'RetrievalMetrics':
        return RetrievalMetrics(
            precision=self.precision / divisor,
            recall=self.recall / divisor,
            f1_score=self.f1_score / divisor,
            hit_rate=self.hit_rate / divisor,
            mean_reciprocal_rank=self.mean_reciprocal_rank / divisor,
            mean_average_precision=self.mean_average_precision / divisor
        )


# Utility functions for retrieval
def create_document(content: str, metadata: Dict[str, Any] = None, doc_id: str = None) -> Document:
    """
    Create a Document object with validation

    Args:
        content: Document content
        metadata: Document metadata
        doc_id: Document ID (optional)

    Returns:
        Document object
    """
    if metadata is None:
        metadata = {}

    return Document(
        content=content,
        metadata=metadata,
        id=doc_id or str(uuid.uuid4())
    )


def create_retrieved_document(
    content: str,
    relevance_score: float,
    query: str,
    rank: int,
    metadata: Dict[str, Any] = None,
    retrieval_method: str = "similarity",
    doc_id: str = None
) -> RetrievedDocument:
    """
    Create a RetrievedDocument object with validation

    Args:
        content: Document content
        relevance_score: Relevance score (0-1)
        query: Query that retrieved this document
        rank: Rank in results
        metadata: Document metadata
        retrieval_method: Method used for retrieval
        doc_id: Document ID (optional)

    Returns:
        RetrievedDocument object
    """
    if metadata is None:
        metadata = {}

    return RetrievedDocument(
        content=content,
        metadata=metadata,
        id=doc_id or str(uuid.uuid4()),
        relevance_score=relevance_score,
        retrieval_method=retrieval_method,
        query=query,
        rank=rank
    )


def create_retrieval_query(
    query: str,
    filters: Dict[str, Any] = None,
    max_results: int = None,
    min_score: float = None
) -> RetrievalQuery:
    """
    Create a RetrievalQuery object with validation

    Args:
        query: Query text
        filters: Document filters
        max_results: Maximum results
        min_score: Minimum relevance score

    Returns:
        RetrievalQuery object
    """
    if filters is None:
        filters = {}

    return RetrievalQuery(
        query=query,
        filters=filters,
        max_results=max_results,
        min_score=min_score
    )


def calculate_retrieval_metrics(
    retrieved_docs: List[RetrievedDocument],
    relevant_doc_ids: List[str],
    k: int = 10
) -> RetrievalMetrics:
    """
    Calculate retrieval performance metrics

    Args:
        retrieved_docs: List of retrieved documents
        relevant_doc_ids: List of relevant document IDs
        k: Cut-off for evaluation

    Returns:
        RetrievalMetrics object
    """
    if not retrieved_docs or not relevant_doc_ids:
        return RetrievalMetrics()

    # Get top k retrieved docs
    top_k_docs = retrieved_docs[:k]
    top_k_ids = [doc.id for doc in top_k_docs]

    # Calculate precision@k
    relevant_retrieved = sum(1 for doc_id in top_k_ids if doc_id in relevant_doc_ids)
    precision_at_k = relevant_retrieved / k

    # Calculate recall
    total_relevant = len(relevant_doc_ids)
    recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0

    # Calculate F1 score
    f1_score = 2 * precision_at_k * recall / (precision_at_k + recall) if (precision_at_k + recall) > 0 else 0.0

    # Calculate hit rate (whether any relevant doc was retrieved)
    hit_rate = 1.0 if relevant_retrieved > 0 else 0.0

    # Calculate Mean Reciprocal Rank
    mrr = 0.0
    for i, doc in enumerate(top_k_docs):
        if doc.id in relevant_doc_ids:
            mrr = 1.0 / (i + 1)
            break

    # Calculate Mean Average Precision (simplified version)
    ap = 0.0
    relevant_count = 0
    for i, doc in enumerate(top_k_docs):
        if doc.id in relevant_doc_ids:
            relevant_count += 1
            ap += relevant_count / (i + 1)

    map_score = ap / len(relevant_doc_ids) if relevant_doc_ids else 0.0

    return RetrievalMetrics(
        precision=precision_at_k,
        recall=recall,
        f1_score=f1_score,
        hit_rate=hit_rate,
        mean_reciprocal_rank=mrr,
        mean_average_precision=map_score
    )


def merge_retrieval_results(results: List[RetrievalResult]) -> RetrievalResult:
    """
    Merge multiple retrieval results

    Args:
        results: List of retrieval results

    Returns:
        Merged retrieval result
    """
    if not results:
        raise RetrievalValidationError("Cannot merge empty results list")

    # Combine all documents
    all_docs = []
    seen_ids = set()

    for result in results:
        for doc in result.documents:
            if doc.id not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(doc.id)

    # Sort by relevance score (descending)
    all_docs.sort(key=lambda doc: doc.relevance_score, reverse=True)

    # Update ranks
    for i, doc in enumerate(all_docs):
        doc.rank = i

    # Merge metadata
    merged_metadata = {}
    total_time = 0.0
    for result in results:
        merged_metadata.update(result.metadata)
        total_time += result.search_time

    return RetrievalResult(
        documents=all_docs,
        query=results[0].query,  # Use first query
        total_results=len(all_docs),
        search_time=total_time,
        retrieval_method="merged_" + "_".join(r.retrieval_method for r in results),
        metadata=merged_metadata
    )