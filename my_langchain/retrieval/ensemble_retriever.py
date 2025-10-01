# -*- coding: utf-8 -*-
"""
Ensemble retriever implementation
"""

from typing import List, Dict, Any, Optional, Union
import time
from collections import defaultdict

from .base import BaseRetriever
from .types import (
    RetrievalConfig, RetrievalResult, RetrievalProcessingError,
    Document, RetrievedDocument, RetrievalQuery, merge_retrieval_results
)


class EnsembleRetriever(BaseRetriever):
    """
    Ensemble retriever that combines multiple retrievers

    This retriever combines results from multiple retrievers using
    various fusion strategies like weighted averaging, rank fusion, etc.
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: Optional[List[float]] = None,
        fusion_strategy: str = "weighted_score",
        config: Optional[RetrievalConfig] = None,
        **kwargs
    ):
        """
        Initialize ensemble retriever

        Args:
            retrievers: List of retrievers to ensemble
            weights: Weights for each retriever (must sum to 1.0)
            fusion_strategy: Strategy for combining results
            config: Retrieval configuration
            **kwargs: Additional configuration parameters
        """
        if not retrievers:
            raise ValueError("At least one retriever must be provided")

        self.retrievers = retrievers
        self.fusion_strategy = fusion_strategy

        # Set default weights if not provided
        if weights is None:
            weights = [1.0 / len(retrievers)] * len(retrievers)

        if len(weights) != len(retrievers):
            raise ValueError("Number of weights must match number of retrievers")

        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        self.weights = weights

        # Initialize with default config
        if config is None:
            config = RetrievalConfig(**kwargs)

        super().__init__(config, **kwargs)
        self._last_retrieval_method = f"ensemble_{fusion_strategy}"

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

    def _setup_retriever(self):
        """Setup ensemble-specific configuration"""
        self.result_cache = {}

        # Validate fusion strategy
        valid_strategies = ["weighted_score", "rank_fusion", "reciprocal_rank", "weighted_vote"]
        if self.fusion_strategy not in valid_strategies:
            raise ValueError(f"fusion_strategy must be one of {valid_strategies}")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to all retrievers

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs (from first retriever)
        """
        all_doc_ids = []

        for i, retriever in enumerate(self.retrievers):
            try:
                doc_ids = retriever.add_documents(documents)
                if i == 0:
                    all_doc_ids = doc_ids
            except Exception as e:
                self.logger.error(f"Failed to add documents to retriever {i}: {str(e)}")
                if i == 0:
                    raise RetrievalProcessingError(
                        f"Failed to add documents to primary retriever: {str(e)}",
                        "EnsembleRetriever"
                    ) from e

        return all_doc_ids

    def _retrieve_documents(self, query: str, config: RetrievalConfig) -> List[RetrievedDocument]:
        """
        Retrieve documents by combining results from multiple retrievers

        Args:
            query: Query text
            config: Retrieval configuration

        Returns:
            List of retrieved documents
        """
        # Retrieve from all retrievers
        retriever_results = []
        for i, retriever in enumerate(self.retrievers):
            try:
                start_time = time.time()
                result = retriever.retrieve(query)
                retrieval_time = time.time() - start_time

                retriever_results.append({
                    "retriever": retriever,
                    "result": result,
                    "weight": self.weights[i],
                    "retrieval_time": retrieval_time
                })
            except Exception as e:
                self.logger.error(f"Retriever {i} failed: {str(e)}")
                continue

        if not retriever_results:
            return []

        # Combine results using fusion strategy
        if self.fusion_strategy == "weighted_score":
            self._last_retrieval_method = "ensemble_weighted_score"
            combined_docs = self._weighted_score_fusion(retriever_results, query)
        elif self.fusion_strategy == "rank_fusion":
            self._last_retrieval_method = "ensemble_rank_fusion"
            combined_docs = self._rank_fusion(retriever_results, query)
        elif self.fusion_strategy == "reciprocal_rank":
            self._last_retrieval_method = "ensemble_reciprocal_rank"
            combined_docs = self._reciprocal_rank_fusion(retriever_results, query)
        elif self.fusion_strategy == "weighted_vote":
            self._last_retrieval_method = "ensemble_weighted_vote"
            combined_docs = self._weighted_vote_fusion(retriever_results, query)
        else:
            self._last_retrieval_method = "ensemble_weighted_score"
            combined_docs = self._weighted_score_fusion(retriever_results, query)

        return combined_docs

    def _weighted_score_fusion(
        self,
        retriever_results: List[Dict[str, Any]],
        query: str
    ) -> List[RetrievedDocument]:
        """
        Combine results using weighted score averaging

        Args:
            retriever_results: Results from each retriever
            query: Original query

        Returns:
            Combined list of documents
        """
        # Collect all documents with their scores
        doc_scores = defaultdict(lambda: {"total_score": 0.0, "retrievers": [], "doc": None})

        for retriever_data in retriever_results:
            result = retriever_data["result"]
            weight = retriever_data["weight"]
            retriever = retriever_data["retriever"]

            for doc in result.documents:
                doc_id = doc.id
                doc_scores[doc_id]["total_score"] += doc.relevance_score * weight
                doc_scores[doc_id]["retrievers"].append(retriever.__class__.__name__)
                if doc_scores[doc_id]["doc"] is None:
                    doc_scores[doc_id]["doc"] = doc

        # Create combined documents
        combined_docs = []
        for doc_id, score_data in doc_scores.items():
            doc = score_data["doc"]
            combined_doc = RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                id=doc.id,
                relevance_score=min(score_data["total_score"], 1.0),
                retrieval_method="ensemble_weighted_score",
                query=query,
                rank=0,  # Will be set later
                additional_info={
                    "source_retrievers": score_data["retrievers"],
                    "num_retrievers": len(score_data["retrievers"]),
                    "fusion_score": score_data["total_score"]
                }
            )
            combined_docs.append(combined_doc)

        # Sort by combined score
        combined_docs.sort(key=lambda doc: doc.relevance_score, reverse=True)

        # Set ranks
        for i, doc in enumerate(combined_docs):
            doc.rank = i

        return combined_docs

    def _rank_fusion(
        self,
        retriever_results: List[Dict[str, Any]],
        query: str
    ) -> List[RetrievedDocument]:
        """
        Combine results using rank fusion (Borda count)

        Args:
            retriever_results: Results from each retriever
            query: Original query

        Returns:
            Combined list of documents
        """
        # Collect document ranks
        doc_ranks = defaultdict(lambda: {"total_rank": 0.0, "retrievers": [], "doc": None})

        for retriever_data in retriever_results:
            result = retriever_data["result"]
            weight = retriever_data["weight"]
            retriever = retriever_data["retriever"]

            for rank, doc in enumerate(result.documents):
                doc_id = doc.id
                # Higher rank = lower number, so invert for scoring
                rank_score = (len(result.documents) - rank) / len(result.documents)
                doc_ranks[doc_id]["total_rank"] += rank_score * weight
                doc_ranks[doc_id]["retrievers"].append(retriever.__class__.__name__)
                if doc_ranks[doc_id]["doc"] is None:
                    doc_ranks[doc_id]["doc"] = doc

        # Create combined documents
        combined_docs = []
        for doc_id, rank_data in doc_ranks.items():
            doc = rank_data["doc"]
            combined_doc = RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                id=doc.id,
                relevance_score=min(rank_data["total_rank"], 1.0),
                retrieval_method="ensemble_rank_fusion",
                query=query,
                rank=0,
                additional_info={
                    "source_retrievers": rank_data["retrievers"],
                    "num_retrievers": len(rank_data["retrievers"]),
                    "fusion_rank": rank_data["total_rank"]
                }
            )
            combined_docs.append(combined_doc)

        # Sort by combined rank
        combined_docs.sort(key=lambda doc: doc.relevance_score, reverse=True)

        # Set ranks
        for i, doc in enumerate(combined_docs):
            doc.rank = i

        return combined_docs

    def _reciprocal_rank_fusion(
        self,
        retriever_results: List[Dict[str, Any]],
        query: str,
        k: int = 60
    ) -> List[RetrievedDocument]:
        """
        Combine results using reciprocal rank fusion

        Args:
            retriever_results: Results from each retriever
            query: Original query
            k: Fusion parameter (typically 60)

        Returns:
            Combined list of documents
        """
        # Collect reciprocal rank scores
        doc_scores = defaultdict(lambda: {"total_score": 0.0, "retrievers": [], "doc": None})

        for retriever_data in retriever_results:
            result = retriever_data["result"]
            weight = retriever_data["weight"]
            retriever = retriever_data["retriever"]

            for rank, doc in enumerate(result.documents):
                doc_id = doc.id
                # Reciprocal rank: 1 / (rank + k)
                rr_score = 1.0 / (rank + k)
                doc_scores[doc_id]["total_score"] += rr_score * weight
                doc_scores[doc_id]["retrievers"].append(retriever.__class__.__name__)
                if doc_scores[doc_id]["doc"] is None:
                    doc_scores[doc_id]["doc"] = doc

        # Create combined documents
        combined_docs = []
        for doc_id, score_data in doc_scores.items():
            doc = score_data["doc"]
            combined_doc = RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                id=doc.id,
                relevance_score=min(score_data["total_score"], 1.0),
                retrieval_method="ensemble_reciprocal_rank",
                query=query,
                rank=0,
                additional_info={
                    "source_retrievers": score_data["retrievers"],
                    "num_retrievers": len(score_data["retrievers"]),
                    "rrf_score": score_data["total_score"]
                }
            )
            combined_docs.append(combined_doc)

        # Sort by reciprocal rank fusion score
        combined_docs.sort(key=lambda doc: doc.relevance_score, reverse=True)

        # Set ranks
        for i, doc in enumerate(combined_docs):
            doc.rank = i

        return combined_docs

    def _weighted_vote_fusion(
        self,
        retriever_results: List[Dict[str, Any]],
        query: str
    ) -> List[RetrievedDocument]:
        """
        Combine results using weighted voting

        Args:
            retriever_results: Results from each retriever
            query: Original query

        Returns:
            Combined list of documents
        """
        # Count votes for each document
        doc_votes = defaultdict(lambda: {"votes": 0.0, "retrievers": [], "doc": None})

        for retriever_data in retriever_results:
            result = retriever_data["result"]
            weight = retriever_data["weight"]
            retriever = retriever_data["retriever"]

            # Top documents get votes based on their position
            for rank, doc in enumerate(result.documents):
                doc_id = doc.id
                # Vote strength decreases with rank
                vote_strength = weight * (1.0 / (rank + 1))
                doc_votes[doc_id]["votes"] += vote_strength
                doc_votes[doc_id]["retrievers"].append(retriever.__class__.__name__)
                if doc_votes[doc_id]["doc"] is None:
                    doc_votes[doc_id]["doc"] = doc

        # Create combined documents
        combined_docs = []
        for doc_id, vote_data in doc_votes.items():
            doc = vote_data["doc"]
            combined_doc = RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                id=doc.id,
                relevance_score=min(vote_data["votes"], 1.0),
                retrieval_method="ensemble_weighted_vote",
                query=query,
                rank=0,
                additional_info={
                    "source_retrievers": vote_data["retrievers"],
                    "num_retrievers": len(vote_data["retrievers"]),
                    "vote_score": vote_data["votes"]
                }
            )
            combined_docs.append(combined_doc)

        # Sort by vote score
        combined_docs.sort(key=lambda doc: doc.relevance_score, reverse=True)

        # Set ranks
        for i, doc in enumerate(combined_docs):
            doc.rank = i

        return combined_docs

    def get_document_count(self) -> int:
        """Get total number of documents (average across retrievers)"""
        if not self.retrievers:
            return 0

        total_count = 0
        valid_retrievers = 0

        for retriever in self.retrievers:
            try:
                count = retriever.get_document_count()
                total_count += count
                valid_retrievers += 1
            except Exception:
                continue

        return total_count // valid_retrievers if valid_retrievers > 0 else 0

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ensemble retriever

        Returns:
            Dictionary with ensemble statistics
        """
        retriever_stats = []
        total_docs = 0

        for i, retriever in enumerate(self.retrievers):
            try:
                info = retriever.get_retriever_info()
                retriever_stats.append({
                    "index": i,
                    "type": info.get("retriever_type", "Unknown"),
                    "weight": self.weights[i],
                    "document_count": info.get("document_count", 0)
                })
                total_docs += info.get("document_count", 0)
            except Exception as e:
                retriever_stats.append({
                    "index": i,
                    "type": "Error",
                    "weight": self.weights[i],
                    "error": str(e)
                })

        return {
            "fusion_strategy": self.fusion_strategy,
            "num_retrievers": len(self.retrievers),
            "weights": self.weights,
            "total_documents": total_docs,
            "retriever_stats": retriever_stats,
            "cache_size": len(self.result_cache)
        }

    def compare_retrievers(self, query: str) -> Dict[str, RetrievalResult]:
        """
        Compare results from individual retrievers

        Args:
            query: Query to test

        Returns:
            Dictionary mapping retriever names to their results
        """
        comparison = {}

        for i, retriever in enumerate(self.retrievers):
            try:
                result = retriever.retrieve(query)
                # Create unique key by appending index if needed
                base_name = retriever.__class__.__name__
                key = f"{base_name}_{i}" if base_name in comparison else base_name
                comparison[key] = result
            except Exception as e:
                self.logger.error(f"Retriever {i} failed during comparison: {str(e)}")

        return comparison

    def clear_cache(self):
        """Clear ensemble cache and all retriever caches"""
        self.result_cache.clear()

        for retriever in self.retrievers:
            try:
                if hasattr(retriever, 'clear_cache'):
                    retriever.clear_cache()
            except Exception as e:
                self.logger.error(f"Failed to clear cache for retriever: {str(e)}")

    def get_retriever_info(self) -> Dict[str, Any]:
        """Get information about the ensemble retriever"""
        base_info = super().get_retriever_info()
        base_info.update({
            "ensemble_type": "EnsembleRetriever",
            "fusion_strategy": self.fusion_strategy,
            "num_retrievers": len(self.retrievers),
            "weights": self.weights,
            "retriever_types": [r.__class__.__name__ for r in self.retrievers],
            "ensemble_stats": self.get_ensemble_stats()
        })
        return base_info

    def set_fusion_strategy(self, strategy: str):
        """
        Change the fusion strategy

        Args:
            strategy: New fusion strategy
        """
        valid_strategies = ["weighted_score", "rank_fusion", "reciprocal_rank", "weighted_vote"]
        if strategy not in valid_strategies:
            raise ValueError(f"fusion_strategy must be one of {valid_strategies}")

        self.fusion_strategy = strategy
        self.clear_cache()  # Clear cache since strategy changed

    def set_weights(self, weights: List[float]):
        """
        Update retriever weights

        Args:
            weights: New weights for retrievers
        """
        if len(weights) != len(self.retrievers):
            raise ValueError("Number of weights must match number of retrievers")

        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        self.weights = weights
        self.clear_cache()  # Clear cache since weights changed