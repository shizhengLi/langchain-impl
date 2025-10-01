# -*- coding: utf-8 -*-
"""
Document retriever implementation
"""

from typing import List, Dict, Any, Optional, Callable
import re
import math
from collections import Counter, defaultdict

from .base import BaseRetriever
from .types import (
    RetrievalConfig, RetrievalResult, RetrievalProcessingError,
    Document, RetrievedDocument, RetrievalQuery
)


class DocumentRetriever(BaseRetriever):
    """
    Document-based retriever using keyword matching and text similarity

    This retriever performs document retrieval based on keyword matching,
    TF-IDF scoring, and other text-based similarity measures.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None, **kwargs):
        """
        Initialize document retriever

        Args:
            config: Retrieval configuration
            **kwargs: Additional configuration parameters
        """
        super().__init__(config, **kwargs)
        self.documents: List[Document] = []
        self.doc_index: Dict[str, int] = {}  # Document ID to index mapping
        self.term_frequency: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.document_frequency: Dict[str, int] = defaultdict(int)
        self.total_docs = 0

    def _setup_retriever(self):
        """Setup document-specific configuration"""
        self.min_term_length = 1
        self.max_term_length = 50
        self.stop_words = self._get_default_stop_words()

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the retriever

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs
        """
        doc_ids = []
        for doc in documents:
            doc_id = self._add_single_document(doc)
            doc_ids.append(doc_id)

        self._rebuild_index()
        return doc_ids

    def _add_single_document(self, document: Document) -> str:
        """
        Add a single document to the retriever

        Args:
            document: Document to add

        Returns:
            Document ID
        """
        # Check if document already exists
        if document.id in self.doc_index:
            return document.id

        # Add document
        index = len(self.documents)
        self.documents.append(document)
        self.doc_index[document.id] = index

        # Update term frequencies
        terms = self._extract_terms(document.content)
        for term in terms:
            self.term_frequency[document.id][term] += 1

        return document.id

    def _rebuild_index(self):
        """Rebuild the document frequency index"""
        self.document_frequency.clear()
        self.total_docs = len(self.documents)

        # Count document frequencies
        all_terms = set()
        for doc in self.documents:
            terms = set(self._extract_terms(doc.content))
            all_terms.update(terms)

        for term in all_terms:
            self.document_frequency[term] = sum(
                1 for doc in self.documents
                if term in self._extract_terms(doc.content)
            )

    def _retrieve_documents(self, query: str, config: RetrievalConfig) -> List[RetrievedDocument]:
        """
        Retrieve documents for a query

        Args:
            query: Query text
            config: Retrieval configuration

        Returns:
            List of retrieved documents
        """
        if not self.documents:
            return []

        # Extract query terms
        query_terms = self._extract_terms(query)
        if not query_terms:
            return []

        # Calculate relevance scores for all documents
        scored_docs = []
        for i, doc in enumerate(self.documents):
            # Apply filters if specified
            if config.filter_dict and not doc.matches_filter(config.filter_dict):
                continue

            # Calculate relevance score
            if config.search_type == "similarity":
                score = self._calculate_similarity_score(query_terms, doc)
            elif config.search_type == "tfidf":
                score = self._calculate_tfidf_score(query_terms, doc)
            elif config.search_type == "bm25":
                score = self._calculate_bm25_score(query_terms, doc)
            else:
                score = self._calculate_similarity_score(query_terms, doc)

            if score > 0:
                retrieved_doc = RetrievedDocument(
                    content=doc.content,
                    metadata=doc.metadata,
                    id=doc.id,
                    relevance_score=min(score, 1.0),  # Normalize to [0, 1]
                    retrieval_method="document_" + config.search_type,
                    query=query,
                    rank=0  # Will be set later
                )
                scored_docs.append(retrieved_doc)

        # Sort by relevance score (descending)
        scored_docs.sort(key=lambda doc: doc.relevance_score, reverse=True)

        # Set ranks
        for i, doc in enumerate(scored_docs):
            doc.rank = i

        return scored_docs

    def _extract_terms(self, text: str) -> List[str]:
        """
        Extract terms from text

        Args:
            text: Text to extract terms from

        Returns:
            List of terms
        """
        # Convert to lowercase and extract alphanumeric terms
        terms = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

        # Filter terms
        filtered_terms = []
        for term in terms:
            if (self.min_term_length <= len(term) <= self.max_term_length and
                term not in self.stop_words):
                filtered_terms.append(term)

        return filtered_terms

    def _calculate_similarity_score(self, query_terms: List[str], document: Document) -> float:
        """
        Calculate simple similarity score based on term overlap

        Args:
            query_terms: List of query terms
            document: Document to score

        Returns:
            Similarity score
        """
        doc_terms = self._extract_terms(document.content)
        query_term_set = set(query_terms)
        doc_term_set = set(doc_terms)

        # Jaccard similarity
        intersection = len(query_term_set & doc_term_set)
        union = len(query_term_set | doc_term_set)

        if union == 0:
            return 0.0

        return intersection / union

    def _calculate_tfidf_score(self, query_terms: List[str], document: Document) -> float:
        """
        Calculate TF-IDF score

        Args:
            query_terms: List of query terms
            document: Document to score

        Returns:
            TF-IDF score
        """
        doc_terms = self._extract_terms(document.content)
        doc_term_counts = Counter(doc_terms)
        doc_length = len(doc_terms)

        if doc_length == 0:
            return 0.0

        score = 0.0
        for term in query_terms:
            # Term frequency in document
            tf = doc_term_counts.get(term, 0) / doc_length

            # Inverse document frequency
            df = self.document_frequency.get(term, 0)
            idf = math.log(self.total_docs / (df + 1))

            score += tf * idf

        # Normalize by query length
        if len(query_terms) > 0:
            score = score / len(query_terms)

        return score

    def _calculate_bm25_score(self, query_terms: List[str], document: Document) -> float:
        """
        Calculate BM25 score

        Args:
            query_terms: List of query terms
            document: Document to score

        Returns:
            BM25 score
        """
        doc_terms = self._extract_terms(document.content)
        doc_term_counts = Counter(doc_terms)
        doc_length = len(doc_terms)

        if doc_length == 0:
            return 0.0

        # BM25 parameters
        k1 = 1.2
        b = 0.75

        # Average document length
        avg_doc_length = sum(len(self._extract_terms(doc.content)) for doc in self.documents) / len(self.documents)

        score = 0.0
        for term in query_terms:
            tf = doc_term_counts.get(term, 0)
            df = self.document_frequency.get(term, 0)

            if tf == 0:
                continue

            # IDF
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5))

            # BM25 term score
            term_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))
            score += term_score

        return score

    def _rerank_documents(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        Rerank documents using additional heuristics

        Args:
            documents: List of documents to rerank

        Returns:
            Reranked list of documents
        """
        if len(documents) <= 1:
            return documents

        # Calculate additional scores
        for doc in documents:
            # Length penalty (prefer not too short, not too long)
            content_length = len(doc.content)
            length_score = 1.0

            if content_length < 50:
                length_score = 0.5  # Penalty for very short
            elif content_length > 2000:
                length_score = 0.8  # Penalty for very long

            # Term density (query terms / total terms)
            query_terms = set(self._extract_terms(doc.query))
            doc_terms = self._extract_terms(doc.content)
            if doc_terms:
                density = len(query_terms & set(doc_terms)) / len(doc_terms)
                density_score = min(density * 5, 1.0)  # Cap at 1.0
            else:
                density_score = 0.0

            # Combine scores
            combined_score = (
                0.7 * doc.relevance_score +
                0.2 * length_score +
                0.1 * density_score
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

    def _get_default_stop_words(self) -> set:
        """
        Get default stop words list

        Returns:
            Set of stop words
        """
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'i', 'you', 'your', 'we', 'our',
            'can', 'or', 'but', 'not', 'this', 'they', 'have', 'had', 'what',
            'when', 'where', 'who', 'which', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'too', 'very', 'just', 'should', 'now'
        }

    def get_document_count(self) -> int:
        """Get total number of documents"""
        return len(self.documents)

    def get_term_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about terms in the corpus

        Returns:
            Dictionary with term statistics
        """
        if not self.documents:
            return {}

        # Count all terms
        all_term_counts = Counter()
        for doc in self.documents:
            terms = self._extract_terms(doc.content)
            all_term_counts.update(terms)

        # Calculate statistics
        total_terms = sum(all_term_counts.values())
        unique_terms = len(all_term_counts)
        avg_doc_length = sum(len(self._extract_terms(doc.content)) for doc in self.documents) / len(self.documents)

        return {
            "total_documents": len(self.documents),
            "total_terms": total_terms,
            "unique_terms": unique_terms,
            "avg_document_length": avg_doc_length,
            "most_common_terms": all_term_counts.most_common(10),
            "vocabulary_size": unique_terms
        }

    def search_by_term(self, term: str, limit: int = 10) -> List[Document]:
        """
        Search documents containing a specific term

        Args:
            term: Term to search for
            limit: Maximum number of results

        Returns:
            List of documents containing the term
        """
        term = term.lower()
        matching_docs = []

        for doc in self.documents:
            if term in self._extract_terms(doc.content):
                matching_docs.append(doc)

        return matching_docs[:limit]

    def get_similar_documents(self, document: Document, limit: int = 5) -> List[Document]:
        """
        Find documents similar to a given document

        Args:
            document: Reference document
            limit: Maximum number of results

        Returns:
            List of similar documents
        """
        doc_terms = self._extract_terms(document.content)
        if not doc_terms:
            return []

        # Calculate similarity scores
        scored_docs = []
        for doc in self.documents:
            if doc.id == document.id:
                continue  # Skip the same document

            score = self._calculate_similarity_score(doc_terms, doc)
            if score > 0:
                scored_docs.append((doc, score))

        # Sort by score and return top results
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:limit]]

    def clear_documents(self):
        """Clear all documents from the retriever"""
        self.documents.clear()
        self.doc_index.clear()
        self.term_frequency.clear()
        self.document_frequency.clear()
        self.total_docs = 0

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the retriever

        Args:
            doc_id: ID of document to remove

        Returns:
            True if document was removed, False if not found
        """
        if doc_id not in self.doc_index:
            return False

        index = self.doc_index[doc_id]
        del self.documents[index]
        del self.doc_index[doc_id]

        # Rebuild index
        self._rebuild_index()
        return True

    def update_document(self, document: Document) -> bool:
        """
        Update an existing document

        Args:
            document: Updated document

        Returns:
            True if document was updated, False if not found
        """
        if document.id not in self.doc_index:
            return False

        # Remove old document and add new one
        self.remove_document(document.id)
        self._add_single_document(document)
        self._rebuild_index()
        return True