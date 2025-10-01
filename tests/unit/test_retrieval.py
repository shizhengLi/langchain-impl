# -*- coding: utf-8 -*-
"""
Tests for retrieval module
"""

import pytest
import time
from my_langchain.retrieval.types import (
    RetrievalConfig, RetrievalResult, RetrievalError, RetrievalValidationError,
    RetrievalProcessingError, Document, RetrievedDocument, RetrievalQuery,
    create_document, create_retrieved_document, create_retrieval_query,
    calculate_retrieval_metrics, merge_retrieval_results
)
from my_langchain.retrieval.base import BaseRetriever
from my_langchain.retrieval.document_retriever import DocumentRetriever
from my_langchain.retrieval.vector_retriever import VectorRetriever
from my_langchain.retrieval.ensemble_retriever import EnsembleRetriever
from my_langchain.embeddings import MockEmbedding
from my_langchain.vectorstores import InMemoryVectorStore
from my_langchain.vectorstores.types import VectorStoreConfig


class TestRetrievalConfig:
    """Test retrieval configuration"""

    def test_default_config(self):
        """Test default configuration"""
        config = RetrievalConfig()
        assert config.top_k == 5
        assert config.score_threshold is None
        assert config.search_type == "similarity"
        assert config.mmr_lambda == 0.5
        assert config.fetch_k == 20
        assert config.filter_dict == {}
        assert config.normalize_scores is True
        assert config.enable_caching is True
        assert config.rerank is False

    def test_custom_config(self):
        """Test custom configuration"""
        config = RetrievalConfig(
            top_k=10,
            score_threshold=0.7,
            search_type="mmr",
            mmr_lambda=0.8,
            rerank=True
        )
        assert config.top_k == 10
        assert config.score_threshold == 0.7
        assert config.search_type == "mmr"
        assert config.mmr_lambda == 0.8
        assert config.rerank is True

    def test_config_validation(self):
        """Test configuration validation"""
        # Valid configurations
        RetrievalConfig(top_k=1)
        RetrievalConfig(mmr_lambda=0.0)
        RetrievalConfig(mmr_lambda=1.0)
        RetrievalConfig(score_threshold=0.0)
        RetrievalConfig(score_threshold=1.0)
        RetrievalConfig(search_type="hybrid")

        # Invalid configurations
        with pytest.raises(ValueError):
            RetrievalConfig(top_k=0)

        with pytest.raises(ValueError):
            RetrievalConfig(mmr_lambda=-0.1)

        with pytest.raises(ValueError):
            RetrievalConfig(mmr_lambda=1.1)

        with pytest.raises(ValueError):
            RetrievalConfig(fetch_k=0)

        with pytest.raises(ValueError):
            RetrievalConfig(score_threshold=-0.1)

        with pytest.raises(ValueError):
            RetrievalConfig(score_threshold=1.1)

        with pytest.raises(ValueError):
            RetrievalConfig(search_type="invalid")


class TestDocument:
    """Test Document class"""

    def test_document_creation(self):
        """Test document creation"""
        doc = Document(
            content="Hello world",
            metadata={"source": "test"},
            id="doc_1"
        )
        assert doc.content == "Hello world"
        assert doc.metadata["source"] == "test"
        assert doc.id == "doc_1"

    def test_document_validation(self):
        """Test document validation"""
        # Valid document
        Document(content="Hello world")

        # Invalid content (empty)
        with pytest.raises(ValueError):
            Document(content="")

        with pytest.raises(ValueError):
            Document(content="   ")

    def test_document_str(self):
        """Test document string representation"""
        doc = Document(content="Hello world", id="test_document_id")
        str_repr = str(doc)
        assert "Document" in str_repr
        assert "test_document_id" in str_repr or "test_doc" in str_repr  # Handle truncated ID
        assert "content_length=11" in str_repr

    def test_get_text_snippet(self):
        """Test text snippet extraction"""
        doc = Document(content="This is a long document content for testing")

        # Shorter than max length
        snippet = doc.get_text_snippet(100)
        assert snippet == doc.content

        # Longer than max length
        snippet = doc.get_text_snippet(20)
        assert len(snippet) <= 23  # 20 + "..."
        assert snippet.endswith("...")

    def test_matches_filter(self):
        """Test filter matching"""
        doc = Document(
            content="Test content",
            metadata={"source": "test", "type": "pdf", "year": 2023}
        )

        # Match filter
        assert doc.matches_filter({"source": "test"})
        assert doc.matches_filter({"type": "pdf"})
        assert doc.matches_filter({"year": 2023})

        # Multiple filter conditions
        assert doc.matches_filter({"source": "test", "type": "pdf"})

        # No match
        assert not doc.matches_filter({"source": "other"})
        assert not doc.matches_filter({"type": "doc"})

        # Empty filter
        assert doc.matches_filter({})

        # Filter with non-existent key
        assert not doc.matches_filter({"missing": "value"})


class TestRetrievedDocument:
    """Test RetrievedDocument class"""

    def test_retrieved_document_creation(self):
        """Test retrieved document creation"""
        doc = RetrievedDocument(
            content="Hello world",
            metadata={"source": "test"},
            id="doc_1",
            relevance_score=0.85,
            retrieval_method="similarity",
            query="test query",
            rank=1
        )
        assert doc.content == "Hello world"
        assert doc.relevance_score == 0.85
        assert doc.retrieval_method == "similarity"
        assert doc.query == "test query"
        assert doc.rank == 1

    def test_retrieved_document_validation(self):
        """Test retrieved document validation"""
        # Valid document
        RetrievedDocument(
            content="Hello",
            relevance_score=0.5,
            retrieval_method="test",
            query="test",
            rank=0
        )

        # Invalid relevance score
        with pytest.raises(ValueError):
            RetrievedDocument(
                content="Hello",
                relevance_score=-0.1,
                retrieval_method="test",
                query="test",
                rank=0
            )

        with pytest.raises(ValueError):
            RetrievedDocument(
                content="Hello",
                relevance_score=1.1,
                retrieval_method="test",
                query="test",
                rank=0
            )

        # Invalid rank
        with pytest.raises(ValueError):
            RetrievedDocument(
                content="Hello",
                relevance_score=0.5,
                retrieval_method="test",
                query="test",
                rank=-1
            )

    def test_retrieved_document_str(self):
        """Test retrieved document string representation"""
        doc = RetrievedDocument(
            content="Hello world",
            relevance_score=0.85,
            retrieval_method="test",
            query="test",
            rank=1,
            id="test_doc_id"
        )
        str_repr = str(doc)
        assert "RetrievedDocument" in str_repr
        assert "test_doc_id" in str_repr or "test_doc" in str_repr  # Handle truncated ID
        assert "score=0.850" in str_repr
        assert "rank=1" in str_repr


class TestRetrievalQuery:
    """Test RetrievalQuery class"""

    def test_retrieval_query_creation(self):
        """Test retrieval query creation"""
        query = RetrievalQuery(
            query="test query",
            metadata={"user": "test"},
            filters={"type": "pdf"},
            max_results=10,
            min_score=0.5
        )
        assert query.query == "test query"
        assert query.metadata["user"] == "test"
        assert query.filters["type"] == "pdf"
        assert query.max_results == 10
        assert query.min_score == 0.5

    def test_retrieval_query_validation(self):
        """Test retrieval query validation"""
        # Valid query
        RetrievalQuery(query="test")

        # Invalid query (empty)
        with pytest.raises(ValueError):
            RetrievalQuery(query="")

        with pytest.raises(ValueError):
            RetrievalQuery(query="   ")

    def test_retrieval_query_str(self):
        """Test retrieval query string representation"""
        query = RetrievalQuery(query="this is a very long test query that should definitely be truncated in the string representation for testing purposes")
        str_repr = str(query)
        assert "RetrievalQuery" in str_repr
        assert "this is a very long test query that should definit..." in str_repr
        assert "max_results=None" in str_repr


class TestRetrievalResult:
    """Test RetrievalResult class"""

    def test_retrieval_result_creation(self):
        """Test retrieval result creation"""
        docs = [
            RetrievedDocument(
                content="Doc 1",
                relevance_score=0.9,
                retrieval_method="test",
                query="test",
                rank=0,
                id="doc_1"
            ),
            RetrievedDocument(
                content="Doc 2",
                relevance_score=0.8,
                retrieval_method="test",
                query="test",
                rank=1,
                id="doc_2"
            )
        ]

        result = RetrievalResult(
            documents=docs,
            query="test query",
            total_results=2,
            search_time=0.5,
            retrieval_method="test_method"
        )
        assert len(result.documents) == 2
        assert result.query == "test query"
        assert result.total_results == 2
        assert result.search_time == 0.5
        assert result.retrieval_method == "test_method"

    def test_retrieval_result_validation(self):
        """Test retrieval result validation"""
        doc = RetrievedDocument(
            content="Test",
            relevance_score=0.5,
            retrieval_method="test",
            query="test",
            rank=0
        )

        # Valid result
        RetrievalResult(
            documents=[doc],
            query="test",
            total_results=1,
            search_time=0.1,
            retrieval_method="test"
        )

        # Valid empty documents (no results found)
        RetrievalResult(
            documents=[],
            query="test",
            total_results=0,
            search_time=0.1,
            retrieval_method="test"
        )

        # Invalid total_results
        with pytest.raises(ValueError):
            RetrievalResult(
                documents=[doc],
                query="test",
                total_results=0,
                search_time=0.1,
                retrieval_method="test"
            )

        # Invalid search_time
        with pytest.raises(ValueError):
            RetrievalResult(
                documents=[doc],
                query="test",
                total_results=1,
                search_time=0,
                retrieval_method="test"
            )

    def test_retrieval_result_operations(self):
        """Test retrieval result operations"""
        docs = [
            RetrievedDocument(
                content=f"Doc {i}",
                relevance_score=0.9 - i * 0.1,
                retrieval_method="test",
                query="test",
                rank=i,
                id=f"doc_{i}"
            )
            for i in range(3)
        ]

        result = RetrievalResult(
            documents=docs,
            query="test",
            total_results=3,
            search_time=0.5,
            retrieval_method="test"
        )

        # Test length
        assert len(result) == 3

        # Test indexing
        assert result[0].content == "Doc 0"
        assert result[1].content == "Doc 1"

        # Test iteration
        contents = [doc.content for doc in result]
        assert contents == ["Doc 0", "Doc 1", "Doc 2"]

    def test_get_top_k(self):
        """Test getting top k documents"""
        docs = [
            RetrievedDocument(
                content=f"Doc {i}",
                relevance_score=0.9 - i * 0.1,
                retrieval_method="test",
                query="test",
                rank=i,
                id=f"doc_{i}"
            )
            for i in range(5)
        ]

        result = RetrievalResult(
            documents=docs,
            query="test",
            total_results=5,
            search_time=0.5,
            retrieval_method="test"
        )

        top_3 = result.get_top_k(3)
        assert len(top_3) == 3
        assert top_3[0].content == "Doc 0"
        assert top_3[1].content == "Doc 1"
        assert top_3[2].content == "Doc 2"

    def test_get_documents_above_threshold(self):
        """Test filtering by score threshold"""
        docs = [
            RetrievedDocument(
                content=f"Doc {i}",
                relevance_score=0.9 - i * 0.2,
                retrieval_method="test",
                query="test",
                rank=i,
                id=f"doc_{i}"
            )
            for i in range(4)
        ]

        result = RetrievalResult(
            documents=docs,
            query="test",
            total_results=4,
            search_time=0.5,
            retrieval_method="test"
        )

        above_threshold = result.get_documents_above_threshold(0.6)
        assert len(above_threshold) == 2
        assert above_threshold[0].content == "Doc 0"
        assert above_threshold[1].content == "Doc 1"

    def test_get_average_score(self):
        """Test getting average score"""
        docs = [
            RetrievedDocument(
                content="Doc 1",
                relevance_score=0.8,
                retrieval_method="test",
                query="test",
                rank=0,
                id="doc_1"
            ),
            RetrievedDocument(
                content="Doc 2",
                relevance_score=0.6,
                retrieval_method="test",
                query="test",
                rank=1,
                id="doc_2"
            )
        ]

        result = RetrievalResult(
            documents=docs,
            query="test",
            total_results=2,
            search_time=0.5,
            retrieval_method="test"
        )

        avg_score = result.get_average_score()
        assert avg_score == 0.7

    def test_get_score_distribution(self):
        """Test getting score distribution"""
        docs = [
            RetrievedDocument(
                content="Doc 1",
                relevance_score=0.9,
                retrieval_method="test",
                query="test",
                rank=0,
                id="doc_1"
            ),
            RetrievedDocument(
                content="Doc 2",
                relevance_score=0.7,
                retrieval_method="test",
                query="test",
                rank=1,
                id="doc_2"
            ),
            RetrievedDocument(
                content="Doc 3",
                relevance_score=0.5,
                retrieval_method="test",
                query="test",
                rank=2,
                id="doc_3"
            )
        ]

        result = RetrievalResult(
            documents=docs,
            query="test",
            total_results=3,
            search_time=0.5,
            retrieval_method="test"
        )

        distribution = result.get_score_distribution()
        assert distribution["min"] == 0.5
        assert distribution["max"] == 0.9
        assert abs(distribution["avg"] - 0.7) < 1e-10
        assert distribution["median"] == 0.7

    def test_filter_by_metadata(self):
        """Test filtering by metadata"""
        docs = [
            RetrievedDocument(
                content="Doc 1",
                relevance_score=0.9,
                retrieval_method="test",
                query="test",
                rank=0,
                metadata={"type": "pdf"},
                id="doc_1"
            ),
            RetrievedDocument(
                content="Doc 2",
                relevance_score=0.8,
                retrieval_method="test",
                query="test",
                rank=1,
                metadata={"type": "doc"},
                id="doc_2"
            )
        ]

        result = RetrievalResult(
            documents=docs,
            query="test",
            total_results=2,
            search_time=0.5,
            retrieval_method="test"
        )

        filtered = result.filter_by_metadata("type", "pdf")
        assert len(filtered.documents) == 1
        assert filtered.documents[0].content == "Doc 1"


class TestDocumentRetriever:
    """Test DocumentRetriever class"""

    def test_document_retriever_creation(self):
        """Test document retriever creation"""
        retriever = DocumentRetriever()
        assert retriever.config.top_k == 5
        assert retriever.config.search_type == "similarity"
        assert len(retriever.documents) == 0

    def test_add_documents(self):
        """Test adding documents"""
        retriever = DocumentRetriever()
        docs = [
            Document(content="Hello world", metadata={"source": "test1"}),
            Document(content="Goodbye world", metadata={"source": "test2"})
        ]

        doc_ids = retriever.add_documents(docs)
        assert len(doc_ids) == 2
        assert len(retriever.documents) == 2

        # Test document count
        assert retriever.get_document_count() == 2

    def test_retrieve_documents(self):
        """Test document retrieval"""
        retriever = DocumentRetriever()
        docs = [
            Document(content="The quick brown fox jumps over the lazy dog"),
            Document(content="Python is a programming language"),
            Document(content="Machine learning is a subset of AI")
        ]

        retriever.add_documents(docs)

        # Search for documents containing "python"
        result = retriever.retrieve("python")
        assert len(result.documents) >= 1
        assert "Python" in result.documents[0].content
        assert result.query == "python"

    def test_search_with_filters(self):
        """Test search with metadata filters"""
        retriever = DocumentRetriever()
        docs = [
            Document(content="Document 1", metadata={"type": "pdf", "year": 2023}),
            Document(content="Document 2", metadata={"type": "doc", "year": 2023}),
            Document(content="Document 3", metadata={"type": "pdf", "year": 2022})
        ]

        retriever.add_documents(docs)

        # Search with filter
        config = RetrievalConfig(filter_dict={"type": "pdf"})
        result = retriever.retrieve_with_config("document", config)

        assert len(result.documents) == 2
        for doc in result.documents:
            assert doc.metadata.get("type") == "pdf"

    def test_different_search_types(self):
        """Test different search types"""
        retriever = DocumentRetriever()
        docs = [
            Document(content="The quick brown fox"),
            Document(content="A fast brown dog"),
            Document(content="Python programming")
        ]

        retriever.add_documents(docs)

        # Test similarity search
        config = RetrievalConfig(search_type="similarity")
        result1 = retriever.retrieve_with_config("quick", config)

        # Test TF-IDF search
        config = RetrievalConfig(search_type="tfidf")
        result2 = retriever.retrieve_with_config("quick", config)

        # Test BM25 search
        config = RetrievalConfig(search_type="bm25")
        result3 = retriever.retrieve_with_config("quick", config)

        assert len(result1.documents) > 0
        assert len(result2.documents) > 0
        assert len(result3.documents) > 0

    def test_term_statistics(self):
        """Test term statistics"""
        retriever = DocumentRetriever()
        docs = [
            Document(content="The quick brown fox"),
            Document(content="The quick blue dog"),
            Document(content="A slow red cat")
        ]

        retriever.add_documents(docs)
        stats = retriever.get_term_statistics()

        assert "total_documents" in stats
        assert "total_terms" in stats
        assert "unique_terms" in stats
        assert "avg_document_length" in stats
        assert "most_common_terms" in stats

        assert stats["total_documents"] == 3
        assert len(stats["most_common_terms"]) <= 10  # Should be 8 unique terms

    def test_search_by_term(self):
        """Test searching by specific term"""
        retriever = DocumentRetriever()
        docs = [
            Document(content="The quick brown fox"),
            Document(content="A quick blue dog"),
            Document(content="A slow red cat")
        ]

        retriever.add_documents(docs)
        results = retriever.search_by_term("quick")

        assert len(results) == 2
        for doc in results:
            assert "quick" in doc.content.lower()

    def test_get_similar_documents(self):
        """Test finding similar documents"""
        retriever = DocumentRetriever()
        docs = [
            Document(content="The quick brown fox jumps"),
            Document(content="A fast brown dog runs"),
            Document(content="Python programming tutorial"),
            Document(content="The quick brown cat sleeps")
        ]

        retriever.add_documents(docs)

        # Find documents similar to the first one
        reference_doc = docs[0]
        similar_docs = retriever.get_similar_documents(reference_doc, limit=2)

        assert len(similar_docs) <= 2
        # Should find documents with similar terms
        for doc in similar_docs:
            assert doc.id != reference_doc.id

    def test_remove_document(self):
        """Test removing documents"""
        retriever = DocumentRetriever()
        docs = [
            Document(content="Document 1", id="doc1"),
            Document(content="Document 2", id="doc2"),
            Document(content="Document 3", id="doc3")
        ]

        retriever.add_documents(docs)
        assert retriever.get_document_count() == 3

        # Remove a document
        removed = retriever.remove_document("doc2")
        assert removed is True
        assert retriever.get_document_count() == 2

        # Try to remove non-existent document
        removed = retriever.remove_document("nonexistent")
        assert removed is False

    def test_update_document(self):
        """Test updating documents"""
        retriever = DocumentRetriever()
        doc = Document(content="Original content", id="doc1")
        retriever.add_documents([doc])

        # Update document
        updated_doc = Document(content="Updated content", id="doc1")
        updated = retriever.update_document(updated_doc)
        assert updated is True

        # Verify update
        results = retriever.retrieve("updated")
        assert len(results.documents) == 1
        assert results.documents[0].content == "Updated content"

    def test_clear_documents(self):
        """Test clearing all documents"""
        retriever = DocumentRetriever()
        docs = [
            Document(content="Document 1"),
            Document(content="Document 2")
        ]

        retriever.add_documents(docs)
        assert retriever.get_document_count() == 2

        retriever.clear_documents()
        assert retriever.get_document_count() == 0
        assert len(retriever.documents) == 0

    def test_batch_retrieve(self):
        """Test batch retrieval"""
        retriever = DocumentRetriever()
        docs = [
            Document(content="The quick brown fox"),
            Document(content="Python programming language"),
            Document(content="Machine learning algorithms")
        ]

        retriever.add_documents(docs)

        queries = ["python", "fox", "algorithms"]
        results = retriever.batch_retrieve(queries)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, RetrievalResult)
            assert result.query in queries


class TestVectorRetriever:
    """Test VectorRetriever class"""

    def setup_method(self):
        """Setup method to create common test objects"""
        self.embedding_model = MockEmbedding(embedding_dimension=10)
        self.vector_config = VectorStoreConfig(dimension=10)
        self.vector_store = InMemoryVectorStore(config=self.vector_config)

    def test_vector_retriever_creation(self):
        """Test vector retriever creation"""
        retriever = VectorRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store
        )

        assert retriever.embedding_model is not None
        assert retriever.vector_store is not None
        assert retriever.config.top_k == 5

    def test_add_documents(self):
        """Test adding documents"""
        retriever = VectorRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store
        )

        docs = [
            Document(content="Hello world", metadata={"source": "test1"}),
            Document(content="Goodbye world", metadata={"source": "test2"})
        ]

        doc_ids = retriever.add_documents(docs)
        assert len(doc_ids) == 2
        assert retriever.get_document_count() == 2

    def test_vector_retrieval(self):
        """Test vector retrieval"""
        retriever = VectorRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store
        )

        docs = [
            Document(content="The quick brown fox"),
            Document(content="Python programming"),
            Document(content="Machine learning")
        ]

        retriever.add_documents(docs)

        # Search for similar content
        result = retriever.retrieve("python")
        assert len(result.documents) >= 1
        assert result.query == "python"

        # Check that retrieved documents have proper metadata
        for doc in result.documents:
            assert doc.retrieval_method.startswith("vector_")
            assert doc.rank >= 0

    def test_mmr_search(self):
        """Test MMR search"""
        retriever = VectorRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store
        )

        docs = [
            Document(content="Document 1 about machine learning"),
            Document(content="Document 2 about machine learning and AI"),
            Document(content="Document 3 about python programming"),
            Document(content="Document 4 about data science")
        ]

        retriever.add_documents(docs)

        # Search with MMR
        config = RetrievalConfig(search_type="mmr", top_k=3, mmr_lambda=0.5)
        result = retriever.retrieve_with_config("machine learning", config)

        assert len(result.documents) <= 3
        assert result.retrieval_method == "vector_mmr"

    def test_search_by_embedding(self):
        """Test search by pre-computed embedding"""
        retriever = VectorRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store
        )

        docs = [
            Document(content="Test document 1"),
            Document(content="Test document 2")
        ]

        retriever.add_documents(docs)

        # Generate query embedding
        query_embedding = self.embedding_model.embed_query("test")

        # Search using embedding
        results = retriever.search_by_embedding(query_embedding, k=2)
        assert len(results) <= 2
        assert results[0].query == "<embedding>"

    def test_embedding_stats(self):
        """Test embedding statistics"""
        retriever = VectorRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store
        )

        docs = [Document(content=f"Document {i}") for i in range(5)]
        retriever.add_documents(docs)

        stats = retriever.get_embedding_stats()
        assert "vector_count" in stats
        assert "embedding_model" in stats
        assert "cache_size" in stats
        assert "embedding_dimension" in stats

        assert stats["vector_count"] == 5
        assert stats["embedding_dimension"] == 10

    def test_clear_cache(self):
        """Test cache clearing"""
        retriever = VectorRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store
        )

        # Add documents and perform search to populate cache
        docs = [Document(content="Test document")]
        retriever.add_documents(docs)
        retriever.retrieve("test")

        # Clear cache
        retriever.clear_cache()
        stats = retriever.get_embedding_stats()
        assert stats["cache_size"] == 0


class TestEnsembleRetriever:
    """Test EnsembleRetriever class"""

    def test_ensemble_retriever_creation(self):
        """Test ensemble retriever creation"""
        retriever1 = DocumentRetriever()
        retriever2 = DocumentRetriever()

        ensemble = EnsembleRetriever(
            retrievers=[retriever1, retriever2],
            weights=[0.6, 0.4],
            fusion_strategy="weighted_score"
        )

        assert len(ensemble.retrievers) == 2
        assert ensemble.weights == [0.6, 0.4]
        assert ensemble.fusion_strategy == "weighted_score"

    def test_ensemble_validation(self):
        """Test ensemble validation"""
        retriever1 = DocumentRetriever()
        retriever2 = DocumentRetriever()

        # No retrievers
        with pytest.raises(ValueError):
            EnsembleRetriever(retrievers=[])

        # Mismatched weights
        with pytest.raises(ValueError):
            EnsembleRetriever(retrievers=[retriever1, retriever2], weights=[0.5])

        # Weights don't sum to 1
        with pytest.raises(ValueError):
            EnsembleRetriever(retrievers=[retriever1, retriever2], weights=[0.6, 0.3])

        # Invalid fusion strategy
        with pytest.raises(ValueError):
            EnsembleRetriever(
                retrievers=[retriever1, retriever2],
                fusion_strategy="invalid"
            )

    def test_ensemble_retrieval(self):
        """Test ensemble retrieval"""
        retriever1 = DocumentRetriever()
        retriever2 = DocumentRetriever()

        docs = [
            Document(content="The quick brown fox"),
            Document(content="Python programming"),
            Document(content="Machine learning")
        ]

        retriever1.add_documents(docs)
        retriever2.add_documents(docs)

        ensemble = EnsembleRetriever(
            retrievers=[retriever1, retriever2],
            weights=[0.5, 0.5]
        )

        result = ensemble.retrieve("python")
        assert len(result.documents) >= 1
        assert result.query == "python"
        assert result.retrieval_method.startswith("ensemble_")

    def test_different_fusion_strategies(self):
        """Test different fusion strategies"""
        retriever1 = DocumentRetriever()
        retriever2 = DocumentRetriever()

        docs = [Document(content=f"Test document {i}") for i in range(3)]
        retriever1.add_documents(docs)
        retriever2.add_documents(docs)

        strategies = ["weighted_score", "rank_fusion", "reciprocal_rank", "weighted_vote"]

        for strategy in strategies:
            ensemble = EnsembleRetriever(
                retrievers=[retriever1, retriever2],
                fusion_strategy=strategy
            )

            result = ensemble.retrieve("test")
            assert len(result.documents) >= 1
            assert strategy in result.retrieval_method

    def test_ensemble_stats(self):
        """Test ensemble statistics"""
        retriever1 = DocumentRetriever()
        retriever2 = DocumentRetriever()

        ensemble = EnsembleRetriever(
            retrievers=[retriever1, retriever2],
            weights=[0.7, 0.3]
        )

        stats = ensemble.get_ensemble_stats()
        assert "fusion_strategy" in stats
        assert "num_retrievers" in stats
        assert "weights" in stats
        assert "retriever_stats" in stats

        assert stats["num_retrievers"] == 2
        assert stats["weights"] == [0.7, 0.3]
        assert len(stats["retriever_stats"]) == 2

    def test_compare_retrievers(self):
        """Test retriever comparison"""
        retriever1 = DocumentRetriever()
        retriever2 = DocumentRetriever()

        docs = [
            Document(content="Python programming tutorial"),
            Document(content="Machine learning basics")
        ]

        retriever1.add_documents(docs)
        retriever2.add_documents(docs)

        ensemble = EnsembleRetriever(retrievers=[retriever1, retriever2])
        comparison = ensemble.compare_retrievers("python")

        assert len(comparison) == 2
        assert "DocumentRetriever" in comparison
        for result in comparison.values():
            assert isinstance(result, RetrievalResult)

    def test_set_fusion_strategy(self):
        """Test changing fusion strategy"""
        retriever1 = DocumentRetriever()
        retriever2 = DocumentRetriever()

        ensemble = EnsembleRetriever(
            retrievers=[retriever1, retriever2],
            fusion_strategy="weighted_score"
        )

        assert ensemble.fusion_strategy == "weighted_score"

        ensemble.set_fusion_strategy("rank_fusion")
        assert ensemble.fusion_strategy == "rank_fusion"

        # Test invalid strategy
        with pytest.raises(ValueError):
            ensemble.set_fusion_strategy("invalid")

    def test_set_weights(self):
        """Test changing weights"""
        retriever1 = DocumentRetriever()
        retriever2 = DocumentRetriever()

        ensemble = EnsembleRetriever(
            retrievers=[retriever1, retriever2],
            weights=[0.5, 0.5]
        )

        assert ensemble.weights == [0.5, 0.5]

        ensemble.set_weights([0.7, 0.3])
        assert ensemble.weights == [0.7, 0.3]

        # Test invalid weights
        with pytest.raises(ValueError):
            ensemble.set_weights([0.7])

        with pytest.raises(ValueError):
            ensemble.set_weights([0.6, 0.3])


class TestUtilityFunctions:
    """Test utility functions"""

    def test_create_document(self):
        """Test document creation utility"""
        doc = create_document("Hello world", {"source": "test"}, "doc_1")
        assert doc.content == "Hello world"
        assert doc.metadata["source"] == "test"
        assert doc.id == "doc_1"

    def test_create_retrieved_document(self):
        """Test retrieved document creation utility"""
        doc = create_retrieved_document(
            content="Hello world",
            relevance_score=0.85,
            query="test",
            rank=1,
            metadata={"source": "test"},
            retrieval_method="similarity"
        )
        assert doc.content == "Hello world"
        assert doc.relevance_score == 0.85
        assert doc.query == "test"
        assert doc.rank == 1

    def test_create_retrieval_query(self):
        """Test retrieval query creation utility"""
        query = create_retrieval_query(
            query="test query",
            filters={"type": "pdf"},
            max_results=10
        )
        assert query.query == "test query"
        assert query.filters["type"] == "pdf"
        assert query.max_results == 10

    def test_calculate_retrieval_metrics(self):
        """Test retrieval metrics calculation"""
        docs = [
            RetrievedDocument(
                content="Doc 1",
                relevance_score=0.9,
                retrieval_method="test",
                query="test",
                rank=0,
                id="doc_1"
            ),
            RetrievedDocument(
                content="Doc 2",
                relevance_score=0.8,
                retrieval_method="test",
                query="test",
                rank=1,
                id="doc_2"
            ),
            RetrievedDocument(
                content="Doc 3",
                relevance_score=0.7,
                retrieval_method="test",
                query="test",
                rank=2,
                id="doc_3"
            )
        ]

        relevant_docs = ["doc_1", "doc_3"]
        metrics = calculate_retrieval_metrics(docs, relevant_docs, k=3)

        assert hasattr(metrics, 'precision')
        assert hasattr(metrics, 'recall')
        assert hasattr(metrics, 'f1_score')
        assert hasattr(metrics, 'hit_rate')
        assert hasattr(metrics, 'mean_reciprocal_rank')
        assert hasattr(metrics, 'mean_average_precision')

    def test_merge_retrieval_results(self):
        """Test merging retrieval results"""
        docs1 = [
            RetrievedDocument(
                content="Doc 1",
                relevance_score=0.9,
                retrieval_method="method1",
                query="test",
                rank=0,
                id="doc_1"
            )
        ]

        docs2 = [
            RetrievedDocument(
                content="Doc 2",
                relevance_score=0.8,
                retrieval_method="method2",
                query="test",
                rank=0,
                id="doc_2"
            ),
            RetrievedDocument(
                content="Doc 3",
                relevance_score=0.7,
                retrieval_method="method2",
                query="test",
                rank=1,
                id="doc_3"
            )
        ]

        result1 = RetrievalResult(
            documents=docs1,
            query="test",
            total_results=1,
            search_time=0.1,
            retrieval_method="method1"
        )

        result2 = RetrievalResult(
            documents=docs2,
            query="test",
            total_results=2,
            search_time=0.2,
            retrieval_method="method2"
        )

        merged = merge_retrieval_results([result1, result2])
        assert len(merged.documents) == 3
        assert merged.total_results == 3
        assert merged.search_time == pytest.approx(0.3)
        assert "merged" in merged.retrieval_method

        # Test empty results
        with pytest.raises(RetrievalValidationError):
            merge_retrieval_results([])


class TestErrorHandling:
    """Test error handling in retrieval"""

    def test_retrieval_errors(self):
        """Test retrieval error types"""

        # Test base error
        error = RetrievalError("Test error", "TestRetriever", {"context": "test"})
        assert str(error) == "Test error"
        assert error.retriever_type == "TestRetriever"
        assert error.context == {"context": "test"}

        # Test validation error
        validation_error = RetrievalValidationError("Validation failed")
        assert isinstance(validation_error, RetrievalError)

        # Test processing error
        processing_error = RetrievalProcessingError("Processing failed", "TestRetriever")
        assert isinstance(processing_error, RetrievalError)

    def test_document_retriever_error_handling(self):
        """Test document retriever error handling"""
        retriever = DocumentRetriever()

        # Empty query
        with pytest.raises(RetrievalValidationError):
            retriever.retrieve("")

        # Invalid query type
        with pytest.raises(RetrievalValidationError):
            retriever.retrieve(123)

    def test_vector_retriever_error_handling(self):
        """Test vector retriever error handling"""
        # Create fresh instances for error testing
        embedding_model = MockEmbedding(embedding_dimension=10)
        vector_config = VectorStoreConfig(dimension=10)
        vector_store = InMemoryVectorStore(config=vector_config)

        retriever = VectorRetriever(
            embedding_model=embedding_model,
            vector_store=vector_store
        )

        # Empty query
        with pytest.raises(RetrievalValidationError):
            retriever.retrieve("")

        # Invalid query type
        with pytest.raises(RetrievalValidationError):
            retriever.retrieve(123)