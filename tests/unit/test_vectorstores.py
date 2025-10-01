# -*- coding: utf-8 -*-
"""
Unit tests for vector store module
"""
import pytest
import numpy as np
from typing import Dict, Any, List

from my_langchain.vectorstores import (
    BaseVectorStore, InMemoryVectorStore, FAISSVectorStore
)
from my_langchain.vectorstores.types import (
    VectorStoreConfig, VectorStoreResult, VectorStoreQuery, Vector, Document,
    VectorStoreError, VectorStoreValidationError, VectorStoreRetrievalError,
    DistanceMetric, IndexStatus
)


class TestVectorStoreConfig:
    """Test VectorStoreConfig class"""

    def test_default_config(self):
        """Test default configuration"""
        config = VectorStoreConfig(dimension=128)
        assert config.dimension == 128
        assert config.metric == "cosine"
        assert config.index_type == "flat"
        assert config.ef_construction is None
        assert config.ef_search is None
        assert config.nlist is None
        assert config.nprobe is None
        assert config.metadata == {}

    def test_custom_config(self):
        """Test custom configuration"""
        config = VectorStoreConfig(
            dimension=256,
            metric="euclidean",
            index_type="hnsw",
            ef_construction=200,
            ef_search=50,
            nlist=100,
            nprobe=10,
            metadata={"version": "1.0", "optimized": True}
        )
        assert config.dimension == 256
        assert config.metric == "euclidean"
        assert config.index_type == "hnsw"
        assert config.ef_construction == 200
        assert config.ef_search == 50
        assert config.nlist == 100
        assert config.nprobe == 10
        assert config.metadata["version"] == "1.0"


class TestVector:
    """Test Vector class"""

    def test_vector_creation(self):
        """Test vector creation"""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        vector = Vector(
            id="test_vector",
            embedding=embedding,
            metadata={"source": "test", "type": "embedding"}
        )
        assert vector.id == "test_vector"
        assert vector.embedding == embedding
        assert vector.metadata["source"] == "test"
        assert vector.metadata["type"] == "embedding"

    def test_vector_without_metadata(self):
        """Test vector without metadata"""
        embedding = [1.0, 2.0, 3.0]
        vector = Vector(id="simple", embedding=embedding)
        assert vector.id == "simple"
        assert vector.embedding == embedding
        assert vector.metadata == {}

    def test_vector_repr(self):
        """Test vector string representation"""
        embedding = [0.1] * 10
        vector = Vector(id="repr_test", embedding=embedding)
        repr_str = repr(vector)
        assert "Vector(id=repr_test" in repr_str
        assert "dimension=10" in repr_str


class TestDocument:
    """Test Document class"""

    def test_document_creation(self):
        """Test document creation"""
        content = "This is a test document content."
        document = Document(
            id="doc1",
            content=content,
            metadata={"title": "Test Doc", "author": "Test Author"},
            embedding=[0.1, 0.2, 0.3]
        )
        assert document.id == "doc1"
        assert document.content == content
        assert document.metadata["title"] == "Test Doc"
        assert document.embedding == [0.1, 0.2, 0.3]

    def test_document_without_embedding(self):
        """Test document without embedding"""
        content = "Document without embedding."
        document = Document(id="doc2", content=content)
        assert document.id == "doc2"
        assert document.content == content
        assert document.embedding is None

    def test_document_repr(self):
        """Test document string representation"""
        content = "Short content"
        document = Document(id="repr_doc", content=content)
        repr_str = repr(document)
        assert "Document(id=repr_doc" in repr_str
        assert "Short content" in repr_str


class TestVectorStoreQuery:
    """Test VectorStoreQuery class"""

    def test_query_creation(self):
        """Test query creation"""
        query_vector = [0.1] * 128
        query = VectorStoreQuery(
            query_vector=query_vector,
            top_k=5,
            include_metadata=True,
            filter_dict={"category": "test"},
            score_threshold=0.8
        )
        assert query.query_vector == query_vector
        assert query.top_k == 5
        assert query.include_metadata is True
        assert query.filter_dict["category"] == "test"
        assert query.score_threshold == 0.8

    def test_query_defaults(self):
        """Test query defaults"""
        query_vector = [0.1] * 64
        query = VectorStoreQuery(query_vector=query_vector)
        assert query.top_k == 10
        assert query.include_metadata is True
        assert query.filter_dict is None
        assert query.score_threshold is None


class TestInMemoryVectorStore:
    """Test InMemoryVectorStore class"""

    def test_store_creation(self):
        """Test store creation"""
        config = VectorStoreConfig(dimension=128, metric="cosine")
        store = InMemoryVectorStore(config=config)
        assert store.config.dimension == 128
        assert store.config.metric == "cosine"
        assert store.index_status == IndexStatus.EMPTY
        assert store.vector_count == 0

    def test_add_vectors(self):
        """Test adding vectors"""
        config = VectorStoreConfig(dimension=4)
        store = InMemoryVectorStore(config=config)

        vectors = [
            Vector(id="v1", embedding=[1.0, 0.0, 0.0, 0.0]),
            Vector(id="v2", embedding=[0.0, 1.0, 0.0, 0.0]),
            Vector(id="v3", embedding=[0.0, 0.0, 1.0, 0.0])
        ]

        added_ids = store.add_vectors(vectors)
        assert len(added_ids) == 3
        assert "v1" in added_ids
        assert "v2" in added_ids
        assert "v3" in added_ids
        assert store.vector_count == 3
        assert store.index_status == IndexStatus.READY

    def test_add_vectors_without_ids(self):
        """Test adding vectors without IDs"""
        config = VectorStoreConfig(dimension=3)
        store = InMemoryVectorStore(config=config)

        vectors = [
            Vector(id="", embedding=[1.0, 0.0, 0.0]),  # Empty ID should be generated
            Vector(id="", embedding=[0.0, 1.0, 0.0])   # Empty ID should be generated
        ]

        added_ids = store.add_vectors(vectors)
        assert len(added_ids) == 2
        assert store.vector_count == 2

        # Check that IDs were generated
        for vector_id in added_ids:
            assert vector_id is not None
            assert len(vector_id) > 0

    def test_add_documents(self):
        """Test adding documents"""
        config = VectorStoreConfig(dimension=4)
        store = InMemoryVectorStore(config=config)

        documents = [
            Document(
                id="doc1",
                content="First document",
                embedding=[1.0, 0.0, 0.0, 0.0],
                metadata={"category": "A"}
            ),
            Document(
                id="doc2",
                content="Second document",
                embedding=[0.0, 1.0, 0.0, 0.0],
                metadata={"category": "B"}
            )
        ]

        added_ids = store.add_documents(documents)
        assert len(added_ids) == 2
        assert store.vector_count == 2

    def test_search_basic(self):
        """Test basic search"""
        config = VectorStoreConfig(dimension=3)
        store = InMemoryVectorStore(config=config)

        # Add test vectors
        vectors = [
            Vector(id="v1", embedding=[1.0, 0.0, 0.0]),
            Vector(id="v2", embedding=[0.0, 1.0, 0.0]),
            Vector(id="v3", embedding=[0.9, 0.1, 0.0])  # Similar to v1
        ]
        store.add_vectors(vectors)

        # Search for similar to v1
        query = VectorStoreQuery(query_vector=[1.0, 0.0, 0.0], top_k=2)
        result = store.search(query)

        assert len(result.vectors) == 2
        assert result.scores[0] > result.scores[1]  # First should be most similar
        assert result.total_count == 3
        assert result.query_time > 0

    def test_search_with_filter(self):
        """Test search with metadata filter"""
        config = VectorStoreConfig(dimension=3)
        store = InMemoryVectorStore(config=config)

        vectors = [
            Vector(id="v1", embedding=[1.0, 0.0, 0.0], metadata={"category": "A"}),
            Vector(id="v2", embedding=[0.0, 1.0, 0.0], metadata={"category": "B"}),
            Vector(id="v3", embedding=[0.9, 0.1, 0.0], metadata={"category": "A"})
        ]
        store.add_vectors(vectors)

        # Search with filter
        query = VectorStoreQuery(
            query_vector=[1.0, 0.0, 0.0],
            top_k=5,
            filter_dict={"category": "A"}
        )
        result = store.search(query)

        # Should only return vectors with category "A"
        assert len(result.vectors) == 2
        for vector in result.vectors:
            assert vector.metadata["category"] == "A"

    def test_search_with_score_threshold(self):
        """Test search with score threshold"""
        config = VectorStoreConfig(dimension=3)
        store = InMemoryVectorStore(config=config)

        vectors = [
            Vector(id="v1", embedding=[1.0, 0.0, 0.0]),
            Vector(id="v2", embedding=[0.0, 1.0, 0.0]),
            Vector(id="v3", embedding=[0.0, 0.0, 1.0])
        ]
        store.add_vectors(vectors)

        # Search with high threshold
        query = VectorStoreQuery(
            query_vector=[1.0, 0.0, 0.0],
            top_k=5,
            score_threshold=0.9
        )
        result = store.search(query)

        # Should only return the exact match
        assert len(result.vectors) == 1
        assert result.vectors[0].id == "v1"

    def test_delete_vectors(self):
        """Test deleting vectors"""
        config = VectorStoreConfig(dimension=3)
        store = InMemoryVectorStore(config=config)

        vectors = [
            Vector(id="v1", embedding=[1.0, 0.0, 0.0]),
            Vector(id="v2", embedding=[0.0, 1.0, 0.0]),
            Vector(id="v3", embedding=[0.0, 0.0, 1.0])
        ]
        store.add_vectors(vectors)

        # Delete one vector
        success = store.delete_vectors(["v2"])
        assert success is True
        assert store.vector_count == 2
        assert store.get_vector("v2") is None

        # Verify remaining vectors
        assert store.get_vector("v1") is not None
        assert store.get_vector("v3") is not None

    def test_get_vector(self):
        """Test getting specific vector"""
        config = VectorStoreConfig(dimension=3)
        store = InMemoryVectorStore(config=config)

        vector = Vector(id="test", embedding=[1.0, 2.0, 3.0])
        store.add_vectors([vector])

        retrieved = store.get_vector("test")
        assert retrieved is not None
        assert retrieved.id == "test"
        assert retrieved.embedding == [1.0, 2.0, 3.0]

        # Test non-existent vector
        non_existent = store.get_vector("non_existent")
        assert non_existent is None

    def test_update_vector(self):
        """Test updating vector"""
        config = VectorStoreConfig(dimension=3)
        store = InMemoryVectorStore(config=config)

        vector = Vector(id="test", embedding=[1.0, 2.0, 3.0])
        store.add_vectors([vector])

        # Update vector
        new_vector = Vector(id="test", embedding=[4.0, 5.0, 6.0])
        success = store.update_vector("test", new_vector)
        assert success is True

        # Verify update
        retrieved = store.get_vector("test")
        assert retrieved.embedding == [4.0, 5.0, 6.0]

    def test_similarity_search_convenience(self):
        """Test similarity search convenience method"""
        config = VectorStoreConfig(dimension=3)
        store = InMemoryVectorStore(config=config)

        vectors = [
            Vector(id="v1", embedding=[1.0, 0.0, 0.0]),
            Vector(id="v2", embedding=[0.0, 1.0, 0.0])
        ]
        store.add_vectors(vectors)

        # Use convenience method
        result = store.similarity_search(
            query_vector=[1.0, 0.0, 0.0],
            top_k=1
        )

        assert len(result.vectors) == 1
        assert result.vectors[0].id == "v1"

    def test_max_marginal_relevance_search(self):
        """Test maximal marginal relevance search"""
        config = VectorStoreConfig(dimension=3)
        store = InMemoryVectorStore(config=config)

        # Add similar vectors - some very similar, some different
        vectors = [
            Vector(id="v1", embedding=[1.0, 0.0, 0.0]),
            Vector(id="v2", embedding=[0.95, 0.05, 0.0]),  # Very similar to v1
            Vector(id="v3", embedding=[0.0, 1.0, 0.0]),  # Different from v1
            Vector(id="v4", embedding=[0.9, 0.1, 0.0])   # Similar to v1
        ]
        store.add_vectors(vectors)

        # Use MMR search with query that has some similarity to all vectors
        result = store.max_marginal_relevance_search(
            query_vector=[0.7, 0.7, 0.0],  # Query that has some similarity to all vectors
            top_k=3,
            lambda_mult=0.5
        )

        assert len(result.vectors) == 3
        # Should return diverse results
        result_ids = [v.id for v in result.vectors]

        # Verify that we got diverse results (not just the 3 most similar)
        # Check that at least one different vector is included
        similar_vectors = {"v1", "v2", "v4"}  # These are all similar to each other
        has_diverse_result = any(v_id not in similar_vectors for v_id in result_ids) or len(result_ids) < 3
        assert has_diverse_result, f"Expected diverse results, got: {result_ids}"

    def test_clear_store(self):
        """Test clearing the store"""
        config = VectorStoreConfig(dimension=3)
        store = InMemoryVectorStore(config=config)

        vectors = [
            Vector(id="v1", embedding=[1.0, 0.0, 0.0]),
            Vector(id="v2", embedding=[0.0, 1.0, 0.0])
        ]
        store.add_vectors(vectors)

        assert store.vector_count == 2

        store.clear()
        assert store.vector_count == 0
        assert store.index_status == IndexStatus.EMPTY

    def test_get_all_vectors(self):
        """Test getting all vectors"""
        config = VectorStoreConfig(dimension=3)
        store = InMemoryVectorStore(config=config)

        vectors = [
            Vector(id="v1", embedding=[1.0, 0.0, 0.0]),
            Vector(id="v2", embedding=[0.0, 1.0, 0.0])
        ]
        store.add_vectors(vectors)

        all_vectors = store.get_all_vectors()
        assert len(all_vectors) == 2
        vector_ids = {v.id for v in all_vectors}
        assert vector_ids == {"v1", "v2"}

    def test_get_stats(self):
        """Test getting store statistics"""
        config = VectorStoreConfig(dimension=128)
        store = InMemoryVectorStore(config=config)

        vectors = [
            Vector(id="v1", embedding=[0.1] * 128),
            Vector(id="v2", embedding=[0.2] * 128)
        ]
        store.add_vectors(vectors)

        stats = store.get_stats()
        assert stats["vector_count"] == 2
        assert stats["dimension"] == 128
        assert stats["memory_usage_mb"] > 0

    def test_different_distance_metrics(self):
        """Test different distance metrics"""
        vectors = [
            Vector(id="v1", embedding=[1.0, 0.0]),
            Vector(id="v2", embedding=[0.0, 1.0])
        ]

        # Test cosine similarity
        config_cosine = VectorStoreConfig(dimension=2, metric="cosine")
        store_cosine = InMemoryVectorStore(config=config_cosine)
        store_cosine.add_vectors(vectors)

        result_cosine = store_cosine.similarity_search([1.0, 0.0], top_k=1)
        assert result_cosine.vectors[0].id == "v1"

        # Test Euclidean distance
        config_euclidean = VectorStoreConfig(dimension=2, metric="euclidean")
        store_euclidean = InMemoryVectorStore(config=config_euclidean)
        store_euclidean.add_vectors(vectors)

        result_euclidean = store_euclidean.similarity_search([1.0, 0.0], top_k=1)
        assert result_euclidean.vectors[0].id == "v1"


class TestVectorStoreErrorHandling:
    """Test vector store error handling"""

    def test_vector_validation_error(self):
        """Test VectorValidationError"""
        config = VectorStoreConfig(dimension=3)
        store = InMemoryVectorStore(config=config)

        # Test wrong dimension
        wrong_vector = Vector(id="wrong", embedding=[1.0, 2.0])  # 2D instead of 3D
        with pytest.raises(VectorStoreValidationError):
            store.add_vectors([wrong_vector])

        # Test empty embedding
        empty_vector = Vector(id="empty", embedding=[])
        with pytest.raises(VectorStoreValidationError):
            store.add_vectors([empty_vector])

    def test_query_validation_error(self):
        """Test query validation"""
        config = VectorStoreConfig(dimension=3)
        store = InMemoryVectorStore(config=config)

        # Test wrong query dimension
        query = VectorStoreQuery(query_vector=[1.0, 2.0])  # 2D instead of 3D
        with pytest.raises(VectorStoreValidationError):
            store.search(query)

        # Test empty query
        empty_query = VectorStoreQuery(query_vector=[])
        with pytest.raises(VectorStoreValidationError):
            store.search(empty_query)

        # Test invalid top_k
        invalid_query = VectorStoreQuery(query_vector=[1.0, 2.0, 3.0], top_k=0)
        with pytest.raises(VectorStoreValidationError):
            store.search(invalid_query)


class TestVectorStoreIntegration:
    """Test vector store integration scenarios"""

    def test_large_scale_operations(self):
        """Test operations with many vectors"""
        config = VectorStoreConfig(dimension=64)
        store = InMemoryVectorStore(config=config)

        # Add many vectors
        vectors = []
        for i in range(100):
            embedding = np.random.random(64).tolist()
            vector = Vector(
                id=f"vector_{i}",
                embedding=embedding,
                metadata={"batch": i // 10}
            )
            vectors.append(vector)

        added_ids = store.add_vectors(vectors)
        assert len(added_ids) == 100
        assert store.vector_count == 100

        # Search with metadata filter
        query = VectorStoreQuery(
            query_vector=np.random.random(64).tolist(),
            top_k=10,
            filter_dict={"batch": 5}
        )
        result = store.search(query)

        # Should find vectors from batch 5 (indices 50-59)
        assert len(result.vectors) == 10
        for vector in result.vectors:
            assert vector.metadata["batch"] == 5

    def test_concurrent_operations(self):
        """Test concurrent read/write operations"""
        config = VectorStoreConfig(dimension=32)
        store = InMemoryVectorStore(config=config)

        # Add initial vectors
        vectors = []
        for i in range(50):
            embedding = np.random.random(32).tolist()
            vectors.append(Vector(id=f"initial_{i}", embedding=embedding))
        store.add_vectors(vectors)

        # Perform multiple searches
        for i in range(10):
            query_vector = np.random.random(32).tolist()
            result = store.similarity_search(query_vector, top_k=5)
            assert len(result.vectors) == 5

        # Add more vectors
        more_vectors = []
        for i in range(50, 100):
            embedding = np.random.random(32).tolist()
            more_vectors.append(Vector(id=f"later_{i}", embedding=embedding))
        store.add_vectors(more_vectors)

        # Verify all vectors are present
        assert store.vector_count == 100

    def test_memory_efficiency(self):
        """Test memory efficiency"""
        config = VectorStoreConfig(dimension=256)
        store = InMemoryVectorStore(config=config)

        # Add vectors and monitor memory usage
        vectors = []
        for i in range(200):
            embedding = np.random.random(256).tolist()
            vectors.append(Vector(id=f"memory_test_{i}", embedding=embedding))

        store.add_vectors(vectors)
        stats = store.get_stats()

        # Check that memory usage is reasonable
        memory_mb = stats["memory_usage_mb"]
        assert 0 < memory_mb < 1000  # Should be less than 1GB for 200 256D vectors

    def test_store_persistence_simulation(self):
        """Test store persistence (simulation)"""
        config = VectorStoreConfig(dimension=16)
        store = InMemoryVectorStore(config=config)

        # Add data
        documents = [
            Document(
                id="doc1",
                content="First document about AI",
                embedding=np.random.random(16).tolist(),
                metadata={"topic": "AI"}
            ),
            Document(
                id="doc2",
                content="Second document about ML",
                embedding=np.random.random(16).tolist(),
                metadata={"topic": "ML"}
            )
        ]
        store.add_documents(documents)

        # Simulate getting store state
        store_info = store.get_store_info()
        assert store_info["vector_count"] == 2

        # Simulate recreating store from config
        new_store = InMemoryVectorStore(config=config)
        assert new_store.vector_count == 0

        # Re-add data
        new_store.add_documents(documents)
        assert new_store.vector_count == 2