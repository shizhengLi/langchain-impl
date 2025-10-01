# -*- coding: utf-8 -*-
"""
Tests for embeddings module
"""

import pytest
import time
from my_langchain.embeddings.types import (
    EmbeddingConfig, EmbeddingResult, EmbeddingError, EmbeddingValidationError,
    EmbeddingProcessingError, Embedding, EmbeddingUsage,
    estimate_token_count, validate_embedding_vector, normalize_vector,
    cosine_similarity, create_embedding, merge_embedding_results
)
from my_langchain.embeddings.base import BaseEmbedding
from my_langchain.embeddings.mock_embedding import MockEmbedding


class TestEmbeddingConfig:
    """Test embedding configuration"""

    def test_default_config(self):
        """Test default configuration"""
        config = EmbeddingConfig()
        assert config.model_name == "text-embedding-ada-002"
        assert config.embedding_dimension == 1536
        assert config.batch_size == 100
        assert config.max_tokens == 8192
        assert config.normalize_embeddings is True
        assert config.show_progress is False
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

    def test_custom_config(self):
        """Test custom configuration"""
        config = EmbeddingConfig(
            model_name="mock-model",
            embedding_dimension=512,
            batch_size=50,
            normalize_embeddings=False
        )
        assert config.model_name == "mock-model"
        assert config.embedding_dimension == 512
        assert config.batch_size == 50
        assert config.normalize_embeddings is False

    def test_config_validation(self):
        """Test configuration validation"""
        # Valid configurations
        EmbeddingConfig(embedding_dimension=100)
        EmbeddingConfig(batch_size=10)
        EmbeddingConfig(max_tokens=1000)
        EmbeddingConfig(timeout=60.0)
        EmbeddingConfig(max_retries=5)

        # Invalid configurations
        with pytest.raises(ValueError):
            EmbeddingConfig(embedding_dimension=0)

        with pytest.raises(ValueError):
            EmbeddingConfig(embedding_dimension=-1)

        with pytest.raises(ValueError):
            EmbeddingConfig(batch_size=0)

        with pytest.raises(ValueError):
            EmbeddingConfig(max_tokens=0)

        with pytest.raises(ValueError):
            EmbeddingConfig(timeout=0)

        with pytest.raises(ValueError):
            EmbeddingConfig(timeout=-1)

        with pytest.raises(ValueError):
            EmbeddingConfig(max_retries=-1)


class TestEmbedding:
    """Test Embedding class"""

    def test_embedding_creation(self):
        """Test embedding creation"""
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        embedding = Embedding(
            vector=vector,
            text="Hello world",
            model_name="test-model",
            embedding_dimension=5,
            token_count=2
        )

        assert embedding.vector == vector
        assert embedding.text == "Hello world"
        assert embedding.model_name == "test-model"
        assert embedding.embedding_dimension == 5
        assert embedding.token_count == 2

    def test_embedding_validation(self):
        """Test embedding validation"""
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Valid embedding
        Embedding(
            vector=vector,
            text="Hello",
            model_name="test",
            embedding_dimension=5
        )

        # Invalid vector (empty)
        with pytest.raises(ValueError):
            Embedding(
                vector=[],
                text="Hello",
                model_name="test",
                embedding_dimension=0
            )

        # Dimension mismatch
        with pytest.raises(ValueError):
            Embedding(
                vector=vector,
                text="Hello",
                model_name="test",
                embedding_dimension=10  # Different from vector length
            )

    def test_embedding_normalize(self):
        """Test embedding normalization"""
        vector = [3.0, 4.0]  # Should normalize to [0.6, 0.8]
        embedding = Embedding(
            vector=vector,
            text="Test",
            model_name="test",
            embedding_dimension=2
        )

        normalized = embedding.normalize()
        expected_length = (0.6**2 + 0.8**2) ** 0.5
        actual_length = sum(x**2 for x in normalized.vector) ** 0.5

        assert abs(actual_length - expected_length) < 1e-10
        assert len(normalized.vector) == len(embedding.vector)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        # Identical vectors should have similarity 1
        vector1 = [1.0, 0.0, 0.0]
        vector2 = [1.0, 0.0, 0.0]

        embedding1 = Embedding(vector=vector1, text="test1", model_name="test", embedding_dimension=3)
        embedding2 = Embedding(vector=vector2, text="test2", model_name="test", embedding_dimension=3)

        similarity = embedding1.cosine_similarity(embedding2)
        assert abs(similarity - 1.0) < 1e-10

        # Orthogonal vectors should have similarity 0
        vector3 = [0.0, 1.0, 0.0]
        embedding3 = Embedding(vector=vector3, text="test3", model_name="test", embedding_dimension=3)

        similarity = embedding1.cosine_similarity(embedding3)
        assert abs(similarity - 0.0) < 1e-10

        # Different dimensions should raise error
        embedding4 = Embedding(vector=[1.0, 0.0], text="test4", model_name="test", embedding_dimension=2)
        with pytest.raises(ValueError):
            embedding1.cosine_similarity(embedding4)

    def test_euclidean_distance(self):
        """Test Euclidean distance calculation"""
        vector1 = [0.0, 0.0]
        vector2 = [3.0, 4.0]

        embedding1 = Embedding(vector=vector1, text="test1", model_name="test", embedding_dimension=2)
        embedding2 = Embedding(vector=vector2, text="test2", model_name="test", embedding_dimension=2)

        distance = embedding1.euclidean_distance(embedding2)
        assert abs(distance - 5.0) < 1e-10  # 3-4-5 triangle

    def test_manhattan_distance(self):
        """Test Manhattan distance calculation"""
        vector1 = [0.0, 0.0]
        vector2 = [3.0, 4.0]

        embedding1 = Embedding(vector=vector1, text="test1", model_name="test", embedding_dimension=2)
        embedding2 = Embedding(vector=vector2, text="test2", model_name="test", embedding_dimension=2)

        distance = embedding1.manhattan_distance(embedding2)
        assert abs(distance - 7.0) < 1e-10  # |3| + |4| = 7


class TestEmbeddingResult:
    """Test EmbeddingResult class"""

    def test_embedding_result_creation(self):
        """Test embedding result creation"""
        embeddings = [
            Embedding(vector=[0.1, 0.2], text="Hello", model_name="test", embedding_dimension=2),
            Embedding(vector=[0.3, 0.4], text="World", model_name="test", embedding_dimension=2)
        ]

        result = EmbeddingResult(
            embeddings=embeddings,
            model_name="test-model",
            total_tokens=4,
            total_time=0.5,
            batch_count=1
        )

        assert len(result.embeddings) == 2
        assert result.model_name == "test-model"
        assert result.total_tokens == 4
        assert result.total_time == 0.5
        assert result.batch_count == 1

    def test_embedding_result_validation(self):
        """Test embedding result validation"""
        # Empty embeddings should raise error
        with pytest.raises(ValueError):
            EmbeddingResult(
                embeddings=[],
                model_name="test",
                batch_count=0
            )

        # Negative batch count should raise error
        with pytest.raises(ValueError):
            EmbeddingResult(
                embeddings=[Embedding(vector=[0.1], text="test", model_name="test", embedding_dimension=1)],
                model_name="test",
                batch_count=-1
            )

    def test_embedding_result_operations(self):
        """Test embedding result operations"""
        embeddings = [
            Embedding(vector=[0.1, 0.2], text="Hello", model_name="test", embedding_dimension=2),
            Embedding(vector=[0.3, 0.4], text="World", model_name="test", embedding_dimension=2),
            Embedding(vector=[0.5, 0.6], text="Test", model_name="test", embedding_dimension=2, token_count=1)
        ]

        result = EmbeddingResult(
            embeddings=embeddings,
            model_name="test-model",
            total_tokens=5,
            batch_count=1
        )

        # Test length
        assert len(result) == 3

        # Test indexing
        assert result[0].text == "Hello"
        assert result[1].text == "World"

        # Test iteration
        texts = [emb.text for emb in result]
        assert texts == ["Hello", "World", "Test"]

        # Test get average embedding
        avg_embedding = result.get_average_embedding()
        expected_avg = [(0.1 + 0.3 + 0.5) / 3, (0.2 + 0.4 + 0.6) / 3]
        assert all(abs(a - b) < 1e-10 for a, b in zip(avg_embedding, expected_avg))

        # Test get embedding by text
        found_embedding = result.get_embedding_by_text("World")
        assert found_embedding is not None
        assert found_embedding.text == "World"

        not_found = result.get_embedding_by_text("Not found")
        assert not_found is None

    def test_filter_by_token_count(self):
        """Test filtering by token count"""
        embeddings = [
            Embedding(vector=[0.1], text="Short", model_name="test", embedding_dimension=1, token_count=1),
            Embedding(vector=[0.2], text="Medium", model_name="test", embedding_dimension=1, token_count=5),
            Embedding(vector=[0.3], text="Long", model_name="test", embedding_dimension=1, token_count=10)
        ]

        result = EmbeddingResult(
            embeddings=embeddings,
            model_name="test",
            batch_count=1
        )

        # Filter by minimum tokens
        filtered = result.filter_by_token_count(min_tokens=3)
        assert len(filtered.embeddings) == 2
        assert all(emb.token_count >= 3 for emb in filtered.embeddings)

        # Filter by maximum tokens
        filtered = result.filter_by_token_count(max_tokens=6)
        assert len(filtered.embeddings) == 2
        assert all(emb.token_count <= 6 for emb in filtered.embeddings)

        # Filter by range
        filtered = result.filter_by_token_count(min_tokens=3, max_tokens=8)
        assert len(filtered.embeddings) == 1
        assert filtered.embeddings[0].text == "Medium"


class TestMockEmbedding:
    """Test MockEmbedding class"""

    def test_mock_embedding_creation(self):
        """Test mock embedding creation"""
        embedding = MockEmbedding()
        assert embedding.config.model_name == "text-embedding-ada-002"
        assert embedding.config.embedding_dimension == 384  # Mock default
        assert embedding.seed == 42

    def test_mock_embedding_custom_config(self):
        """Test mock embedding with custom config"""
        config = EmbeddingConfig(
            model_name="mock-model",
            embedding_dimension=128,
            batch_size=10
        )
        embedding = MockEmbedding(config=config, seed=123)
        assert embedding.config.model_name == "mock-model"
        assert embedding.config.embedding_dimension == 128
        assert embedding.seed == 123

    def test_embed_single_text(self):
        """Test embedding single text"""
        embedding = MockEmbedding(embedding_dimension=10)
        result = embedding.embed_text("Hello world")

        assert isinstance(result, Embedding)
        assert result.text == "Hello world"
        assert len(result.vector) == 10
        assert result.model_name == embedding.config.model_name
        assert result.processing_time is not None
        assert result.token_count is not None

    def test_embed_multiple_texts(self):
        """Test embedding multiple texts"""
        embedding = MockEmbedding(embedding_dimension=10)
        texts = ["Hello", "World", "Test"]
        result = embedding.embed_texts(texts)

        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 3
        assert result.model_name == embedding.config.model_name
        assert result.total_time is not None
        assert result.batch_count >= 1

        # Check each embedding
        for i, emb in enumerate(result.embeddings):
            assert emb.text == texts[i]
            assert len(emb.vector) == 10

    def test_deterministic_embeddings(self):
        """Test that embeddings are deterministic"""
        embedding1 = MockEmbedding(embedding_dimension=10, seed=42)
        embedding2 = MockEmbedding(embedding_dimension=10, seed=42)

        text = "Hello world"
        result1 = embedding1.embed_text(text)
        result2 = embedding2.embed_text(text)

        assert result1.vector == result2.vector

    def test_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings"""
        embedding = MockEmbedding(embedding_dimension=10)

        result1 = embedding.embed_text("Hello")
        result2 = embedding.embed_text("World")

        assert result1.vector != result2.vector

    def test_embed_query(self):
        """Test embed_query convenience method"""
        embedding = MockEmbedding(embedding_dimension=5)
        vector = embedding.embed_query("Query text")

        assert isinstance(vector, list)
        assert len(vector) == 5

    def test_embed_documents(self):
        """Test embed_documents convenience method"""
        embedding = MockEmbedding(embedding_dimension=5)
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        vectors = embedding.embed_documents(documents)

        assert isinstance(vectors, list)
        assert len(vectors) == 3
        assert all(isinstance(vec, list) for vec in vectors)
        assert all(len(vec) == 5 for vec in vectors)

    def test_batch_processing(self):
        """Test batch processing"""
        config = EmbeddingConfig(batch_size=2)
        embedding = MockEmbedding(config=config, embedding_dimension=5)

        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        result = embedding.embed_texts(texts)

        assert len(result.embeddings) == 5
        assert result.batch_count == 3  # 2+2+1

    def test_calculate_similarity(self):
        """Test similarity calculation"""
        embedding = MockEmbedding(embedding_dimension=10)

        similarity = embedding.calculate_similarity("Hello", "Hello")
        assert abs(similarity - 1.0) < 1e-10

        similarity = embedding.calculate_similarity("Hello", "World")
        assert -1.0 <= similarity <= 1.0

    def test_find_most_similar(self):
        """Test finding most similar texts"""
        embedding = MockEmbedding(embedding_dimension=10)

        candidates = ["Apple", "Orange", "Banana", "Grape"]
        results = embedding.find_most_similar("Apple", candidates, top_k=3)

        assert len(results) <= 3
        assert all("text" in result and "similarity" in result for result in results)
        assert results[0]["text"] == "Apple"  # Most similar should be the text itself

    def test_mock_responses(self):
        """Test predefined mock responses"""
        embedding = MockEmbedding(embedding_dimension=3)

        # Set predefined responses
        custom_vectors = {
            "Custom text 1": [1.0, 0.0, 0.0],
            "Custom text 2": [0.0, 1.0, 0.0]
        }
        embedding.set_mock_responses(custom_vectors)

        # Test custom response
        result1 = embedding.embed_text("Custom text 1")
        assert result1.vector == [1.0, 0.0, 0.0]

        result2 = embedding.embed_text("Custom text 2")
        assert result2.vector == [0.0, 1.0, 0.0]

        # Test normal response for undefined text
        result3 = embedding.embed_text("Normal text")
        assert len(result3.vector) == 3
        assert result3.vector != [1.0, 0.0, 0.0]

    def test_similarity_matrix(self):
        """Test similarity matrix creation"""
        embedding = MockEmbedding(embedding_dimension=5)

        texts = ["A", "B", "C"]
        matrix = embedding.create_similarity_matrix(texts)

        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)

        # Diagonal should be 1.0 (self-similarity)
        for i in range(3):
            assert abs(matrix[i][i] - 1.0) < 1e-10

    def test_embedding_quality_test(self):
        """Test embedding quality testing"""
        embedding = MockEmbedding(embedding_dimension=10)

        test_texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        quality_metrics = embedding.test_embedding_quality(test_texts)

        assert "avg_cross_similarity" in quality_metrics
        assert "similarity_variance" in quality_metrics
        assert "avg_self_similarity" in quality_metrics
        assert "embedding_count" in quality_metrics
        assert "dimension" in quality_metrics

        assert quality_metrics["avg_self_similarity"] == 1.0
        assert quality_metrics["embedding_count"] == 4
        assert quality_metrics["dimension"] == 10

    def test_class_methods(self):
        """Test class factory methods"""
        # Test small model
        small = MockEmbedding.create_small_model()
        assert small.config.embedding_dimension == 128

        # Test medium model
        medium = MockEmbedding.create_medium_model()
        assert medium.config.embedding_dimension == 384

        # Test large model
        large = MockEmbedding.create_large_model()
        assert large.config.embedding_dimension == 1536

        # Test custom size
        custom = MockEmbedding.create_medium_model(embedding_dimension=256)
        assert custom.config.embedding_dimension == 256


class TestBaseEmbedding:
    """Test BaseEmbedding abstract class functionality"""

    def test_embedding_validation(self):
        """Test input validation"""

        class TestEmbedding(BaseEmbedding):
            def _embed_single_text(self, text: str):
                return [0.1] * self.config.embedding_dimension

            def _embed_batch(self, texts):
                return [[0.1] * self.config.embedding_dimension] * len(texts)

        embedding = TestEmbedding(embedding_dimension=5)

        # Valid text
        result = embedding.embed_text("Hello")
        assert isinstance(result, Embedding)

        # Invalid text types
        with pytest.raises(EmbeddingValidationError):
            embedding.embed_text(123)

        with pytest.raises(EmbeddingValidationError):
            embedding.embed_text("")

        with pytest.raises(EmbeddingValidationError):
            embedding.embed_text("   ")

        # Invalid text lists
        with pytest.raises(EmbeddingValidationError):
            embedding.embed_texts([])

        with pytest.raises(EmbeddingValidationError):
            embedding.embed_texts([123])

    def test_model_info(self):
        """Test model info method"""

        class TestEmbedding(BaseEmbedding):
            def _embed_single_text(self, text: str):
                return [0.1] * self.config.embedding_dimension

            def _embed_batch(self, texts):
                return [[0.1] * self.config.embedding_dimension] * len(texts)

        config = EmbeddingConfig(
            model_name="test-model",
            embedding_dimension=256,
            batch_size=50
        )
        embedding = TestEmbedding(config=config)

        info = embedding.get_model_info()
        assert info["model_name"] == "test-model"
        assert info["embedding_dimension"] == 256
        assert info["batch_size"] == 50

    def test_dimension_methods(self):
        """Test dimension-related methods"""

        class TestEmbedding(BaseEmbedding):
            def _embed_single_text(self, text: str):
                return [0.1] * self.config.embedding_dimension

            def _embed_batch(self, texts):
                return [[0.1] * self.config.embedding_dimension] * len(texts)

        embedding = TestEmbedding(embedding_dimension=128)
        assert embedding.get_embedding_dimension() == 128


class TestUtilityFunctions:
    """Test utility functions"""

    def test_estimate_token_count(self):
        """Test token count estimation"""
        # Simple tests
        assert estimate_token_count("") == 0
        assert estimate_token_count("Hello") >= 1
        assert estimate_token_count("Hello world") >= 1

        # Long text
        long_text = "word " * 100
        tokens = estimate_token_count(long_text)
        assert tokens >= 25  # Should be around 100 tokens, but we use rough estimation

    def test_validate_embedding_vector(self):
        """Test embedding vector validation"""
        # Valid vectors
        validate_embedding_vector([0.1, 0.2, 0.3])
        validate_embedding_vector([1.0, -1.0, 0.0])
        validate_embedding_vector([0], expected_dimension=1)

        # Invalid vectors
        with pytest.raises(EmbeddingValidationError):
            validate_embedding_vector([])

        with pytest.raises(EmbeddingValidationError):
            validate_embedding_vector([0.1, "invalid"])

        with pytest.raises(EmbeddingValidationError):
            validate_embedding_vector([float('nan')])

        with pytest.raises(EmbeddingValidationError):
            validate_embedding_vector([float('inf')])

        with pytest.raises(EmbeddingValidationError):
            validate_embedding_vector([0.1, 0.2], expected_dimension=3)

    def test_normalize_vector(self):
        """Test vector normalization"""
        # Normal vector
        vector = [3.0, 4.0]
        normalized = normalize_vector(vector)

        expected_length = 1.0
        actual_length = sum(x**2 for x in normalized) ** 0.5
        assert abs(actual_length - expected_length) < 1e-10

        # Zero vector
        zero_vector = [0.0, 0.0]
        normalized_zero = normalize_vector(zero_vector)
        assert normalized_zero == zero_vector

    def test_cosine_similarity_function(self):
        """Test cosine similarity utility function"""
        # Identical vectors
        vector1 = [1.0, 0.0, 0.0]
        vector2 = [1.0, 0.0, 0.0]
        similarity = cosine_similarity(vector1, vector2)
        assert abs(similarity - 1.0) < 1e-10

        # Orthogonal vectors
        vector3 = [0.0, 1.0, 0.0]
        similarity = cosine_similarity(vector1, vector3)
        assert abs(similarity - 0.0) < 1e-10

        # Different dimensions
        with pytest.raises(ValueError):
            cosine_similarity([1.0], [1.0, 0.0])

    def test_create_embedding(self):
        """Test create_embedding utility function"""
        vector = [0.1, 0.2, 0.3]
        embedding = create_embedding(
            vector=vector,
            text="Test",
            model_name="test-model",
            processing_time=0.5,
            token_count=2
        )

        assert isinstance(embedding, Embedding)
        assert embedding.vector == vector
        assert embedding.text == "Test"
        assert embedding.model_name == "test-model"
        assert embedding.processing_time == 0.5
        assert embedding.token_count == 2

    def test_merge_embedding_results(self):
        """Test merging embedding results"""
        # Create two results
        embeddings1 = [
            Embedding(vector=[0.1], text="Text 1", model_name="test", embedding_dimension=1)
        ]
        result1 = EmbeddingResult(
            embeddings=embeddings1,
            model_name="test",
            total_tokens=5,
            total_time=0.5,
            batch_count=1,
            metadata={"source": "test1"}
        )

        embeddings2 = [
            Embedding(vector=[0.2], text="Text 2", model_name="test", embedding_dimension=1)
        ]
        result2 = EmbeddingResult(
            embeddings=embeddings2,
            model_name="test",
            total_tokens=3,
            total_time=0.3,
            batch_count=1,
            metadata={"source": "test2"}
        )

        # Merge results
        merged = merge_embedding_results([result1, result2])

        assert len(merged.embeddings) == 2
        assert merged.total_tokens == 8
        assert merged.total_time == 0.8
        assert merged.batch_count == 2
        assert merged.metadata["source"] == "test2"  # Last one wins

        # Test empty list
        with pytest.raises(EmbeddingValidationError):
            merge_embedding_results([])


class TestEmbeddingUsage:
    """Test EmbeddingUsage class"""

    def test_usage_creation(self):
        """Test usage creation"""
        usage = EmbeddingUsage(prompt_tokens=10, total_tokens=15)
        assert usage.prompt_tokens == 10
        assert usage.total_tokens == 15

    def test_usage_addition(self):
        """Test usage addition"""
        usage1 = EmbeddingUsage(prompt_tokens=10, total_tokens=15)
        usage2 = EmbeddingUsage(prompt_tokens=5, total_tokens=8)

        combined = usage1 + usage2
        assert combined.prompt_tokens == 15
        assert combined.total_tokens == 23


class TestErrorHandling:
    """Test error handling in embeddings"""

    def test_embedding_errors(self):
        """Test embedding error types"""

        # Test base error
        error = EmbeddingError("Test error", "TestEmbedding", {"context": "test"})
        assert str(error) == "Test error"
        assert error.embedding_type == "TestEmbedding"
        assert error.context == {"context": "test"}

        # Test validation error
        validation_error = EmbeddingValidationError("Validation failed")
        assert isinstance(validation_error, EmbeddingError)

        # Test processing error
        processing_error = EmbeddingProcessingError("Processing failed", "TestEmbedding")
        assert isinstance(processing_error, EmbeddingError)

    def test_mock_embedding_error_handling(self):
        """Test mock embedding error handling"""
        embedding = MockEmbedding(embedding_dimension=5)

        # Empty text list should raise validation error
        with pytest.raises(EmbeddingValidationError):
            embedding.embed_texts([])

        # Very long text should raise validation error
        very_long_text = "word " * 10000  # This should exceed token limit
        with pytest.raises(EmbeddingValidationError):
            embedding.embed_text(very_long_text)