# -*- coding: utf-8 -*-
"""
Mock embedding implementation for testing
"""

import hashlib
import math
from typing import List, Dict, Any

from .base import BaseEmbedding
from .types import EmbeddingConfig, EmbeddingProcessingError


class MockEmbedding(BaseEmbedding):
    """
    Mock embedding implementation for testing and development

    This implementation generates deterministic embeddings based on text content
    using a hash function. It's useful for testing and development purposes.
    """

    def __init__(self, config: EmbeddingConfig = None, **kwargs):
        """
        Initialize mock embedding

        Args:
            config: Embedding configuration
            **kwargs: Additional configuration parameters
        """
        # Set default dimension for mock embeddings
        if config is None and 'embedding_dimension' not in kwargs:
            kwargs['embedding_dimension'] = 384  # Common small dimension

        super().__init__(config, **kwargs)

        # Seed for reproducible results
        self.seed = kwargs.get('seed', 42)

    def _embed_single_text(self, text: str) -> List[float]:
        """
        Generate deterministic embedding for a single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check for predefined response first
        mock_response = self.get_mock_response(text)
        if mock_response:
            return mock_response

        return self._generate_embedding_from_text(text)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate deterministic embeddings for multiple texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return [self._embed_single_text(text) for text in texts]

    def _generate_embedding_from_text(self, text: str) -> List[float]:
        """
        Generate deterministic embedding vector from text

        Uses a hash function to create reproducible embeddings that are
        similar for similar texts.

        Args:
            text: Text to generate embedding for

        Returns:
            Embedding vector
        """
        dimension = self.config.embedding_dimension

        # Create hash of text
        hash_obj = hashlib.sha256(f"{text}_{self.seed}".encode('utf-8'))
        hash_bytes = hash_obj.digest()

        # Generate base embedding from hash
        embedding = []
        for i in range(dimension):
            # Use different parts of hash for different dimensions
            byte_index = (i * 4) % len(hash_bytes)
            next_byte_index = (byte_index + 1) % len(hash_bytes)

            # Combine bytes to create a float between -1 and 1
            combined = (hash_bytes[byte_index] << 8) | hash_bytes[next_byte_index]
            value = (combined / 65535.0) * 2 - 1  # Normalize to [-1, 1]

            embedding.append(value)

        # Add text-based variations for more realistic embeddings
        embedding = self._add_text_characteristics(embedding, text)

        # Normalize if required
        if self.config.normalize_embeddings:
            embedding = self._normalize_vector(embedding)

        return embedding

    def _add_text_characteristics(self, base_embedding: List[float], text: str) -> List[float]:
        """
        Add text-specific characteristics to base embedding

        This method adds variations based on text properties like length,
        character composition, etc., to make embeddings more realistic.

        Args:
            base_embedding: Base embedding vector
            text: Original text

        Returns:
            Modified embedding vector
        """
        embedding = base_embedding.copy()
        dimension = len(embedding)

        # Add length-based variations
        length_factor = math.log(len(text) + 1) / 10
        for i in range(min(dimension // 4, len(embedding))):
            embedding[i] += length_factor * 0.1

        # Add character composition variations
        char_stats = self._get_char_statistics(text)

        # Vowel ratio affects certain dimensions
        vowel_factor = char_stats['vowel_ratio']
        for i in range(dimension // 4, dimension // 2):
            embedding[i] += vowel_factor * 0.2

        # Digit ratio affects other dimensions
        digit_factor = char_stats['digit_ratio']
        for i in range(dimension // 2, 3 * dimension // 4):
            embedding[i] += digit_factor * 0.3

        # Special character ratio affects remaining dimensions
        special_factor = char_stats['special_ratio']
        for i in range(3 * dimension // 4, dimension):
            embedding[i] += special_factor * 0.1

        return embedding

    def _get_char_statistics(self, text: str) -> Dict[str, float]:
        """
        Get character statistics for text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with character statistics
        """
        if not text:
            return {'vowel_ratio': 0, 'digit_ratio': 0, 'special_ratio': 0}

        vowels = set('aeiouAEIOU')
        digits = set('0123456789')
        special = set('!@#$%^&*()_+-=[]{}|;:,.<>?')

        vowel_count = sum(1 for char in text if char in vowels)
        digit_count = sum(1 for char in text if char in digits)
        special_count = sum(1 for char in text if char in special)

        total_chars = len(text)

        return {
            'vowel_ratio': vowel_count / total_chars,
            'digit_ratio': digit_count / total_chars,
            'special_ratio': special_count / total_chars
        }

    def simulate_delay(self, delay: float = None):
        """
        Simulate processing delay for testing

        Args:
            delay: Delay in seconds (default: random between 0.1 and 0.5)
        """
        import random
        import time

        if delay is None:
            delay = random.uniform(0.1, 0.5)

        time.sleep(delay)

    def set_mock_responses(self, responses: Dict[str, List[float]]):
        """
        Set predefined mock responses for specific texts

        Args:
            responses: Dictionary mapping texts to embedding vectors
        """
        self._mock_responses = responses

    def get_mock_response(self, text: str) -> List[float]:
        """
        Get predefined mock response if available

        Args:
            text: Text to get response for

        Returns:
            Embedding vector or None if not found
        """
        if hasattr(self, '_mock_responses') and text in self._mock_responses:
            return self._mock_responses[text]
        return None

    
    def create_similarity_matrix(self, texts: List[str]) -> List[List[float]]:
        """
        Create similarity matrix for a list of texts

        Args:
            texts: List of texts

        Returns:
            Similarity matrix (list of lists)
        """
        embeddings = self.embed_texts(texts)

        matrix = []
        for i, emb1 in enumerate(embeddings.embeddings):
            row = []
            for j, emb2 in enumerate(embeddings.embeddings):
                similarity = emb1.cosine_similarity(emb2)
                row.append(similarity)
            matrix.append(row)

        return matrix

    def test_embedding_quality(self, test_texts: List[str]) -> Dict[str, float]:
        """
        Test embedding quality with various metrics

        Args:
            test_texts: List of test texts

        Returns:
            Dictionary with quality metrics
        """
        if len(test_texts) < 2:
            raise EmbeddingProcessingError("Need at least 2 texts for quality testing")

        embeddings = self.embed_texts(test_texts)

        # Calculate average similarity between different texts
        similarities = []
        for i in range(len(embeddings.embeddings)):
            for j in range(i + 1, len(embeddings.embeddings)):
                sim = embeddings.embeddings[i].cosine_similarity(embeddings.embeddings[j])
                similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        # Calculate variance in similarities
        variance = sum((s - avg_similarity) ** 2 for s in similarities) / len(similarities) if similarities else 0

        # Test self-similarity (should be 1.0)
        self_similarities = []
        for embedding in embeddings.embeddings:
            self_sim = embedding.cosine_similarity(embedding)
            self_similarities.append(self_sim)

        avg_self_similarity = sum(self_similarities) / len(self_similarities)

        return {
            "avg_cross_similarity": avg_similarity,
            "similarity_variance": variance,
            "avg_self_similarity": avg_self_similarity,
            "embedding_count": len(embeddings.embeddings),
            "dimension": self.config.embedding_dimension
        }

    @classmethod
    def create_small_model(cls, **kwargs) -> 'MockEmbedding':
        """
        Create a small mock embedding model

        Args:
            **kwargs: Additional configuration

        Returns:
            MockEmbedding instance with small dimension
        """
        kwargs['embedding_dimension'] = kwargs.get('embedding_dimension', 128)
        return cls(**kwargs)

    @classmethod
    def create_medium_model(cls, **kwargs) -> 'MockEmbedding':
        """
        Create a medium mock embedding model

        Args:
            **kwargs: Additional configuration

        Returns:
            MockEmbedding instance with medium dimension
        """
        kwargs['embedding_dimension'] = kwargs.get('embedding_dimension', 384)
        return cls(**kwargs)

    @classmethod
    def create_large_model(cls, **kwargs) -> 'MockEmbedding':
        """
        Create a large mock embedding model

        Args:
            **kwargs: Additional configuration

        Returns:
            MockEmbedding instance with large dimension
        """
        kwargs['embedding_dimension'] = kwargs.get('embedding_dimension', 1536)
        return cls(**kwargs)