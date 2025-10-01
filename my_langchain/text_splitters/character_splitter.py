# -*- coding: utf-8 -*-
"""
Character-based text splitter implementation
"""

from typing import List, Optional

from .base import BaseTextSplitter
from .types import TextSplitterConfig, SplitStrategy, TextSplitterProcessingError


class CharacterTextSplitter(BaseTextSplitter):
    """
    Character-based text splitter

    This splitter divides text based on a single character separator.
    It's useful when you want to split on specific delimiters like newlines,
    spaces, or custom separators.
    """

    def __init__(self, separator: str = "\n\n", **kwargs):
        """
        Initialize character text splitter

        Args:
            separator: Character separator to split on
            **kwargs: Additional configuration parameters
        """
        # Handle chunk_overlap relative to chunk_size
        if 'chunk_size' in kwargs and 'chunk_overlap' not in kwargs:
            # Set default overlap to 10% of chunk_size if not specified
            chunk_size = kwargs['chunk_size']
            kwargs['chunk_overlap'] = min(50, chunk_size // 10)

        config = TextSplitterConfig(separator=separator, strategy=SplitStrategy.CHARACTER, **kwargs)
        super().__init__(config=config)

    def split_text(self, text: str) -> List[str]:
        """
        Split text using the configured separator

        Args:
            text: Text to split

        Returns:
            List of text chunks

        Raises:
            TextSplitterProcessingError: If splitting fails
        """
        try:
            if not text:
                return []

            # Split using the separator
            splits = self._split_text_with_separator(text, self.config.separator)

            # Filter out empty splits if strip_whitespace is enabled
            if self.config.strip_whitespace:
                splits = [split.strip() for split in splits if split.strip()]

            # Split chunks that are too large
            splits = self._split_large_chunks(splits)

            # Only merge chunks that are too small if we need to reach chunk size
            # and we have more than one chunk (to avoid merging everything back)
            if len(splits) > 1:
                # Check if we need merging based on average chunk size
                avg_size = sum(len(s) for s in splits) / len(splits)
                if avg_size < self.config.chunk_size // 2:
                    splits = self._merge_small_chunks(splits)

            # Apply overlap if configured
            if self.config.chunk_overlap > 0 and len(splits) > 1:
                splits = self._apply_overlap(splits)

            return splits

        except Exception as e:
            raise TextSplitterProcessingError(
                f"Failed to split text: {str(e)}",
                self.__class__.__name__
            ) from e

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap between consecutive chunks

        Args:
            chunks: List of text chunks

        Returns:
            List of chunks with overlap
        """
        if self.config.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = overlapped[-1]
                overlap_start = max(0, len(prev_chunk) - self.config.chunk_overlap)
                overlap_text = prev_chunk[overlap_start:]

                # Combine overlap with current chunk
                combined_chunk = overlap_text + chunk

                # If combined chunk is too large, trim it
                if self.length_function(combined_chunk) > self.config.chunk_size:
                    # Trim from the beginning
                    excess = self.length_function(combined_chunk) - self.config.chunk_size
                    combined_chunk = combined_chunk[excess:]

                overlapped.append(combined_chunk)

        return overlapped

    def split_on_newlines(self, text: str) -> List[str]:
        """
        Convenience method to split on newlines

        Args:
            text: Text to split

        Returns:
            List of chunks split on newlines
        """
        original_separator = self.config.separator
        self.config.separator = "\n"
        try:
            result = self.split_text(text)
            return result
        finally:
            self.config.separator = original_separator

    def split_on_spaces(self, text: str) -> List[str]:
        """
        Convenience method to split on spaces

        Args:
            text: Text to split

        Returns:
            List of chunks split on spaces
        """
        original_separator = self.config.separator
        self.config.separator = " "
        try:
            result = self.split_text(text)
            return result
        finally:
            self.config.separator = original_separator

    def split_on_sentences(self, text: str) -> List[str]:
        """
        Convenience method to split on sentence boundaries

        Args:
            text: Text to split

        Returns:
            List of chunks split on sentences
        """
        # Use regex pattern to match sentence boundaries
        original_separator = self.config.separator
        original_is_regex = self.config.is_separator_regex
        self.config.separator = r'[.!?]+\s+'
        self.config.is_separator_regex = True
        try:
            result = self.split_text(text)
            # Clean up leading/trailing whitespace
            result = [chunk.strip() for chunk in result if chunk.strip()]
            return result
        finally:
            self.config.separator = original_separator
            self.config.is_separator_regex = original_is_regex

    def split_on_paragraphs(self, text: str) -> List[str]:
        """
        Convenience method to split on paragraph boundaries

        Args:
            text: Text to split

        Returns:
            List of chunks split on paragraphs
        """
        original_separator = self.config.separator
        self.config.separator = "\n\n"
        try:
            result = self.split_text(text)
            return result
        finally:
            self.config.separator = original_separator

    def split_on_double_newlines(self, text: str) -> List[str]:
        """
        Convenience method to split on double newlines (paragraphs)

        Args:
            text: Text to split

        Returns:
            List of chunks split on double newlines
        """
        return self.split_on_paragraphs(text)

    def get_chunk_count_estimate(self, text: str) -> int:
        """
        Estimate the number of chunks for a given text

        Args:
            text: Text to estimate for

        Returns:
            Estimated number of chunks
        """
        if not text:
            return 0

        # Quick estimation based on average chunk size
        effective_chunk_size = self.config.chunk_size - self.config.chunk_overlap
        estimated_chunks = max(1, len(text) // effective_chunk_size)

        # Adjust for separator density
        if self.config.separator:
            separator_count = text.count(self.config.separator)
            if separator_count > 0:
                estimated_chunks = min(estimated_chunks, separator_count + 1)

        return estimated_chunks