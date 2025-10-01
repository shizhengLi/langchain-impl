# -*- coding: utf-8 -*-
"""
Recursive character-based text splitter implementation
"""

from typing import List, Optional, Dict, Any

from .base import BaseTextSplitter
from .types import TextSplitterConfig, SplitStrategy, TextSplitterProcessingError


class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """
    Recursive character-based text splitter

    This splitter tries to split text using a list of separators in order of preference.
    It recursively tries different separators to achieve the desired chunk size while maintaining
    semantic coherence. This is the recommended approach for most text splitting use cases.
    """

    def __init__(self, separators: Optional[List[str]] = None, **kwargs):
        """
        Initialize recursive character text splitter

        Args:
            separators: List of separators to try in order (default tries paragraphs, sentences, etc.)
            **kwargs: Additional configuration parameters
        """
        if separators is None:
            # Default separators that work well for most text
            separators = [
                "\n\n",  # Paragraphs
                "\n",    # Lines
                ". ",    # Sentences (with space)
                "? ",    # Questions (with space)
                "! ",    # Exclamations (with space)
                " ",     # Words
                ""       # Characters (last resort)
            ]

        # Handle chunk_overlap relative to chunk_size
        if 'chunk_size' in kwargs and 'chunk_overlap' not in kwargs:
            # Set default overlap to 10% of chunk_size if not specified
            chunk_size = kwargs['chunk_size']
            kwargs['chunk_overlap'] = min(50, chunk_size // 10)

        config = TextSplitterConfig(
            separators=separators,
            strategy=SplitStrategy.RECURSIVE,
            **kwargs
        )
        super().__init__(config=config)

    def split_text(self, text: str) -> List[str]:
        """
        Split text using recursive separator approach

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

            # Apply recursive splitting
            splits = self._recursive_split_with_separators(text, self.config.separators)

            # Filter out empty splits if strip_whitespace is enabled
            if self.config.strip_whitespace:
                splits = [split.strip() for split in splits if split.strip()]

            # Apply overlap if configured
            if self.config.chunk_overlap > 0 and len(splits) > 1:
                splits = self._apply_overlap_to_splits(splits)

            return splits

        except Exception as e:
            raise TextSplitterProcessingError(
                f"Failed to split text: {str(e)}",
                self.__class__.__name__
            ) from e

    def _recursive_split_with_separators(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using a list of separators

        Args:
            text: Text to split
            separators: List of separators to try in order

        Returns:
            List of text chunks
        """
        # If text is small enough, return as is
        if self.length_function(text) <= self.config.chunk_size:
            return [text]

        # If no separators left, split by character
        if not separators:
            return self._split_text_into_char_chunks(text)

        # Try the current separator
        separator = separators[0]
        remaining_separators = separators[1:]

        # Split using the current separator
        splits = self._split_text_with_separator(text, separator)

        # Check if any split is small enough
        good_splits = []
        for split in splits:
            if self.length_function(split) <= self.config.chunk_size:
                good_splits.append(split)
            else:
                # If split is too large, recursively split it with remaining separators
                if remaining_separators:
                    sub_splits = self._recursive_split_with_separators(split, remaining_separators)
                    good_splits.extend(sub_splits)
                else:
                    # Last resort: split into character chunks
                    char_chunks = self._split_text_into_char_chunks(split)
                    good_splits.extend(char_chunks)

        # If we got good splits, return them
        if len(good_splits) > 1:
            return good_splits
        elif len(good_splits) == 1:
            # Only one good split, try the next separator
            if remaining_separators:
                return self._recursive_split_with_separators(text, remaining_separators)
            else:
                return good_splits
        else:
            # No good splits, try the next separator
            if remaining_separators:
                return self._recursive_split_with_separators(text, remaining_separators)
            else:
                # Last resort: split into character chunks
                return self._split_text_into_char_chunks(text)

    def _split_text_into_char_chunks(self, text: str) -> List[str]:
        """
        Split text into character-based chunks as last resort

        Args:
            text: Text to split

        Returns:
            List of character-based chunks
        """
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]

            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)

            start = end - overlap if overlap > 0 else end

        return chunks

    def _apply_overlap_to_splits(self, splits: List[str]) -> List[str]:
        """
        Apply overlap between consecutive splits

        Args:
            splits: List of text splits

        Returns:
            List of splits with overlap
        """
        if self.config.chunk_overlap <= 0 or len(splits) <= 1:
            return splits

        overlapped = []
        for i, split in enumerate(splits):
            if i == 0:
                overlapped.append(split)
            else:
                # Get overlap from previous split
                prev_split = overlapped[-1]
                overlap_chars = self._get_overlap_chars(prev_split, split)
                combined = overlap_chars + split

                # If combined is too large, trim it
                if self.length_function(combined) > self.config.chunk_size:
                    excess = self.length_function(combined) - self.config.chunk_size
                    combined = combined[excess:]

                overlapped.append(combined)

        return overlapped

    def _get_overlap_chars(self, prev_text: str, current_text: str) -> str:
        """
        Get overlapping characters between two texts

        Args:
            prev_text: Previous text
            current_text: Current text

        Returns:
            Overlapping characters
        """
        overlap_size = self.config.chunk_overlap
        if overlap_size <= 0:
            return ""

        # Get overlap from end of previous text
        overlap = prev_text[-overlap_size:] if len(prev_text) >= overlap_size else prev_text

        # Try to find the best overlap point that maintains word boundaries
        if self.config.strip_whitespace:
            # Try to end at a word boundary
            for i in range(len(overlap), 0, -1):
                if overlap[i-1].isspace():
                    return overlap[i:]
            # If no space found, return character-based overlap
            return overlap

        return overlap

    @classmethod
    def from_language(cls, language: str = "python", **kwargs) -> 'RecursiveCharacterTextSplitter':
        """
        Create a splitter configured for a specific programming language

        Args:
            language: Programming language name
            **kwargs: Additional configuration parameters

        Returns:
            Configured RecursiveCharacterTextSplitter
        """
        # Create temporary instance to get language separators
        temp_instance = cls()
        language_separators = temp_instance._get_language_separators(language)

        return cls(
            separators=language_separators,
            **kwargs
        )

    def _get_language_separators(self, language: str) -> List[str]:
        """
        Get appropriate separators for a programming language

        Args:
            language: Programming language name

        Returns:
            List of separators for the language
        """
        language = language.lower()

        separators_map = {
            "python": [
                "\nclass ",    # Class definitions
                "\ndef ",      # Function definitions
                "\n\tdef ",    # Method definitions
                "\n\n",        # Double newlines
                "\n",          # Single newlines
                ". ",          # Sentences
                " ",           # Spaces
                ""             # Characters
            ],
            "javascript": [
                "\nfunction ", # Function definitions
                "\nconst ",    # Constants
                "\nlet ",      # Let declarations
                "\nvar ",      # Var declarations
                "\n\n",        # Double newlines
                "\n",          # Single newlines
                "}",           # Closing braces
                "{",           # Opening braces
                "; ",          # Semicolons
                " ",           # Spaces
                ""             # Characters
            ],
            "java": [
                "\npublic class ",  # Class definitions
                "\nprivate class ", # Private class definitions
                "\npublic ",        # Public methods
                "\nprivate ",       # Private methods
                "\n\n",            # Double newlines
                "\n",              # Single newlines
                "}",               # Closing braces
                "{",               # Opening braces
                "; ",              # Semicolons
                " ",               # Spaces
                ""                 # Characters
            ],
            "markdown": [
                "\n## ",       # Level 2 headers
                "\n# ",        # Level 1 headers
                "\n\n",        # Paragraphs
                "\n- ",        # List items
                "\n",          # Single newlines
                "```",         # Code blocks
                "**",          # Bold
                "*",           # Italic
                " ",           # Spaces
                ""             # Characters
            ],
            "html": [
                "\n</div>",     # Closing div tags
                "\n</p>",       # Closing paragraph tags
                "\n</section>", # Closing section tags
                "\n\n",         # Paragraphs
                "\n",           # Single newlines
                ">",            # Tag closures
                " ",            # Spaces
                ""              # Characters
            ],
            "css": [
                "\n}",          # Closing braces
                "\n\n",         # Double newlines
                "\n",           # Single newlines
                "{",            # Opening braces
                ";",            # Semicolons
                " ",            # Spaces
                ""              # Characters
            ]
        }

        return separators_map.get(language, self.config.separators)

    def get_separators_by_semantic_importance(self, text: str) -> List[str]:
        """
        Analyze text and return separators ordered by semantic importance

        Args:
            text: Text to analyze

        Returns:
            List of separators ordered by importance
        """
        separators = self.config.separators.copy()
        separator_scores = []

        for separator in separators:
            if separator == "":
                # Character-level splitting is always last resort
                score = 0
            else:
                # Score based on frequency and semantic value
                count = text.count(separator)
                score = count * self._get_separator_semantic_weight(separator)

            separator_scores.append((separator, score))

        # Sort by score (descending)
        separator_scores.sort(key=lambda x: x[1], reverse=True)

        return [sep for sep, score in separator_scores]

    def _get_separator_semantic_weight(self, separator: str) -> float:
        """
        Get semantic weight for a separator

        Args:
            separator: Separator to weight

        Returns:
            Semantic weight
        """
        weight_map = {
            "\n\n": 3.0,      # Paragraphs are very important
            "\n": 2.0,        # Lines are important
            ". ": 2.5,        # Sentences are important
            "? ": 2.5,        # Questions are important
            "! ": 2.5,        # Exclamations are important
            ";": 1.5,         # Semicolons are somewhat important
            ":": 1.5,         # Colons are somewhat important
            ",": 1.0,         # Commas are less important
            " ": 0.5,         # Spaces are least important
        }

        return weight_map.get(separator, 1.0)

    def estimate_split_quality(self, chunks: List[str]) -> Dict[str, float]:
        """
        Estimate the quality of text splits

        Args:
            chunks: List of text chunks

        Returns:
            Dictionary with quality metrics
        """
        if not chunks:
            return {"avg_chunk_size": 0, "size_variance": 0, "semantic_coherence": 0}

        # Average chunk size
        sizes = [self.length_function(chunk) for chunk in chunks]
        avg_size = sum(sizes) / len(sizes)

        # Size variance (lower is better)
        size_variance = sum((size - avg_size) ** 2 for size in sizes) / len(sizes)

        # Semantic coherence (higher is better)
        semantic_score = self._calculate_semantic_coherence(chunks)

        # Coverage efficiency (how close we are to target chunk size)
        target_size = self.config.chunk_size
        coverage_efficiency = 1.0 - abs(avg_size - target_size) / target_size

        return {
            "avg_chunk_size": avg_size,
            "size_variance": size_variance,
            "semantic_coherence": semantic_score,
            "coverage_efficiency": coverage_efficiency,
            "total_chunks": len(chunks)
        }

    def _calculate_semantic_coherence(self, chunks: List[str]) -> float:
        """
        Calculate semantic coherence score for chunks

        Args:
            chunks: List of text chunks

        Returns:
            Semantic coherence score (0-1, higher is better)
        """
        if len(chunks) <= 1:
            return 1.0

        coherence_scores = []

        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # Check if chunks end at natural boundaries
            current_ends_well = self._ends_at_boundary(current_chunk)
            next_starts_well = self._starts_at_boundary(next_chunk)

            coherence = 1.0 if current_ends_well and next_starts_well else 0.5
            coherence_scores.append(coherence)

        return sum(coherence_scores) / len(coherence_scores)

    def _ends_at_boundary(self, text: str) -> bool:
        """
        Check if text ends at a natural boundary

        Args:
            text: Text to check

        Returns:
            True if ends at boundary
        """
        text = text.strip()
        if not text:
            return True

        # Check for sentence-ending punctuation
        if text[-1] in '.!?':
            return True

        # Check for code-related endings
        if text[-1] in ')}]':
            return True

        return False

    def _starts_at_boundary(self, text: str) -> bool:
        """
        Check if text starts at a natural boundary

        Args:
            text: Text to check

        Returns:
            True if starts at boundary
        """
        text = text.strip()
        if not text:
            return True

        # Check for capital letters (start of sentence)
        if text[0].isupper():
            return True

        # Check for list items
        if text.startswith(('-', '*', '#', '1.', '2.', '3.')):
            return True

        return False