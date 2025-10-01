# -*- coding: utf-8 -*-
"""
Base text splitter implementation
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
import re
import time

from .types import (
    Document, Chunk, SplitResult, TextSplitterConfig, SplitStrategy,
    TextSplitterError, TextSplitterValidationError, TextSplitterProcessingError
)


class BaseTextSplitter(ABC):
    """
    Base text splitter providing common functionality

    This class defines the interface that all text splitter implementations must follow
    and provides common utility methods for text processing and validation.
    """

    def __init__(self, config: Optional[TextSplitterConfig] = None, **kwargs):
        """
        Initialize text splitter

        Args:
            config: Text splitter configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or TextSplitterConfig(**kwargs)
        self.length_function = self._get_length_function()

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks

        Args:
            text: Text to split

        Returns:
            List of text chunks

        Raises:
            TextSplitterProcessingError: If splitting fails
        """
        pass

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Split a list of documents into chunks

        Args:
            documents: List of documents to split

        Returns:
            List of chunks

        Raises:
            TextSplitterValidationError: If documents are invalid
        """
        self._validate_documents(documents)

        all_chunks = []
        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)

        return all_chunks

    def split_document(self, document: Document) -> List[Chunk]:
        """
        Split a single document into chunks

        Args:
            document: Document to split

        Returns:
            List of chunks

        Raises:
            TextSplitterProcessingError: If splitting fails
        """
        try:
            text_chunks = self.split_text(document.content)
            return self._create_chunks_from_text(text_chunks, document)
        except Exception as e:
            raise TextSplitterProcessingError(
                f"Failed to split document: {str(e)}",
                self.__class__.__name__,
                document
            ) from e

    def create_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[Document]:
        """
        Create documents from texts

        Args:
            texts: List of texts
            metadatas: Optional list of metadata dictionaries

        Returns:
            List of documents
        """
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            documents.append(Document(content=text, metadata=metadata))

        return documents

    def split_and_create_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[Document]:
        """
        Split texts and create documents from chunks

        Args:
            texts: List of texts to split
            metadatas: Optional list of metadata dictionaries

        Returns:
            List of documents created from chunks
        """
        documents = self.create_documents(texts, metadatas)
        chunks = self.split_documents(documents)

        # Convert chunks back to documents
        result_documents = []
        for chunk in chunks:
            metadata = chunk.metadata.copy()
            if chunk.source_document_id:
                metadata["source_document_id"] = chunk.source_document_id
            if chunk.chunk_index is not None:
                metadata["chunk_index"] = chunk.chunk_index

            result_documents.append(
                Document(
                    content=chunk.content,
                    metadata=metadata
                )
            )

        return result_documents

    def transform_documents(self, documents: List[Document]) -> List[Document]:
        """
        Transform documents by splitting them and creating new documents from chunks

        Args:
            documents: List of documents to transform

        Returns:
            List of transformed documents
        """
        return self.split_and_create_documents([doc.content for doc in documents],
                                            [doc.metadata for doc in documents])

    def _get_length_function(self) -> Callable[[str], int]:
        """
        Get the length function based on configuration

        Returns:
            Length function
        """
        if self.config.length_function == "len":
            return len
        elif self.config.length_function == "token_estimate":
            from .types import estimate_token_count
            return lambda text: estimate_token_count(text)
        else:
            # Try to get a custom function
            try:
                # This could be extended to support custom functions
                return len
            except Exception:
                return len

    def _validate_documents(self, documents: List[Document]) -> None:
        """
        Validate documents before processing

        Args:
            documents: Documents to validate

        Raises:
            TextSplitterValidationError: If documents are invalid
        """
        if not documents:
            raise TextSplitterValidationError(
                "No documents provided",
                self.__class__.__name__,
                documents
            )

        for i, doc in enumerate(documents):
            if not isinstance(doc, Document):
                raise TextSplitterValidationError(
                    f"Document {i} is not a Document instance",
                    self.__class__.__name__,
                    documents
                )

            if not doc.content:
                raise TextSplitterValidationError(
                    f"Document {i} has empty content",
                    self.__class__.__name__,
                    doc
                )

    def _create_chunks_from_text(self, text_chunks: List[str], source_document: Document) -> List[Chunk]:
        """
        Create Chunk objects from text chunks

        Args:
            text_chunks: List of text chunks
            source_document: Source document

        Returns:
            List of Chunk objects
        """
        chunks = []
        current_char_pos = 0

        for i, text_chunk in enumerate(text_chunks):
            # Find the position of this chunk in the original text
            if text_chunk in source_document.content:
                start_char = source_document.content.find(text_chunk, current_char_pos)
                end_char = start_char + len(text_chunk)
                current_char_pos = end_char
            else:
                # Fallback if chunk not found exactly
                start_char = end_char = None

            # Create chunk metadata
            chunk_metadata = source_document.metadata.copy()
            chunk_metadata.update({
                "source_document_id": source_document.id,
                "chunk_index": i,
                "chunk_size": len(text_chunk)
            })

            chunk = Chunk(
                content=text_chunk,
                metadata=chunk_metadata,
                source_document_id=source_document.id,
                chunk_index=i,
                start_char=start_char,
                end_char=end_char
            )

            chunks.append(chunk)

        return chunks

    def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
        """
        Split text using a specific separator

        Args:
            text: Text to split
            separator: Separator to use

        Returns:
            List of text chunks
        """
        if not text:
            return []

        if separator == "":
            # Character-level splitting
            return list(text)

        if self.config.is_separator_regex:
            separator_pattern = re.compile(separator)
            splits = separator_pattern.split(text)
        else:
            splits = text.split(separator)

        # Filter out empty splits if strip_whitespace is enabled
        if self.config.strip_whitespace:
            splits = [split.strip() for split in splits if split.strip()]

        # Add separator back if keep_separator is enabled
        if self.config.keep_separator and separator and len(splits) > 1:
            result = []
            for i, split in enumerate(splits):
                result.append(split)
                if i < len(splits) - 1:
                    result.append(separator)
            return result

        return splits

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge chunks that are too small

        Args:
            chunks: List of text chunks

        Returns:
            List of merged chunks
        """
        if not chunks:
            return []

        merged = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            if self.length_function(current_chunk) < self.config.chunk_size // 2:
                # Merge small chunk with next one
                current_chunk += self.config.separator + next_chunk
            else:
                merged.append(current_chunk)
                current_chunk = next_chunk

        if current_chunk:
            merged.append(current_chunk)

        return merged

    def _split_large_chunks(self, chunks: List[str]) -> List[str]:
        """
        Split chunks that are too large

        Args:
            chunks: List of text chunks

        Returns:
            List of split chunks
        """
        result = []

        for chunk in chunks:
            if self.length_function(chunk) <= self.config.chunk_size:
                result.append(chunk)
            else:
                # Split large chunk recursively
                split_chunks = self._recursive_split(chunk)
                result.extend(split_chunks)

        return result

    def _recursive_split(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        """
        Recursively split text using multiple separators

        Args:
            text: Text to split
            separators: List of separators to try

        Returns:
            List of text chunks
        """
        if separators is None:
            separators = self.config.separators

        # If text is small enough, return as is
        if self.length_function(text) <= self.config.chunk_size:
            return [text]

        # Try each separator
        for separator in separators:
            if separator == "":
                # Last resort: split by character
                return self._split_by_character(text)

            splits = self._split_text_with_separator(text, separator)

            # Filter and process splits
            good_splits = []
            for split in splits:
                if self.length_function(split) <= self.config.chunk_size:
                    good_splits.append(split)
                else:
                    # Recursively split large parts
                    sub_splits = self._recursive_split(split, separators)
                    good_splits.extend(sub_splits)

            if len(good_splits) > 1:
                return self._merge_overlapping_chunks(good_splits)

        # If no separator worked, split by character
        return self._split_by_character(text)

    def _split_by_character(self, text: str) -> List[str]:
        """
        Split text by character

        Args:
            text: Text to split

        Returns:
            List of character-based chunks
        """
        chunks = []
        current_pos = 0

        while current_pos < len(text):
            end_pos = min(current_pos + self.config.chunk_size, len(text))
            chunks.append(text[current_pos:end_pos])
            current_pos = end_pos - self.config.chunk_overlap

        return [chunk for chunk in chunks if chunk.strip()]

    def _merge_overlapping_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge chunks with overlap

        Args:
            chunks: List of text chunks

        Returns:
            List of merged chunks
        """
        if self.config.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        merged = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                merged.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = merged[-1]
                overlap_start = len(prev_chunk) - self.config.chunk_overlap
                if overlap_start > 0:
                    overlap = prev_chunk[overlap_start:]
                    merged.append(overlap + chunk)
                else:
                    merged.append(chunk)

        return merged

    def get_split_result(self, chunks: List[Chunk]) -> SplitResult:
        """
        Create a SplitResult from chunks

        Args:
            chunks: List of chunks

        Returns:
            SplitResult object
        """
        metadata = {
            "splitter": self.__class__.__name__,
            "strategy": self.config.strategy,
            "config": self.config.model_dump(),
            "processing_time": time.time()
        }

        return SplitResult(
            chunks=chunks,
            metadata=metadata
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(strategy={self.config.strategy}, chunk_size={self.config.chunk_size})"