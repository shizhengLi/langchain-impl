# -*- coding: utf-8 -*-
"""
Text splitter types and data structures
"""

from typing import Any, Dict, List, Optional, Union, Sequence
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class Document(BaseModel):
    """
    Document representation for text splitting
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique document identifier")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    page_content: Optional[str] = Field(default=None, description="Page content (deprecated, use content)")

    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(id={self.id[:8]}..., content='{content_preview}')"

    def __str__(self) -> str:
        return self.content


class Chunk(BaseModel):
    """
    Text chunk representation
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique chunk identifier")
    content: str = Field(..., description="Chunk content")
    page_content: Optional[str] = Field(default=None, description="Page content (deprecated, use content)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    source_document_id: Optional[str] = Field(default=None, description="ID of source document")
    chunk_index: Optional[int] = Field(default=None, description="Index of chunk in document")
    start_char: Optional[int] = Field(default=None, description="Start character position in source")
    end_char: Optional[int] = Field(default=None, description="End character position in source")

    def __repr__(self) -> str:
        content_preview = self.content[:30] + "..." if len(self.content) > 30 else self.content
        return f"Chunk(id={self.id[:8]}..., content='{content_preview}')"

    def __str__(self) -> str:
        return self.content


class SplitStrategy(str, Enum):
    """Split strategy enumeration"""
    CHARACTER = "character"
    RECURSIVE = "recursive"
    TOKEN = "token"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"
    CODE = "code"


class SplitResult(BaseModel):
    """
    Result of text splitting operation
    """
    chunks: List[Chunk] = Field(default_factory=list, description="List of text chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Split result metadata")
    total_chunks: int = Field(default=0, description="Total number of chunks")
    total_characters: int = Field(default=0, description="Total characters across all chunks")
    total_tokens: Optional[int] = Field(default=None, description="Estimated total tokens")

    def __repr__(self) -> str:
        return f"SplitResult(chunks={len(self.chunks)}, chars={self.total_characters})"

    def model_post_init(self, __context) -> None:
        """Calculate derived fields after initialization"""
        self.total_chunks = len(self.chunks)
        self.total_characters = sum(len(chunk.content) for chunk in self.chunks)


class TextSplitterConfig(BaseModel):
    """
    Configuration for text splitters
    """
    chunk_size: int = Field(default=1000, gt=0, description="Maximum size of each chunk")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap between chunks")
    length_function: str = Field(default="len", description="Function to measure length")
    separator: str = Field(default="\n\n", description="Separator to split on")
    keep_separator: bool = Field(default=False, description="Whether to keep separator in chunks")
    strip_whitespace: bool = Field(default=True, description="Whether to strip whitespace")
    separators: List[str] = Field(default_factory=lambda: ["\n\n", "\n", " ", ""], description="List of separators to try")
    strategy: SplitStrategy = Field(default=SplitStrategy.RECURSIVE, description="Splitting strategy")

    # Language-specific separators
    is_separator_regex: bool = Field(default=False, description="Whether separators are regex patterns")

    # Markdown-specific options
    markdown_options: Dict[str, Any] = Field(default_factory=dict, description="Markdown splitting options")

    # Code-specific options
    code_options: Dict[str, Any] = Field(default_factory=dict, description="Code splitting options")

    def model_post_init(self, __context) -> None:
        """Validate configuration after initialization"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})")

    def __repr__(self) -> str:
        return f"TextSplitterConfig(strategy={self.strategy}, chunk_size={self.chunk_size})"


# Error types
class TextSplitterError(Exception):
    """
    Base text splitter error class
    """
    message: str
    splitter_name: str
    details: Dict[str, Any] = {}

    def __init__(self, message: str, splitter_name: str, details: Dict[str, Any] = None):
        self.message = message
        self.splitter_name = splitter_name
        self.details = details or {}
        super().__init__(message)


class TextSplitterValidationError(TextSplitterError):
    """
    Text splitter validation error
    """
    input_data: Any

    def __init__(self, message: str, splitter_name: str, input_data: Any = None):
        self.input_data = input_data
        validation_details = {
            "input_data": input_data,
            "validation_error": True
        }
        super().__init__(message, splitter_name, validation_details)


class TextSplitterProcessingError(TextSplitterError):
    """
    Text splitter processing error
    """
    document: Optional[Document] = None

    def __init__(self, message: str, splitter_name: str, document: Document = None):
        self.document = document
        processing_details = {
            "document_id": document.id if document else None,
            "processing_error": True
        }
        super().__init__(message, splitter_name, processing_details)


# Helper functions
def create_document(content: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
    """
    Create a document from content

    Args:
        content: Document content
        metadata: Optional metadata

    Returns:
        Document instance
    """
    return Document(content=content, metadata=metadata or {})


def create_chunk(content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> Chunk:
    """
    Create a chunk from content

    Args:
        content: Chunk content
        metadata: Optional metadata
        **kwargs: Additional chunk fields

    Returns:
        Chunk instance
    """
    chunk_metadata = metadata or {}
    return Chunk(content=content, metadata=chunk_metadata, **kwargs)


def estimate_token_count(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate token count for text

    Args:
        text: Text to estimate tokens for
        model: Model name for tokenization

    Returns:
        Estimated token count
    """
    # Simple estimation: ~4 characters per token for English
    # This is a rough approximation - real tokenization would use tiktoken
    return len(text) // 4


def merge_chunks(chunks: List[Chunk], separator: str = "\n\n") -> str:
    """
    Merge chunks back into a single text

    Args:
        chunks: List of chunks to merge
        separator: Separator to use between chunks

    Returns:
        Merged text
    """
    if not chunks:
        return ""

    # Sort chunks by their index if available
    sorted_chunks = sorted(chunks, key=lambda c: c.chunk_index or 0)
    contents = [chunk.content for chunk in sorted_chunks]

    return separator.join(contents)


def filter_chunks_by_size(chunks: List[Chunk], min_size: int = 0, max_size: Optional[int] = None) -> List[Chunk]:
    """
    Filter chunks by size

    Args:
        chunks: List of chunks to filter
        min_size: Minimum chunk size
        max_size: Maximum chunk size

    Returns:
        Filtered chunks
    """
    filtered = []

    for chunk in chunks:
        size = len(chunk.content)

        if size >= min_size:
            if max_size is None or size <= max_size:
                filtered.append(chunk)

    return filtered