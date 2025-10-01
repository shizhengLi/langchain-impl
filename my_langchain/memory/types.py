# -*- coding: utf-8 -*-
"""
Memory types and data structures
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    """
    Memory configuration
    """
    max_messages: Optional[int] = Field(
        default=None,
        description="Maximum number of messages to store"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to store"
    )
    return_messages: bool = Field(
        default=True,
        description="Whether to return messages in context"
    )
    summary_frequency: int = Field(
        default=10,
        description="Frequency of summarization (number of messages)"
    )
    llm_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="LLM configuration for summarization"
    )


class ChatMessage(BaseModel):
    """
    Chat message data structure
    """
    role: str = Field(..., description="Message role: user, assistant, system, etc.")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Message timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional message metadata"
    )


class MemoryResult(BaseModel):
    """
    Memory operation result
    """
    messages: List[ChatMessage] = Field(
        default_factory=list,
        description="Messages in memory"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Memory summary if available"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional memory metadata"
    )
    token_count: int = Field(
        default=0,
        description="Total token count"
    )


class MemorySearchResult(BaseModel):
    """
    Memory search result
    """
    messages: List[ChatMessage] = Field(
        default_factory=list,
        description="Search results"
    )
    score: float = Field(
        default=0.0,
        description="Search relevance score"
    )
    query: str = Field(..., description="Search query")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Search metadata"
    )


class MemoryContext(BaseModel):
    """
    Memory context for conversation
    """
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="Conversation history"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Conversation summary"
    )
    context_window: List[ChatMessage] = Field(
        default_factory=list,
        description="Current context window"
    )
    total_tokens: int = Field(
        default=0,
        description="Total tokens in context"
    )


# Error types
class MemoryError(Exception):
    """
    Base memory error class
    """
    message: str
    details: Dict[str, Any] = {}

    def __init__(self, message: str, details: Dict[str, Any] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class MemoryValidationError(MemoryError):
    """
    Memory validation error
    """
    def __init__(self, message: str, data: Any = None):
        details = {"validation_error": True}
        if data is not None:
            details["data"] = data
        super().__init__(message, details)


class MemorySearchError(MemoryError):
    """
    Memory search error
    """
    query: str

    def __init__(self, message: str, query: str, details: Dict[str, Any] = None):
        self.query = query
        search_details = {"query": query, "search_error": True}
        if details:
            search_details.update(details)
        super().__init__(message, search_details)


class MemoryCapacityError(MemoryError):
    """
    Memory capacity error
    """
    limit: Union[int, str]

    def __init__(self, message: str, limit: Union[int, str], details: Dict[str, Any] = None):
        self.limit = limit
        capacity_details = {"limit": limit, "capacity_error": True}
        if details:
            capacity_details.update(details)
        super().__init__(message, capacity_details)