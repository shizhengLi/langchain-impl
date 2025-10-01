# -*- coding: utf-8 -*-
"""
Base memory implementation
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from my_langchain.base.base import BaseComponent
from my_langchain.memory.types import (
    MemoryConfig, MemoryResult, MemoryContext, MemorySearchResult,
    ChatMessage, MemoryError, MemoryValidationError
)
from pydantic import ConfigDict, Field


class BaseMemory(BaseComponent):
    """
    Base memory implementation for conversation history management

    Provides common functionality for memory operations,
    including validation, error handling, and token estimation.
    """

    config: MemoryConfig = Field(..., description="Memory configuration")

    def __init__(self, config: Optional[MemoryConfig] = None, **kwargs):
        """
        Initialize memory

        Args:
            config: Memory configuration
            **kwargs: Additional parameters
        """
        if config is None:
            config = MemoryConfig()

        super().__init__(config=config, **kwargs)

    @abstractmethod
    def add_message(self, message: Union[ChatMessage, Dict[str, Any]], **kwargs) -> None:
        """
        Add a message to memory

        Args:
            message: Message to add (ChatMessage or dict)
            **kwargs: Additional parameters
        """
        pass

    @abstractmethod
    def add_messages(self, messages: List[Union[ChatMessage, Dict[str, Any]]], **kwargs) -> None:
        """
        Add multiple messages to memory

        Args:
            messages: List of messages to add
            **kwargs: Additional parameters
        """
        pass

    @abstractmethod
    def get_memory(self, **kwargs) -> MemoryResult:
        """
        Get current memory state

        Args:
            **kwargs: Additional parameters

        Returns:
            Current memory result
        """
        pass

    @abstractmethod
    def get_context(self, **kwargs) -> MemoryContext:
        """
        Get conversation context

        Args:
            **kwargs: Additional parameters

        Returns:
            Memory context
        """
        pass

    @abstractmethod
    def clear(self, **kwargs) -> None:
        """
        Clear all memory

        Args:
            **kwargs: Additional parameters
        """
        pass

    @abstractmethod
    def search(self, query: str, **kwargs) -> List[MemorySearchResult]:
        """
        Search memory

        Args:
            query: Search query
            **kwargs: Additional parameters

        Returns:
            List of search results
        """
        pass

    def run(self, **kwargs) -> Any:
        """Implement BaseComponent run method"""
        return self.get_memory(**kwargs)

    def _validate_message(self, message: Union[ChatMessage, Dict[str, Any]]) -> ChatMessage:
        """
        Validate and normalize message

        Args:
            message: Message to validate

        Returns:
            Validated ChatMessage

        Raises:
            MemoryValidationError: If message is invalid
        """
        if isinstance(message, ChatMessage):
            return message
        elif isinstance(message, dict):
            try:
                return ChatMessage(**message)
            except Exception as e:
                raise MemoryValidationError(
                    f"Invalid message dict: {str(e)}",
                    data=message
                )
        else:
            raise MemoryValidationError(
                f"Message must be ChatMessage or dict, got {type(message)}",
                data=message
            )

    def _check_capacity(self, messages: List[ChatMessage]) -> None:
        """
        Check if adding messages would exceed capacity

        Args:
            messages: Messages to check

        Raises:
            MemoryCapacityError: If capacity would be exceeded
        """
        if self.config.max_messages is not None:
            current_count = len(self.get_memory().messages)
            if current_count + len(messages) > self.config.max_messages:
                raise MemoryCapacityError(
                    f"Would exceed max message limit of {self.config.max_messages}",
                    limit=self.config.max_messages,
                    details={
                        "current_count": current_count,
                        "new_messages": len(messages)
                    }
                )

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Simple token estimation: ~4 chars per token for English
        return max(1, len(text) // 4)

    def _count_message_tokens(self, message: ChatMessage) -> int:
        """
        Count tokens in a message

        Args:
            message: Message to count

        Returns:
            Token count
        """
        tokens = self._estimate_tokens(message.content)
        # Add overhead for role and metadata
        tokens += 5
        return tokens

    def _count_total_tokens(self, messages: List[ChatMessage]) -> int:
        """
        Count total tokens in messages

        Args:
            messages: Messages to count

        Returns:
            Total token count
        """
        return sum(self._count_message_tokens(msg) for msg in messages)

    def _format_message_dict(self, message: ChatMessage) -> Dict[str, Any]:
        """
        Format message as dictionary for LLM consumption

        Args:
            message: Message to format

        Returns:
            Formatted message dictionary
        """
        return {
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            **message.metadata
        }

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory information

        Returns:
            Memory information dictionary
        """
        memory_result = self.get_memory()
        return {
            "memory_type": self.__class__.__name__,
            "config": self.config.model_dump(),
            "message_count": len(memory_result.messages),
            "token_count": memory_result.token_count,
            "has_summary": memory_result.summary is not None
        }

    def _merge_messages(self, existing: List[ChatMessage], new: List[ChatMessage]) -> List[ChatMessage]:
        """
        Merge message lists with timestamp ordering

        Args:
            existing: Existing messages
            new: New messages to add

        Returns:
            Merged and sorted message list
        """
        merged = existing + new
        merged.sort(key=lambda x: x.timestamp)
        return merged