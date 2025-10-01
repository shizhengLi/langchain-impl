# -*- coding: utf-8 -*-
"""
Chat message history implementation
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from my_langchain.memory.base import BaseMemory
from my_langchain.memory.types import (
    MemoryConfig, MemoryResult, MemoryContext, MemorySearchResult,
    ChatMessage, MemoryError, MemoryValidationError
)


class ChatMessageHistory(BaseMemory):
    """
    Simple chat message history storage

    Provides basic conversation history management with message
    storage, retrieval, and search capabilities.
    """

    def __init__(self, config: Optional[MemoryConfig] = None, **kwargs):
        """
        Initialize chat message history

        Args:
            config: Memory configuration
            **kwargs: Additional parameters
        """
        super().__init__(config=config, **kwargs)
        self._messages: List[ChatMessage] = []

    def add_message(self, message: Union[ChatMessage, Dict[str, Any]], **kwargs) -> None:
        """
        Add a message to history

        Args:
            message: Message to add
            **kwargs: Additional parameters
        """
        validated_message = self._validate_message(message)

        # Check capacity
        self._check_capacity([validated_message])

        # Add message
        self._messages.append(validated_message)

    def add_messages(self, messages: List[Union[ChatMessage, Dict[str, Any]]], **kwargs) -> None:
        """
        Add multiple messages to history

        Args:
            messages: List of messages to add
            **kwargs: Additional parameters
        """
        validated_messages = []
        for msg in messages:
            validated_msg = self._validate_message(msg)
            validated_messages.append(validated_msg)

        # Check capacity
        self._check_capacity(validated_messages)

        # Add messages
        self._messages.extend(validated_messages)

    def get_memory(self, **kwargs) -> MemoryResult:
        """
        Get current memory state

        Args:
            **kwargs: Additional parameters

        Returns:
            Current memory result
        """
        # Apply token limit if configured
        messages = self._apply_token_limit(self._messages)

        # Count tokens
        token_count = self._count_total_tokens(messages)

        return MemoryResult(
            messages=messages.copy(),
            summary=None,
            metadata={
                "message_count": len(messages),
                "total_messages": len(self._messages),
                "has_token_limit": self.config.max_tokens is not None,
                "has_message_limit": self.config.max_messages is not None
            },
            token_count=token_count
        )

    def get_context(self, **kwargs) -> MemoryContext:
        """
        Get conversation context

        Args:
            **kwargs: Additional parameters

        Returns:
            Memory context
        """
        memory_result = self.get_memory(**kwargs)

        return MemoryContext(
            history=memory_result.messages,
            summary=memory_result.summary,
            context_window=memory_result.messages if self.config.return_messages else [],
            total_tokens=memory_result.token_count
        )

    def clear(self, **kwargs) -> None:
        """
        Clear all messages

        Args:
            **kwargs: Additional parameters
        """
        self._messages.clear()

    def search(self, query: str, **kwargs) -> List[MemorySearchResult]:
        """
        Search messages by content

        Args:
            query: Search query
            **kwargs: Additional parameters

        Returns:
            List of search results
        """
        results = []
        query_lower = query.lower()

        for message in self._messages:
            # Simple text search
            if query_lower in message.content.lower():
                # Simple relevance scoring based on position and content match
                score = self._calculate_relevance_score(query, message)

                result = MemorySearchResult(
                    messages=[message],
                    score=score,
                    query=query,
                    metadata={
                        "message_role": message.role,
                        "timestamp": message.timestamp,
                        "search_type": "content_match"
                    }
                )
                results.append(result)

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def get_messages_by_role(self, role: str) -> List[ChatMessage]:
        """
        Get messages by role

        Args:
            role: Message role to filter by

        Returns:
            List of messages with specified role
        """
        return [msg for msg in self._messages if msg.role == role]

    def get_last_n_messages(self, n: int) -> List[ChatMessage]:
        """
        Get last n messages

        Args:
            n: Number of messages to get

        Returns:
            List of last n messages
        """
        return self._messages[-n:] if n > 0 else []

    def get_messages_after_timestamp(self, timestamp: datetime) -> List[ChatMessage]:
        """
        Get messages after specified timestamp

        Args:
            timestamp: Timestamp to filter by

        Returns:
            List of messages after timestamp
        """
        return [msg for msg in self._messages if msg.timestamp > timestamp]

    def remove_message(self, index: int) -> ChatMessage:
        """
        Remove message by index

        Args:
            index: Index of message to remove

        Returns:
            Removed message

        Raises:
            IndexError: If index is out of range
        """
        if 0 <= index < len(self._messages):
            return self._messages.pop(index)
        else:
            raise IndexError(f"Message index {index} out of range")

    def get_message_count(self) -> int:
        """
        Get total message count

        Returns:
            Total number of messages
        """
        return len(self._messages)

    def _apply_token_limit(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        Apply token limit to messages

        Args:
            messages: Messages to apply limit to

        Returns:
            Messages after applying token limit
        """
        if self.config.max_tokens is None:
            return messages

        # Keep removing oldest messages until under token limit
        result_messages = messages.copy()
        total_tokens = self._count_total_tokens(result_messages)

        while total_tokens > self.config.max_tokens and len(result_messages) > 1:
            # Remove oldest message
            removed = result_messages.pop(0)
            total_tokens -= self._count_message_tokens(removed)

        return result_messages

    def _calculate_relevance_score(self, query: str, message: ChatMessage) -> float:
        """
        Calculate relevance score for search result

        Args:
            query: Search query
            message: Message to score

        Returns:
            Relevance score (0.0 to 1.0)
        """
        query_lower = query.lower()
        content_lower = message.content.lower()

        # Exact match gets highest score
        if query_lower == content_lower:
            return 1.0

        # Count occurrences of query terms
        query_words = query_lower.split()
        content_words = content_lower.split()

        matches = 0
        for q_word in query_words:
            for c_word in content_words:
                if q_word in c_word or c_word in q_word:
                    matches += 1
                    break

        # Calculate score based on match ratio and recency
        match_ratio = matches / max(len(query_words), 1)

        # Boost score for more recent messages
        now = datetime.now()
        hours_ago = (now - message.timestamp).total_seconds() / 3600
        recency_boost = max(0.1, 1.0 - (hours_ago / (24 * 7)))  # Decay over a week

        return min(1.0, match_ratio * recency_boost)

    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get conversation summary statistics

        Returns:
            Dictionary with conversation statistics
        """
        if not self._messages:
            return {
                "total_messages": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "system_messages": 0,
                "total_tokens": 0,
                "first_message_time": None,
                "last_message_time": None
            }

        role_counts = {}
        total_tokens = 0
        timestamps = []

        for message in self._messages:
            role_counts[message.role] = role_counts.get(message.role, 0) + 1
            total_tokens += self._count_message_tokens(message)
            timestamps.append(message.timestamp)

        return {
            "total_messages": len(self._messages),
            "role_counts": role_counts,
            "total_tokens": total_tokens,
            "first_message_time": min(timestamps),
            "last_message_time": max(timestamps),
            "conversation_duration": (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) > 1 else 0
        }