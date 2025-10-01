# -*- coding: utf-8 -*-
"""
Conversation buffer memory implementation
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from my_langchain.memory.chat_message_history import ChatMessageHistory
from my_langchain.memory.types import (
    MemoryConfig, MemoryResult, MemoryContext, MemorySearchResult,
    ChatMessage, MemoryError, MemoryValidationError
)


class ConversationBufferMemory(ChatMessageHistory):
    """
    Enhanced conversation memory with buffer management

    Extends ChatMessageHistory with additional buffer management
    features like automatic pruning and context window management.
    """

    def __init__(self, config: Optional[MemoryConfig] = None, **kwargs):
        """
        Initialize conversation buffer memory

        Args:
            config: Memory configuration
            **kwargs: Additional parameters
        """
        super().__init__(config=config, **kwargs)
        self._context_buffer: List[ChatMessage] = []
        self._summary_buffer: Optional[str] = None

    def add_message(self, message: Union[ChatMessage, Dict[str, Any]], **kwargs) -> None:
        """
        Add a message to buffer

        Args:
            message: Message to add
            **kwargs: Additional parameters
        """
        validated_message = self._validate_message(message)

        # Check capacity
        self._check_capacity([validated_message])

        # Add to main storage
        self._messages.append(validated_message)

        # Update context buffer
        self._update_context_buffer()

    def add_messages(self, messages: List[Union[ChatMessage, Dict[str, Any]]], **kwargs) -> None:
        """
        Add multiple messages to buffer

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

        # Update context buffer
        self._update_context_buffer()

    def get_memory(self, **kwargs) -> MemoryResult:
        """
        Get current memory state with buffer management

        Args:
            **kwargs: Additional parameters

        Returns:
            Current memory result
        """
        # Use context buffer instead of raw messages
        messages = self._get_buffered_messages()

        # Count tokens
        token_count = self._count_total_tokens(messages)

        return MemoryResult(
            messages=messages.copy(),
            summary=self._summary_buffer,
            metadata={
                "message_count": len(messages),
                "total_messages": len(self._messages),
                "buffer_type": "conversation_buffer",
                "has_summary": self._summary_buffer is not None,
                "has_token_limit": self.config.max_tokens is not None,
                "has_message_limit": self.config.max_messages is not None
            },
            token_count=token_count
        )

    def get_context(self, **kwargs) -> MemoryContext:
        """
        Get optimized conversation context

        Args:
            **kwargs: Additional parameters

        Returns:
            Memory context with optimized window
        """
        memory_result = self.get_memory(**kwargs)

        # Create context window based on configuration
        context_window = self._create_context_window(memory_result.messages)

        return MemoryContext(
            history=memory_result.messages,
            summary=memory_result.summary,
            context_window=context_window,
            total_tokens=memory_result.token_count
        )

    def set_summary(self, summary: str) -> None:
        """
        Set conversation summary

        Args:
            summary: Conversation summary
        """
        self._summary_buffer = summary

    def get_summary(self) -> Optional[str]:
        """
        Get conversation summary

        Returns:
            Current summary or None
        """
        return self._summary_buffer

    def clear_summary(self) -> None:
        """Clear conversation summary"""
        self._summary_buffer = None

    def prune_old_messages(self, keep_count: int) -> int:
        """
        Prune old messages to keep only the most recent ones

        Args:
            keep_count: Number of messages to keep

        Returns:
            Number of messages removed
        """
        if keep_count >= len(self._messages):
            return 0

        old_count = len(self._messages)
        self._messages = self._messages[-keep_count:]
        self._update_context_buffer()

        return old_count - len(self._messages)

    def get_context_window_messages(self, window_size: Optional[int] = None) -> List[ChatMessage]:
        """
        Get messages for context window

        Args:
            window_size: Size of context window (uses config if None)

        Returns:
            Messages for context window
        """
        if window_size is None:
            window_size = self._calculate_optimal_window_size()

        return self._get_last_n_messages_from_buffer(window_size)

    def should_summarize(self) -> bool:
        """
        Check if conversation should be summarized based on configuration

        Returns:
            True if summarization is recommended
        """
        if self.config.summary_frequency <= 0:
            return False

        return len(self._messages) >= self.config.summary_frequency

    def summarize_recent_messages(self, count: int) -> Optional[str]:
        """
        Create summary of recent messages

        Args:
            count: Number of recent messages to summarize

        Returns:
            Summary string or None if no messages
        """
        if not self._messages:
            return None

        recent_messages = self._messages[-count:] if count > 0 else self._messages

        # Simple summarization logic (can be enhanced with LLM)
        user_messages = [msg for msg in recent_messages if msg.role == "user"]
        assistant_messages = [msg for msg in recent_messages if msg.role == "assistant"]

        summary_parts = []

        if user_messages:
            topics = []
            for msg in user_messages:
                # Extract key topics (simple implementation)
                words = msg.content.lower().split()
                topics.extend([w for w in words if len(w) > 4][:3])

            if topics:
                summary_parts.append(f"Topics discussed: {', '.join(set(topics))}")

        summary_parts.append(f"Total exchanges: {len(user_messages)}")

        return " | ".join(summary_parts)

    def _update_context_buffer(self) -> None:
        """Update the context buffer based on current messages"""
        self._context_buffer = self._apply_token_limit(self._messages)

    def _get_buffered_messages(self) -> List[ChatMessage]:
        """
        Get messages from context buffer

        Returns:
            Buffered messages
        """
        return self._context_buffer.copy()

    def _create_context_window(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        Create optimized context window from messages

        Args:
            messages: Messages to create window from

        Returns:
            Context window messages
        """
        if not self.config.return_messages:
            return []

        # If we have a summary, include it and fewer messages
        if self._summary_buffer:
            # Create a system message with summary
            summary_message = ChatMessage(
                role="system",
                content=f"Previous conversation summary: {self._summary_buffer}",
                timestamp=datetime.now(),
                metadata={"type": "summary"}
            )

            # Return summary + recent messages
            window_size = self._calculate_optimal_window_size(with_summary=True)
            recent_messages = self._get_last_n_messages_from_buffer(window_size - 1)

            return [summary_message] + recent_messages
        else:
            # Return recent messages only
            window_size = self._calculate_optimal_window_size()
            return self._get_last_n_messages_from_buffer(window_size)

    def _calculate_optimal_window_size(self, with_summary: bool = False) -> int:
        """
        Calculate optimal context window size

        Args:
            with_summary: Whether including summary in window

        Returns:
            Optimal window size
        """
        # Default window size
        default_size = 10

        # Adjust for token limits
        if self.config.max_tokens:
            # Estimate average tokens per message
            if self._messages:
                avg_tokens = self._count_total_tokens(self._messages) / len(self._messages)
                estimated_size = int(self.config.max_tokens / (avg_tokens * 1.5))  # 1.5x safety margin
                default_size = min(default_size, max(1, estimated_size))

        # Adjust if we have a summary (can fit more messages)
        if with_summary and self._summary_buffer:
            default_size = max(default_size - 2, 3)  # Reserve space for summary

        # Apply message limit if configured
        if self.config.max_messages:
            default_size = min(default_size, self.config.max_messages)

        return default_size

    def _get_last_n_messages_from_buffer(self, n: int) -> List[ChatMessage]:
        """
        Get last n messages from buffer

        Args:
            n: Number of messages to get

        Returns:
            Last n messages from buffer
        """
        if n <= 0:
            return []

        return self._context_buffer[-n:] if self._context_buffer else []

    def get_buffer_info(self) -> Dict[str, Any]:
        """
        Get buffer information

        Returns:
            Dictionary with buffer statistics
        """
        base_info = self.get_conversation_summary()

        buffer_info = {
            "buffer_size": len(self._context_buffer),
            "has_summary": self._summary_buffer is not None,
            "summary_length": len(self._summary_buffer) if self._summary_buffer else 0,
            "should_summarize": self.should_summarize(),
            "optimal_window_size": self._calculate_optimal_window_size(),
            "buffer_utilization": len(self._context_buffer) / max(self.config.max_messages or 10, 1)
        }

        base_info.update(buffer_info)
        return base_info