# -*- coding: utf-8 -*-
"""
Conversation summary memory implementation
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from my_langchain.memory.buffer_memory import ConversationBufferMemory
from my_langchain.memory.types import (
    MemoryConfig, MemoryResult, MemoryContext, MemorySearchResult,
    ChatMessage, MemoryError, MemoryValidationError
)


class ConversationSummaryMemory(ConversationBufferMemory):
    """
    Advanced conversation memory with automatic summarization

    Provides intelligent conversation summarization using LLM
    to maintain long-term context while respecting token limits.
    """

    def __init__(self, config: Optional[MemoryConfig] = None, llm=None, **kwargs):
        """
        Initialize conversation summary memory

        Args:
            config: Memory configuration
            llm: LLM instance for summarization
            **kwargs: Additional parameters
        """
        super().__init__(config=config, **kwargs)
        self._llm = llm
        self._summary_prompt_template = (
            "Create a concise summary of the following conversation, "
            "preserving key information and context:\n\n{conversation}\n\n"
            "Summary:"
        )
        self._full_summary = ""
        self._recent_messages: List[ChatMessage] = []
        self._last_summary_index = 0

    def get_summary(self) -> Optional[str]:
        """
        Get full conversation summary

        Returns:
            Full conversation summary or None
        """
        return self._full_summary if self._full_summary else None

    def add_message(self, message: Union[ChatMessage, Dict[str, Any]], **kwargs) -> None:
        """
        Add a message and potentially update summary

        Args:
            message: Message to add
            **kwargs: Additional parameters
        """
        validated_message = self._validate_message(message)

        # Check capacity
        self._check_capacity([validated_message])

        # Add to main storage
        self._messages.append(validated_message)

        # Add to recent messages
        self._recent_messages.append(validated_message)

        # Check if we should summarize
        if self._should_trigger_summarization():
            self._update_summary()

    def add_messages(self, messages: List[Union[ChatMessage, Dict[str, Any]]], **kwargs) -> None:
        """
        Add multiple messages and potentially update summary

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
        self._recent_messages.extend(validated_messages)

        # Check if we should summarize
        if self._should_trigger_summarization():
            self._update_summary()

    def get_memory(self, **kwargs) -> MemoryResult:
        """
        Get current memory state with summary

        Args:
            **kwargs: Additional parameters

        Returns:
            Current memory result with summary
        """
        # Combine full summary with recent messages
        messages = self._get_summary_context()

        # Count tokens
        token_count = self._count_total_tokens(messages)

        return MemoryResult(
            messages=messages.copy(),
            summary=self._full_summary,
            metadata={
                "message_count": len(messages),
                "total_messages": len(self._messages),
                "recent_messages": len(self._recent_messages),
                "buffer_type": "conversation_summary",
                "full_summary_length": len(self._full_summary),
                "last_summary_index": self._last_summary_index,
                "has_llm": self._llm is not None
            },
            token_count=token_count
        )

    def get_context(self, **kwargs) -> MemoryContext:
        """
        Get conversation context with summary

        Args:
            **kwargs: Additional parameters

        Returns:
            Memory context with summary
        """
        memory_result = self.get_memory(**kwargs)

        # Create context window (summary + recent messages)
        context_window = memory_result.messages

        return MemoryContext(
            history=memory_result.messages,
            summary=memory_result.summary,
            context_window=context_window,
            total_tokens=memory_result.token_count
        )

    def force_summarization(self) -> Optional[str]:
        """
        Force immediate summarization of recent messages

        Returns:
            New summary or None if no messages to summarize
        """
        if self._recent_messages:
            return self._update_summary()
        return None

    def clear_summary(self) -> None:
        """Clear full summary and reset"""
        self._full_summary = ""
        self._recent_messages = []
        self._last_summary_index = 0

    def set_llm(self, llm) -> None:
        """
        Set LLM for summarization

        Args:
            llm: LLM instance
        """
        self._llm = llm

    def get_summarization_stats(self) -> Dict[str, Any]:
        """
        Get summarization statistics

        Returns:
            Dictionary with summarization statistics
        """
        return {
            "full_summary_length": len(self._full_summary),
            "recent_messages_count": len(self._recent_messages),
            "last_summary_index": self._last_summary_index,
            "total_messages": len(self._messages),
            "summarization_ratio": len(self._recent_messages) / max(len(self._messages), 1),
            "should_summarize": self._should_trigger_summarization(),
            "has_llm": self._llm is not None
        }

    def _should_trigger_summarization(self) -> bool:
        """
        Check if summarization should be triggered

        Returns:
            True if summarization should be triggered
        """
        if not self._recent_messages:
            return False

        # Check message count
        if len(self._recent_messages) >= self.config.summary_frequency:
            return True

        # Check token count
        if self.config.max_tokens:
            current_tokens = self._count_total_tokens(self._recent_messages)
            if current_tokens > self.config.max_tokens * 0.7:  # 70% threshold
                return True

        return False

    def _update_summary(self) -> Optional[str]:
        """
        Update summary with recent messages

        Returns:
            New summary or None if summarization failed
        """
        if not self._recent_messages:
            return None

        try:
            new_summary = self._create_summary()
            if new_summary:
                # Combine with existing summary
                if self._full_summary:
                    self._full_summary = self._combine_summaries(self._full_summary, new_summary)
                else:
                    self._full_summary = new_summary

                # Clear recent messages and update index
                self._last_summary_index = len(self._messages)
                self._recent_messages = []

                return new_summary
        except Exception as e:
            # Log error but don't crash
            print(f"Summarization failed: {e}")

        return None

    def _create_summary(self) -> Optional[str]:
        """
        Create summary of recent messages

        Returns:
            Summary string or None if failed
        """
        if not self._recent_messages:
            return None

        # Format conversation for summarization
        conversation_text = self._format_conversation_for_summary()

        if self._llm:
            # Use LLM for summarization
            try:
                prompt = self._summary_prompt_template.format(conversation=conversation_text)
                summary = self._llm.invoke(prompt)
                return summary.strip() if summary else None
            except Exception:
                # Fallback to simple summarization
                pass

        # Simple rule-based summarization
        return self._create_simple_summary()

    def _format_conversation_for_summary(self) -> str:
        """
        Format recent messages for summarization

        Returns:
            Formatted conversation string
        """
        lines = []
        for message in self._recent_messages:
            timestamp = message.timestamp.strftime("%H:%M")
            lines.append(f"[{timestamp}] {message.role}: {message.content}")
        return "\n".join(lines)

    def _create_simple_summary(self) -> str:
        """
        Create simple rule-based summary

        Returns:
            Simple summary string
        """
        user_messages = [msg for msg in self._recent_messages if msg.role == "user"]
        assistant_messages = [msg for msg in self._recent_messages if msg.role == "assistant"]

        summary_parts = []

        # Count exchanges
        exchanges = min(len(user_messages), len(assistant_messages))
        if exchanges > 0:
            summary_parts.append(f"{exchanges} conversation exchanges")

        # Extract key topics from user messages
        if user_messages:
            topics = self._extract_topics(user_messages)
            if topics:
                summary_parts.append(f"topics: {', '.join(topics[:3])}")  # Limit to 3 topics

        # Time range
        if len(self._recent_messages) > 1:
            first_time = min(msg.timestamp for msg in self._recent_messages)
            last_time = max(msg.timestamp for msg in self._recent_messages)
            duration = (last_time - first_time).total_seconds() / 60  # minutes
            if duration > 1:
                summary_parts.append(f"spanning {duration:.0f} minutes")

        return ". ".join(summary_parts) if summary_parts else "Brief conversation"

    def _extract_topics(self, messages: List[ChatMessage]) -> List[str]:
        """
        Extract key topics from messages

        Args:
            messages: Messages to extract topics from

        Returns:
            List of topics
        """
        topics = []
        for message in messages:
            # Simple topic extraction (can be enhanced)
            words = message.content.lower().split()
            # Filter for meaningful words
            meaningful_words = [
                w for w in words
                if len(w) > 4 and w.isalpha() and w not in
                ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'said']
            ]
            topics.extend(meaningful_words[:2])  # Take 2 words per message

        # Count frequency and return most common
        from collections import Counter
        if topics:
            counter = Counter(topics)
            return [topic for topic, _ in counter.most_common(5)]

        return []

    def _combine_summaries(self, existing_summary: str, new_summary: str) -> str:
        """
        Combine existing summary with new summary

        Args:
            existing_summary: Existing summary
            new_summary: New summary to add

        Returns:
            Combined summary
        """
        # Simple combination (can be enhanced with LLM)
        if len(existing_summary) + len(new_summary) < 500:  # Keep it concise
            return f"{existing_summary}. {new_summary}"
        else:
            # If getting too long, prioritize recent information
            return new_summary

    def _get_summary_context(self) -> List[ChatMessage]:
        """
        Get messages for context (summary + recent)

        Returns:
            List of messages for context
        """
        messages = []

        # Add summary as system message if available
        if self._full_summary:
            summary_message = ChatMessage(
                role="system",
                content=f"Conversation summary: {self._full_summary}",
                timestamp=datetime.now(),
                metadata={"type": "full_summary"}
            )
            messages.append(summary_message)

        # Add recent messages
        if self.config.return_messages:
            messages.extend(self._recent_messages)

        return messages

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get comprehensive memory information

        Returns:
            Memory information dictionary
        """
        base_info = super().get_memory_info()

        summary_info = {
            "full_summary": self._full_summary,
            "recent_messages_count": len(self._recent_messages),
            "last_summary_index": self._last_summary_index,
            "summarization_stats": self.get_summarization_stats()
        }

        base_info.update(summary_info)
        return base_info