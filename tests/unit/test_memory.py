# -*- coding: utf-8 -*-
"""
Unit tests for memory module
"""
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from my_langchain.memory import (
    ChatMessageHistory, ConversationBufferMemory, ConversationSummaryMemory
)
from my_langchain.memory.types import (
    MemoryConfig, ChatMessage, MemoryResult, MemoryContext,
    MemoryError, MemoryValidationError, MemoryCapacityError
)


class TestMemoryConfig:
    """Test MemoryConfig class"""

    def test_default_config(self):
        """Test default configuration"""
        config = MemoryConfig()
        assert config.max_messages is None
        assert config.max_tokens is None
        assert config.return_messages is True
        assert config.summary_frequency == 10
        assert config.llm_config is None

    def test_custom_config(self):
        """Test custom configuration"""
        config = MemoryConfig(
            max_messages=5,
            max_tokens=100,
            return_messages=False,
            summary_frequency=5,
            llm_config={"temperature": 0.0}
        )
        assert config.max_messages == 5
        assert config.max_tokens == 100
        assert config.return_messages is False
        assert config.summary_frequency == 5
        assert config.llm_config == {"temperature": 0.0}


class TestChatMessage:
    """Test ChatMessage class"""

    def test_message_creation(self):
        """Test message creation"""
        message = ChatMessage(
            role="user",
            content="Hello, world!"
        )
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert isinstance(message.timestamp, datetime)
        assert message.metadata == {}

    def test_message_with_metadata(self):
        """Test message with metadata"""
        metadata = {"source": "test", "importance": 0.8}
        message = ChatMessage(
            role="assistant",
            content="I can help you!",
            metadata=metadata
        )
        assert message.metadata == metadata

    def test_message_validation(self):
        """Test message validation"""
        # Valid message
        message = ChatMessage(role="user", content="test")
        assert message is not None

        # Empty content should be allowed
        message = ChatMessage(role="user", content="")
        assert message.content == ""

    def test_message_serialization(self):
        """Test message serialization"""
        message = ChatMessage(
            role="user",
            content="Test message",
            metadata={"key": "value"}
        )

        data = message.model_dump()
        assert data["role"] == "user"
        assert data["content"] == "Test message"
        assert "timestamp" in data
        assert data["metadata"]["key"] == "value"


class TestChatMessageHistory:
    """Test ChatMessageHistory class"""

    def test_history_creation(self):
        """Test history creation"""
        history = ChatMessageHistory()
        assert history.get_message_count() == 0
        assert isinstance(history.config, MemoryConfig)

    def test_history_with_config(self):
        """Test history with custom config"""
        config = MemoryConfig(max_messages=2)
        history = ChatMessageHistory(config=config)
        assert history.config.max_messages == 2

    def test_add_message(self):
        """Test adding single message"""
        history = ChatMessageHistory()

        message = ChatMessage(role="user", content="Hello")
        history.add_message(message)

        assert history.get_message_count() == 1
        memory = history.get_memory()
        assert len(memory.messages) == 1
        assert memory.messages[0].content == "Hello"

    def test_add_message_dict(self):
        """Test adding message as dictionary"""
        history = ChatMessageHistory()

        message_dict = {
            "role": "assistant",
            "content": "Hi there!",
            "metadata": {"source": "test"}
        }
        history.add_message(message_dict)

        assert history.get_message_count() == 1
        memory = history.get_memory()
        assert memory.messages[0].role == "assistant"
        assert memory.messages[0].metadata["source"] == "test"

    def test_add_multiple_messages(self):
        """Test adding multiple messages"""
        history = ChatMessageHistory()

        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi!"),
            {"role": "user", "content": "How are you?"}
        ]

        history.add_messages(messages)

        assert history.get_message_count() == 3
        memory = history.get_memory()
        assert len(memory.messages) == 3

    def test_message_validation(self):
        """Test message validation"""
        history = ChatMessageHistory()

        # Invalid message type
        with pytest.raises(MemoryValidationError):
            history.add_message("invalid message")

        # Invalid message dict
        with pytest.raises(MemoryValidationError):
            history.add_message({"invalid": "dict"})

    def test_max_messages_limit(self):
        """Test max messages limit"""
        config = MemoryConfig(max_messages=2)
        history = ChatMessageHistory(config=config)

        # Add messages up to limit
        history.add_message(ChatMessage(role="user", content="1"))
        history.add_message(ChatMessage(role="assistant", content="2"))
        assert history.get_message_count() == 2

        # Adding one more should raise error
        with pytest.raises(MemoryCapacityError):
            history.add_message(ChatMessage(role="user", content="3"))

    def test_clear_history(self):
        """Test clearing history"""
        history = ChatMessageHistory()
        history.add_message(ChatMessage(role="user", content="Hello"))
        assert history.get_message_count() == 1

        history.clear()
        assert history.get_message_count() == 0

    def test_search_messages(self):
        """Test message search"""
        history = ChatMessageHistory()

        # Add some messages
        history.add_message(ChatMessage(role="user", content="What is Python?"))
        history.add_message(ChatMessage(role="assistant", content="Python is a programming language"))
        history.add_message(ChatMessage(role="user", content="Tell me more"))

        # Search
        results = history.search("Python")
        assert len(results) == 2
        assert "Python" in results[0].messages[0].content or "Python" in results[1].messages[0].content

    def test_get_messages_by_role(self):
        """Test getting messages by role"""
        history = ChatMessageHistory()

        history.add_message(ChatMessage(role="user", content="User 1"))
        history.add_message(ChatMessage(role="assistant", content="Assistant 1"))
        history.add_message(ChatMessage(role="user", content="User 2"))

        user_messages = history.get_messages_by_role("user")
        assert len(user_messages) == 2

        assistant_messages = history.get_messages_by_role("assistant")
        assert len(assistant_messages) == 1

    def test_get_last_n_messages(self):
        """Test getting last n messages"""
        history = ChatMessageHistory()

        for i in range(5):
            history.add_message(ChatMessage(role="user", content=f"Message {i}"))

        last_3 = history.get_last_n_messages(3)
        assert len(last_3) == 3
        assert last_3[0].content == "Message 2"
        assert last_3[2].content == "Message 4"

    def test_get_messages_after_timestamp(self):
        """Test getting messages after timestamp"""
        history = ChatMessageHistory()
        base_time = datetime.now()

        # Add messages with delays
        history.add_message(ChatMessage(role="user", content="First"))
        history.add_message(ChatMessage(role="assistant", content="Second"))

        # Get messages after base_time
        later_messages = history.get_messages_after_timestamp(base_time)
        assert len(later_messages) == 2

    def test_remove_message(self):
        """Test removing message by index"""
        history = ChatMessageHistory()

        history.add_message(ChatMessage(role="user", content="Message 1"))
        history.add_message(ChatMessage(role="assistant", content="Message 2"))
        history.add_message(ChatMessage(role="user", content="Message 3"))

        # Remove middle message
        removed = history.remove_message(1)
        assert removed.content == "Message 2"
        assert history.get_message_count() == 2

        remaining_messages = history.get_memory().messages
        assert remaining_messages[0].content == "Message 1"
        assert remaining_messages[1].content == "Message 3"

    def test_remove_message_invalid_index(self):
        """Test removing message with invalid index"""
        history = ChatMessageHistory()

        with pytest.raises(IndexError):
            history.remove_message(0)

    def test_conversation_summary(self):
        """Test conversation summary"""
        history = ChatMessageHistory()

        # Empty history
        summary = history.get_conversation_summary()
        assert summary["total_messages"] == 0
        assert summary["total_tokens"] == 0

        # Add some messages
        history.add_message(ChatMessage(role="user", content="Hello"))
        history.add_message(ChatMessage(role="assistant", content="Hi there!"))

        summary = history.get_conversation_summary()
        assert summary["total_messages"] == 2
        assert summary["role_counts"]["user"] == 1
        assert summary["role_counts"]["assistant"] == 1
        assert summary["total_tokens"] > 0

    def test_get_memory_and_context(self):
        """Test getting memory and context"""
        history = ChatMessageHistory()

        history.add_message(ChatMessage(role="user", content="Hello"))
        history.add_message(ChatMessage(role="assistant", content="Hi!"))

        # Test get_memory
        memory = history.get_memory()
        assert isinstance(memory, MemoryResult)
        assert len(memory.messages) == 2
        assert memory.token_count > 0
        assert memory.summary is None

        # Test get_context
        context = history.get_context()
        assert isinstance(context, MemoryContext)
        assert len(context.history) == 2
        assert len(context.context_window) == 2
        assert context.total_tokens > 0

    def test_token_limit(self):
        """Test token limit functionality"""
        config = MemoryConfig(max_tokens=50)  # Very small limit
        history = ChatMessageHistory(config=config)

        # Add messages that exceed token limit
        long_message = "This is a very long message that should exceed the token limit"
        history.add_message(ChatMessage(role="user", content=long_message))
        history.add_message(ChatMessage(role="assistant", content="Another long message"))
        history.add_message(ChatMessage(role="user", content="Third message"))

        memory = history.get_memory()
        # Should apply token limit but still have some messages
        assert len(memory.messages) >= 1
        # Token count should be reasonable
        assert memory.token_count <= config.max_tokens * 1.5  # Allow some tolerance


class TestConversationBufferMemory:
    """Test ConversationBufferMemory class"""

    def test_buffer_creation(self):
        """Test buffer memory creation"""
        buffer = ConversationBufferMemory()
        assert buffer.get_message_count() == 0
        assert buffer.get_summary() is None

    def test_buffer_with_config(self):
        """Test buffer with custom config"""
        config = MemoryConfig(max_messages=3, summary_frequency=2)
        buffer = ConversationBufferMemory(config=config)
        assert buffer.config.max_messages == 3
        assert buffer.config.summary_frequency == 2

    def test_add_message_buffer_update(self):
        """Test that adding messages updates buffer"""
        buffer = ConversationBufferMemory()

        buffer.add_message(ChatMessage(role="user", content="Hello"))

        memory = buffer.get_memory()
        assert len(memory.messages) == 1
        assert memory.metadata["buffer_type"] == "conversation_buffer"

    def test_set_and_get_summary(self):
        """Test setting and getting summary"""
        buffer = ConversationBufferMemory()

        summary = "The user asked about AI and I provided information"
        buffer.set_summary(summary)

        assert buffer.get_summary() == summary

        memory = buffer.get_memory()
        assert memory.summary == summary
        assert memory.metadata["has_summary"] is True

    def test_clear_summary(self):
        """Test clearing summary"""
        buffer = ConversationBufferMemory()
        buffer.set_summary("Some summary")
        assert buffer.get_summary() is not None

        buffer.clear_summary()
        assert buffer.get_summary() is None

    def test_prune_old_messages(self):
        """Test pruning old messages"""
        buffer = ConversationBufferMemory()

        # Add 5 messages
        for i in range(5):
            buffer.add_message(ChatMessage(role="user", content=f"Message {i}"))

        assert buffer.get_message_count() == 5

        # Keep only last 2
        removed_count = buffer.prune_old_messages(2)
        assert removed_count == 3
        assert buffer.get_message_count() == 2

        remaining = buffer.get_memory().messages
        assert remaining[0].content == "Message 3"
        assert remaining[1].content == "Message 4"

    def test_get_context_window_messages(self):
        """Test getting context window messages"""
        buffer = ConversationBufferMemory()

        # Add 10 messages
        for i in range(10):
            buffer.add_message(ChatMessage(role="user", content=f"Message {i}"))

        # Get 5 messages for context window
        context_messages = buffer.get_context_window_messages(5)
        assert len(context_messages) == 5
        assert context_messages[0].content == "Message 5"  # Last 5 messages

    def test_should_summarize(self):
        """Test summarization trigger"""
        config = MemoryConfig(summary_frequency=3)
        buffer = ConversationBufferMemory(config=config)

        # Should not summarize yet
        assert buffer.should_summarize() is False

        # Add 2 messages
        buffer.add_message(ChatMessage(role="user", content="1"))
        buffer.add_message(ChatMessage(role="assistant", content="2"))
        assert buffer.should_summarize() is False

        # Add 3rd message
        buffer.add_message(ChatMessage(role="user", content="3"))
        assert buffer.should_summarize() is True

    def test_summarize_recent_messages(self):
        """Test summarizing recent messages"""
        buffer = ConversationBufferMemory()

        # Add messages
        buffer.add_message(ChatMessage(role="user", content="What is machine learning?"))
        buffer.add_message(ChatMessage(role="assistant", content="Machine learning is a subset of AI"))

        summary = buffer.summarize_recent_messages(2)
        assert summary is not None
        assert len(summary) > 0
        assert "topics:" in summary.lower() or "exchanges" in summary.lower()

    def test_context_creation_with_summary(self):
        """Test context creation with summary"""
        buffer = ConversationBufferMemory()

        # Set summary and add messages
        buffer.set_summary("Previous discussion about AI")
        buffer.add_message(ChatMessage(role="user", content="Tell me more"))

        context = buffer.get_context()
        assert context.summary == "Previous discussion about AI"
        assert len(context.context_window) > 0

    def test_get_buffer_info(self):
        """Test getting buffer information"""
        buffer = ConversationBufferMemory()

        # Add some data
        buffer.add_message(ChatMessage(role="user", content="Hello"))
        buffer.set_summary("Test summary")

        info = buffer.get_buffer_info()
        assert info["buffer_size"] == 1
        assert info["has_summary"] is True
        assert info["summary_length"] > 0
        assert "optimal_window_size" in info
        assert "buffer_utilization" in info


class TestConversationSummaryMemory:
    """Test ConversationSummaryMemory class"""

    def test_summary_memory_creation(self):
        """Test summary memory creation"""
        memory = ConversationSummaryMemory()
        assert memory.get_message_count() == 0
        assert memory.get_summary() is None
        assert len(memory._recent_messages) == 0

    def test_summary_memory_with_llm(self):
        """Test summary memory with LLM"""
        from my_langchain.llms import MockLLM

        llm = MockLLM(temperature=0.0)
        memory = ConversationSummaryMemory(llm=llm)

        assert memory._llm is not None

    def test_add_message_triggers_summarization(self):
        """Test that adding messages can trigger summarization"""
        config = MemoryConfig(summary_frequency=2)
        memory = ConversationSummaryMemory(config=config)

        # Add first message - should not summarize yet
        memory.add_message(ChatMessage(role="user", content="Hello"))
        assert len(memory._recent_messages) == 1
        assert memory._full_summary == ""

        # Add second message - should trigger summarization
        memory.add_message(ChatMessage(role="assistant", content="Hi there!"))
        # Recent messages should be cleared after summarization
        assert len(memory._recent_messages) == 0

    def test_force_summarization(self):
        """Test forcing summarization"""
        memory = ConversationSummaryMemory()

        memory.add_message(ChatMessage(role="user", content="Hello"))
        memory.add_message(ChatMessage(role="assistant", content="Hi!"))

        # Force summarization
        summary = memory.force_summarization()
        assert summary is not None
        assert len(memory._recent_messages) == 0
        assert len(memory._full_summary) > 0

    def test_clear_summary(self):
        """Test clearing summary"""
        memory = ConversationSummaryMemory()

        # Add messages and create summary
        memory.add_message(ChatMessage(role="user", content="Hello"))
        memory.add_message(ChatMessage(role="assistant", content="Hi!"))
        memory.force_summarization()

        assert len(memory._full_summary) > 0

        # Clear summary
        memory.clear_summary()
        assert memory._full_summary == ""
        assert memory._last_summary_index == 0

    def test_set_llm(self):
        """Test setting LLM"""
        from my_langchain.llms import MockLLM

        memory = ConversationSummaryMemory()
        llm = MockLLM(temperature=0.5)

        memory.set_llm(llm)
        assert memory._llm is not None
        assert memory._llm.temperature == 0.5

    def test_get_summarization_stats(self):
        """Test getting summarization statistics"""
        config = MemoryConfig(summary_frequency=3)
        memory = ConversationSummaryMemory(config=config)

        # Add some messages
        memory.add_message(ChatMessage(role="user", content="Hello"))
        memory.add_message(ChatMessage(role="assistant", content="Hi!"))

        stats = memory.get_summarization_stats()
        assert stats["recent_messages_count"] == 2
        assert stats["total_messages"] == 2
        assert stats["full_summary_length"] == 0
        # With 2 messages and frequency of 3, should not summarize yet
        assert stats["should_summarize"] is False

    def test_get_memory_with_summary(self):
        """Test getting memory with summary"""
        memory = ConversationSummaryMemory()

        # Add messages and summarize
        memory.add_message(ChatMessage(role="user", content="What is AI?"))
        memory.add_message(ChatMessage(role="assistant", content="AI is artificial intelligence"))
        memory.force_summarization()

        memory_result = memory.get_memory()
        assert memory_result.summary is not None
        assert len(memory_result.summary) > 0
        assert memory_result.metadata["buffer_type"] == "conversation_summary"

    def test_simple_summarization(self):
        """Test simple rule-based summarization"""
        memory = ConversationSummaryMemory()

        # Add conversation with topics
        memory.add_message(ChatMessage(role="user", content="Tell me about machine learning"))
        memory.add_message(ChatMessage(role="assistant", content="Machine learning is a subset of AI"))

        summary = memory.force_summarization()
        assert summary is not None
        assert len(summary) > 0

    def test_topic_extraction(self):
        """Test topic extraction from messages"""
        memory = ConversationSummaryMemory()

        messages = [
            ChatMessage(role="user", content="What is machine learning and Python programming?"),
            ChatMessage(role="assistant", content="Machine learning involves algorithms and data")
        ]

        topics = memory._extract_topics(messages)
        assert isinstance(topics, list)
        assert len(topics) > 0

    def test_combine_summaries(self):
        """Test combining summaries"""
        memory = ConversationSummaryMemory()

        existing = "User asked about machine learning"
        new = "User also inquired about neural networks"

        combined = memory._combine_summaries(existing, new)
        assert existing in combined
        assert new in combined

    def test_get_memory_info(self):
        """Test getting comprehensive memory info"""
        memory = ConversationSummaryMemory()

        # Add some data
        memory.add_message(ChatMessage(role="user", content="Hello"))
        memory.force_summarization()

        info = memory.get_memory_info()
        assert "summarization_stats" in info
        assert info["summarization_stats"]["total_messages"] == 1
        assert info["summarization_stats"]["recent_messages_count"] == 0


class TestMemoryErrorHandling:
    """Test memory error handling"""

    def test_memory_validation_error(self):
        """Test memory validation error"""
        error = MemoryValidationError("Invalid message", data={"test": "data"})
        assert error.message == "Invalid message"
        assert error.details["data"] == {"test": "data"}

    def test_memory_error_inheritance(self):
        """Test memory error inheritance"""
        error = MemoryValidationError("Test error")
        assert isinstance(error, Exception)
        assert isinstance(error, MemoryError)
        assert str(error) == "Test error"

    def test_chat_message_history_token_limit_error(self):
        """Test token limit enforcement"""
        config = MemoryConfig(max_tokens=10)  # Very small limit
        history = ChatMessageHistory(config=config)

        # Add very long message
        long_content = "This is an extremely long message that should definitely exceed the token limit"
        history.add_message(ChatMessage(role="user", content=long_content))

        # Should work
        memory = history.get_memory()
        assert len(memory.messages) >= 1

        # Add another long message - should enforce limit
        history.add_message(ChatMessage(role="assistant", content=long_content))
        memory = history.get_memory()

        # Should have applied token limit
        assert memory.token_count <= config.max_tokens or len(memory.messages) < 2


class TestMemoryIntegration:
    """Test memory integration scenarios"""

    def test_memory_config_inheritance(self):
        """Test that config is properly inherited"""
        config = MemoryConfig(max_messages=5, return_messages=False)

        histories = [
            ChatMessageHistory(config=config),
            ConversationBufferMemory(config=config),
            ConversationSummaryMemory(config=config)
        ]

        for history in histories:
            assert history.config.max_messages == 5
            assert history.config.return_messages is False

    def test_memory_with_complex_messages(self):
        """Test memory with complex messages including metadata"""
        memory = ConversationBufferMemory()

        complex_message = ChatMessage(
            role="user",
            content="Help me understand AI",
            metadata={
                "topic": "AI",
                "importance": 0.9,
                "language": "en",
                "session_id": "abc123"
            }
        )

        memory.add_message(complex_message)

        retrieved = memory.get_memory()
        assert len(retrieved.messages) == 1
        assert retrieved.messages[0].metadata["topic"] == "AI"
        assert retrieved.messages[0].metadata["importance"] == 0.9

    def test_memory_consistency_across_types(self):
        """Test consistent behavior across memory types"""
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
            ChatMessage(role="user", content="How are you?")
        ]

        memories = [
            ChatMessageHistory(),
            ConversationBufferMemory(),
            ConversationSummaryMemory()
        ]

        results = []
        for memory in memories:
            memory.add_messages(messages)
            memory_result = memory.get_memory()
            results.append(len(memory_result.messages))

        # All should have the same number of messages (SummaryMemory may differ due to summarization)
        assert results[0] == 3  # ChatMessageHistory
        assert results[1] == 3  # ConversationBufferMemory
        # ConversationSummaryMemory might have different due to internal logic
        assert results[2] >= 1  # At least some messages should remain

    def test_memory_performance_with_large_data(self):
        """Test memory performance with larger datasets"""
        import time

        memory = ChatMessageHistory()

        # Add many messages
        start_time = time.time()
        for i in range(100):
            memory.add_message(ChatMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}: This is test content"
            ))
        add_time = time.time() - start_time

        # Test retrieval performance
        start_time = time.time()
        memory_result = memory.get_memory()
        retrieval_time = time.time() - start_time

        # Assertions
        assert len(memory_result.messages) == 100
        assert add_time < 1.0  # Should be fast
        assert retrieval_time < 0.1  # Retrieval should be very fast
        assert memory_result.token_count > 0

    def test_memory_thread_safety(self):
        """Test basic thread safety (simplified test)"""
        import threading

        memory = ChatMessageHistory()
        errors = []

        def add_messages(thread_id):
            try:
                for i in range(10):
                    memory.add_message(ChatMessage(
                        role="user",
                        content=f"Thread {thread_id}, Message {i}"
                    ))
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_messages, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check for errors
        assert len(errors) == 0, f"Thread safety errors: {errors}"

        # Check that all messages were added
        final_count = memory.get_message_count()
        assert final_count == 30  # 3 threads * 10 messages each