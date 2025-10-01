# -*- coding: utf-8 -*-
"""
Integration tests for memory module
"""
import pytest
import asyncio
from datetime import datetime, timedelta

from my_langchain.memory import (
    ChatMessageHistory, ConversationBufferMemory, ConversationSummaryMemory
)
from my_langchain.memory.types import MemoryConfig, ChatMessage
from my_langchain.llms import MockLLM


class TestMemoryIntegration:
    """Integration tests for memory functionality"""

    def test_conversation_flow_integration(self):
        """Test complete conversation flow across memory types"""
        # Simulate a conversation
        conversation = [
            {"role": "user", "content": "Hello, I need help with Python programming"},
            {"role": "assistant", "content": "I'd be happy to help you with Python! What specific topic are you interested in?"},
            {"role": "user", "content": "I want to learn about data structures"},
            {"role": "assistant", "content": "Great choice! Data structures are fundamental. Let's start with lists and dictionaries."},
            {"role": "user", "content": "Can you give me an example of a list?"}
        ]

        # Test with different memory types
        memories = {
            "basic": ChatMessageHistory(),
            "buffer": ConversationBufferMemory(),
            "summary": ConversationSummaryMemory()
        }

        results = {}
        for name, memory in memories.items():
            # Add conversation
            memory.add_messages(conversation)

            # Get memory state
            memory_result = memory.get_memory()
            context = memory.get_context()

            results[name] = {
                "message_count": len(memory_result.messages),
                "token_count": memory_result.token_count,
                "has_summary": memory_result.summary is not None,
                "context_length": len(context.context_window)
            }

        # Verify all memories captured the conversation
        assert results["basic"]["message_count"] == 5
        assert results["buffer"]["message_count"] == 5
        assert results["summary"]["message_count"] >= 1  # May have summary

        # Verify token counts are reasonable
        for name, result in results.items():
            assert result["token_count"] > 0
            assert result["context_length"] > 0

    def test_memory_with_token_management(self):
        """Test memory behavior under token constraints"""
        # Create memory with small token limit
        config = MemoryConfig(max_tokens=100)
        memory = ConversationBufferMemory(config=config)

        # Add many short messages
        for i in range(10):
            memory.add_message(ChatMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"This is message number {i}"
            ))

        memory_result = memory.get_memory()

        # Should have applied token limit
        assert len(memory_result.messages) < 10
        assert memory_result.token_count <= config.max_tokens * 1.5  # Allow tolerance

    def test_memory_search_and_filtering(self):
        """Test memory search and filtering functionality"""
        memory = ChatMessageHistory()

        # Add diverse messages
        messages = [
            ChatMessage(role="user", content="What is machine learning?"),
            ChatMessage(role="assistant", content="Machine learning is a subset of AI"),
            ChatMessage(role="user", content="Tell me about neural networks"),
            ChatMessage(role="assistant", content="Neural networks mimic brain structure"),
            ChatMessage(role="user", content="How about deep learning?"),
            ChatMessage(role="assistant", content="Deep learning uses multiple layers")
        ]

        memory.add_messages(messages)

        # Test search
        ml_results = memory.search("machine learning")
        assert len(ml_results) >= 2  # Should find both user and assistant messages

        # Test role filtering
        user_messages = memory.get_messages_by_role("user")
        assistant_messages = memory.get_messages_by_role("assistant")
        assert len(user_messages) == 3
        assert len(assistant_messages) == 3

        # Test time-based filtering
        cutoff_time = datetime.now() - timedelta(minutes=1)
        recent_messages = memory.get_messages_after_timestamp(cutoff_time)
        assert len(recent_messages) == 6  # All messages should be recent

    def test_memory_with_summarization_workflow(self):
        """Test summarization workflow"""
        config = MemoryConfig(summary_frequency=3)  # Summarize every 3 messages
        llm = MockLLM(temperature=0.0)
        memory = ConversationSummaryMemory(config=config, llm=llm)

        # Add conversation that triggers summarization
        messages = [
            ChatMessage(role="user", content="I'm learning about web development"),
            ChatMessage(role="assistant", content="That's great! What aspect interests you?"),
            ChatMessage(role="user", content="Frontend development with HTML and CSS")
        ]

        memory.add_messages(messages)

        # Should have triggered summarization
        stats = memory.get_summarization_stats()
        assert stats["recent_messages_count"] == 0  # Should be cleared after summarization
        assert len(memory._full_summary) > 0

        # Add more messages
        memory.add_message(ChatMessage(
            role="assistant",
            content="HTML and CSS are fundamental for web development"
        ))

        # Should have recent messages again
        stats = memory.get_summarization_stats()
        assert stats["recent_messages_count"] == 1

    def test_memory_performance_under_load(self):
        """Test memory performance with large datasets"""
        import time

        memory = ChatMessageHistory()

        # Performance test setup
        num_messages = 1000
        start_time = time.time()

        # Add many messages
        for i in range(num_messages):
            memory.add_message(ChatMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"This is test message {i} with some content to process"
            ))

        add_time = time.time() - start_time

        # Test retrieval performance
        start_time = time.time()
        memory_result = memory.get_memory()
        retrieval_time = time.time() - start_time

        # Test search performance
        start_time = time.time()
        search_results = memory.search("test message")
        search_time = time.time() - start_time

        # Performance assertions
        assert add_time < 2.0  # Should add 1000 messages quickly
        assert retrieval_time < 0.1  # Retrieval should be fast
        assert search_time < 0.5  # Search should be reasonable
        assert len(memory_result.messages) == num_messages
        assert len(search_results) > 0

    def test_memory_configuration_integration(self):
        """Test different memory configurations"""
        configs = [
            MemoryConfig(max_messages=5),
            MemoryConfig(max_tokens=100),
            MemoryConfig(return_messages=False),
            MemoryConfig(summary_frequency=2)
        ]

        for config in configs:
            memory = ConversationBufferMemory(config=config)

            # Add test messages - handle capacity limits
            added_count = 0
            for i in range(7):
                try:
                    memory.add_message(ChatMessage(
                        role="user",
                        content=f"Test message {i}"
                    ))
                    added_count += 1
                except Exception:
                    # Expected for max_messages limit
                    break

            # Verify configuration is respected
            memory_result = memory.get_memory()

            if config.max_messages:
                assert len(memory_result.messages) <= config.max_messages
                assert added_count <= config.max_messages

            if config.max_tokens:
                # Should be reasonable token count
                assert memory_result.token_count > 0

            if not config.return_messages:
                # Context window should be empty
                context = memory.get_context()
                assert len(context.context_window) == 0

    def test_memory_error_handling_integration(self):
        """Test error handling in realistic scenarios"""
        memory = ChatMessageHistory()

        # Test with invalid messages
        with pytest.raises(Exception):  # Should handle gracefully
            memory.add_message(None)

        with pytest.raises(Exception):  # Should handle gracefully
            memory.add_message({"invalid": "message"})

        # Test capacity limits
        config = MemoryConfig(max_messages=2)
        limited_memory = ChatMessageHistory(config=config)

        limited_memory.add_message(ChatMessage(role="user", content="1"))
        limited_memory.add_message(ChatMessage(role="user", content="2"))

        # Should raise error on third message
        with pytest.raises(Exception):
            limited_memory.add_message(ChatMessage(role="user", content="3"))

    def test_memory_state_persistence_simulation(self):
        """Test memory state persistence simulation"""
        # Simulate saving and loading memory state
        original_memory = ConversationBufferMemory()

        # Add conversation
        conversation = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
            ChatMessage(role="user", content="How are you?"),
        ]
        original_memory.add_messages(conversation)

        # Get memory state (simulating serialization)
        memory_state = original_memory.get_memory()

        # Create new memory and restore state (simulating deserialization)
        restored_memory = ConversationBufferMemory()
        for message in memory_state.messages:
            restored_memory.add_message(message)

        # Verify restoration
        original_result = original_memory.get_memory()
        restored_result = restored_memory.get_memory()

        assert len(original_result.messages) == len(restored_result.messages)
        assert original_result.token_count == restored_result.token_count

    def test_memory_concurrent_access_simulation(self):
        """Test memory concurrent access simulation"""
        import threading
        import time

        memory = ChatMessageHistory()
        errors = []
        results = []

        def worker(worker_id):
            try:
                for i in range(10):
                    memory.add_message(ChatMessage(
                        role="user",
                        content=f"Worker {worker_id}, Message {i}"
                    ))
                    time.sleep(0.001)  # Small delay
                results.append(worker_id)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert memory.get_message_count() == 50  # 5 workers * 10 messages

    def test_memory_integration_with_llm(self):
        """Test memory integration with LLM"""
        llm = MockLLM(temperature=0.0)
        memory = ConversationSummaryMemory(llm=llm)

        # Simulate conversation that would benefit from summarization
        conversation = [
            ChatMessage(role="user", content="I'm working on a machine learning project"),
            ChatMessage(role="assistant", content="That sounds interesting! What kind of project?"),
            ChatMessage(role="user", content="Image classification using neural networks"),
            ChatMessage(role="assistant", content="Great choice! What dataset are you using?"),
            ChatMessage(role="user", content="CIFAR-10 dataset for object recognition"),
        ]

        memory.add_messages(conversation)

        # Force summarization
        summary = memory.force_summarization()
        assert summary is not None
        assert len(summary) > 0

        # Get memory with summary
        memory_result = memory.get_memory()
        assert memory_result.summary is not None
        assert len(memory_result.messages) >= 1

    def test_memory_context_window_management(self):
        """Test context window management"""
        config = MemoryConfig(max_tokens=200, return_messages=True)
        memory = ConversationBufferMemory(config=config)

        # Add conversation that exceeds token limit
        long_conversation = []
        for i in range(20):
            long_conversation.append(ChatMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"This is a longer message {i} with more content to test token management and context window behavior"
            ))

        memory.add_messages(long_conversation)

        # Get context
        context = memory.get_context()

        # Should have applied token limits
        assert len(context.context_window) < 20
        assert context.total_tokens <= config.max_tokens * 1.5  # Allow tolerance

        # Context should be the most recent messages
        if len(context.context_window) > 1:
            # Messages should be in chronological order
            for i in range(len(context.context_window) - 1):
                assert (context.context_window[i].timestamp <=
                       context.context_window[i + 1].timestamp)

    def test_memory_comprehensive_workflow(self):
        """Test comprehensive memory workflow"""
        # Create memory with advanced configuration
        config = MemoryConfig(
            max_messages=10,
            max_tokens=500,
            return_messages=True,
            summary_frequency=5
        )
        llm = MockLLM(temperature=0.0)
        memory = ConversationSummaryMemory(config=config, llm=llm)

        # Simulate complex conversation
        phases = [
            # Phase 1: Introduction
            [
                ChatMessage(role="user", content="Hi, I need help with my AI project"),
                ChatMessage(role="assistant", content="Hello! I'd be happy to help with your AI project"),
            ],
            # Phase 2: Problem discussion
            [
                ChatMessage(role="user", content="I'm building a recommendation system"),
                ChatMessage(role="assistant", content="Recommendation systems are fascinating! What approach are you considering?"),
                ChatMessage(role="user", content="Collaborative filtering seems most appropriate"),
            ],
            # Phase 3: Technical details
            [
                ChatMessage(role="assistant", content="Collaborative filtering is a good choice for many scenarios"),
                ChatMessage(role="user", content="What about cold start problems?"),
                ChatMessage(role="assistant", content="Cold start is indeed a challenge in collaborative filtering"),
                ChatMessage(role="user", content="Should I consider hybrid approaches?"),
            ]
        ]

        # Process each phase
        for i, phase in enumerate(phases):
            memory.add_messages(phase)

            # Check memory state after each phase
            memory_result = memory.get_memory()
            context = memory.get_context()

            # Basic sanity checks
            assert len(memory_result.messages) > 0
            assert memory_result.token_count > 0
            assert len(context.context_window) > 0

            # Check summarization progress
            if i > 0:  # After first phase
                stats = memory.get_summarization_stats()
                # Should have some summary by now
                if stats["full_summary_length"] > 0:
                    assert memory_result.summary is not None

        # Final comprehensive check
        final_memory = memory.get_memory()
        final_context = memory.get_context()

        assert len(final_memory.messages) >= 1  # Should have at least some context
        assert final_memory.token_count > 0
        assert len(final_context.context_window) >= 1
        assert final_context.total_tokens > 0

        # Test search functionality
        search_results = memory.search("recommendation")
        assert len(search_results) > 0

        # Test role filtering
        user_messages = memory.get_messages_by_role("user")
        assistant_messages = memory.get_messages_by_role("assistant")
        assert len(user_messages) > 0
        assert len(assistant_messages) > 0

        # Test memory info
        memory_info = memory.get_memory_info()
        assert "summarization_stats" in memory_info
        assert memory_info["memory_type"] == "ConversationSummaryMemory"