"""
Tests for chain module
"""
import pytest
from unittest.mock import Mock, AsyncMock

from my_langchain.chains.types import (
    ChainConfig, ChainResult, ChainInput,
    ChainError, ChainValidationError, ChainExecutionError
)
from my_langchain.chains.base import BaseChain
from my_langchain.chains.llm_chain import LLMChain
from my_langchain.chains.sequential_chain import SequentialChain
from my_langchain.chains.simple_chain import SimpleChain
from my_langchain.llms import MockLLM
from my_langchain.prompts import PromptTemplate


class MockChain(BaseChain):
    """Mock chain for testing"""

    return_value: str = "test_result"
    should_fail: bool = False

    def __init__(self, return_value="test_result", should_fail=False, **kwargs):
        super().__init__(**kwargs)
        self.return_value = return_value
        self.should_fail = should_fail

    def _run(self, inputs):
        if self.should_fail:
            raise ValueError("Mock chain failure")
        return {"output": self.return_value}

    async def _arun(self, inputs):
        if self.should_fail:
            raise ValueError("Mock chain failure")
        return {"output": self.return_value}


class TestChainConfig:
    """Test chain configuration"""

    def test_config_creation(self):
        """Test creating chain configuration"""
        config = ChainConfig(
            verbose=True,
            return_intermediate_steps=True,
            input_key="test_input",
            output_key="test_output"
        )

        assert config.verbose is True
        assert config.return_intermediate_steps is True
        assert config.input_key == "test_input"
        assert config.output_key == "test_output"

    def test_config_defaults(self):
        """Test default configuration values"""
        config = ChainConfig()

        assert config.verbose is False
        assert config.return_intermediate_steps is False
        assert config.input_key is None
        assert config.output_key is None


class TestBaseChain:
    """Test base chain functionality"""

    def test_mock_chain_creation(self):
        """Test creating mock chain"""
        chain = MockChain(return_value="hello")

        assert chain.return_value == "hello"
        assert chain.should_fail is False

    def test_chain_run_with_dict_input(self):
        """Test running chain with dictionary input"""
        chain = MockChain(return_value="test_result")
        result = chain.run({"input": "test"})

        assert result == "test_result"

    def test_chain_run_with_string_input(self):
        """Test running chain with string input"""
        chain = MockChain(return_value="test_result")
        result = chain.run("test_input")

        assert result == "test_result"

    def test_chain_run_with_config(self):
        """Test running chain with configuration override"""
        config = ChainConfig(verbose=True, return_intermediate_steps=True)
        chain = MockChain(config=config, return_value="test")
        result = chain.run({"input": "test"})

        assert isinstance(result, ChainResult)
        assert result.output == "test"
        assert len(result.intermediate_steps) > 0

    def test_chain_arun(self):
        """Test running chain asynchronously"""
        import asyncio

        async def test_arun():
            chain = MockChain(return_value="async_result")
            result = await chain.arun({"input": "test"})
            assert result == "async_result"

        asyncio.run(test_arun())

    def test_chain_invoke(self):
        """Test chain invoke method"""
        chain = MockChain(return_value="invoke_result")
        result = chain.invoke("test_input")
        assert result == "invoke_result"

    def test_chain_error_handling(self):
        """Test chain error handling"""
        chain = MockChain(should_fail=True)

        with pytest.raises(ChainExecutionError):
            chain.run({"input": "test"})

    def test_chain_info(self):
        """Test getting chain information"""
        chain = MockChain(return_value="test")
        info = chain.get_chain_info()

        assert "chain_type" in info
        assert "config" in info
        assert "input_keys" in info
        assert "output_keys" in info
        assert info["chain_type"] == "MockChain"


class TestLLMChain:
    """Test LLM chain"""

    def test_llm_chain_creation(self):
        """Test creating LLM chain"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)
        prompt = PromptTemplate(
            template="Question: {question}\nAnswer:",
            input_variables=["question"]
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        assert chain.llm == llm
        assert chain.prompt == prompt
        assert chain.output_key == "text"

    def test_llm_chain_run(self):
        """Test running LLM chain"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)
        prompt = PromptTemplate(
            template="What is {topic}?",
            input_variables=["topic"]
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run({"topic": "AI"})

        assert isinstance(result, str)
        assert len(result) > 0

    def test_llm_chain_run_with_string(self):
        """Test running LLM chain with string input"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)
        prompt = PromptTemplate(
            template="Hello, {name}!",
            input_variables=["name"]
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run("Alice")

        assert isinstance(result, str)
        assert "Alice" in result

    def test_llm_chain_arun(self):
        """Test running LLM chain asynchronously"""
        import asyncio

        async def test_arun():
            llm = MockLLM(temperature=0.0, response_delay=0.0)
            prompt = PromptTemplate(
                template="Tell me about {subject}",
                input_variables=["subject"]
            )

            chain = LLMChain(llm=llm, prompt=prompt)
            result = await chain.arun({"subject": "science"})

            assert isinstance(result, str)
            assert len(result) > 0

        asyncio.run(test_arun())

    def test_llm_chain_apply(self):
        """Test applying LLM chain to multiple inputs"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)
        prompt = PromptTemplate(
            template="Hello, {name}!",
            input_variables=["name"]
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        inputs = [{"name": "Alice"}, {"name": "Bob"}, "Charlie"]
        results = chain.apply(inputs)

        assert len(results) == 3
        assert all(isinstance(result, str) for result in results)

    def test_llm_chain_aapply(self):
        """Test applying LLM chain asynchronously to multiple inputs"""
        import asyncio

        async def test_aapply():
            llm = MockLLM(temperature=0.0, response_delay=0.0)
            prompt = PromptTemplate(
                template="Process: {item}",
                input_variables=["item"]
            )

            chain = LLMChain(llm=llm, prompt=prompt)
            inputs = [{"item": "A"}, {"item": "B"}]
            results = await chain.aapply(inputs)

            assert len(results) == 2
            assert all(isinstance(result, str) for result in results)

        asyncio.run(test_aapply())

    def test_llm_chain_get_set_methods(self):
        """Test LLM chain getter and setter methods"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)
        prompt = PromptTemplate(template="Test: {input}", input_variables=["input"])
        chain = LLMChain(llm=llm, prompt=prompt)

        # Test get_prompt
        assert chain.get_prompt() == prompt

        # Test update_prompt
        new_prompt = PromptTemplate(template="New: {input}", input_variables=["input"])
        chain.update_prompt(new_prompt)
        assert chain.prompt == new_prompt

        # Test update_llm
        new_llm = MockLLM(temperature=0.5, response_delay=0.0)
        chain.update_llm(new_llm)
        assert chain.llm == new_llm

    def test_llm_chain_validation_error(self):
        """Test LLM chain validation error"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)

        # Test with None prompt (should still raise an error)
        with pytest.raises(Exception):  # Any validation error is acceptable
            LLMChain(llm=llm, prompt=None)

        # Test that static prompts are now allowed
        static_prompt = PromptTemplate(template="Static text")  # No input variables
        chain = LLMChain(llm=llm, prompt=static_prompt)
        assert chain is not None
        assert chain.prompt == static_prompt


class TestSequentialChain:
    """Test sequential chain"""

    def test_sequential_chain_creation(self):
        """Test creating sequential chain"""
        chain1 = MockChain(return_value="step1")
        chain2 = MockChain(return_value="step2")

        seq_chain = SequentialChain(
            chains=[chain1, chain2],
            input_variables=["input"],
            output_variables=["final_output"]
        )

        assert len(seq_chain.chains) == 2
        assert seq_chain.input_variables == ["input"]
        assert seq_chain.output_variables == ["final_output"]

    def test_sequential_chain_run(self):
        """Test running sequential chain"""
        chain1 = MockChain(return_value="intermediate")
        chain2 = MockChain(return_value="final")

        seq_chain = SequentialChain(chains=[chain1, chain2])
        result = seq_chain.run({"input": "test"})

        # Should return final output
        assert "output" in result

    def test_sequential_chain_return_all(self):
        """Test sequential chain returning all outputs"""
        chain1 = MockChain(return_value="step1_result")
        chain2 = MockChain(return_value="step2_result")

        seq_chain = SequentialChain(
            chains=[chain1, chain2],
            return_all=True
        )
        result = seq_chain.run({"input": "test"})

        # Should return all intermediate outputs
        assert isinstance(result, dict)
        assert len(result) >= 1

    def test_sequential_chain_arun(self):
        """Test running sequential chain asynchronously"""
        import asyncio

        async def test_arun():
            chain1 = MockChain(return_value="async_step1")
            chain2 = MockChain(return_value="async_step2")

            seq_chain = SequentialChain(chains=[chain1, chain2])
            result = await seq_chain.arun({"input": "test"})

            assert isinstance(result, dict)

        asyncio.run(test_arun())

    def test_sequential_chain_add_remove(self):
        """Test adding and removing chains"""
        chain1 = MockChain(return_value="step1")
        seq_chain = SequentialChain(chains=[chain1])

        # Add chain
        chain2 = MockChain(return_value="step2")
        seq_chain.add_chain(chain2)
        assert seq_chain.get_chain_count() == 2

        # Remove chain
        seq_chain.remove_chain(0)
        assert seq_chain.get_chain_count() == 1

    def test_sequential_chain_get_chain_at(self):
        """Test getting chain at specific index"""
        chain1 = MockChain(return_value="step1")
        chain2 = MockChain(return_value="step2")

        seq_chain = SequentialChain(chains=[chain1, chain2])
        retrieved_chain = seq_chain.get_chain_at(0)

        assert retrieved_chain == chain1
        assert seq_chain.get_chain_at(10) is None  # Invalid index

    def test_sequential_chain_empty_error(self):
        """Test error when creating empty sequential chain"""
        with pytest.raises(ValueError):
            SequentialChain(chains=[])

    def test_sequential_chain_error_propagation(self):
        """Test error propagation in sequential chain"""
        chain1 = MockChain(return_value="step1")
        chain2 = MockChain(should_fail=True)

        seq_chain = SequentialChain(chains=[chain1, chain2])

        with pytest.raises(ChainError):
            seq_chain.run({"input": "test"})


class TestSimpleChain:
    """Test simple chain"""

    def test_simple_chain_creation(self):
        """Test creating simple chain"""
        def test_func(x):
            return x.upper()

        chain = SimpleChain(
            func=test_func,
            input_keys=["text"],
            output_keys=["uppercase"]
        )

        assert chain.func == test_func
        assert chain.input_keys == ["text"]
        assert chain.output_keys == ["uppercase"]

    def test_simple_chain_run(self):
        """Test running simple chain"""
        def add_prefix(text):
            return f"Hello, {text}!"

        chain = SimpleChain(
            func=add_prefix,
            input_keys=["text"],
            output_keys=["greeting"]
        )

        result = chain.run({"text": "World"})
        assert result == "Hello, World!"

    def test_simple_chain_run_with_defaults(self):
        """Test running simple chain with default keys"""
        def echo_func(input):
            return f"Echo: {input}"

        chain = SimpleChain(func=echo_func)
        result = chain.run({"input": "test"})
        assert result == "Echo: test"

    def test_simple_chain_multiple_outputs(self):
        """Test simple chain with multiple outputs"""
        def split_name(name):
            parts = name.split()
            return parts[0], parts[1]

        chain = SimpleChain(
            func=split_name,
            input_keys=["full_name"],
            output_keys=["first_name", "last_name"]
        )

        result = chain.run({"full_name": "John Doe"})
        assert result["first_name"] == "John"
        assert result["last_name"] == "Doe"

    def test_simple_chain_arun(self):
        """Test running simple chain asynchronously"""
        import asyncio

        async def async_uppercase(text):
            await asyncio.sleep(0.01)  # Simulate async work
            return text.upper()

        async def test_arun():
            chain = SimpleChain(
                func=async_uppercase,
                input_keys=["text"],
                output_keys=["uppercase"]
            )
            result = await chain.arun({"text": "hello"})
            assert result == "HELLO"

        asyncio.run(test_arun())

    def test_simple_chain_sync_to_async(self):
        """Test running sync function in async context"""
        import asyncio

        def sync_func(text):
            return f"Async: {text}"

        async def test_sync_to_async():
            chain = SimpleChain(func=sync_func)
            result = await chain.arun({"text": "test"})
            assert result == "Async: test"

        asyncio.run(test_sync_to_async())

    def test_simple_chain_from_function(self):
        """Test creating simple chain from function"""
        def multiply(x, y):
            return x * y

        chain = SimpleChain.from_function(
            multiply,
            input_keys=["x", "y"],
            output_keys=["product"]
        )

        result = chain.run({"x": 3, "y": 4})
        assert result == 12

    def test_simple_chain_set_methods(self):
        """Test setter methods"""
        def dummy_func(x):
            return x

        chain = SimpleChain(func=dummy_func)

        # Set input keys
        chain.set_input_keys(["input_text"])
        assert chain.input_keys == ["input_text"]

        # Set output keys
        chain.set_output_keys(["output_text"])
        assert chain.output_keys == ["output_text"]

    def test_simple_chain_error_handling(self):
        """Test simple chain error handling"""
        def failing_func(x):
            raise ValueError("Test error")

        chain = SimpleChain(func=failing_func)

        with pytest.raises(ChainExecutionError):
            chain.run({"input": "test"})