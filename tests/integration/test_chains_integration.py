"""
Integration tests for chain module
"""
import pytest
import asyncio

from my_langchain.chains import LLMChain, SequentialChain, SimpleChain
from my_langchain.llms import MockLLM
from my_langchain.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate


class TestChainIntegration:
    """Integration tests for chain functionality"""

    def test_complete_llm_chain_workflow(self):
        """Test complete LLM chain workflow"""
        # Create LLM
        llm = MockLLM(temperature=0.0, response_delay=0.0)

        # Create prompt template
        prompt = PromptTemplate(
            template="You are a helpful assistant. Answer the following question:\n\nQuestion: {question}\n\nAnswer:",
            input_variables=["question"]
        )

        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt)

        # Run chain
        result = chain.run({"question": "What is the capital of France?"})

        # Verify result
        assert isinstance(result, str)
        assert len(result) > 0

    def test_llm_chain_with_different_prompt_types(self):
        """Test LLM chain with different prompt types"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)

        # Test with simple prompt
        simple_prompt = PromptTemplate(
            template="Process: {input}",
            input_variables=["input"]
        )
        simple_chain = LLMChain(llm=llm, prompt=simple_prompt)
        result1 = simple_chain.run({"input": "test data"})

        # Test with chat prompt
        chat_prompt = ChatPromptTemplate()
        chat_prompt.system_message("You are a helpful assistant.")
        chat_prompt.user_message("Help me with {topic}")
        chat_chain = LLMChain(llm=llm, prompt=chat_prompt)
        result2 = chat_chain.run({"topic": "Python programming"})

        assert isinstance(result1, str) and isinstance(result2, str)
        assert len(result1) > 0 and len(result2) > 0

    def test_sequential_chain_integration(self):
        """Test sequential chain with multiple LLM chains"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)

        # First chain: Summarize input
        summary_prompt = PromptTemplate(
            template="Summarize the following text in one sentence:\n\n{text}\n\nSummary:",
            input_variables=["text"]
        )
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

        # Second chain: Generate questions about summary
        question_prompt = PromptTemplate(
            template="Generate one interesting question about the following summary:\n\n{summary}\n\nQuestion:",
            input_variables=["summary"]
        )
        question_chain = LLMChain(llm=llm, prompt=question_prompt, output_key="question")

        # Create sequential chain
        seq_chain = SequentialChain(
            chains=[summary_chain, question_chain],
            input_variables=["text"],
            output_variables=["summary", "question"]
        )

        # Run chain
        long_text = "Artificial intelligence is a branch of computer science that aims to create intelligent machines."
        result = seq_chain.run({"text": long_text})

        # Verify result
        assert isinstance(result, dict)
        assert "summary" in result
        assert "question" in result
        assert isinstance(result["summary"], str)
        assert isinstance(result["question"], str)

    def test_mixed_chain_types_integration(self):
        """Test mixing different chain types"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)

        # Step 1: Simple transformation
        def preprocess_text(text):
            return text.strip().lower()

        preprocess_chain = SimpleChain(
            func=preprocess_text,
            input_keys=["raw_text"],
            output_keys=["processed_text"]
        )

        # Step 2: LLM processing
        prompt = PromptTemplate(
            template="Analyze the following text: {processed_text}\n\nAnalysis:",
            input_variables=["processed_text"]
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt, output_key="analysis")

        # Step 3: Post-processing
        def extract_key_points(analysis):  # Fix: rename parameter to match input key
            # Simple key point extraction
            sentences = analysis.split('.')
            return [s.strip() for s in sentences if s.strip()][:3]

        postprocess_chain = SimpleChain(
            func=extract_key_points,
            input_keys=["analysis"],  # This matches the output from llm_chain
            output_keys=["key_points"]
        )

        # Create sequential chain
        mixed_chain = SequentialChain(
            chains=[preprocess_chain, llm_chain, postprocess_chain],
            input_variables=["raw_text"],
            return_all=True
        )

        # Run chain
        result = mixed_chain.run({
            "raw_text": "  ARTIFICIAL INTELLIGENCE is revolutionizing TECHNOLOGY.  "
        })

        # Verify result
        assert isinstance(result, dict)
        assert "processed_text" in result
        assert "analysis" in result
        assert "key_points" in result
        assert "artificial intelligence" in result["processed_text"]
        assert isinstance(result["key_points"], list)

    def test_chain_with_few_shot_prompt(self):
        """Test chain with few-shot prompt template"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)

        # Create few-shot template
        example_prompt = PromptTemplate(
            template="Question: {question}\nAnswer: {answer}",
            input_variables=["question", "answer"]
        )

        examples = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is the capital of Spain?", "answer": "Madrid"}
        ]

        few_shot_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Answer the following questions based on examples:",
            suffix="Question: {input_question}\nAnswer:",
            example_separator="\n---\n"
        )

        # Create chain
        chain = LLMChain(llm=llm, prompt=few_shot_template)

        # Run chain
        result = chain.run({"input_question": "What is 3+3?"})

        assert isinstance(result, str)
        assert len(result) > 0

    def test_chain_batch_processing(self):
        """Test chain batch processing"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)

        prompt = PromptTemplate(
            template="Generate a creative name for: {topic}",
            input_variables=["topic"]
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        # Process multiple inputs
        inputs = [
            {"topic": "a coffee shop"},
            {"topic": "a tech startup"},
            {"topic": "a pet store"}
        ]

        results = chain.apply(inputs)

        # Verify results
        assert len(results) == 3
        assert all(isinstance(result, str) for result in results)
        assert all(len(result) > 0 for result in results)

    def test_async_chain_workflow(self):
        """Test async chain workflow"""
        import asyncio

        async def test_async_workflow():
            llm = MockLLM(temperature=0.0, response_delay=0.0)

            # Create chain
            prompt = PromptTemplate(
                template="Write a haiku about {topic}:",
                input_variables=["topic"]
            )
            chain = LLMChain(llm=llm, prompt=prompt)

            # Run async
            result = await chain.arun({"topic": "ocean"})

            # Verify result
            assert isinstance(result, str)
            assert len(result) > 0

        asyncio.run(test_async_workflow())

    def test_chain_error_handling_integration(self):
        """Test error handling in chain integration"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)

        # Create chain with None prompt (should raise an error)
        with pytest.raises(Exception):  # Any validation error is acceptable
            LLMChain(llm=llm, prompt=None)

        # Test that static prompts are now allowed
        static_prompt = PromptTemplate(
            template="This template has no variables",
            input_variables=[]
        )
        chain = LLMChain(llm=llm, prompt=static_prompt)
        assert chain is not None
        assert chain.prompt == static_prompt

    def test_chain_configuration_integration(self):
        """Test chain configuration integration"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)
        prompt = PromptTemplate(
            template="Process: {input}",
            input_variables=["input"]
        )

        # Create chain with verbose configuration
        from my_langchain.chains.types import ChainConfig
        config = ChainConfig(verbose=True, return_intermediate_steps=True)
        chain = LLMChain(llm=llm, prompt=prompt, config=config)

        # Run chain
        result = chain.run({"input": "test"})

        # Should return ChainResult with intermediate steps
        assert hasattr(result, 'output')
        assert hasattr(result, 'intermediate_steps')
        assert len(result.intermediate_steps) > 0

    def test_chain_memory_integration_simulation(self):
        """Test chain with simulated memory"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)

        # Simulate conversation history
        conversation_history = []

        def add_memory_step(step_info):
            conversation_history.append(step_info)
            return conversation_history

        # Create memory simulation chain
        memory_chain = SimpleChain(
            func=add_memory_step,
            input_keys=["step"],
            output_keys=["updated_history"]
        )

        # Create response generation chain
        prompt = PromptTemplate(
            template="Given this conversation history: {updated_history}\n\nUser: {message}\n\nAssistant:",
            input_variables=["updated_history", "message"]  # Fix: use updated_history instead of history
        )
        response_chain = LLMChain(llm=llm, prompt=prompt)

        # Create sequential workflow
        workflow_chain = SequentialChain(
            chains=[memory_chain, response_chain],
            return_all=True
        )

        # Simulate conversation turn
        result = workflow_chain.run({
            "step": "User asked about AI",
            "message": "What is artificial intelligence?"
        })

        # Verify simulation
        assert isinstance(result, dict)
        assert "updated_history" in result
        assert "text" in result  # LLMChain outputs "text" by default
        assert isinstance(result["updated_history"], list)
        assert len(result["updated_history"]) > 0

    def test_complex_multi_step_workflow(self):
        """Test complex multi-step workflow"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)

        # Step 1: Text preprocessing
        def clean_text(text):
            return ' '.join(text.strip().split())

        clean_chain = SimpleChain(
            func=clean_text,
            input_keys=["raw_text"],
            output_keys=["clean_text"]
        )

        # Step 2: Topic extraction
        topic_prompt = PromptTemplate(
            template="Extract the main topic from this text: {clean_text}\n\nTopic:",
            input_variables=["clean_text"]
        )
        topic_chain = LLMChain(llm=llm, prompt=topic_prompt, output_key="topic")

        # Step 3: Generate explanation
        explanation_prompt = PromptTemplate(
            template="Explain {topic} in simple terms:\n\nExplanation:",
            input_variables=["topic"]
        )
        explanation_chain = LLMChain(llm=llm, prompt=explanation_prompt, output_key="explanation")

        # Step 4: Create summary
        summary_prompt = PromptTemplate(
            template="Create a one-sentence summary:\nTopic: {topic}\nExplanation: {explanation}\n\nSummary:",
            input_variables=["topic", "explanation"]
        )
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

        # Create complex workflow
        complex_chain = SequentialChain(
            chains=[clean_chain, topic_chain, explanation_chain, summary_chain],
            input_variables=["raw_text"],
            output_variables=["clean_text", "topic", "explanation", "summary"]
        )

        # Run complex workflow
        messy_text = "   Machine Learning   is   a   subset   of   artificial   intelligence...   "
        result = complex_chain.run({"raw_text": messy_text})

        # Verify complete workflow
        assert isinstance(result, dict)
        assert "clean_text" in result
        assert "topic" in result
        assert "explanation" in result
        assert "summary" in result
        assert "machine learning" in result["clean_text"].lower()
        assert isinstance(result["topic"], str)
        assert isinstance(result["explanation"], str)
        assert isinstance(result["summary"], str)

    def test_chain_performance_considerations(self):
        """Test chain performance and efficiency"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)
        prompt = PromptTemplate(
            template="Process: {input}",
            input_variables=["input"]
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        # Test multiple rapid executions
        import time
        start_time = time.time()

        for i in range(10):
            result = chain.run({"input": f"test_{i}"})

        end_time = time.time()
        total_time = end_time - start_time

        # Verify performance (should be fast with MockLLM)
        assert total_time < 5.0  # Should complete within 5 seconds
        assert len([result for result in [chain.run({"input": "test"}) for _ in range(5)]]) == 5

    def test_chain_output_key_mapping(self):
        """Test chain output key mapping and transformation"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)

        # Chain with custom output key
        prompt = PromptTemplate(
            template="Generate a number between 1 and 100:",
            input_variables=[]
        )
        chain = LLMChain(llm=llm, prompt=prompt, output_key="random_number")

        result = chain.run({})

        # Should be accessible via the custom key
        assert isinstance(result, str)

    def test_chain_input_output_validation(self):
        """Test chain input and output validation"""
        llm = MockLLM(temperature=0.0, response_delay=0.0)
        prompt = PromptTemplate(
            template="Hello, {name}!",
            input_variables=["name"]
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        # Valid input
        result1 = chain.run({"name": "Alice"})
        assert isinstance(result1, str)

        # Missing required input
        with pytest.raises(Exception):
            chain.run({})  # Missing 'name'

        # String input (should use first variable)
        result2 = chain.run("Bob")
        assert isinstance(result2, str)
        assert "Bob" in result2