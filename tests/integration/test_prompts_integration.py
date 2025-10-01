"""
Integration tests for prompt template system
"""
import pytest
import asyncio
from my_langchain.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from my_langchain.prompts.types import PromptTemplateError, VariableMissingError
from my_langchain.llms import MockLLM


class TestPromptTemplateIntegration:
    """Integration tests for prompt templates"""

    def test_complete_prompt_workflow(self):
        """Test complete prompt template workflow"""
        # Create template
        template = PromptTemplate(
            template="Task: {task}\nContext: {context}\nQuestion: {question}\nAnswer:",
            input_variables=["task", "context", "question"]
        )

        # Format template
        prompt = template.format(
            task="Answer questions about AI",
            context="Artificial Intelligence is a field of computer science",
            question="What is machine learning?"
        )

        # Verify structure
        assert "Task: Answer questions about AI" in prompt
        assert "Context: Artificial Intelligence is a field of computer science" in prompt
        assert "Question: What is machine learning?" in prompt
        assert prompt.endswith("Answer:")

    def test_prompt_with_llm_integration(self):
        """Test prompt template with LLM integration"""
        # Create LLM
        llm = MockLLM(temperature=0.0, response_delay=0.0)

        # Create prompt template
        template = PromptTemplate(
            template="Question: {question}\nAnswer:",
            input_variables=["question"]
        )

        # Generate prompt
        prompt = template.format(question="What is the capital of France?")

        # Get response from LLM
        response = llm.generate(prompt)

        # Verify integration
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Question: What is the capital of France?" in prompt

    def test_complex_multi_step_template(self):
        """Test complex multi-step template chain"""
        # Step 1: Create context template
        context_template = PromptTemplate(
            template="Topic: {topic}\nKey Points: {points}\nBackground: {background}",
            input_variables=["topic", "points", "background"]
        )

        # Step 2: Create question template
        question_template = PromptTemplate(
            template="{context}\n\nBased on the above context, {question}",
            input_variables=["context", "question"]
        )

        # Generate context
        context = context_template.format(
            topic="Climate Change",
            points="Global warming, rising sea levels, extreme weather",
            background="Climate change refers to long-term shifts in global temperatures"
        )

        # Generate final prompt
        final_prompt = question_template.format(
            context=context,
            question="what are the main causes of climate change?"
        )

        # Verify multi-step process
        assert "Topic: Climate Change" in final_prompt
        assert "Key Points: Global warming, rising sea levels, extreme weather" in final_prompt
        assert "what are the main causes of climate change?" in final_prompt

    def test_template_error_handling_workflow(self):
        """Test error handling in template workflow"""
        template = PromptTemplate(
            template="User: {user}\nMessage: {message}",
            input_variables=["user", "message"],
            strict_variables=True
        )

        # Test successful formatting
        result = template.format(user="Alice", message="Hello!")
        assert "User: Alice" in result
        assert "Message: Hello!" in result

        # Test error case
        with pytest.raises(VariableMissingError):
            template.format(user="Bob")  # Missing 'message'

    def test_template_partial_workflow(self):
        """Test partial variables workflow"""
        # Base template with partial variables
        base_template = PromptTemplate(
            template="System: {system_message}\nUser: {user_input}\nAssistant:",
            input_variables=["user_input"],
            partial_variables={"system_message": "You are a helpful assistant."}
        )

        # Create specialized templates
        customer_service = base_template.partial(
            system_message="You are a customer service representative."
        )

        technical_support = customer_service.partial(
            system_message="You are a technical support specialist."
        )

        # Test different specializations
        result1 = base_template.format(user_input="Help me!")
        assert "You are a helpful assistant." in result1

        result2 = customer_service.format(user_input="I have a complaint.")
        assert "You are a customer service representative." in result2

        result3 = technical_support.format(user_input="My computer is broken.")
        assert "You are a technical support specialist." in result3


class TestFewShotIntegration:
    """Integration tests for few-shot templates"""

    def test_few_shot_with_llm_workflow(self):
        """Test few-shot template with LLM workflow"""
        # Create example template
        example_template = PromptTemplate(
            template="Question: {question}\nAnswer: {answer}",
            input_variables=["question", "answer"]
        )

        # Create examples
        examples = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 5+3?", "answer": "8"},
            {"question": "What is 10-3?", "answer": "7"}
        ]

        # Create few-shot template
        few_shot_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_template,
            prefix="Here are some examples of math problems:",
            suffix="Question: {new_question}\nAnswer:",
            example_separator="\n---\n"
        )

        # Generate prompt with new question
        prompt = few_shot_template.format(new_question="What is 6+4?")

        # Verify structure
        assert "Here are some examples of math problems:" in prompt
        assert "What is 2+2?" in prompt
        assert "What is 5+3?" in prompt
        assert "What is 10-3?" in prompt
        assert "What is 6+4?" in prompt

        # Test with LLM
        llm = MockLLM(temperature=0.0, response_delay=0.0)
        response = llm.generate(prompt)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_dynamic_few_shot_workflow(self):
        """Test dynamic few-shot example management"""
        example_template = PromptTemplate(
            template="Input: {input_text}\nOutput: {output_text}",
            input_variables=["input_text", "output_text"]
        )

        few_shot_template = FewShotPromptTemplate(
            examples=[],
            example_prompt=example_template
        )

        # Start with empty template
        assert few_shot_template.get_example_count() == 0

        # Add examples dynamically
        examples_to_add = [
            {"input_text": "hello", "output_text": "hi"},
            {"input_text": "goodbye", "output_text": "bye"},
            {"input_text": "thank you", "output_text": "you're welcome"}
        ]

        for example in examples_to_add:
            few_shot_template.add_example(example)

        # Test with different numbers of examples
        for max_examples in range(1, 4):
            selected = few_shot_template.select_examples({}, max_examples)
            assert len(selected) == max_examples

        # Test formatted output
        prompt = few_shot_template.format(input_text="please")
        assert len([line for line in prompt.split('\n') if 'Input:' in line]) == 3


class TestChatIntegration:
    """Integration tests for chat templates"""

    def test_chat_conversation_workflow(self):
        """Test complete chat conversation workflow"""
        chat_template = ChatPromptTemplate()

        # Build conversation
        chat_template.system_message("You are a helpful AI assistant.")
        chat_template.user_message("Hello, my name is {user_name}.")
        chat_template.assistant_message("Nice to meet you, {user_name}! How can I help you today?")
        chat_template.user_message("I need help with {topic}.")

        # Format conversation
        conversation = chat_template.format(user_name="Alice", topic="Python programming")

        # Verify conversation structure
        lines = conversation.split('\n')
        assert any("system: You are a helpful AI assistant." in line for line in lines)
        assert any("user: Hello, my name is Alice." in line for line in lines)
        assert any("assistant: Nice to meet you, Alice!" in line for line in lines)
        assert any("user: I need help with Python programming." in line for line in lines)

    def test_chat_with_structured_messages(self):
        """Test chat template with structured message output"""
        chat_template = ChatPromptTemplate()
        chat_template.add_message_template("system", "You are a {role} assistant.")
        chat_template.add_message_template("user", "Help me with {task}.")

        # Get structured messages
        messages = chat_template.format_messages(role="technical", task="debugging code")

        # Verify message structure
        assert len(messages) == 2
        assert messages[0].role.value == "system"
        assert messages[0].content == "You are a technical assistant."
        assert messages[1].role.value == "user"
        assert messages[1].content == "Help me with debugging code."

    def test_multi_turn_chat_simulation(self):
        """Test multi-turn chat simulation"""
        chat_template = ChatPromptTemplate()

        # Initial setup
        chat_template.system_message("You are a helpful tutor.")

        # Simulate multiple turns
        turns = [
            {"role": "user", "content": "Can you explain {concept}?"},
            {"role": "assistant", "content": "Certainly! {concept} is {explanation}."},
            {"role": "user", "content": "Can you give me an example?"},
            {"role": "assistant", "content": "Here's an example: {example}."}
        ]

        for turn in turns:
            chat_template.add_message_template(turn["role"], turn["content"])

        # Format with specific content
        conversation = chat_template.format(
            concept="photosynthesis",
            explanation="the process by which plants convert sunlight into energy",
            example="a tree using sunlight to make food from carbon dioxide and water"
        )

        # Verify all turns are present
        assert "Can you explain photosynthesis?" in conversation
        assert "Certainly! photosynthesis is the process by which plants convert sunlight into energy." in conversation
        assert "Can you give me an example?" in conversation
        assert "Here's an example: a tree using sunlight to make food from carbon dioxide and water." in conversation

    def test_chat_template_error_handling(self):
        """Test error handling in chat templates"""
        chat_template = ChatPromptTemplate()

        # Add valid messages
        chat_template.system_message("System message")
        chat_template.user_message("Hello, {name}!")

        # Test successful formatting
        result = chat_template.format(name="Bob")
        assert "Hello, Bob!" in result

        # Test formatting with missing variable (should not raise error in chat templates)
        result = chat_template.format()
        assert "Hello, {name}!" in result  # Template preserved when variable missing


class TestPromptTemplatePerformance:
    """Performance tests for prompt templates"""

    def test_large_template_formatting(self):
        """Test formatting large templates efficiently"""
        # Create template with many variables
        variables = [f"var_{i}" for i in range(100)]
        template_parts = [f"{{{var}}}" for var in variables]
        template_text = " ".join(template_parts)

        large_template = PromptTemplate(
            template=template_text,
            input_variables=variables
        )

        # Time the formatting
        import time
        start_time = time.time()

        # Create input values
        input_values = {var: f"value_{i}" for i, var in enumerate(variables)}
        result = large_template.format(**input_values)

        end_time = time.time()
        formatting_time = end_time - start_time

        # Verify result and performance
        assert len(result) > 0
        assert formatting_time < 1.0  # Should complete within 1 second
        assert all(f"value_{i}" in result for i in range(100))

    def test_many_small_templates(self):
        """Test creating and formatting many small templates"""
        templates = []
        for i in range(100):
            template = PromptTemplate(
                template="Template {i}: Value is {value}",
                input_variables=["value"]
            )
            templates.append(template)

        # Format all templates
        import time
        start_time = time.time()

        results = []
        for i, template in enumerate(templates):
            result = template.format(value=f"result_{i}")
            results.append(result)

        end_time = time.time()
        total_time = end_time - start_time

        # Verify results and performance
        assert len(results) == 100
        assert total_time < 2.0  # Should complete within 2 seconds
        assert all(f"Template {i}: Value is result_{i}" in results[i] for i in range(100))