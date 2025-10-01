"""
Tests for prompt template module
"""
import pytest
from unittest.mock import Mock, patch

from my_langchain.prompts.types import (
    PromptTemplateConfig, PromptTemplateResult,
    PromptTemplateError, TemplateValidationError, VariableMissingError
)
from my_langchain.prompts.prompt_template import PromptTemplate
from my_langchain.prompts.few_shot import FewShotPromptTemplate
from my_langchain.prompts.chat import ChatPromptTemplate, ChatMessage, ChatMessageType


class TestPromptTemplateConfig:
    """Test prompt template configuration"""

    def test_config_creation(self):
        """Test creating prompt template configuration"""
        config = PromptTemplateConfig(
            template_format="f-string",
            validate_template=True,
            strict_variables=True
        )

        assert config.template_format == "f-string"
        assert config.validate_template is True
        assert config.strict_variables is True

    def test_config_defaults(self):
        """Test default configuration values"""
        config = PromptTemplateConfig()

        assert config.template_format == "f-string"
        assert config.validate_template is True
        assert config.strict_variables is True


class TestPromptTemplate:
    """Test basic prompt template"""

    def test_simple_template(self):
        """Test simple template formatting"""
        template = PromptTemplate(
            template="Hello, {name}! How are you?",
            input_variables=["name"]
        )

        result = template.format(name="Alice")
        assert result == "Hello, Alice! How are you?"

    def test_auto_extract_variables(self):
        """Test automatic variable extraction"""
        template = PromptTemplate(
            template="What is {capital} of {country}?"
        )

        assert set(template.input_variables) == {"capital", "country"}

    def test_multiple_variables(self):
        """Test template with multiple variables"""
        template = PromptTemplate(
            template="User: {user}\nQuestion: {question}\nContext: {context}",
            input_variables=["user", "question", "context"]
        )

        result = template.format(
            user="Bob",
            question="What is AI?",
            context="Artificial Intelligence"
        )
        assert "User: Bob" in result
        assert "Question: What is AI?" in result
        assert "Context: Artificial Intelligence" in result

    def test_partial_variables(self):
        """Test template with partial variables"""
        template = PromptTemplate(
            template="Hello, {name}! Today is {day}.",
            partial_variables={"day": "Monday"},
            input_variables=["name"]
        )

        result = template.format(name="Charlie")
        assert result == "Hello, Charlie! Today is Monday."

    def test_partial_method(self):
        """Test partial method to create new template"""
        template1 = PromptTemplate(
            template="Task: {task}, Priority: {priority}, Due: {due_date}",
            input_variables=["task", "priority", "due_date"]
        )

        template2 = template1.partial(priority="High", due_date="2024-01-01")
        result = template2.format(task="Complete project")
        assert result == "Task: Complete project, Priority: High, Due: 2024-01-01"

    def test_missing_variable_error(self):
        """Test error when required variables are missing"""
        template = PromptTemplate(
            template="Hello, {name}!",
            input_variables=["name"],
            strict_variables=True
        )

        with pytest.raises(VariableMissingError):
            template.format()  # Missing 'name'

    def test_template_validation_error(self):
        """Test template validation error"""
        # Invalid f-string syntax
        with pytest.raises(TemplateValidationError):
            PromptTemplate(
                template="Hello, {name!}",  # Invalid format
                input_variables=["name"]
            )

    def test_format_with_result(self):
        """Test formatting with detailed result"""
        template = PromptTemplate(
            template="Answer: {answer}",
            input_variables=["answer"]
        )

        result = template.format_with_result(answer="42")
        assert isinstance(result, PromptTemplateResult)
        assert result.text == "Answer: 42"
        assert result.variables["answer"] == "42"
        assert result.missing_variables == []

    def test_invoke_method(self):
        """Test invoke method with different input types"""
        template = PromptTemplate(
            template="Value: {value}",
            input_variables=["value"]
        )

        # Test with dict
        result1 = template.invoke({"value": "test"})
        assert result1 == "Value: test"

        # Test with single value
        result2 = template.invoke("test")
        assert result2 == "Value: test"

    def test_empty_template(self):
        """Test empty template"""
        template = PromptTemplate(template="")

        result = template.format()
        assert result == ""

    def test_template_with_no_variables(self):
        """Test template with no variables"""
        template = PromptTemplate(template="Hello, world!")

        result = template.format()
        assert result == "Hello, world!"

    def test_complex_f_string_variables(self):
        """Test complex f-string variable extraction"""
        template = PromptTemplate(
            template="Hello, {name}! Age: {age:02d}. Score: {score:.1f}"
        )

        # Should extract simple variable names only
        expected_vars = {"name", "age", "score"}
        assert set(template.input_variables) == expected_vars


class TestFewShotPromptTemplate:
    """Test few-shot prompt template"""

    def test_few_shot_template(self):
        """Test basic few-shot template"""
        example_prompt = PromptTemplate(
            template="Question: {question}\nAnswer: {answer}",
            input_variables=["question", "answer"]
        )

        examples = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3+3?", "answer": "6"}
        ]

        few_shot_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Answer the following questions:",
            suffix="Question: {input_question}\nAnswer:",
            example_separator="\n---\n"
        )

        result = few_shot_template.format(input_question="What is 4+4?")
        assert "What is 2+2?" in result
        assert "What is 3+3?" in result
        assert "What is 4+4?" in result
        assert "Answer the following questions:" in result

    def test_add_examples(self):
        """Test adding examples to few-shot template"""
        example_prompt = PromptTemplate(
            template="Input: {input}\nOutput: {output}",
            input_variables=["input", "output"]
        )

        few_shot_template = FewShotPromptTemplate(
            examples=[],
            example_prompt=example_prompt
        )

        # Add examples
        few_shot_template.add_example({"input": "A", "output": "B"})
        few_shot_template.add_examples([{"input": "C", "output": "D"}])

        assert few_shot_template.get_example_count() == 2

    def test_clear_examples(self):
        """Test clearing examples"""
        example_prompt = PromptTemplate(
            template="Q: {q}\nA: {a}",
            input_variables=["q", "a"]
        )

        examples = [{"q": "test", "a": "test_answer"}]
        few_shot_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt
        )

        assert few_shot_template.get_example_count() == 1

        few_shot_template.clear_examples()
        assert few_shot_template.get_example_count() == 0

    def test_select_examples(self):
        """Test example selection"""
        example_prompt = PromptTemplate(
            template="Example: {text}",
            input_variables=["text"]
        )

        examples = [
            {"text": f"example_{i}"} for i in range(10)
        ]

        few_shot_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt
        )

        selected = few_shot_template.select_examples({}, 3)
        assert len(selected) == 3
        assert selected[0]["text"] == "example_0"


class TestChatPromptTemplate:
    """Test chat prompt template"""

    def test_chat_template_with_messages(self):
        """Test chat template with predefined messages"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, {name}!"}
        ]

        chat_template = ChatPromptTemplate(
            messages=[ChatMessage(role=ChatMessageType.SYSTEM, content="You are a helpful assistant.")],
            message_templates=[
                {"role": "user", "content": "Hello, {name}!"}
            ]
        )

        result = chat_template.format(name="Alice")
        assert "system: You are a helpful assistant." in result
        assert "user: Hello, Alice!" in result

    def test_add_message_methods(self):
        """Test convenience methods for adding messages"""
        chat_template = ChatPromptTemplate()

        chat_template.system_message("You are a helpful assistant.")
        chat_template.user_message("Hello, {name}!")
        chat_template.assistant_message("Hi there!")

        result = chat_template.format(name="Bob")
        assert "system: You are a helpful assistant." in result
        assert "user: Hello, Bob!" in result
        assert "assistant: Hi there!" in result

    def test_format_messages(self):
        """Test formatting to structured messages"""
        chat_template = ChatPromptTemplate()
        chat_template.add_message_template("user", "Hello, {name}!")

        messages = chat_template.format_messages(name="Charlie")
        assert len(messages) == 1
        assert messages[0].role == ChatMessageType.USER
        assert messages[0].content == "Hello, Charlie!"

    def test_from_messages_class_method(self):
        """Test creating template from messages"""
        messages = [
            ChatMessage(role=ChatMessageType.SYSTEM, content="System message"),
            ChatMessage(role=ChatMessageType.USER, content="User message")
        ]

        chat_template = ChatPromptTemplate.from_messages(messages)
        assert len(chat_template.messages) == 2

    def test_chat_message_types(self):
        """Test different message types"""
        chat_template = ChatPromptTemplate()

        # Test all valid roles
        for role_type in ChatMessageType:
            chat_template.add_message(role_type, f"Test {role_type.value} message")

        result = chat_template.format()
        for role_type in ChatMessageType:
            assert f"{role_type.value}: Test {role_type.value} message" in result

    def test_chat_template_variable_extraction(self):
        """Test variable extraction from chat templates"""
        message_templates = [
            {"role": "user", "content": "Hello, {name}!"},
            {"role": "assistant", "content": "Hi {name}, I'm {assistant_name}."}
        ]

        chat_template = ChatPromptTemplate(message_templates=message_templates)
        expected_vars = {"name", "assistant_name"}
        assert set(chat_template.input_variables) == expected_vars

    def test_chat_message_model(self):
        """Test chat message data model"""
        message = ChatMessage(
            role=ChatMessageType.USER,
            content="Hello, world!",
            name="user123"
        )

        assert message.role == ChatMessageType.USER
        assert message.content == "Hello, world!"
        assert message.name == "user123"