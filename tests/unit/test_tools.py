# -*- coding: utf-8 -*-
"""
Unit tests for tool module
"""
import pytest
from typing import Dict, Any

from my_langchain.tools import (
    BaseTool, Tool, SearchTool, CalculatorTool, WikipediaTool,
    PythonREPLTool, ShellTool
)
from my_langchain.tools.types import (
    ToolConfig, ToolResult, ToolInput, ToolSchema,
    ToolError, ToolValidationError, ToolExecutionError, ToolTimeoutError
)


class TestToolConfig:
    """Test ToolConfig class"""

    def test_default_config(self):
        """Test default configuration"""
        config = ToolConfig(
            name="test_tool",
            description="Test tool description"
        )
        assert config.name == "test_tool"
        assert config.description == "Test tool description"
        assert config.return_direct is False
        assert config.verbose is False
        assert config.handle_error is True
        assert config.max_execution_time is None
        assert config.metadata == {}

    def test_custom_config(self):
        """Test custom configuration"""
        config = ToolConfig(
            name="custom_tool",
            description="Custom tool",
            return_direct=True,
            verbose=True,
            handle_error=False,
            max_execution_time=30.0,
            metadata={"version": "1.0", "author": "test"}
        )
        assert config.name == "custom_tool"
        assert config.return_direct is True
        assert config.verbose is True
        assert config.handle_error is False
        assert config.max_execution_time == 30.0
        assert config.metadata["version"] == "1.0"


class TestToolResult:
    """Test ToolResult class"""

    def test_successful_result(self):
        """Test successful result"""
        result = ToolResult(
            output="Success",
            success=True,
            execution_time=0.5
        )
        assert result.output == "Success"
        assert result.success is True
        assert result.error is None
        assert result.execution_time == 0.5

    def test_failed_result(self):
        """Test failed result"""
        result = ToolResult(
            output="Error occurred",
            success=False,
            error="Something went wrong",
            execution_time=1.0
        )
        assert result.output == "Error occurred"
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.execution_time == 1.0


class TestBaseTool:
    """Test BaseTool class"""

    def test_base_tool_creation(self):
        """Test base tool creation"""
        class TestTool(BaseTool):
            def _run(self, input: str) -> str:
                return f"Processed: {input}"

        config = ToolConfig(name="test", description="Test tool")
        tool = TestTool(config=config)

        assert tool.config.name == "test"
        assert tool.config.description == "Test tool"

    def test_base_tool_invoke(self):
        """Test base tool invoke"""
        class TestTool(BaseTool):
            def _run(self, input: str) -> str:
                return f"Processed: {input}"

        config = ToolConfig(name="test", description="Test tool")
        tool = TestTool(config=config)

        result = tool.invoke("test input")
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "Processed: test input"

    def test_base_tool_with_string_input(self):
        """Test tool with string input"""
        class TestTool(BaseTool):
            def _run(self, input: str) -> str:
                return input.upper()

        tool = TestTool(config=ToolConfig(name="upper", description="Uppercase tool"))

        result = tool.invoke("hello")
        assert result.success is True
        assert result.output == "HELLO"

    def test_base_tool_with_dict_input(self):
        """Test tool with dict input"""
        class TestTool(BaseTool):
            def _run(self, text: str, times: int = 1) -> str:
                return text * times

        tool = TestTool(config=ToolConfig(name="repeat", description="Repeat tool"))

        result = tool.invoke({"text": "hi", "times": 3})
        assert result.success is True
        assert result.output == "hihihi"

    def test_base_tool_error_handling(self):
        """Test tool error handling"""
        class ErrorTool(BaseTool):
            def _run(self, input: str) -> str:
                raise ValueError("Test error")

        tool = ErrorTool(config=ToolConfig(name="error", description="Error tool"))

        result = tool.invoke("test")
        assert result.success is False
        assert "Test error" in result.error

    def test_base_tool_get_schema(self):
        """Test tool schema generation"""
        class TestTool(BaseTool):
            def _run(self, input: str, count: int = 1) -> str:
                return input * count

        tool = TestTool(config=ToolConfig(name="test", description="Test tool"))
        schema = tool.get_schema()

        assert isinstance(schema, ToolSchema)
        assert schema.name == "test"
        assert schema.description == "Test tool"
        assert len(schema.inputs) >= 2  # input and count parameters


class TestSearchTool:
    """Test SearchTool class"""

    def test_search_tool_creation(self):
        """Test search tool creation"""
        tool = SearchTool()
        assert tool.config.name == "search"
        assert "Search for information" in tool.config.description

    def test_search_tool_exact_match(self):
        """Test exact match search"""
        tool = SearchTool()
        result = tool.invoke({"query": "python"})
        assert result.success is True
        assert "Python is a high-level programming language" in result.output

    def test_search_tool_partial_match(self):
        """Test partial match search"""
        tool = SearchTool()
        result = tool.invoke({"query": "learning"})
        assert result.success is True
        assert "machine learning" in result.output.lower() or "deep learning" in result.output.lower()

    def test_search_tool_no_match(self):
        """Test no match search"""
        tool = SearchTool()
        result = tool.invoke({"query": "nonexistentterm"})
        assert result.success is True
        assert "No results found" in result.output

    def test_search_tool_empty_input(self):
        """Test empty input"""
        tool = SearchTool()
        result = tool.invoke({"query": ""})
        assert result.success is True
        assert "Please provide a search query" in result.output


class TestCalculatorTool:
    """Test CalculatorTool class"""

    def test_calculator_tool_creation(self):
        """Test calculator tool creation"""
        tool = CalculatorTool()
        assert tool.config.name == "calculator"
        assert "mathematical expressions" in tool.config.description

    def test_calculator_simple_addition(self):
        """Test simple addition"""
        tool = CalculatorTool()
        result = tool.invoke({"expression": "2+2"})
        assert result.success is True
        assert "4" in result.output

    def test_calculator_complex_expression(self):
        """Test complex expression"""
        tool = CalculatorTool()
        result = tool.invoke({"expression": "(5+3)*2-4"})
        assert result.success is True
        assert "12" in result.output

    def test_calculator_invalid_expression(self):
        """Test invalid expression"""
        tool = CalculatorTool()
        result = tool.invoke({"expression": "2+++"})
        assert result.success is True
        # Should handle error gracefully and return error message
        assert "Error" in result.output or "Invalid" in result.output

    def test_calculator_dangerous_expression(self):
        """Test dangerous expression rejection"""
        tool = CalculatorTool()
        result = tool.invoke({"expression": "__import__('os')"})
        assert result.success is True
        assert "Invalid expression" in result.output


class TestWikipediaTool:
    """Test WikipediaTool class"""

    def test_wikipedia_tool_creation(self):
        """Test Wikipedia tool creation"""
        tool = WikipediaTool()
        assert tool.config.name == "wikipedia"
        assert "Search Wikipedia" in tool.config.description

    def test_wikipedia_exact_match(self):
        """Test exact match"""
        tool = WikipediaTool()
        result = tool.invoke({"query": "python programming"})
        assert result.success is True
        assert "Python (programming language)" in result.output

    def test_wikipedia_no_match(self):
        """Test no match"""
        tool = WikipediaTool()
        result = tool.invoke({"query": "nonexistent article"})
        assert result.success is True
        assert "No Wikipedia article found" in result.output

    def test_wikipedia_empty_input(self):
        """Test empty input"""
        tool = WikipediaTool()
        result = tool.invoke({"query": ""})
        assert result.success is True
        assert "Please provide a search query" in result.output


class TestPythonREPLTool:
    """Test PythonREPLTool class"""

    def test_python_repl_tool_creation(self):
        """Test Python REPL tool creation"""
        tool = PythonREPLTool()
        assert tool.config.name == "python_repl"
        assert "Execute Python code" in tool.config.description

    def test_python_repl_forbidden_keywords(self):
        """Test forbidden keyword rejection"""
        tool = PythonREPLTool()
        result = tool.invoke({"code": "import os"})
        assert result.success is True
        assert "not allowed" in result.output

    def test_python_repl_empty_input(self):
        """Test empty input"""
        tool = PythonREPLTool()
        result = tool.invoke({"code": ""})
        assert result.success is True
        assert "Please provide Python code" in result.output


class TestShellTool:
    """Test ShellTool class"""

    def test_shell_tool_creation(self):
        """Test shell tool creation"""
        tool = ShellTool()
        assert tool.config.name == "shell"
        assert "Execute shell commands" in tool.config.description

    def test_shell_echo_command(self):
        """Test echo command"""
        tool = ShellTool()
        result = tool.invoke({"command": "echo hello world"})
        assert result.success is True
        assert "hello world" in result.output

    def test_shell_help_command(self):
        """Test help command"""
        tool = ShellTool()
        result = tool.invoke({"command": "help"})
        assert result.success is True
        assert "Available commands" in result.output
        assert "echo" in result.output

    def test_shell_forbidden_command(self):
        """Test forbidden command"""
        tool = ShellTool()
        result = tool.invoke({"command": "rm -rf /"})
        assert result.success is True
        assert "not allowed" in result.output


class TestToolErrorHandling:
    """Test tool error handling"""

    def test_tool_validation_error(self):
        """Test ToolValidationError"""
        error = ToolValidationError("Invalid input", "test_tool", {"input": "bad"})
        assert error.message == "Invalid input"
        assert error.tool_name == "test_tool"
        assert error.input_data == {"input": "bad"}

    def test_tool_execution_error(self):
        """Test ToolExecutionError"""
        cause = ValueError("Original error")
        error = ToolExecutionError("Execution failed", "test_tool", cause)
        assert error.message == "Execution failed"
        assert error.tool_name == "test_tool"
        assert error.cause == cause

    def test_tool_timeout_error(self):
        """Test ToolTimeoutError"""
        error = ToolTimeoutError("Timeout", "test_tool", 30.0)
        assert error.message == "Timeout"
        assert error.tool_name == "test_tool"
        assert error.timeout_seconds == 30.0


class TestToolIntegration:
    """Test tool integration scenarios"""

    def test_tool_configuration_variations(self):
        """Test tool with different configurations"""
        configs = [
            ToolConfig(name="test1", description="Test 1", verbose=True),
            ToolConfig(name="test2", description="Test 2", handle_error=False),
            ToolConfig(name="test3", description="Test 3", max_execution_time=1.0),
            ToolConfig(name="test4", description="Test 4", return_direct=True),
        ]

        for config in configs:
            tool = SearchTool(config=config)
            assert tool.config == config

    def test_tool_composition(self):
        """Test using multiple tools together"""
        search = SearchTool()
        calculator = CalculatorTool()

        # Search for information
        search_result = search.invoke({"query": "algorithm"})
        assert search_result.success is True

        # Use calculator for computation
        calc_result = calculator.invoke({"expression": "100 * 1.5"})
        assert calc_result.success is True
        assert "150" in calc_result.output

    def test_tool_metadata_handling(self):
        """Test tool with metadata"""
        metadata = {
            "version": "1.0.0",
            "author": "Test Author",
            "category": "utility",
            "tags": ["math", "calculation"]
        }

        config = ToolConfig(
            name="meta_tool",
            description="Tool with metadata",
            metadata=metadata
        )

        tool = CalculatorTool(config=config)
        assert tool.config.metadata == metadata
        assert tool.config.metadata["version"] == "1.0.0"
        assert "math" in tool.config.metadata["tags"]

    def test_tool_performance_considerations(self):
        """Test tool performance characteristics"""
        import time

        calculator = CalculatorTool()
        search = SearchTool()

        # Measure response times
        start = time.time()
        calc_result = calculator.invoke({"expression": "1000*1000"})
        calc_time = time.time() - start

        start = time.time()
        search_result = search.invoke({"query": "python"})
        search_time = time.time() - start

        # Both should be reasonably fast
        assert calc_time < 1.0
        assert search_time < 1.0

        # Results should still be correct
        assert calc_result.success is True
        assert search_result.success is True