# -*- coding: utf-8 -*-
"""
Built-in tool implementations
"""

import ast
import operator
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict

from my_langchain.tools.base import BaseTool
from my_langchain.tools.types import (
    ToolConfig, ToolValidationError, ToolExecutionError, ToolPermissionError
)


class SearchTool(BaseTool):
    """
    Simple search tool using basic text matching
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, config: Optional[ToolConfig] = None, **kwargs):
        if config is None:
            config = ToolConfig(
                name="search",
                description="Search for information in a simple knowledge base"
            )
        super().__init__(config=config, **kwargs)

        # Simple knowledge base - make private to avoid Pydantic conflicts
        self._knowledge_base = {
            "python": "Python is a high-level programming language known for its simplicity and readability.",
            "machine learning": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "artificial intelligence": "AI is the simulation of human intelligence in machines.",
            "deep learning": "Deep learning is a subset of machine learning using neural networks with multiple layers.",
            "data science": "Data science is an interdisciplinary field that uses scientific methods to extract knowledge from data.",
            "algorithm": "An algorithm is a finite sequence of well-defined instructions to solve a problem.",
            "programming": "Programming is the process of creating computer software.",
            "computer science": "Computer science is the study of computation, information, and automation."
        }

    def _run(self, query: str) -> str:
        """Search for information in the knowledge base"""
        if not query or not query.strip():
            return "Please provide a search query."

        query = query.lower().strip()
        results = []

        # Search for exact matches first
        if query in self._knowledge_base:
            results.append(f"Exact match: {self._knowledge_base[query]}")

        # Search for partial matches
        for key, value in self._knowledge_base.items():
            if query in key and key != query:
                results.append(f"Related match ({key}): {value}")
            elif query in value.lower():
                results.append(f"Content match: {value}")

        if not results:
            return f"No results found for '{query}'. Try searching for: Python, machine learning, AI, or data science."

        return "\n\n".join(results[:3])  # Limit to top 3 results


class CalculatorTool(BaseTool):
    """
    Simple calculator tool for mathematical expressions
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, config: Optional[ToolConfig] = None, **kwargs):
        if config is None:
            config = ToolConfig(
                name="calculator",
                description="Calculate mathematical expressions safely"
            )
        super().__init__(config=config, **kwargs)

        # Safe operators - make private to avoid Pydantic conflicts
        self._operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        # Safe functions - make private to avoid Pydantic conflicts
        self._functions = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
        }

    def _run(self, expression: str) -> str:
        """Safely evaluate mathematical expression"""
        if not expression or not expression.strip():
            return "Please provide a mathematical expression."

        try:
            # Remove whitespace and validate characters
            expression = expression.strip()
            if not re.match(r'^[\d\s\+\-\*\/\(\)\.\,\[\]]+$', expression):
                return "Invalid expression. Only numbers and basic operators (+, -, *, /) are allowed."

            # Parse and evaluate
            node = ast.parse(expression, mode='eval')
            result = self._evaluate_node(node.body)

            # Format result
            if isinstance(result, float):
                if result.is_integer():
                    return str(int(result))
                else:
                    return f"{result:.6f}"
            else:
                return str(result)

        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

    def _evaluate_node(self, node):
        """Recursively evaluate AST node"""
        if isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Constant):  # Python >= 3.8
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                raise ValueError("Only numbers are allowed")
        elif isinstance(node, ast.BinOp):
            left = self._evaluate_node(node.left)
            right = self._evaluate_node(node.right)
            op_type = type(node.op)
            if op_type in self._operators:
                return self._operators[op_type](left, right)
            else:
                raise ValueError(f"Operator {op_type} not allowed")
        elif isinstance(node, ast.UnaryOp):
            operand = self._evaluate_node(node.operand)
            op_type = type(node.op)
            if op_type in self._operators:
                return self._operators[op_type](operand)
            else:
                raise ValueError(f"Unary operator {op_type} not allowed")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in self._functions:
                    args = [self._evaluate_node(arg) for arg in node.args]
                    return self._functions[func_name](*args)
                else:
                    raise ValueError(f"Function {func_name} not allowed")
            else:
                raise ValueError("Only named functions are allowed")
        else:
            raise ValueError(f"Expression type {type(node)} not allowed")


class WikipediaTool(BaseTool):
    """
    Mock Wikipedia tool for demonstration
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, config: Optional[ToolConfig] = None, **kwargs):
        if config is None:
            config = ToolConfig(
                name="wikipedia",
                description="Search Wikipedia for information about topics"
            )
        super().__init__(config=config, **kwargs)

        # Mock Wikipedia data - make private to avoid Pydantic conflicts
        self._mock_data = {
            "python programming": {
                "title": "Python (programming language)",
                "summary": "Python is a high-level, interpreted programming language with dynamic semantics.",
                "content": "Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability and allows programmers to express concepts in fewer lines of code."
            },
            "machine learning": {
                "title": "Machine learning",
                "summary": "Machine learning is a method of data analysis that automates analytical model building.",
                "content": "Machine learning is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."
            },
            "artificial intelligence": {
                "title": "Artificial intelligence",
                "summary": "AI is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans.",
                "content": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents."
            }
        }

    def _run(self, query: str) -> str:
        """Search Wikipedia mock data"""
        if not query or not query.strip():
            return "Please provide a search query for Wikipedia."

        query = query.lower().strip()

        # Try to find exact match
        if query in self._mock_data:
            article = self._mock_data[query]
            return f"**{article['title']}**\n\n{article['summary']}\n\n{article['content']}"

        # Try to find partial match
        for key, article in self._mock_data.items():
            if query in key:
                return f"**{article['title']}** (related search)\n\n{article['summary']}\n\n{article['content']}"

        return f"No Wikipedia article found for '{query}'. Try: Python programming, machine learning, or artificial intelligence."


class PythonREPLTool(BaseTool):
    """
    Python REPL tool for executing Python code
    """

    def __init__(self, config: Optional[ToolConfig] = None, **kwargs):
        if config is None:
            config = ToolConfig(
                name="python_repl",
                description="Execute Python code in a REPL environment"
            )
        super().__init__(config=config, **kwargs)

    def _run(self, code: str) -> str:
        """Execute Python code safely"""
        if not code or not code.strip():
            return "Please provide Python code to execute."

        try:
            # Basic safety checks
            forbidden_keywords = ['import', 'exec', 'eval', 'open', 'file', '__import__']
            code_lower = code.lower()
            for keyword in forbidden_keywords:
                if keyword in code_lower:
                    return f"For security reasons, the keyword '{keyword}' is not allowed in this tool."

            # Capture output
            import io
            import sys
            from contextlib import redirect_stdout, redirect_stderr

            old_stdout = sys.stdout
            old_stderr = sys.stderr

            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            try:
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture

                # Execute the code
                exec_globals = {}
                exec_locals = {}

                # Compile and execute
                compiled_code = compile(code, '<string>', 'exec')
                exec(compiled_code, exec_globals, exec_locals)

                # Get output
                stdout_output = stdout_capture.getvalue()
                stderr_output = stderr_capture.getvalue()

                # Look for result in locals
                result = None
                for var_name, var_value in exec_locals.items():
                    if not var_name.startswith('_') and var_name not in ['code']:
                        result = var_value
                        break

                # Format output
                output_parts = []
                if stdout_output.strip():
                    output_parts.append(f"Output:\n{stdout_output.strip()}")

                if stderr_output.strip():
                    output_parts.append(f"Error:\n{stderr_output.strip()}")

                if result is not None:
                    output_parts.append(f"Result: {repr(result)}")

                if output_parts:
                    return "\n\n".join(output_parts)
                else:
                    return "Code executed successfully (no output or result)."

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        except Exception as e:
            return f"Error executing Python code: {str(e)}"


class ShellTool(BaseTool):
    """
    Shell command tool (mock/safe implementation)
    """

    def __init__(self, config: Optional[ToolConfig] = None, **kwargs):
        if config is None:
            config = ToolConfig(
                name="shell",
                description="Execute shell commands (limited safe commands only)"
            )
        super().__init__(config=config, **kwargs)

        # Safe commands - make private to avoid Pydantic conflicts
        self._safe_commands = {
            'echo': self._echo_command,
            'date': self._date_command,
            'pwd': self._pwd_command,
            'ls': self._ls_command,
            'whoami': self._whoami_command,
            'help': self._help_command
        }

    def _run(self, command: str) -> str:
        """Execute safe shell command"""
        if not command or not command.strip():
            return "Please provide a shell command."

        command = command.strip()
        parts = command.split()
        command_name = parts[0]

        if command_name not in self._safe_commands:
            available = ', '.join(self._safe_commands.keys())
            return f"Command '{command_name}' not allowed. Available commands: {available}"

        try:
            return self._safe_commands[command_name](parts[1:])
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _echo_command(self, args: List[str]) -> str:
        """Echo command"""
        return ' '.join(args)

    def _date_command(self, args: List[str]) -> str:
        """Date command"""
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def _pwd_command(self, args: List[str]) -> str:
        """Current directory command"""
        import os
        return os.getcwd()

    def _ls_command(self, args: List[str]) -> str:
        """List directory command"""
        import os
        try:
            path = args[0] if args else '.'
            items = os.listdir(path)
            if not items:
                return f"Directory '{path}' is empty."
            return '\n'.join(sorted(items))
        except FileNotFoundError:
            return f"Directory '{args[0] if args else '.'}' not found."

    def _whoami_command(self, args: List[str]) -> str:
        """Current user command"""
        import os
        return os.environ.get('USER', os.environ.get('USERNAME', 'unknown'))

    def _help_command(self, args: List[str]) -> str:
        """Help command"""
        available = ', '.join(self._safe_commands.keys())
        return f"Available commands: {available}\n\nThis is a safe mock shell tool for demonstration."