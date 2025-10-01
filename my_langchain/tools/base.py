# -*- coding: utf-8 -*-
"""
Base tool implementation
"""

import time
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable

from my_langchain.base.base import BaseComponent
from my_langchain.tools.types import (
    ToolConfig, ToolResult, ToolSchema, ToolInput,
    ToolError, ToolValidationError, ToolExecutionError, ToolTimeoutError,
    InputType, OutputType
)
from pydantic import BaseModel, ConfigDict, Field


class BaseTool(BaseComponent):
    """
    Base tool implementation providing common functionality

    This class defines the interface that all tool implementations must follow
    and provides common utility methods for tool execution, validation, and error handling.
    """

    config: ToolConfig = Field(..., description="Tool configuration")
    args_schema: Optional[type[BaseModel]] = Field(
        default=None,
        description="Pydantic model for input validation"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, config: Optional[ToolConfig] = None, **kwargs):
        """
        Initialize tool

        Args:
            config: Tool configuration
            **kwargs: Additional parameters
        """
        if config is None:
            config = self._create_default_config()

        super().__init__(config=config, **kwargs)

    @abstractmethod
    def _run(self, **kwargs) -> Any:
        """
        Run the tool implementation

        Args:
            **kwargs: Tool input arguments

        Returns:
            Tool output

        Raises:
            ToolExecutionError: If tool execution fails
        """
        pass

    def _arun(self, **kwargs) -> Any:
        """
        Async version of _run (optional to implement)

        Args:
            **kwargs: Tool input arguments

        Returns:
            Tool output

        Raises:
            ToolExecutionError: If tool execution fails
        """
        # Default implementation runs sync version
        return self._run(**kwargs)

    def invoke(self, input_data: Union[str, Dict[str, Any]], **kwargs) -> Any:
        """
        Invoke the tool with input data

        Args:
            input_data: Input data (string or dict)
            **kwargs: Additional arguments

        Returns:
            Tool output

        Raises:
            ToolValidationError: If input is invalid
            ToolExecutionError: If execution fails
            ToolTimeoutError: If execution times out
        """
        start_time = time.time()

        try:
            # Parse and validate input
            parsed_input = self._parse_input(input_data)
            validated_input = self._validate_input(parsed_input)

            # Execute with timeout if configured
            if self.config.max_execution_time:
                result = self._execute_with_timeout(validated_input)
            else:
                result = self._run(**validated_input)

            # Create successful result
            execution_time = time.time() - start_time
            return ToolResult(
                output=result,
                success=True,
                execution_time=execution_time,
                metadata={"tool_name": self.config.name}
            )

        except Exception as e:
            execution_time = time.time() - start_time

            if isinstance(e, ToolError):
                # Re-raise tool errors
                if not self.config.handle_error:
                    raise
                return ToolResult(
                    output=str(e),
                    success=False,
                    error=str(e),
                    execution_time=execution_time,
                    metadata={"tool_name": self.config.name, "error_type": type(e).__name__}
                )

            # Handle other exceptions
            if self.config.handle_error:
                error_msg = f"Tool execution failed: {str(e)}"
                if self.config.verbose:
                    print(f"[{self.config.name}] Error: {error_msg}")
                return ToolResult(
                    output=error_msg,
                    success=False,
                    error=error_msg,
                    execution_time=execution_time,
                    metadata={"tool_name": self.config.name, "error_type": type(e).__name__}
                )
            else:
                raise ToolExecutionError(
                    f"Tool execution failed: {str(e)}",
                    tool_name=self.config.name,
                    cause=e
                )

    async def ainvoke(self, input_data: Union[str, Dict[str, Any]], **kwargs) -> Any:
        """
        Async version of invoke

        Args:
            input_data: Input data (string or dict)
            **kwargs: Additional arguments

        Returns:
            Tool output

        Raises:
            ToolValidationError: If input is invalid
            ToolExecutionError: If execution fails
            ToolTimeoutError: If execution times out
        """
        start_time = time.time()

        try:
            # Parse and validate input
            parsed_input = self._parse_input(input_data)
            validated_input = self._validate_input(parsed_input)

            # Execute with timeout if configured
            if self.config.max_execution_time:
                result = await self._aexecute_with_timeout(validated_input)
            else:
                result = await self._arun(**validated_input)

            # Create successful result
            execution_time = time.time() - start_time
            return ToolResult(
                output=result,
                success=True,
                execution_time=execution_time,
                metadata={"tool_name": self.config.name}
            )

        except Exception as e:
            execution_time = time.time() - start_time

            if isinstance(e, ToolError):
                # Re-raise tool errors
                if not self.config.handle_error:
                    raise
                return ToolResult(
                    output=str(e),
                    success=False,
                    error=str(e),
                    execution_time=execution_time,
                    metadata={"tool_name": self.config.name, "error_type": type(e).__name__}
                )

            # Handle other exceptions
            if self.config.handle_error:
                error_msg = f"Tool execution failed: {str(e)}"
                if self.config.verbose:
                    print(f"[{self.config.name}] Error: {error_msg}")
                return ToolResult(
                    output=error_msg,
                    success=False,
                    error=error_msg,
                    execution_time=execution_time,
                    metadata={"tool_name": self.config.name, "error_type": type(e).__name__}
                )
            else:
                raise ToolExecutionError(
                    f"Tool execution failed: {str(e)}",
                    tool_name=self.config.name,
                    cause=e
                )

    def run(self, inputs: Union[str, Dict[str, Any]], **kwargs) -> Any:
        """
        Implement BaseComponent run method

        Args:
            inputs: Input data
            **kwargs: Additional parameters

        Returns:
            Tool output or ToolResult
        """
        result = self.invoke(inputs, **kwargs)
        return result.output if result.success else result

    def _parse_input(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse input data into dictionary format

        Args:
            input_data: Input data to parse

        Returns:
            Parsed input dictionary

        Raises:
            ToolValidationError: If parsing fails
        """
        if isinstance(input_data, str):
            # If args_schema is defined, try to parse as JSON
            if self.args_schema:
                try:
                    import json
                    return json.loads(input_data)
                except json.JSONDecodeError:
                    # If JSON parsing fails, treat as single string input
                    return {"input": input_data}
            else:
                # No schema defined, treat as single string input
                return {"input": input_data}

        elif isinstance(input_data, dict):
            return input_data

        else:
            # Try to convert to string and treat as single input
            return {"input": str(input_data)}

    def _validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data against schema

        Args:
            input_data: Input data to validate

        Returns:
            Validated input data

        Raises:
            ToolValidationError: If validation fails
        """
        if self.args_schema:
            try:
                # Use Pydantic model for validation
                validated = self.args_schema(**input_data)
                return validated.model_dump()
            except Exception as e:
                raise ToolValidationError(
                    f"Input validation failed: {str(e)}",
                    tool_name=self.config.name,
                    input_data=input_data
                )

        # No validation schema, return as-is
        return input_data

    def _execute_with_timeout(self, input_data: Dict[str, Any]) -> Any:
        """
        Execute tool with timeout

        Args:
            input_data: Validated input data

        Returns:
            Tool output

        Raises:
            ToolTimeoutError: If execution times out
        """
        import signal
        import threading

        result_container = [None]
        exception_container = [None]

        def target():
            try:
                result_container[0] = self._run(**input_data)
            except Exception as e:
                exception_container[0] = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.config.max_execution_time)

        if thread.is_alive():
            raise ToolTimeoutError(
                f"Tool execution timed out after {self.config.max_execution_time} seconds",
                tool_name=self.config.name,
                timeout_seconds=self.config.max_execution_time
            )

        if exception_container[0]:
            raise exception_container[0]

        return result_container[0]

    async def _aexecute_with_timeout(self, input_data: Dict[str, Any]) -> Any:
        """
        Async execute tool with timeout

        Args:
            input_data: Validated input data

        Returns:
            Tool output

        Raises:
            ToolTimeoutError: If execution times out
        """
        import asyncio

        try:
            return await asyncio.wait_for(
                self._arun(**input_data),
                timeout=self.config.max_execution_time
            )
        except asyncio.TimeoutError:
            raise ToolTimeoutError(
                f"Tool execution timed out after {self.config.max_execution_time} seconds",
                tool_name=self.config.name,
                timeout_seconds=self.config.max_execution_time
            )

    def _create_default_config(self) -> ToolConfig:
        """
        Create default configuration for the tool

        Returns:
            Default ToolConfig
        """
        return ToolConfig(
            name=getattr(self, 'name', self.__class__.__name__),
            description=getattr(self, 'description', f"{self.__class__.__name__} tool"),
        )

    def get_schema(self) -> ToolSchema:
        """
        Get tool schema

        Returns:
            Tool schema
        """
        inputs = []

        if self.args_schema:
            # Extract input schema from Pydantic model
            schema = self.args_schema.model_json_schema()
            properties = schema.get("properties", {})
            required_fields = schema.get("required", [])

            for field_name, field_info in properties.items():
                tool_input = ToolInput(
                    name=field_name,
                    type=field_info.get("type", "string"),
                    description=field_info.get("description", ""),
                    required=field_name in required_fields,
                    default=field_info.get("default")
                )
                inputs.append(tool_input)
        else:
            # Try to extract from function signature
            sig = inspect.signature(self._run)
            for param_name, param in sig.parameters.items():
                if param_name != "self":
                    tool_input = ToolInput(
                        name=param_name,
                        type="string",  # Default to string
                        description=f"Parameter {param_name}",
                        required=param.default == inspect.Parameter.empty
                    )
                    inputs.append(tool_input)

        return ToolSchema(
            name=self.config.name,
            description=self.config.description,
            inputs=inputs,
            output_type="string",
            metadata=self.config.metadata
        )

    @property
    def name(self) -> str:
        """Get tool name"""
        return self.config.name

    @property
    def description(self) -> str:
        """Get tool description"""
        return self.config.description

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}')"

    def __str__(self) -> str:
        return f"{self.config.name}: {self.config.description}"


class Tool(BaseTool):
    """
    Simple tool wrapper for functions
    """

    def __init__(
        self,
        name: str,
        func: Callable,
        description: str,
        args_schema: Optional[type[BaseModel]] = None,
        **kwargs
    ):
        """
        Initialize tool from function

        Args:
            name: Tool name
            func: Function to wrap
            description: Tool description
            args_schema: Optional input schema
            **kwargs: Additional configuration
        """
        self._func = func  # Make private to avoid Pydantic conflicts
        self._description = description

        # Create config
        config = ToolConfig(
            name=name,
            description=description,
            **kwargs
        )

        super().__init__(config=config, args_schema=args_schema)

    def _create_default_config(self) -> ToolConfig:
        """Create default config"""
        return ToolConfig(
            name=self.config.name,
            description=self._description
        )

    def _run(self, **kwargs) -> Any:
        """Run the wrapped function"""
        return self._func(**kwargs)

    def get_schema(self) -> ToolSchema:
        """Get tool schema from function signature"""
        if self.args_schema:
            return super().get_schema()

        # Extract schema from function signature
        sig = inspect.signature(self._func)
        inputs = []

        for param_name, param in sig.parameters.items():
            param_type = "string"
            param_desc = f"Parameter {param_name}"
            param_required = param.default == inspect.Parameter.empty
            param_default = None if param_required else param.default

            # Try to infer type
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "float"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif hasattr(param.annotation, '__origin__'):
                    if param.annotation.__origin__ == list:
                        param_type = "array"
                    elif param.annotation.__origin__ == dict:
                        param_type = "object"

            tool_input = ToolInput(
                name=param_name,
                type=param_type,
                description=param_desc,
                required=param_required,
                default=param_default
            )
            inputs.append(tool_input)

        return ToolSchema(
            name=self.config.name,
            description=self.config.description,
            inputs=inputs,
            output_type="any",
            metadata=self.config.metadata
        )