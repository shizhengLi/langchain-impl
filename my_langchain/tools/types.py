# -*- coding: utf-8 -*-
"""
Tool types and data structures
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field


class ToolConfig(BaseModel):
    """
    Tool configuration
    """
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    return_direct: bool = Field(
        default=False,
        description="Whether the tool output should be returned directly"
    )
    verbose: bool = Field(
        default=False,
        description="Whether to print verbose output"
    )
    handle_error: bool = Field(
        default=True,
        description="Whether to handle errors gracefully"
    )
    max_execution_time: Optional[float] = Field(
        default=None,
        description="Maximum execution time in seconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class ToolResult(BaseModel):
    """
    Result of tool execution
    """
    output: Any = Field(..., description="Output of the tool")
    success: bool = Field(default=True, description="Whether execution was successful")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class ToolInput(BaseModel):
    """
    Tool input specification
    """
    name: str = Field(..., description="Name of the input parameter")
    type: str = Field(..., description="Type of the input parameter")
    description: str = Field(..., description="Description of the input parameter")
    required: bool = Field(default=True, description="Whether the parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value if not required")


class ToolSchema(BaseModel):
    """
    Tool input schema
    """
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of the tool")
    inputs: List[ToolInput] = Field(default_factory=list, description="Input parameters")
    output_type: str = Field(default="string", description="Type of output")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional schema metadata"
    )


# Error types
class ToolError(Exception):
    """
    Base tool error class
    """
    message: str
    tool_name: str
    details: Dict[str, Any] = {}

    def __init__(self, message: str, tool_name: str, details: Dict[str, Any] = None):
        self.message = message
        self.tool_name = tool_name
        self.details = details or {}
        super().__init__(message)


class ToolValidationError(ToolError):
    """
    Tool validation error
    """
    input_data: Any

    def __init__(self, message: str, tool_name: str, input_data: Any = None):
        self.input_data = input_data
        validation_details = {
            "input_data": input_data,
            "validation_error": True
        }
        super().__init__(message, tool_name, validation_details)


class ToolExecutionError(ToolError):
    """
    Tool execution error
    """
    cause: Optional[Exception] = None

    def __init__(self, message: str, tool_name: str, cause: Exception = None, details: Dict[str, Any] = None):
        self.cause = cause
        execution_details = {
            "cause": str(cause) if cause else None,
            "execution_error": True
        }
        if details:
            execution_details.update(details)
        super().__init__(message, tool_name, execution_details)


class ToolTimeoutError(ToolError):
    """
    Tool timeout error
    """
    timeout_seconds: float

    def __init__(self, message: str, tool_name: str, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        timeout_details = {
            "timeout_seconds": timeout_seconds,
            "timeout_error": True
        }
        super().__init__(message, tool_name, timeout_details)


class ToolPermissionError(ToolError):
    """
    Tool permission error
    """
    required_permission: str

    def __init__(self, message: str, tool_name: str, required_permission: str):
        self.required_permission = required_permission
        permission_details = {
            "required_permission": required_permission,
            "permission_error": True
        }
        super().__init__(message, tool_name, permission_details)


# Enums
class ToolType(str, Enum):
    """Tool type enumeration"""
    SEARCH = "search"
    CALCULATOR = "calculator"
    CODE = "code"
    DATABASE = "database"
    API = "api"
    FILE = "file"
    SYSTEM = "system"
    CUSTOM = "custom"


class ToolStatus(str, Enum):
    """Tool status enumeration"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    DISABLED = "disabled"


class InputType(str, Enum):
    """Input type enumeration"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ANY = "any"


class OutputType(str, Enum):
    """Output type enumeration"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ANY = "any"
    STREAM = "stream"