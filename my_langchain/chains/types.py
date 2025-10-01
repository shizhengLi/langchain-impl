# -*- coding: utf-8 -*-
"""
Chain types and data structures
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ChainConfig(BaseModel):
    """Configuration for chains"""
    verbose: bool = Field(default=False, description="Enable verbose logging")
    memory: Optional[Dict[str, Any]] = Field(default=None, description="Memory configuration")
    return_intermediate_steps: bool = Field(default=False, description="Return intermediate steps")
    input_key: Optional[str] = Field(default=None, description="Primary input key")
    output_key: Optional[str] = Field(default=None, description="Primary output key")


class ChainResult(BaseModel):
    """Result from chain execution"""
    output: Any = Field(..., description="Chain output")
    intermediate_steps: List[Dict[str, Any]] = Field(default_factory=list, description="Intermediate steps")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")


class ChainInput(BaseModel):
    """Input for chain execution"""
    inputs: Dict[str, Any] = Field(..., description="Input values")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Runtime configuration")


class ChainError(Exception):
    """Base error class for chains"""
    def __init__(self, message: str, error_type: str = "CHAIN_ERROR", details: Optional[Dict] = None):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}


class ChainValidationError(ChainError):
    """Raised when chain validation fails"""
    def __init__(self, message: str, chain_inputs: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_type="CHAIN_VALIDATION_ERROR",
            details={"chain_inputs": chain_inputs}
        )


class ChainExecutionError(ChainError):
    """Raised when chain execution fails"""
    def __init__(self, message: str, step: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(
            message=message,
            error_type="CHAIN_EXECUTION_ERROR",
            details={"step": step, "cause": str(cause) if cause else None}
        )


class ChainTimeoutError(ChainError):
    """Raised when chain execution times out"""
    def __init__(self, timeout: float):
        super().__init__(
            message=f"Chain execution timed out after {timeout} seconds",
            error_type="CHAIN_TIMEOUT_ERROR",
            details={"timeout": timeout}
        )