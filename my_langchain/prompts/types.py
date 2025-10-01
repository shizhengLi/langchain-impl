# -*- coding: utf-8 -*-
"""
Prompt template types and data structures
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class PromptTemplateConfig(BaseModel):
    """Configuration for prompt templates"""
    template_format: str = Field(default="f-string", description="Template format (f-string, jinja2)")
    validate_template: bool = Field(default=True, description="Whether to validate template variables")
    strict_variables: bool = Field(default=True, description="Whether to require all variables to be provided")


class PromptInputVariables(BaseModel):
    """Input variables for prompt templates"""
    variables: Dict[str, Any] = Field(default_factory=dict, description="Template variables")
    partial_variables: Optional[Dict[str, Any]] = Field(default=None, description="Partial variables that will be merged")


class PromptTemplateResult(BaseModel):
    """Result of prompt template formatting"""
    text: str = Field(..., description="Formatted prompt text")
    variables: Dict[str, Any] = Field(..., description="Used variables")
    missing_variables: List[str] = Field(default_factory=list, description="Missing variables")
    template_name: Optional[str] = Field(default=None, description="Template name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PromptTemplateError(Exception):
    """Base error class for prompt templates"""
    def __init__(self, message: str, error_type: str = "PROMPT_ERROR", details: Optional[Dict] = None):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}


class TemplateValidationError(PromptTemplateError):
    """Raised when template validation fails"""
    def __init__(self, message: str, template: str, variables: List[str]):
        super().__init__(
            message=message,
            error_type="TEMPLATE_VALIDATION_ERROR",
            details={"template": template, "expected_variables": variables}
        )


class VariableMissingError(PromptTemplateError):
    """Raised when required variables are missing"""
    def __init__(self, message: str, missing_vars: List[str]):
        super().__init__(
            message=message,
            error_type="VARIABLE_MISSING_ERROR",
            details={"missing_variables": missing_vars}
        )


class TemplateFormatError(PromptTemplateError):
    """Raised when template format is invalid"""
    def __init__(self, message: str, template: str, format_type: str):
        super().__init__(
            message=message,
            error_type="TEMPLATE_FORMAT_ERROR",
            details={"template": template, "format": format_type}
        )