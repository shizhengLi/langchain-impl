# -*- coding: utf-8 -*-
"""
Prompt template module

Provides unified interfaces and implementations for various prompt templates.
"""

from .base import BasePromptTemplate
from .prompt_template import PromptTemplate
from .few_shot import FewShotPromptTemplate
from .chat import ChatPromptTemplate, ChatMessage, ChatMessageType
from .types import (
    PromptTemplateConfig, PromptTemplateResult, PromptInputVariables,
    PromptTemplateError, TemplateValidationError, VariableMissingError,
    TemplateFormatError
)

__all__ = [
    "BasePromptTemplate",
    "PromptTemplate",
    "FewShotPromptTemplate",
    "ChatPromptTemplate",
    "ChatMessage",
    "ChatMessageType",
    "PromptTemplateConfig",
    "PromptTemplateResult",
    "PromptInputVariables",
    "PromptTemplateError",
    "TemplateValidationError",
    "VariableMissingError",
    "TemplateFormatError"
]