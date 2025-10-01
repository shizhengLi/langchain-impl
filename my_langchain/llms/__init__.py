# -*- coding: utf-8 -*-
"""
LLM module

Provides unified interfaces and specific implementations for various LLMs.
"""

from .base import BaseLLM
from .mock_llm import MockLLM
from .types import LLMResult, LLMConfig

__all__ = [
    "BaseLLM",
    "MockLLM",
    "LLMResult",
    "LLMConfig"
]