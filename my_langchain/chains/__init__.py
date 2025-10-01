# -*- coding: utf-8 -*-
"""
Chain module

Provides unified interfaces and implementations for various chain types.
"""

from .base import BaseChain
from .llm_chain import LLMChain
from .sequential_chain import SequentialChain
from .simple_chain import SimpleChain
from .types import (
    ChainConfig, ChainResult, ChainInput, ChainError,
    ChainValidationError, ChainExecutionError, ChainTimeoutError
)

__all__ = [
    "BaseChain",
    "LLMChain",
    "SequentialChain",
    "SimpleChain",
    "ChainConfig",
    "ChainResult",
    "ChainInput",
    "ChainError",
    "ChainValidationError",
    "ChainExecutionError",
    "ChainTimeoutError"
]