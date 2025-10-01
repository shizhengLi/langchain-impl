# -*- coding: utf-8 -*-
"""
LLM related data type definitions
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class LLMConfig(BaseModel):
    """LLM configuration class"""

    model_name: str = Field(..., description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation randomness, 0-2")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum generation tokens")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop words")
    timeout: int = Field(default=60, ge=1, description="Request timeout (seconds)")

    class Config:
        extra = "allow"


class LLMResult(BaseModel):
    """LLM generation result class"""

    text: str = Field(..., description="Generated text content")
    prompt: str = Field(..., description="Original prompt")
    model_name: str = Field(..., description="Model name used")
    finish_reason: Optional[str] = Field(default=None, description="Generation finish reason")
    token_usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage")
    generation_time: Optional[float] = Field(default=None, description="Generation time (seconds)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Generation timestamp")

    @property
    def prompt_tokens(self) -> Optional[int]:
        """Get prompt tokens"""
        return self.token_usage.get("prompt_tokens") if self.token_usage else None

    @property
    def completion_tokens(self) -> Optional[int]:
        """Get completion tokens"""
        return self.token_usage.get("completion_tokens") if self.token_usage else None

    @property
    def total_tokens(self) -> Optional[int]:
        """Get total tokens"""
        return self.token_usage.get("total_tokens") if self.token_usage else None


class LLMError(Exception):
    """Base class for LLM related errors"""

    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class LLMTimeoutError(LLMError):
    """LLM timeout error"""

    def __init__(self, message: str = "LLM request timed out", timeout: Optional[int] = None):
        super().__init__(message, "TIMEOUT", {"timeout": timeout})
        self.timeout = timeout


class LLMRateLimitError(LLMError):
    """LLM rate limit error"""

    def __init__(self, message: str = "LLM rate limit exceeded", retry_after: Optional[int] = None):
        super().__init__(message, "RATE_LIMIT", {"retry_after": retry_after})
        self.retry_after = retry_after


class LLMTokenLimitError(LLMError):
    """LLM token limit error"""

    def __init__(self, message: str = "LLM token limit exceeded", limit: Optional[int] = None):
        super().__init__(message, "TOKEN_LIMIT", {"limit": limit})
        self.limit = limit


class LLMInvalidRequestError(LLMError):
    """LLM invalid request error"""

    def __init__(self, message: str = "Invalid LLM request", parameter: Optional[str] = None):
        super().__init__(message, "INVALID_REQUEST", {"parameter": parameter})
        self.parameter = parameter


class LLMServiceUnavailableError(LLMError):
    """LLM service unavailable error"""

    def __init__(self, message: str = "LLM service unavailable"):
        super().__init__(message, "SERVICE_UNAVAILABLE")


class LLMAuthenticationError(LLMError):
    """LLM authentication error"""

    def __init__(self, message: str = "LLM authentication failed"):
        super().__init__(message, "AUTHENTICATION_FAILED")