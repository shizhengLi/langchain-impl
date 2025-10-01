# -*- coding: utf-8 -*-
"""
Base LLM implementation

Provides common functionality and utility methods for LLMs.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

from my_langchain.base.base import BaseLLM as BaseLLMComponent
from my_langchain.llms.types import (
    LLMConfig, LLMResult, LLMError, LLMTimeoutError,
    LLMRateLimitError, LLMTokenLimitError
)
from pydantic import ConfigDict, Field


class BaseLLM(BaseLLMComponent):
    """
    Base implementation class for LLM

    Inherits from BaseComponent and implements common LLM functionality.
    """

    config: LLMConfig = Field(..., description="LLM configuration")

    def __init__(self, config: LLMConfig, **kwargs):
        """
        Initialize LLM

        Args:
            config: LLM configuration
            **kwargs: Additional parameters
        """
        # Initialize with config field
        super().__init__(
            config=config,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **kwargs
        )

    @abstractmethod
    def _generate(self, prompt: str, **kwargs) -> LLMResult:
        """
        Actual text generation method

        Subclasses must implement this method.

        Args:
            prompt: Input prompt
            **kwargs: Other generation parameters

        Returns:
            Generation result
        """
        pass

    @abstractmethod
    async def _agenerate(self, prompt: str, **kwargs) -> LLMResult:
        """
        Async text generation method

        Subclasses must implement this method.

        Args:
            prompt: Input prompt
            **kwargs: Other generation parameters

        Returns:
            Generation result
        """
        pass

    @abstractmethod
    def _generate_batch(self, prompts: List[str], **kwargs) -> List[LLMResult]:
        """
        Batch generation method

        Args:
            prompts: Input prompt list
            **kwargs: Other generation parameters

        Returns:
            Generation result list
        """
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text response

        Args:
            prompt: Input prompt
            **kwargs: Other generation parameters

        Returns:
            Generated text response
        """
        result = self._generate_with_handling(prompt, **kwargs)
        return result.text

    async def agenerate(self, prompt: str, **kwargs) -> str:
        """
        Async generate text response

        Args:
            prompt: Input prompt
            **kwargs: Other generation parameters

        Returns:
            Generated text response
        """
        result = await self._agenerate_with_handling(prompt, **kwargs)
        return result.text

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Batch generate text responses

        Args:
            prompts: Input prompt list
            **kwargs: Other generation parameters

        Returns:
            Generated text response list
        """
        results = self._generate_batch_with_handling(prompts, **kwargs)
        return [result.text for result in results]

    def generate_with_result(self, prompt: str, **kwargs) -> LLMResult:
        """
        Generate text and return full result

        Args:
            prompt: Input prompt
            **kwargs: Other generation parameters

        Returns:
            Complete generation result
        """
        return self._generate_with_handling(prompt, **kwargs)

    async def agenerate_with_result(self, prompt: str, **kwargs) -> LLMResult:
        """
        Async generate text and return full result

        Args:
            prompt: Input prompt
            **kwargs: Other generation parameters

        Returns:
            Complete generation result
        """
        return await self._agenerate_with_handling(prompt, **kwargs)

    def generate_batch_with_result(self, prompts: List[str], **kwargs) -> List[LLMResult]:
        """
        Batch generate text and return full results

        Args:
            prompts: Input prompt list
            **kwargs: Other generation parameters

        Returns:
            Complete generation result list
        """
        return self._generate_batch_with_handling(prompts, **kwargs)

    def _generate_with_handling(self, prompt: str, **kwargs) -> LLMResult:
        """
        Generate method with error handling

        Args:
            prompt: Input prompt
            **kwargs: Other generation parameters

        Returns:
            Generation result

        Raises:
            LLMError: Various LLM related errors
        """
        try:
            start_time = time.time()
            result = self._generate(prompt, **kwargs)
            if result.generation_time is None:
                result.generation_time = time.time() - start_time
            return result
        except LLMError:
            raise
        except asyncio.TimeoutError:
            raise LLMTimeoutError(timeout=self.config.timeout)
        except Exception as e:
            raise LLMError(f"Generation failed: {str(e)}")

    async def _agenerate_with_handling(self, prompt: str, **kwargs) -> LLMResult:
        """
        Async generate method with error handling

        Args:
            prompt: Input prompt
            **kwargs: Other generation parameters

        Returns:
            Generation result

        Raises:
            LLMError: Various LLM related errors
        """
        try:
            start_time = time.time()
            result = await self._agenerate(prompt, **kwargs)
            if result.generation_time is None:
                result.generation_time = time.time() - start_time
            return result
        except LLMError:
            raise
        except asyncio.TimeoutError:
            raise LLMTimeoutError(timeout=self.config.timeout)
        except Exception as e:
            raise LLMError(f"Async generation failed: {str(e)}")

    def _generate_batch_with_handling(self, prompts: List[str], **kwargs) -> List[LLMResult]:
        """
        Batch generate method with error handling

        Args:
            prompts: Input prompt list
            **kwargs: Other generation parameters

        Returns:
            Generation result list

        Raises:
            LLMError: Various LLM related errors
        """
        try:
            start_time = time.time()
            results = self._generate_batch(prompts, **kwargs)

            # Calculate average time if not provided
            if results and not results[0].generation_time:
                total_time = time.time() - start_time
                avg_time = total_time / len(results)
                for result in results:
                    result.generation_time = avg_time

            return results
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Batch generation failed: {str(e)}")

    def _merge_configs(self, **kwargs) -> Dict[str, Any]:
        """
        Merge configuration and runtime parameters

        Args:
            **kwargs: Runtime parameters

        Returns:
            Merged configuration dictionary
        """
        # Extract parameters from config
        config_dict = self.config.model_dump()

        # Update with runtime parameters
        for key, value in kwargs.items():
            if value is not None:
                config_dict[key] = value

        return config_dict

    def _validate_prompt(self, prompt: str) -> None:
        """
        Validate prompt

        Args:
            prompt: Input prompt

        Raises:
            LLMInvalidRequestError: When prompt is invalid
        """
        if not isinstance(prompt, str):
            raise LLMError("Prompt must be a string", "INVALID_REQUEST", {"parameter": "prompt"})

        if not prompt.strip():
            raise LLMError("Prompt cannot be empty", "INVALID_REQUEST", {"parameter": "prompt"})

    def _validate_prompts(self, prompts: List[str]) -> None:
        """
        Validate prompt list

        Args:
            prompts: Input prompt list

        Raises:
            LLMInvalidRequestError: When prompts are invalid
        """
        if not isinstance(prompts, list):
            raise LLMError("Prompts must be a list", "INVALID_REQUEST", {"parameter": "prompts"})

        if not prompts:
            raise LLMError("Prompt list cannot be empty", "INVALID_REQUEST", {"parameter": "prompts"})

        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                raise LLMError(f"Prompt {i+1} must be a string", "INVALID_REQUEST", {"parameter": "prompts"})
            if not prompt.strip():
                raise LLMError(f"Prompt {i+1} cannot be empty", "INVALID_REQUEST", {"parameter": "prompts"})

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information

        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "config": self.config.model_dump(),
            "type": self.__class__.__name__
        }

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text

        This is a simple estimation method, subclasses can override for more accurate estimation.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Simple estimation: split by spaces and punctuation
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return len(tokens)