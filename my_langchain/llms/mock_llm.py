# -*- coding: utf-8 -*-
"""
Mock LLM implementation

Mock LLM implementation for testing and development phases.
"""

import random
import time
import asyncio
from typing import Any, Dict, List

from my_langchain.llms.base import BaseLLM
from my_langchain.llms.types import LLMResult, LLMConfig
from pydantic import Field


class MockLLM(BaseLLM):
    """
    Mock LLM implementation

    Used for testing and development, provides predictable responses.
    """

    response_delay: float = Field(default=0.1, description="Response delay in seconds")
    responses: Dict[str, str] = Field(default_factory=dict, description="Predefined responses")

    def __init__(
        self,
        model_name: str = "mock-gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 100,
        response_delay: float = 0.1,
        responses: Dict[str, str] = None,
        **kwargs
    ):
        """
        Initialize Mock LLM

        Args:
            model_name: Model name
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            response_delay: Response delay (seconds)
            responses: Predefined response mapping
            **kwargs: Other parameters
        """
        config = LLMConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        super().__init__(
            config=config,
            response_delay=response_delay,
            responses=responses or {}
        )

    def _generate(self, prompt: str, **kwargs) -> LLMResult:
        """
        Generate mock response

        Args:
            prompt: Input prompt
            **kwargs: Other parameters

        Returns:
            Mock generation result
        """
        self._validate_prompt(prompt)

        # Simulate delay
        if self.response_delay > 0:
            time.sleep(self.response_delay)

        # Get response text
        if prompt in self.responses:
            response_text = self.responses[prompt]
        else:
            response_text = self._generate_response(prompt, **kwargs)

        # Calculate token count (simple estimation)
        prompt_tokens = self.estimate_tokens(prompt)
        completion_tokens = self.estimate_tokens(response_text)

        return LLMResult(
            text=response_text,
            prompt=prompt,
            model_name=self.model_name,
            finish_reason="stop",
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            metadata={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "mock_model": True
            }
        )

    async def _agenerate(self, prompt: str, **kwargs) -> LLMResult:
        """
        Async generate mock response

        Args:
            prompt: Input prompt
            **kwargs: Other parameters

        Returns:
            Mock generation result
        """
        self._validate_prompt(prompt)

        # Simulate async delay
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)

        # Get response text
        if prompt in self.responses:
            response_text = self.responses[prompt]
        else:
            response_text = self._generate_response(prompt, **kwargs)

        # Calculate token count
        prompt_tokens = self.estimate_tokens(prompt)
        completion_tokens = self.estimate_tokens(response_text)

        return LLMResult(
            text=response_text,
            prompt=prompt,
            model_name=self.model_name,
            finish_reason="stop",
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            metadata={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "mock_model": True,
                "async": True
            }
        )

    def _generate_batch(self, prompts: List[str], **kwargs) -> List[LLMResult]:
        """
        Batch generate mock responses

        Args:
            prompts: Input prompt list
            **kwargs: Other parameters

        Returns:
            Mock generation result list
        """
        self._validate_prompts(prompts)

        results = []
        for prompt in prompts:
            result = self._generate(prompt, **kwargs)
            results.append(result)

        return results

    def _generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response text

        Args:
            prompt: Input prompt
            **kwargs: Other parameters

        Returns:
            Generated response text
        """
        # Get merged configuration
        config = self._merge_configs(**kwargs)
        temperature = config.get("temperature", self.temperature)
        max_tokens = config.get("max_tokens", self.max_tokens)

        # Generate response based on prompt
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["hello", "hi", "你好"]):
            responses = [
                "Hello! How can I help you today?",
                "Hi there! What would you like to know?",
                "你好！今天我能为你做些什么？",
                "您好！有什么我可以帮助您的吗？"
            ]
        elif any(word in prompt_lower for word in ["what", "什么", "how", "如何"]):
            responses = [
                "That's a great question! Let me think about it...",
                "Based on my understanding, I would say...",
                "这是一个很好的问题！让我思考一下...",
                "根据我的理解，我认为..."
            ]
        elif any(word in prompt_lower for word in ["thanks", "thank", "谢谢"]):
            responses = [
                "You're welcome! Is there anything else I can help with?",
                "Happy to help! Feel free to ask if you have more questions.",
                "不客气！还有什么我可以帮助的吗？",
                "很乐意帮助！如果您还有更多问题，请随时询问。"
            ]
        else:
            responses = [
                "I understand your input. Here's what I think...",
                "That's an interesting point. Let me elaborate...",
                "Based on the context, I would suggest...",
                "这是一个有趣的观点。让我详细说明...",
                "根据上下文，我建议..."
            ]

        # Select response based on temperature
        if temperature < 0.3:
            # Low temperature: select first response (more deterministic)
            response = responses[0]
        elif temperature < 0.8:
            # Medium temperature: random selection
            response = random.choice(responses)
        else:
            # High temperature: generate more random response
            base_response = random.choice(responses)
            random_suffixes = [
                " This is quite fascinating!",
                " 这个话题很有意思！",
                " I hope this helps!",
                " 希望这有帮助！",
                " Let me know if you need more details.",
                " 如果需要更多细节，请告诉我。"
            ]
            response = base_response + random.choice(random_suffixes)

        # Limit response length to simulate max_tokens
        if max_tokens and len(response) > max_tokens:
            response = response[:max_tokens] + "..."

        return response

    def set_responses(self, responses: Dict[str, str]) -> None:
        """
        Set predefined response mapping

        Args:
            responses: Response mapping dictionary
        """
        self.responses = responses

    def add_response(self, prompt: str, response: str) -> None:
        """
        Add single predefined response

        Args:
            prompt: Prompt
            response: Response text
        """
        self.responses[prompt] = response

    def clear_responses(self) -> None:
        """Clear predefined responses"""
        self.responses = {}