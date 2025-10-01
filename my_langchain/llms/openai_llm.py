# -*- coding: utf-8 -*-
"""
OpenAI LLM 实现

基于 OpenAI API 的 LLM 实现。
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

from my_langchain.llms.base import BaseLLM
from my_langchain.llms.types import LLMResult, LLMConfig, LLMError, LLMTimeoutError, LLMRateLimitError


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM 实现

    使用 OpenAI API 进行文本生成。
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        初始化 OpenAI LLM

        Args:
            api_key: OpenAI API 密钥
            model_name: 模型名称
            temperature: 温度参数
            max_tokens: 最大令牌数
            organization: OpenAI 组织ID
            base_url: API 基础URL
            **kwargs: 其他参数
        """
        config = LLMConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        super().__init__(config)

        self.api_key = api_key
        self.organization = organization
        self.base_url = base_url

        # 延迟导入 openai，避免在没有安装的情况下出错
        try:
            import openai
            self.openai = openai
            self.client = openai.OpenAI(
                api_key=api_key,
                organization=organization,
                base_url=base_url
            )
        except ImportError:
            raise ImportError(
                "OpenAI 库未安装。请运行: pip install openai"
            )

    def _generate(self, prompt: str, **kwargs) -> LLMResult:
        """
        使用 OpenAI API 生成文本

        Args:
            prompt: 输入提示词
            **kwargs: 其他参数

        Returns:
            生成结果
        """
        self._validate_prompt(prompt)

        try:
            # 合并配置
            config = self._merge_configs(**kwargs)

            # 调用 OpenAI API
            response = self.client.chat.completions.create(
                model=config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=config.get("temperature", self.temperature),
                max_tokens=config.get("max_tokens", self.max_tokens),
                top_p=config.get("top_p"),
                frequency_penalty=config.get("frequency_penalty"),
                presence_penalty=config.get("presence_penalty"),
                stop=config.get("stop"),
                timeout=config.get("timeout", self.config.timeout)
            )

            # 解析响应
            return self._parse_response(response, prompt)

        except self.openai.RateLimitError as e:
            raise LLMRateLimitError(
                str(e),
                retry_after=getattr(e, 'retry_after', None)
            )
        except self.openai.APITimeoutError as e:
            raise LLMTimeoutError(str(e))
        except self.openai.APIError as e:
            raise LLMError(f"OpenAI API 错误: {str(e)}")
        except Exception as e:
            raise LLMError(f"生成失败: {str(e)}")

    async def _agenerate(self, prompt: str, **kwargs) -> LLMResult:
        """
        使用 OpenAI API 异步生成文本

        Args:
            prompt: 输入提示词
            **kwargs: 其他参数

        Returns:
            生成结果
        """
        self._validate_prompt(prompt)

        try:
            # 合并配置
            config = self._merge_configs(**kwargs)

            # 创建异步客户端
            async_client = self.openai.AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url
            )

            # 调用 OpenAI API
            response = await async_client.chat.completions.create(
                model=config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=config.get("temperature", self.temperature),
                max_tokens=config.get("max_tokens", self.max_tokens),
                top_p=config.get("top_p"),
                frequency_penalty=config.get("frequency_penalty"),
                presence_penalty=config.get("presence_penalty"),
                stop=config.get("stop"),
                timeout=config.get("timeout", self.config.timeout)
            )

            # 解析响应
            return self._parse_response(response, prompt)

        except self.openai.RateLimitError as e:
            raise LLMRateLimitError(
                str(e),
                retry_after=getattr(e, 'retry_after', None)
            )
        except self.openai.APITimeoutError as e:
            raise LLMTimeoutError(str(e))
        except self.openai.APIError as e:
            raise LLMError(f"OpenAI API 错误: {str(e)}")
        except Exception as e:
            raise LLMError(f"异步生成失败: {str(e)}")

    def _generate_batch(self, prompts: List[str], **kwargs) -> List[LLMResult]:
        """
        批量生成文本

        对于 OpenAI API，我们逐个处理，因为 API 没有真正的批量端点。

        Args:
            prompts: 输入提示词列表
            **kwargs: 其他参数

        Returns:
            生成结果列表
        """
        self._validate_prompts(prompts)

        results = []
        for prompt in prompts:
            result = self._generate(prompt, **kwargs)
            results.append(result)

        return results

    def _parse_response(self, response: Any, prompt: str) -> LLMResult:
        """
        解析 OpenAI API 响应

        Args:
            response: OpenAI API 响应
            prompt: 原始提示词

        Returns:
            解析后的结果
        """
        choice = response.choices[0]

        return LLMResult(
            text=choice.message.content or "",
            prompt=prompt,
            model_name=response.model,
            finish_reason=choice.finish_reason,
            token_usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None
            } if response.usage else None,
            metadata={
                "system_fingerprint": response.system_fingerprint,
                "created": response.created,
                "id": response.id,
                "provider": "openai"
            }
        )

    def get_available_models(self) -> List[str]:
        """
        获取可用的模型列表

        Returns:
            模型名称列表
        """
        try:
            models = self.client.models.list()
            # 过滤出聊天模型
            chat_models = [
                model.id for model in models.data
                if "gpt" in model.id and ("chat" in model.id or "instruct" in model.id)
            ]
            return sorted(chat_models)
        except Exception as e:
            # 如果无法获取模型列表，返回常用模型
            return [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k"
            ]

    def validate_api_key(self) -> bool:
        """
        验证 API 密钥是否有效

        Returns:
            API 密钥是否有效
        """
        try:
            # 尝试获取模型列表来验证 API 密钥
            self.client.models.list()
            return True
        except Exception:
            return False

    def estimate_tokens(self, text: str) -> int:
        """
        使用 tiktoken 估算令牌数

        Args:
            text: 输入文本

        Returns:
            估算的令牌数
        """
        try:
            import tiktoken

            # 获取模型的编码器
            try:
                encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # 如果找不到模型的编码器，使用默认的
                encoding = tiktoken.get_encoding("cl100k_base")

            # 计算令牌数
            return len(encoding.encode(text))
        except ImportError:
            # 如果没有安装 tiktoken，使用简单估算
            return super().estimate_tokens(text)