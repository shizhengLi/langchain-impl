"""
测试 LLM 模块
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock

from my_langchain.llms.types import (
    LLMConfig, LLMResult, LLMError, LLMTimeoutError,
    LLMRateLimitError, LLMTokenLimitError
)
from my_langchain.llms.mock_llm import MockLLM


class TestLLMConfig:
    """测试 LLM 配置类"""

    def test_llm_config_creation(self):
        """测试创建 LLM 配置"""
        config = LLMConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )

        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.top_p == 1.0
        assert config.frequency_penalty == 0.0

    def test_llm_config_validation(self):
        """测试配置验证"""
        # 测试温度范围
        with pytest.raises(ValueError):
            LLMConfig(model_name="test", temperature=-0.1)

        with pytest.raises(ValueError):
            LLMConfig(model_name="test", temperature=2.1)


class TestLLMResult:
    """测试 LLM 结果类"""

    def test_llm_result_creation(self):
        """测试创建 LLM 结果"""
        result = LLMResult(
            text="Hello world",
            prompt="Say hello",
            model_name="test-model",
            token_usage={"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
        )

        assert result.text == "Hello world"
        assert result.prompt == "Say hello"
        assert result.model_name == "test-model"
        assert result.prompt_tokens == 2
        assert result.completion_tokens == 2
        assert result.total_tokens == 4

    def test_llm_result_properties(self):
        """测试 LLM 结果属性"""
        result = LLMResult(
            text="Hello",
            prompt="Say hello",
            model_name="test-model"
        )

        assert result.prompt_tokens is None
        assert result.completion_tokens is None
        assert result.total_tokens is None


class TestMockLLM:
    """测试 Mock LLM"""

    def test_mock_llm_creation(self):
        """测试创建 Mock LLM"""
        llm = MockLLM(
            model_name="mock-gpt",
            temperature=0.5,
            response_delay=0.0
        )

        assert llm.model_name == "mock-gpt"
        assert llm.temperature == 0.5
        assert llm.response_delay == 0.0

    def test_mock_llm_generate(self):
        """测试 Mock LLM 生成"""
        llm = MockLLM(response_delay=0.0, temperature=0.0)  # 使用确定性温度

        # 测试基本生成
        response = llm.generate("Hello")
        assert isinstance(response, str)
        assert len(response) > 0

        # 测试带结果的生成
        result = llm.generate_with_result("Hello")
        assert isinstance(result, LLMResult)
        assert result.text == response
        assert result.prompt == "Hello"
        assert result.model_name == "mock-gpt-3.5-turbo"  # 默认模型名
        assert result.finish_reason == "stop"

    def test_mock_llm_batch_generate(self):
        """测试 Mock LLM 批量生成"""
        llm = MockLLM(response_delay=0.0)

        prompts = ["Hello", "How are you?", "Goodbye"]
        responses = llm.generate_batch(prompts)

        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0

    def test_mock_llm_predefined_responses(self):
        """测试预定义响应"""
        responses = {
            "Hello": "Hi there!",
            "How are you?": "I'm doing well, thank you!"
        }
        llm = MockLLM(responses=responses, response_delay=0.0)

        # 测试预定义响应
        assert llm.generate("Hello") == "Hi there!"
        assert llm.generate("How are you?") == "I'm doing well, thank you!"

        # 测试未定义的响应
        response = llm.generate("Unknown prompt")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_mock_llm_set_responses(self):
        """测试设置响应"""
        llm = MockLLM(response_delay=0.0)

        # 添加响应
        llm.add_response("Test", "Test response")
        assert llm.generate("Test") == "Test response"

        # 清空响应
        llm.clear_responses()
        response = llm.generate("Test")
        assert response != "Test response"

    def test_mock_llm_validation(self):
        """测试输入验证"""
        llm = MockLLM(response_delay=0.0)

        # 测试空提示词
        with pytest.raises(LLMError):
            llm.generate("")

        # 测试非字符串提示词
        with pytest.raises(LLMError):
            llm.generate(123)

    def test_mock_llm_model_info(self):
        """测试获取模型信息"""
        llm = MockLLM(
            model_name="mock-gpt-4",
            temperature=0.8,
            max_tokens=2000
        )

        info = llm.get_model_info()
        assert info["model_name"] == "mock-gpt-4"
        assert info["config"]["temperature"] == 0.8
        assert info["config"]["max_tokens"] == 2000
        assert info["type"] == "MockLLM"

    def test_mock_llm_token_estimation(self):
        """测试令牌估算"""
        llm = MockLLM()

        # 简单文本
        tokens = llm.estimate_tokens("Hello world")
        assert tokens == 2

        # 空字符串
        tokens = llm.estimate_tokens("")
        assert tokens == 0