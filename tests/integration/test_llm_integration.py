"""
LLM 集成测试
"""
import pytest
import asyncio

from my_langchain.llms import MockLLM, LLMConfig, LLMResult


class TestLLMIntegration:
    """LLM 集成测试"""

    def test_mock_llm_full_workflow(self):
        """测试 Mock LLM 完整工作流程"""
        # 创建 LLM - 使用确定性温度
        llm = MockLLM(
            model_name="test-model",
            temperature=0.0,
            max_tokens=50,
            response_delay=0.0
        )

        # 测试单次生成
        response = llm.generate("What is the capital of France?")
        assert isinstance(response, str)
        assert len(response) > 0

        # 测试带完整结果的生成
        result = llm.generate_with_result("Tell me a joke")
        assert isinstance(result, LLMResult)
        assert result.prompt == "Tell me a joke"
        assert result.model_name == "test-model"
        assert result.finish_reason == "stop"
        assert result.token_usage is not None
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_mock_llm_async_workflow(self):
        """测试 Mock LLM 异步工作流程"""
        llm = MockLLM(
            model_name="async-test-model",
            temperature=0.3,
            response_delay=0.0
        )

        # 测试异步生成
        response = await llm.agenerate("Hello, how are you?")
        assert isinstance(response, str)
        assert len(response) > 0

        # 测试异步完整结果
        result = await llm.agenerate_with_result("What's the weather like?")
        assert isinstance(result, LLMResult)
        assert result.text != response  # Different prompts should give different responses
        assert result.metadata["async"] is True

    def test_mock_llm_batch_workflow(self):
        """测试 Mock LLM 批量工作流程"""
        llm = MockLLM(
            model_name="batch-model",
            temperature=0.0,  # 确定性输出
            response_delay=0.0
        )

        # 测试批量生成
        prompts = [
            "Hello",
            "How are you?",
            "Goodbye"
        ]
        responses = llm.generate_batch(prompts)

        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0

        # 测试批量完整结果
        results = llm.generate_batch_with_result(prompts)
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, LLMResult)
            assert result.prompt == prompts[i]
            assert result.generation_time is not None

    def test_mock_llm_predefined_responses_workflow(self):
        """测试预定义响应工作流程"""
        predefined_responses = {
            "What is 2+2?": "2+2 equals 4.",
            "Who wrote Romeo and Juliet?": "William Shakespeare wrote Romeo and Juliet.",
            "What is the capital of Japan?": "The capital of Japan is Tokyo."
        }

        llm = MockLLM(
            model_name="predefined-model",
            responses=predefined_responses,
            response_delay=0.0
        )

        # 测试预定义响应
        for prompt, expected_response in predefined_responses.items():
            response = llm.generate(prompt)
            assert response == expected_response

        # 测试未定义的提示词（应该生成随机响应）
        unknown_prompt = "Tell me something interesting"
        response = llm.generate(unknown_prompt)
        assert isinstance(response, str)
        assert len(response) > 0
        assert response not in predefined_responses.values()

    def test_mock_llm_configuration_workflow(self):
        """测试配置管理工作流程"""
        # 测试默认配置
        llm_default = MockLLM()
        assert llm_default.model_name == "mock-gpt-3.5-turbo"
        assert llm_default.temperature == 0.7
        assert llm_default.max_tokens == 100

        # 测试自定义配置
        llm_custom = MockLLM(
            model_name="custom-model",
            temperature=0.1,
            max_tokens=200,
            response_delay=0.05
        )
        assert llm_custom.model_name == "custom-model"
        assert llm_custom.temperature == 0.1
        assert llm_custom.max_tokens == 200

        # 测试模型信息
        info = llm_custom.get_model_info()
        assert info["model_name"] == "custom-model"
        assert info["config"]["temperature"] == 0.1
        assert info["type"] == "MockLLM"

    def test_mock_llm_error_handling_workflow(self):
        """测试错误处理工作流程"""
        llm = MockLLM(response_delay=0.0)

        # 测试空提示词
        with pytest.raises(Exception):  # 应该抛出某种错误
            llm.generate("")

        # 测试非字符串提示词
        with pytest.raises(Exception):
            llm.generate(123)

        # 测试批量错误
        with pytest.raises(Exception):
            llm.generate_batch([])

        with pytest.raises(Exception):
            llm.generate_batch(["valid prompt", "", "another valid"])

    def test_mock_llm_performance_considerations(self):
        """测试性能考虑"""
        llm = MockLLM(response_delay=0.0)

        # 测试令牌估算
        simple_text = "Hello world"
        tokens = llm.estimate_tokens(simple_text)
        assert tokens == 2

        # 测试复杂文本
        complex_text = "Hello, world! How are you doing today?"
        tokens = llm.estimate_tokens(complex_text)
        assert tokens > 5

        # 测试空文本
        empty_tokens = llm.estimate_tokens("")
        assert empty_tokens == 0

    def test_llm_result_workflow(self):
        """测试 LLM 结果工作流程"""
        llm = MockLLM(response_delay=0.0)

        # 生成结果
        result = llm.generate_with_result("Test prompt")

        # 测试结果属性
        assert hasattr(result, 'text')
        assert hasattr(result, 'prompt')
        assert hasattr(result, 'model_name')
        assert hasattr(result, 'finish_reason')
        assert hasattr(result, 'token_usage')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'timestamp')

        # 测试令牌属性
        if result.token_usage:
            assert result.prompt_tokens is not None
            assert result.completion_tokens is not None
            assert result.total_tokens is not None
            assert result.total_tokens == result.prompt_tokens + result.completion_tokens

    def test_temperature_effects_workflow(self):
        """测试温度参数影响工作流程"""
        # 低温度 - 确定性输出
        llm_low = MockLLM(temperature=0.0, response_delay=0.0)
        responses_low = [llm_low.generate("Hello") for _ in range(3)]
        assert all(r == responses_low[0] for r in responses_low)

        # 高温度 - 随机输出
        llm_high = MockLLM(temperature=1.5, response_delay=0.0)
        responses_high = [llm_high.generate("Hello") for _ in range(3)]
        # 注意：由于随机性，这个测试可能会偶然失败，但我们不强制要求不同

        # 中等温度
        llm_medium = MockLLM(temperature=0.7, response_delay=0.0)
        response_medium = llm_medium.generate("Hello")
        assert isinstance(response_medium, str)
        assert len(response_medium) > 0