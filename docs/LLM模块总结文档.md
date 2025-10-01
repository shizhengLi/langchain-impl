# LLM 模块总结文档

## 概述

LLM (Large Language Model) 模块是 LangChain 实现的核心组件之一，提供了统一的大语言模型接口和功能。该模块采用抽象基类设计模式，支持多种 LLM 实现，并提供了完整的同步/异步调用接口。

## 架构设计

### 核心组件

1. **数据类型定义** (`types.py`)
   - `LLMConfig`: LLM 配置类，包含模型名称、温度、最大令牌数等参数
   - `LLMResult`: LLM 生成结果类，包含生成文本、令牌使用情况、元数据等
   - 错误类层次结构：`LLMError`, `LLMTimeoutError`, `LLMRateLimitError`, `LLMTokenLimitError`

2. **抽象基类** (`base.py`)
   - `BaseLLM`: 继承自 `BaseLLMComponent`，实现通用 LLM 功能
   - 提供同步/异步生成接口
   - 内置错误处理、配置管理、输入验证

3. **具体实现** (`mock_llm.py`)
   - `MockLLM`: 用于测试和开发的模拟 LLM 实现
   - 支持预定义响应和随机生成
   - 基于温度参数的可控随机性

## 功能特性

### 1. 统一接口设计

```python
# 基础生成接口
llm.generate("Hello, world!")  # 返回字符串
llm.generate_with_result("Hello")  # 返回完整的 LLMResult

# 异步接口
await llm.agenerate("Hello")
await llm.agenerate_with_result("Hello")

# 批量处理
llm.generate_batch(["prompt1", "prompt2"])
llm.generate_batch_with_result(["prompt1", "prompt2"])
```

### 2. 配置管理

```python
config = LLMConfig(
    model_name="gpt-3.5-turbo",
    temperature=0.7,  # 0.0-2.0，控制随机性
    max_tokens=1000,  # 最大生成令牌数
    timeout=30.0      # 请求超时时间
)
```

### 3. 错误处理

模块提供了完整的错误类层次结构：
- `LLMError`: 基础错误类
- `LLMTimeoutError`: 超时错误
- `LLMRateLimitError`: 频率限制错误
- `LLMTokenLimitError`: 令牌限制错误

### 4. 结果跟踪

每个生成结果都包含详细信息：
- 生成文本内容
- 输入提示词
- 模型名称
- 完成原因 (stop, length, etc.)
- 令牌使用统计
- 生成时间
- 元数据信息

## MockLLM 实现细节

### 温度控制

- **低温 (0.0-0.3)**: 确定性输出，选择第一个响应
- **中温 (0.3-0.8)**: 随机选择预定义响应
- **高温 (0.8-2.0)**: 随机选择并添加随机后缀

### 预定义响应支持

```python
responses = {
    "Hello": "Hi there!",
    "How are you?": "I'm doing well!"
}
llm = MockLLM(responses=responses)
```

### 智能响应生成

根据提示词内容智能选择响应类别：
- 问候类：包含 "hello", "hi", "你好"
- 问题类：包含 "what", "什么", "how", "如何"
- 感谢类：包含 "thanks", "thank", "谢谢"
- 通用类：其他情况

## 测试覆盖

### 单元测试 (12个测试用例)

1. **配置测试**
   - 配置创建和验证
   - 温度范围验证 (0.0-2.0)

2. **结果测试**
   - 结果创建和属性访问
   - 令牌统计验证

3. **MockLLM 功能测试**
   - 基本生成功能
   - 批量生成
   - 预定义响应
   - 响应管理 (添加/清空)
   - 输入验证
   - 模型信息获取
   - 令牌估算

### 集成测试 (9个测试用例)

1. **完整工作流程测试**
   - 同步生成流程
   - 异步生成流程
   - 批量处理流程

2. **高级功能测试**
   - 预定义响应工作流程
   - 配置管理工作流程
   - 错误处理工作流程
   - 性能考虑测试
   - LLM 结果工作流程
   - 温度影响测试

### 测试统计

- **总测试用例**: 21个
- **通过率**: 100%
- **覆盖场景**: 配置、生成、异步、批量、错误处理、性能

## 使用示例

### 基础使用

```python
from my_langchain.llms import MockLLM

# 创建 LLM 实例
llm = MockLLM(
    model_name="my-model",
    temperature=0.7,
    max_tokens=100
)

# 生成文本
response = llm.generate("What is the capital of France?")
print(response)

# 获取完整结果
result = llm.generate_with_result("Tell me a joke")
print(f"Model: {result.model_name}")
print(f"Tokens: {result.total_tokens}")
print(f"Time: {result.generation_time}s")
```

### 异步使用

```python
import asyncio
from my_langchain.llms import MockLLM

async def main():
    llm = MockLLM(temperature=0.0)

    # 异步生成
    response = await llm.agenerate("Hello, how are you?")
    print(response)

    # 批量异步生成
    prompts = ["Hello", "How are you?", "Goodbye"]
    responses = await llm.agenerate_batch(prompts)
    for i, resp in enumerate(responses):
        print(f"{i+1}: {resp}")

asyncio.run(main())
```

### 预定义响应

```python
# 设置预定义响应
responses = {
    "What is 2+2?": "2+2 equals 4.",
    "Who wrote Romeo and Juliet?": "William Shakespeare."
}

llm = MockLLM(responses=responses)

# 使用预定义响应
print(llm.generate("What is 2+2?"))  # 输出: 2+2 equals 4.

# 动态添加响应
llm.add_response("New question", "New answer")
print(llm.generate("New question"))  # 输出: New answer
```

## 设计模式和最佳实践

### 1. 抽象工厂模式
- `BaseLLM` 定义抽象接口
- 具体实现类 (如 `MockLLM`) 实现具体逻辑

### 2. 模板方法模式
- 基类提供通用方法和流程
- 子类实现核心生成逻辑

### 3. 错误处理策略
- 自定义异常类层次结构
- 统一的错误处理机制
- 详细的错误信息和建议

### 4. 配置管理
- Pydantic 数据验证
- 配置合并和覆盖
- 运行时参数支持

## 性能特性

1. **令牌估算**: 基于正则表达式的简单令牌计数
2. **批量处理**: 支持多提示词并行处理
3. **异步支持**: 完整的异步接口实现
4. **延迟模拟**: MockLLM 支持响应延迟配置

## 扩展性

该模块设计具有良好的扩展性：

1. **新 LLM 实现**: 继承 `BaseLLM` 类即可添加新的 LLM 实现
2. **配置扩展**: 通过 `LLMConfig` 轻松添加新的配置参数
3. **结果扩展**: `LLMResult` 支持自定义元数据
4. **错误扩展**: 可以轻松添加新的错误类型

## 下一步计划

LLM 模块已完成实现和测试，下一步将实现：

1. **Prompt 模板系统**: 提示词模板管理
2. **Chain 链式调用**: 多组件组合执行
3. **Memory 记忆系统**: 对话历史管理
4. **Agent 智能体系统**: 智能决策和工具调用

## 总结

LLM 模块作为框架的核心组件，提供了完整、可靠、易用的大语言模型接口。通过抽象设计、完善的错误处理、全面的测试覆盖，为后续模块的实现奠定了坚实的基础。该模块不仅支持当前的 Mock 实现，还为未来接入真实 LLM 服务 (如 OpenAI、Claude 等) 做好了准备。