# Chain 链式调用模块总结

## 📊 模块概览

Chain 模块是 LangChain 框架的核心组件，提供了灵活而强大的链式调用机制，支持将多个组件连接成复杂的工作流。

### 🎯 核心特性
- **类型安全**: 使用 Pydantic 进行数据验证
- **异步支持**: 完整的 sync/async 接口
- **错误处理**: 完善的异常处理机制
- **可配置**: 灵活的运行时配置
- **可扩展**: 清晰的抽象层设计

## 🏗️ 架构设计

### 核心组件

#### 1. BaseChain - 抽象基类
提供所有链式调用的通用功能：
- 统一的执行接口 (`run`, `arun`)
- 输入验证和错误处理
- 中间步骤跟踪
- 配置管理

**文件位置**: `my_langchain/chains/base.py`

```python
class BaseChain(BaseChainComponent):
    def __init__(self, config: Optional[ChainConfig] = None, **kwargs)
    def run(self, inputs: Union[Dict[str, Any], str], config: Optional[Dict[str, Any]] = None) -> Any
    async def arun(self, inputs: Union[Dict[str, Any], str], config: Optional[Dict[str, Any]] = None) -> Any
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]
    async def acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]
```

#### 2. LLMChain - LLM调用链
将 Prompt 模板与 LLM 结合，是最常用的链类型：

**文件位置**: `my_langchain/chains/llm_chain.py`

```python
class LLMChain(BaseChain):
    def __init__(self, llm: BaseLLM, prompt: BasePromptTemplate, output_key: str = "text")
    def run(self, inputs: Union[Dict[str, Any], str]) -> Any
    async def arun(self, inputs: Union[Dict[str, Any], str]) -> Any
    def apply(self, inputs_list: List[Union[Dict[str, Any], str]]) -> List[Any]
    async def aapply(self, inputs_list: List[Union[Dict[str, Any], str]]) -> List[Any]
```

**特点**:
- 智能的字符串输入映射
- 支持静态 prompt（无变量）
- 批量处理功能
- 自定义输出键

#### 3. SequentialChain - 顺序执行链
按顺序执行多个链，支持复杂的多步工作流：

**文件位置**: `my_langchain/chains/sequential_chain.py`

```python
class SequentialChain(BaseChain):
    def __init__(self, chains: List[BaseChain], return_all: bool = False)
    def add_chain(self, chain: BaseChain) -> None
    def remove_chain(self, index: int) -> None
    def get_chain_at(self, index: int) -> Optional[BaseChain]
    def run(self, inputs: Union[Dict[str, Any], str]) -> Dict[str, Any]
    async def arun(self, inputs: Union[Dict[str, Any], str]) -> Dict[str, Any]
```

**特点**:
- 链式数据流传递
- 错误传播机制
- 中间结果保留选项
- 动态链管理

#### 4. SimpleChain - 自定义函数链
将任意 Python 函数包装为链：

**文件位置**: `my_langchain/chains/simple_chain.py`

```python
class SimpleChain(BaseChain):
    def __init__(self, func: Callable, input_keys: List[str] = None, output_keys: List[str] = None)
    def run(self, inputs: Union[Dict[str, Any], str]) -> Any
    async def arun(self, inputs: Union[Dict[str, Any], str]) -> Any
    def set_input_keys(self, input_keys: List[str]) -> None
    def set_output_keys(self, output_keys: List[str]) -> None

    @classmethod
    def from_function(cls, func: Callable, input_keys: List[str], output_keys: List[str]) -> 'SimpleChain'
```

**特点**:
- 智能参数传递（关键字/位置）
- 同步/异步函数支持
- 灵活的输入输出映射
- 简化的单输出返回

## 📋 数据类型系统

### Chain 相关类型
**文件位置**: `my_langchain/chains/types.py`

```python
class ChainConfig(BaseModel):
    verbose: bool = False
    memory: Optional[Any] = None
    return_intermediate_steps: bool = False
    input_key: Optional[str] = None
    output_key: Optional[str] = None

class ChainResult(BaseModel):
    output: Any
    intermediate_steps: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    execution_time: float = 0.0

class ChainInput(BaseModel):
    data: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
```

### 错误类型
```python
class ChainError(Exception):
    chain_type: str
    details: Dict[str, Any] = {}

class ChainValidationError(ChainError):
    pass

class ChainExecutionError(ChainError):
    step: str
    cause: Optional[Exception] = None

class ChainTimeoutError(ChainError):
    timeout: float
```

## 🧪 测试覆盖

### 单元测试
**测试文件**: `tests/unit/test_chains.py`
- **35个测试用例，100%通过**
- 测试范围：
  - ChainConfig 配置管理
  - BaseChain 基础功能
  - LLMChain LLM调用链
  - SequentialChain 顺序链
  - SimpleChain 自定义函数链

### 集成测试
**测试文件**: `tests/integration/test_chains_integration.py`
- **14个测试用例，100%通过**
- 测试范围：
  - 完整工作流测试
  - 不同链类型组合
  - 批量处理
  - 异步工作流
  - 错误处理
  - 配置集成
  - 记忆模拟

### 总体测试结果
- **49个测试，100%通过**
- **单元测试：35/35 通过**
- **集成测试：14/14 通过**

## 🚀 使用示例

### 基础 LLMChain
```python
from my_langchain.chains import LLMChain
from my_langchain.llms import MockLLM
from my_langchain.prompts import PromptTemplate

# 创建 LLM 和 Prompt
llm = MockLLM(temperature=0.0)
prompt = PromptTemplate(
    template="Hello, {name}! How can I help you today?",
    input_variables=["name"]
)

# 创建链
chain = LLMChain(llm=llm, prompt=prompt)

# 执行链
result = chain.run("Alice")
print(result)  # "Hello Alice! How can I help you today?"
```

### SequentialChain 工作流
```python
from my_langchain.chains import SequentialChain, SimpleChain, LLMChain

# 步骤1：文本预处理
def preprocess(text):
    return text.strip().lower()

preprocess_chain = SimpleChain(
    func=preprocess,
    input_keys=["raw_text"],
    output_keys=["processed_text"]
)

# 步骤2：LLM分析
prompt = PromptTemplate(
    template="Analyze: {processed_text}",
    input_variables=["processed_text"]
)
llm_chain = LLMChain(llm=llm, prompt=prompt, output_key="analysis")

# 步骤3：后处理
def extract_key_points(analysis):
    return analysis.split('.')[:3]

postprocess_chain = SimpleChain(
    func=extract_key_points,
    input_keys=["analysis"],
    output_keys=["key_points"]
)

# 创建工作流
workflow = SequentialChain(
    chains=[preprocess_chain, llm_chain, postprocess_chain],
    return_all=True
)

# 执行工作流
result = workflow.run({"raw_text": "  HELLO WORLD  "})
print(result["key_points"])
```

### SimpleChain 自定义函数
```python
def multiply(x, y):
    return x * y

# 从函数创建链
chain = SimpleChain.from_function(
    multiply,
    input_keys=["x", "y"],
    output_keys=["product"]
)

result = chain.run({"x": 3, "y": 4})
print(result)  # 12
```

## ⚙️ 配置选项

### ChainConfig
```python
config = ChainConfig(
    verbose=True,                    # 详细输出
    return_intermediate_steps=True,  # 返回中间步骤
    input_key="input",              # 默认输入键
    output_key="output"             # 默认输出键
)

chain = LLMChain(llm=llm, prompt=prompt, config=config)
```

### 运行时配置覆盖
```python
result = chain.run(
    inputs={"topic": "AI"},
    config={"verbose": True}  # 运行时配置覆盖
)
```

## 🔧 高级特性

### 1. 异步执行
```python
async def async_workflow():
    result = await chain.arun("Hello")
    return result

# 执行异步工作流
import asyncio
result = asyncio.run(async_workflow())
```

### 2. 批量处理
```python
inputs = ["Alice", "Bob", "Charlie"]
results = chain.apply(inputs)  # 同步批量
async_results = await chain.aapply(inputs)  # 异步批量
```

### 3. 中间步骤跟踪
```python
config = ChainConfig(return_intermediate_steps=True)
chain = LLMChain(llm=llm, prompt=prompt, config=config)

result = chain.run("test")
print(result.intermediate_steps)  # 查看中间步骤
```

### 4. 错误处理
```python
from my_langchain.chains.types import ChainExecutionError

try:
    result = chain.run(invalid_input)
except ChainExecutionError as e:
    print(f"Chain execution failed: {e}")
    print(f"Step: {e.step}")
    print(f"Cause: {e.cause}")
```

## 🎯 设计亮点

### 1. 类型安全
- 使用 Pydantic 确保数据验证
- 完整的类型注解
- 运行时类型检查

### 2. 灵活性
- 支持字典和字符串输入
- 可配置的输入输出映射
- 动态链组合

### 3. 错误恢复
- 详细的错误信息
- 错误传播机制
- 异常上下文保留

### 4. 性能考虑
- 异步支持
- 批量处理优化
- 最小化数据复制

## 📈 性能特点

- **执行时间跟踪**: 每个链执行都会记录时间
- **内存优化**: 高效的数据传递机制
- **异步友好**: 原生异步支持
- **批量优化**: 批量处理减少开销

## 🔗 与其他模块集成

### LLM 集成
- 与所有 LLM 实现无缝集成
- 统一的调用接口
- 智能参数传递

### Prompt 集成
- 支持所有 Prompt 模板类型
- 自动变量验证
- 灵活的格式化选项

### Memory 集成（待实现）
- 预留 Memory 接口
- 支持对话历史管理
- 状态保持机制

## 🎉 总结

Chain 模块成功实现了：

1. **完整的链式调用机制** - 支持复杂工作流构建
2. **类型安全的设计** - Pydantic 数据验证和类型注解
3. **灵活的输入输出** - 支持多种数据格式和映射
4. **强大的错误处理** - 详细的错误信息和恢复机制
5. **异步支持** - 原生的 async/await 接口
6. **高测试覆盖率** - 49个测试，100%通过
7. **优秀的可扩展性** - 清晰的抽象层设计

该模块为构建复杂的 AI 应用提供了坚实的基础，是整个 LangChain 框架的核心组件之一。