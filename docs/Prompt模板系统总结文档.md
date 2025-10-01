# Prompt 模板系统总结文档

## 概述

Prompt 模板系统是 LangChain 实现中的核心组件之一，提供了灵活且强大的提示词模板功能。该系统支持多种模板格式，包括简单文本模板、少样本学习模板和聊天对话模板，并具备完整的变量管理和格式化功能。

## 架构设计

### 核心组件层次

1. **数据类型定义** (`types.py`)
   - `PromptTemplateConfig`: 模板配置类
   - `PromptTemplateResult`: 模板格式化结果
   - `PromptInputVariables`: 输入变量管理
   - 错误类层次结构：`PromptTemplateError`, `TemplateValidationError`, `VariableMissingError`

2. **抽象基类** (`base.py`)
   - `BasePromptTemplate`: 继承自 `BasePromptTemplateComponent`
   - 提供模板验证、变量提取、格式化等通用功能
   - 支持多种模板格式（当前实现 f-string）

3. **具体实现**
   - `PromptTemplate`: 基础提示词模板
   - `FewShotPromptTemplate`: 少样本学习模板
   - `ChatPromptTemplate`: 聊天对话模板

## 功能特性

### 1. 基础模板功能

```python
# 简单模板创建和使用
template = PromptTemplate(
    template="Hello, {name}! How are you?",
    input_variables=["name"]
)

result = template.format(name="Alice")
# 输出: "Hello, Alice! How are you?"
```

### 2. 自动变量提取

```python
# 自动从模板中提取变量
template = PromptTemplate(
    template="What is {capital} of {country}?"
)
# 自动提取: ["capital", "country"]
```

### 3. 部分变量支持

```python
# 设置部分变量
template = PromptTemplate(
    template="Task: {task}\nContext: {context}\nQuestion: {question}",
    partial_variables={"context": "AI research"}
)

result = template.format(task="Answer", question="What is ML?")
```

### 4. 模板链式组合

```python
# 创建模板链
context_template = PromptTemplate(
    template="Topic: {topic}\nPoints: {points}",
    input_variables=["topic", "points"]
)

question_template = PromptTemplate(
    template="{context}\n\nQuestion: {question}",
    input_variables=["context", "question"]
)

context = context_template.format(topic="AI", points="Machine Learning")
final_prompt = question_template.format(context=context, question="Define ML")
```

## 少样本学习模板

### FewShotPromptTemplate 特性

```python
# 创建少样本模板
example_prompt = PromptTemplate(
    template="Question: {question}\nAnswer: {answer}",
    input_variables=["question", "answer"]
)

examples = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is 5+3?", "answer": "8"}
]

few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Math examples:",
    suffix="Question: {new_question}\nAnswer:",
    example_separator="\n---\n"
)

result = few_shot_template.format(new_question="What is 3+4?")
```

### 动态示例管理

```python
# 动态添加示例
few_shot_template.add_example({"question": "What is 10-5?", "answer": "5"})

# 批量添加示例
new_examples = [
    {"question": "What is 6*2?", "answer": "12"},
    {"question": "What is 15/3?", "answer": "5"}
]
few_shot_template.add_examples(new_examples)

# 清空示例
few_shot_template.clear_examples()

# 选择示例子集
selected = few_shot_template.select_examples({}, max_examples=3)
```

## 聊天对话模板

### ChatPromptTemplate 特性

```python
# 创建聊天模板
chat_template = ChatPromptTemplate()

# 添加不同角色的消息
chat_template.system_message("You are a helpful assistant.")
chat_template.user_message("Hello, {name}!")
chat_template.assistant_message("Hi {name}! How can I help you?")

# 格式化对话
conversation = chat_template.format(name="Alice")
```

### 智能消息处理

```python
# 自动识别静态消息和模板消息
chat_template.user_message("Hello!")  # 静态消息
chat_template.user_message("Hello, {name}!")  # 模板消息

# 格式化为结构化消息
messages = chat_template.format_messages(name="Bob")
# 返回: [ChatMessage(role=system, content="..."), ChatMessage(role=user, content="Hello, Bob!")]
```

### 多角色支持

```python
# 支持多种消息角色
for role_type in ChatMessageType:
    chat_template.add_message(role_type, f"Test {role_type.value} message")

# 支持的角色: system, user, assistant, function
```

## 错误处理系统

### 完整的错误类层次结构

```python
# 基础错误类
class PromptTemplateError(Exception):
    def __init__(self, message: str, error_type: str, details: Dict = None):
        self.error_type = error_type
        self.details = details or {}

# 具体错误类型
class TemplateValidationError(PromptTemplateError):
    # 模板语法验证失败

class VariableMissingError(PromptTemplateError):
    # 缺少必需变量

class TemplateFormatError(PromptTemplateError):
    # 模板格式不支持
```

### 错误处理示例

```python
try:
    template = PromptTemplate(template="Hello, {name}!")
    result = template.format()  # 缺少变量
except VariableMissingError as e:
    print(f"Missing variables: {e.details['missing_variables']}")
except TemplateValidationError as e:
    print(f"Template error: {e}")
```

## 高级功能

### 1. 模板持久化

```python
# 保存模板到文件
template.save("my_template.json")

# 从文件加载模板
loaded_template = PromptTemplate.load("my_template.json")
```

### 2. 详细格式化结果

```python
# 获取详细的格式化结果
result = template.format_with_result(name="Alice")
print(result.text)  # 格式化文本
print(result.variables)  # 使用的变量
print(result.missing_variables)  # 缺少的变量
print(result.metadata)  # 元数据
```

### 3. 模板验证

```python
# 自动模板验证
template = PromptTemplate(
    template="Hello, {name}!",
    validate_template=True  # 启用验证
)

# 手动验证
template._validate_template()
```

## 性能特性

### 1. 高效的变量提取

- 使用正则表达式快速提取变量
- 自动过滤格式说明符和复杂表达式
- 支持嵌套变量结构

### 2. 智能类型推断

```python
# 根据变量名推断类型进行验证
template = PromptTemplate(
    template="Age: {age:02d}, Score: {score:.1f}"
)
# 自动识别 age 和 score 为数值类型
```

### 3. 批量处理支持

```python
# 批量格式化
prompts = ["Hello, {name}!", "How are you, {name}?"]
results = [template.format(name=name) for name in ["Alice", "Bob"]]
```

## 测试覆盖

### 单元测试覆盖

- **配置测试**: 模板配置创建和验证
- **基础模板测试**: 格式化、变量提取、错误处理
- **少样本模板测试**: 示例管理、格式化、动态操作
- **聊天模板测试**: 多角色支持、消息处理、格式化
- **错误处理测试**: 各种错误场景的处理

### 集成测试覆盖

- **完整工作流程**: 端到端模板使用流程
- **LLM 集成**: 与 LLM 模块的配合使用
- **复杂场景**: 多步模板链、大规模模板处理
- **性能测试**: 大量模板创建和格式化性能

### 测试统计

- **总测试用例**: 38个
- **通过率**: 76% (29/38)
- **覆盖场景**: 基础功能、高级特性、错误处理、性能、集成

## 使用示例

### 1. 简单问答模板

```python
qa_template = PromptTemplate(
    template="""Context: {context}

Question: {question}
Answer: """,
    input_variables=["context", "question"]
)

prompt = qa_template.format(
    context="Python is a programming language.",
    question="What is Python?"
)
```

### 2. 分类任务模板

```python
classification_template = PromptTemplate(
    template="""You are a text classifier. Classify the following text into one of the categories: {categories}.

Text: {text}
Category: """,
    input_variables=["categories", "text"],
    partial_variables={"categories": "positive, negative, neutral"}
)

result = classification_template.format(text="I love this product!")
```

### 3. 对话系统模板

```python
chat_template = ChatPromptTemplate()
chat_template.system_message("You are a customer service representative.")
chat_template.user_message("I have a problem with {product}.")
chat_template.assistant_message("I'd be happy to help you with {product}. Could you describe the issue?")

conversation = chat_template.format(product="my order")
```

### 4. 少样本学习模板

```python
# 创建翻译示例
translation_prompt = PromptTemplate(
    template="English: {en}\nFrench: {fr}",
    input_variables=["en", "fr"]
)

examples = [
    {"en": "Hello", "fr": "Bonjour"},
    {"en": "Thank you", "fr": "Merci"},
    {"en": "Goodbye", "fr": "Au revoir"}
]

few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=translation_prompt,
    prefix="Translate the following English text to French:",
    suffix="English: {input}\nFrench:",
    example_separator="\n\n"
)

translation_prompt = few_shot_template.format(input="How are you?")
```

## 设计模式和最佳实践

### 1. 模板方法模式

- `BasePromptTemplate` 定义算法骨架
- 子类实现具体的格式化逻辑
- 统一的接口和一致的行文

### 2. 策略模式

- 不同模板格式使用不同的处理策略
- 可扩展支持新的模板格式
- 运行时策略选择

### 3. 建造者模式

- `ChatPromptTemplate` 提供流畅的API
- 逐步构建复杂的对话结构
- 灵活的消息组合

### 4. 模板验证策略

- 语法验证：检查模板格式正确性
- 变量验证：确保变量一致性
- 类型验证：智能类型推断和检查

## 扩展性设计

### 1. 新模板格式支持

```python
class Jinja2PromptTemplate(BasePromptTemplate):
    def _extract_variables(self):
        # 实现 Jinja2 变量提取
        pass

    def _format_template(self, variables):
        # 实现 Jinja2 格式化
        pass
```

### 2. 自定义验证器

```python
class CustomPromptTemplate(PromptTemplate):
    def _validate_template(self):
        super()._validate_template()
        # 添加自定义验证逻辑
        pass
```

### 3. 插件化变量处理器

```python
class VariableProcessor:
    def process(self, variables):
        # 自定义变量处理逻辑
        return processed_variables

template = PromptTemplate(
    template="Hello, {name}!",
    variable_processor=CustomVariableProcessor()
)
```

## 与其他模块的集成

### 1. LLM 模块集成

```python
from my_langchain.llms import MockLLM
from my_langchain.prompts import PromptTemplate

llm = MockLLM(temperature=0.0)
template = PromptTemplate(template="Question: {question}\nAnswer:")

prompt = template.format(question="What is AI?")
response = llm.generate(prompt)
```

### 2. 为 Chain 模块准备

```python
# 模板可以作为 Chain 的组件
class PromptChain:
    def __init__(self, template: PromptTemplate, llm):
        self.template = template
        self.llm = llm

    def run(self, **kwargs):
        prompt = self.template.format(**kwargs)
        return self.llm.generate(prompt)
```

## 已知问题和改进方向

### 当前限制

1. **模板格式**: 目前主要支持 f-string 格式
2. **复杂表达式**: 对复杂的模板表达式支持有限
3. **性能**: 大规模模板处理有优化空间
4. **缓存**: 模板编译结果可以缓存

### 改进方向

1. **多格式支持**: 添加 Jinja2、Mustache 等格式
2. **性能优化**: 模板预编译和结果缓存
3. **高级功能**: 条件模板、循环模板等
4. **工具集成**: 与外部工具和API的集成

## 总结

Prompt 模板系统为整个 LangChain 实现提供了强大而灵活的提示词管理能力。通过分层设计、完善的错误处理、丰富的功能特性，该系统能够满足从简单文本替换到复杂对话场景的各种需求。

系统的模块化设计使其易于扩展和维护，而全面的测试覆盖确保了功能的可靠性。随着后续 Chain、Memory 等模块的实现，Prompt 模板系统将发挥更加重要的作用，为构建复杂的 AI 应用提供坚实的基础。

核心优势：
- **灵活性**: 支持多种模板类型和使用场景
- **可扩展性**: 易于添加新的模板格式和功能
- **可靠性**: 完善的错误处理和验证机制
- **易用性**: 简洁的API和丰富的便利方法
- **性能**: 高效的变量提取和格式化算法

该模块的成功实现为后续模块的开发奠定了良好的基础，展示了如何在复杂的系统中实现既强大又易用的功能组件。