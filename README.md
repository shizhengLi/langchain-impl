# LangChain 实现项目

一个从零开始实现的 LangChain 框架复现版本，用于学习大语言模型应用开发的核心概念。

## 项目结构

```
langchain-impl/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖
├── setup.py                 # 安装配置
├── pyproject.toml           # 现代Python项目配置
├── pytest.ini              # 测试配置
├── .gitignore              # Git忽略文件
├── my_langchain/           # 主要代码目录
│   ├── __init__.py         # 包初始化
│   ├── base/               # 基础抽象类
│   ├── llms/               # 大语言模型
│   ├── prompts/            # 提示词模板
│   ├── chains/             # 链式调用
│   ├── memory/             # 记忆系统
│   ├── agents/             # 智能体
│   ├── tools/              # 工具系统
│   ├── vector_stores/      # 向量存储
│   ├── embeddings/         # 嵌入模型
│   ├── text_splitters/     # 文本分割
│   └── retrieval/          # 检索系统
├── tests/                  # 测试目录
│   ├── __init__.py
│   ├── unit/               # 单元测试
│   ├── integration/        # 集成测试
│   └── fixtures/           # 测试数据
├── examples/               # 示例代码
│   ├── basic_usage.py
│   ├── chain_example.py
│   └── agent_example.py
└── docs/                   # 文档目录
    ├── architecture.md     # 架构设计文档
    ├── api_reference.md    # API参考
    └── tutorials/          # 教程
```

## 核心组件说明

### 1. 基础抽象层 (base/)
定义所有组件的基础接口，确保系统的可扩展性。

### 2. LLM抽象层 (llms/)
大语言模型的统一接口，支持多种模型提供商。

### 3. 提示词模板 (prompts/)
动态生成和格式化提示词。

### 4. 链式调用 (chains)
将多个组件组合成处理流程。

### 5. 记忆系统 (memory)
为对话和交互提供持久化记忆。

### 6. 智能体系统 (agents)
基于工具的自主决策和执行。

## 开发原则

1. **小步快跑**: 每个功能都有完整的测试覆盖
2. **接口优先**: 先定义接口，再实现具体功能
3. **可扩展性**: 支持自定义组件和扩展
4. **类型安全**: 使用类型注解提高代码质量

## 安装和使用

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest

# 运行示例
python examples/basic_usage.py
```

## 开发进度

- [x] 项目初始化
- [ ] 核心接口设计
- [ ] LLM抽象层实现
- [ ] 提示词模板系统
- [ ] 链式调用机制
- [ ] 记忆系统
- [ ] 智能体系统
- [ ] 工具系统
- [ ] 向量存储
- [ ] 文本分割
- [ ] 嵌入模型
- [ ] 检索系统