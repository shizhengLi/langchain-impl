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
│   ├── agent_example.py
│   ├── retrieval_demo.py   # 检索系统演示
│   └── test_retrieval_demo.py # 检索系统测试脚本
└── docs/                   # 文档目录
    ├── architecture.md     # 架构设计文档
    ├── api_reference.md    # API参考
    ├── retrieval.md        # 检索系统文档
    ├── retrieval_index.md  # 检索模块文档索引
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

### 7. 工具系统 (tools)
可扩展的工具集合，支持各种外部API和功能调用。

### 8. 向量存储 (vector_stores)
高效的向量数据库接口，支持语义检索。

### 9. 嵌入模型 (embeddings)
文本向量化功能，将文本转换为数值表示。

### 10. 文本分割 (text_splitters)
智能文档分割，将长文档分割为合适的片段。

### 11. 检索系统 (retrieval)
完整的RAG检索功能，支持多种检索策略。

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
python examples/retrieval_demo.py
```

## 开发进度

### ✅ 已完成模块 (100%测试通过率)
- [x] 项目初始化 - 基础项目结构和配置
- [x] 核心接口设计 - 抽象基类和接口定义
- [x] LLM抽象层实现 - 大语言模型接口
- [x] 提示词模板系统 - 动态提示词生成
- [x] 链式调用机制 - Chain基础功能
- [x] 记忆系统 - 对话记忆功能
- [x] 智能体系统 - Agent和工具框架
- [x] 工具系统 - 可扩展工具集合
- [x] 向量存储 - 向量数据库接口
- [x] 文本分割 - 智能文档分割
- [x] 嵌入模型 - 文本向量化
- [x] **检索系统 - 完整RAG功能 (57个测试，100%通过)**

### 📚 完整文档
- [x] 模块架构文档
- [x] API参考文档
- [x] 使用示例和教程
- [x] 检索系统完整文档 (`docs/retrieval.md`, `docs/retrieval_index.md`)

### 🎯 特色功能
- **多种检索策略**: DocumentRetriever、VectorRetriever、EnsembleRetriever
- **高级算法**: TF-IDF、BM25、MMR、多种融合策略
- **性能优化**: 缓存机制、批量处理、内存管理
- **生产就绪**: 完整错误处理、监控指标、统计分析