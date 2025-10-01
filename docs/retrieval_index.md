# 检索模块文档索引

## 概述

本模块实现了完整的检索增强生成(RAG)功能，为LangChain项目提供了强大的文档检索能力。该模块支持多种检索策略，包括基于关键词的检索、基于向量的语义检索以及集成多种检索器的混合检索。

## 📚 文档结构

### 核心文档
- **[Retrieval模块完整文档](retrieval.md)** - 详细的模块文档，包含架构设计、使用示例、性能优化等

### 示例代码
- **[Retrieval功能演示](../examples/retrieval_demo.py)** - 完整的功能演示脚本，展示所有检索器的使用方法
- **[Retrieval测试脚本](../examples/test_retrieval_demo.py)** - 验证示例代码正确性的测试脚本

### API参考
- **[类型定义](../my_langchain/retrieval/types.py)** - 核心数据模型和类型定义
- **[基础类](../my_langchain/retrieval/base.py)** - 检索器抽象基类
- **[文档检索器](../my_langchain/retrieval/document_retriever.py)** - 基于关键词的文档检索实现
- **[向量检索器](../my_langchain/retrieval/vector_retriever.py)** - 基于嵌入向量的语义检索实现
- **[集成检索器](../my_langchain/retrieval/ensemble_retriever.py)** - 多检索器融合实现

### 测试覆盖
- **[单元测试](../tests/unit/test_retrieval.py)** - 57个单元测试，100%通过率

## 🚀 快速开始

### 基础文档检索
```python
from my_langchain.retrieval import DocumentRetriever, Document

# 创建检索器
retriever = DocumentRetriever()

# 添加文档
documents = [
    Document(content="Python是一种编程语言"),
    Document(content="Java是另一种编程语言")
]
retriever.add_documents(documents)

# 执行检索
result = retriever.retrieve("Python编程")
for doc in result.documents:
    print(f"Score: {doc.relevance_score:.3f}, Content: {doc.content}")
```

### 向量语义检索
```python
from my_langchain.retrieval import VectorRetriever
from my_langchain.embeddings import MockEmbedding
from my_langchain.vectorstores import InMemoryVectorStore
from my_langchain.vectorstores.types import VectorStoreConfig

# 创建组件
embedding_model = MockEmbedding(embedding_dimension=384)
vector_config = VectorStoreConfig(dimension=384)
vector_store = InMemoryVectorStore(config=vector_config)

# 创建检索器
retriever = VectorRetriever(
    embedding_model=embedding_model,
    vector_store=vector_store
)

# 添加文档并检索
retriever.add_documents(documents)
result = retriever.retrieve("编程语言", search_type="mmr")
```

### 集成检索
```python
from my_langchain.retrieval import EnsembleRetriever

# 创建多个检索器
doc_retriever = DocumentRetriever()
vector_retriever = VectorRetriever(embedding_model, vector_store)

# 创建集成检索器
ensemble = EnsembleRetriever(
    retrievers=[doc_retriever, vector_retriever],
    weights=[0.4, 0.6],
    fusion_strategy="weighted_score"
)

# 执行检索
result = ensemble.retrieve("Python")
comparison = ensemble.compare_retrievers("Python")
```

## 🎯 核心特性

### 多种检索策略
- **DocumentRetriever**: 基于关键词匹配、TF-IDF和BM25算法
- **VectorRetriever**: 基于嵌入向量的语义相似度检索
- **EnsembleRetriever**: 结合多个检索器的结果，支持多种融合策略

### 高级检索算法
- **相似度检索**: 基于余弦相似度的语义匹配
- **MMR检索**: 最大边界相关性，平衡相关性和多样性
- **TF-IDF**: 词频-逆文档频率算法
- **BM25**: 最佳匹配25，改进的TF-IDF算法

### 智能结果融合
- **加权平均**: 基于权重的分数融合
- **排名融合**: Borda计数式的排名融合
- **倒数排名融合**: RRF算法，广泛用于信息检索
- **加权投票**: 基于排名位置的投票机制

## 📊 性能指标

### 测试覆盖率
- **单元测试**: 57个测试用例，100%通过率
- **集成测试**: 覆盖所有核心功能和边界情况
- **性能测试**: 包含缓存、批量操作等优化功能

### 检索质量指标
```python
from my_langchain.retrieval.types import calculate_retrieval_metrics

# 计算检索指标
metrics = calculate_retrieval_metrics(
    retrieved_docs=result.documents,
    relevant_doc_ids=["doc_1", "doc_3", "doc_5"],
    k=10
)

print(f"Precision@10: {metrics.precision:.3f}")
print(f"Recall@10: {metrics.recall:.3f}")
print(f"F1-Score: {metrics.f1_score:.3f}")
print(f"Hit Rate: {metrics.hit_rate:.3f}")
print(f"MRR: {metrics.mean_reciprocal_rank:.3f}")
print(f"MAP: {metrics.mean_average_precision:.3f}")
```

## 🔧 配置选项

### RetrievalConfig参数
```python
config = RetrievalConfig(
    top_k=5,                    # 返回结果数量
    score_threshold=0.7,        # 相似度阈值
    search_type="similarity",   # 检索类型
    mmr_lambda=0.5,            # MMR多样性参数
    fetch_k=20,                # MMR候选文档数
    filter_dict={"type": "pdf"} # 元数据过滤
)
```

## 🚀 性能优化

### 缓存机制
- 查询结果缓存
- 嵌入向量缓存
- 配置参数缓存

### 批量处理
- 批量文档添加
- 批量向量计算
- 并行检索处理

### 内存管理
- LRU缓存策略
- 延迟加载机制
- 资源自动清理

## 🔍 监控和分析

### 统计信息
```python
# 获取检索器统计信息
stats = retriever.get_retriever_info()
print(f"Document count: {stats['document_count']}")
print(f"Retriever type: {stats['retriever_type']}")

# 向量检索器特定统计
if isinstance(retriever, VectorRetriever):
    embedding_stats = retriever.get_embedding_stats()
    print(f"Vector count: {embedding_stats['vector_count']}")
    print(f"Embedding dimension: {embedding_stats['embedding_dimension']}")
```

## 🛠️ 扩展性

### 自定义检索器
```python
from my_langchain.retrieval.base import BaseRetriever

class CustomRetriever(BaseRetriever):
    def _retrieve_documents(self, query: str, config: RetrievalConfig):
        # 实现自定义检索逻辑
        return custom_documents

    def add_documents(self, documents: List[Document]):
        # 实现文档添加逻辑
        return document_ids
```

### 自定义融合策略
```python
def custom_fusion(retriever_results, query):
    # 实现自定义融合算法
    return fused_documents
```

## ❓ 常见问题

### Q: 如何选择合适的检索器？
A:
- **小数据集**: 使用DocumentRetriever，无需向量化
- **语义检索**: 使用VectorRetriever，理解查询意图
- **高精度需求**: 使用EnsembleRetriever，结合多种策略

### Q: 如何优化检索性能？
A:
- 预计算和缓存嵌入向量
- 使用批量处理减少API调用
- 定期清理过期缓存数据
- 调整top_k和score_threshold参数

### Q: 如何处理中文文档？
A: 当前实现主要支持英文文档的精确匹配。对于中文文档，建议使用VectorRetriever进行语义检索，它可以更好地处理多语言文本。

## 📈 版本历史

- **v1.0**: 核心检索功能实现
  - DocumentRetriever、VectorRetriever、EnsembleRetriever
  - 57个单元测试，100%通过率
  - 完整的文档和示例

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 编写测试用例
4. 确保所有测试通过
5. 提交Pull Request

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。