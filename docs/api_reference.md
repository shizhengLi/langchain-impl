# API 参考文档

本文档提供LangChain实现项目的API参考信息。

## 🔍 检索模块 API

### 核心类

#### DocumentRetriever
```python
class DocumentRetriever(BaseRetriever):
    """基于关键词的文档检索器"""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        """初始化文档检索器"""

    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到检索器"""

    def retrieve(self, query: str) -> RetrievalResult:
        """执行文档检索"""

    def get_term_statistics(self) -> Dict[str, Any]:
        """获取词频统计信息"""
```

#### VectorRetriever
```python
class VectorRetriever(BaseRetriever):
    """基于向量的语义检索器"""

    def __init__(self,
                 embedding_model: EmbeddingModel,
                 vector_store: VectorStore,
                 config: Optional[RetrievalConfig] = None):
        """初始化向量检索器"""

    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档并计算嵌入向量"""

    def retrieve(self, query: str) -> RetrievalResult:
        """执行语义检索"""

    def get_embedding_stats(self) -> Dict[str, Any]:
        """获取嵌入向量统计信息"""
```

#### EnsembleRetriever
```python
class EnsembleRetriever(BaseRetriever):
    """集成多个检索器的融合检索器"""

    def __init__(self,
                 retrievers: List[BaseRetriever],
                 weights: Optional[List[float]] = None,
                 fusion_strategy: str = "weighted_score",
                 config: Optional[RetrievalConfig] = None):
        """初始化集成检索器"""

    def compare_retrievers(self, query: str) -> Dict[str, RetrievalResult]:
        """比较不同检索器的结果"""

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """获取集成检索统计信息"""
```

### 数据模型

#### Document
```python
class Document(BaseModel):
    """文档数据模型"""
    content: str                           # 文档内容
    metadata: Dict[str, Any]              # 元数据
    id: str                               # 文档ID

    def get_text_snippet(self, max_length: int = 100) -> str:
        """获取文档片段"""

    def matches_filter(self, filter_dict: Dict[str, Any]) -> bool:
        """检查文档是否匹配过滤条件"""
```

#### RetrievalConfig
```python
class RetrievalConfig(BaseModel):
    """检索配置"""
    top_k: int = 5                       # 返回结果数量
    score_threshold: Optional[float] = None  # 相似度阈值
    search_type: str = "similarity"       # 搜索类型
    mmr_lambda: float = 0.5              # MMR多样性参数
    fetch_k: int = 20                    # MMR候选文档数
    filter_dict: Dict[str, Any] = {}     # 元数据过滤
    enable_caching: bool = True          # 启用缓存
```

#### RetrievalResult
```python
class RetrievalResult(BaseModel):
    """检索结果"""
    documents: List[RetrievedDocument]   # 检索到的文档
    query: str                           # 查询文本
    total_results: int                   # 总结果数
    search_time: float                   # 检索耗时
    retrieval_method: str                # 检索方法

    def get_top_k(self, k: int) -> List[RetrievedDocument]:
        """获取前k个结果"""

    def get_average_score(self) -> float:
        """获取平均分数"""
```

## 🧠 LLM 模块 API

### 基础接口

#### BaseLLM
```python
class BaseLLM(ABC):
    """大语言模型基类"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResult:
        """生成文本"""

    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs) -> LLMResult:
        """异步生成文本"""

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
```

#### MockLLM
```python
class MockLLM(BaseLLM):
    """模拟大语言模型，用于测试"""

    def __init__(self, responses: Optional[List[str]] = None):
        """初始化模拟LLM"""

    def generate(self, prompt: str, **kwargs) -> LLMResult:
        """生成模拟响应"""
```

## 🔗 Chain 模块 API

### 基础类

#### BaseChain
```python
class BaseChain(ABC):
    """链式调用基类"""

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行链"""

    @abstractmethod
    async def arun(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """异步执行链"""

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """调用链的便捷方法"""
```

#### LLMChain
```python
class LLMChain(BaseChain):
    """LLM链，结合提示词模板和LLM"""

    def __init__(self,
                 llm: BaseLLM,
                 prompt: PromptTemplate,
                 output_parser: Optional[OutputParser] = None):
        """初始化LLM链"""

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """运行LLM链"""
```

## 💾 Memory 模块 API

### 基础接口

#### BaseMemory
```python
class BaseMemory(ABC):
    """记忆系统基类"""

    @abstractmethod
    def add_message(self, message: str, role: str = "user"):
        """添加消息"""

    @abstractmethod
    def get_messages(self) -> List[Dict[str, str]]:
        """获取消息历史"""

    @abstractmethod
    def clear(self):
        """清空记忆"""
```

## 🛠️ Tool 模块 API

### 基础类

#### BaseTool
```python
class BaseTool(ABC):
    """工具基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""

    @abstractmethod
    def run(self, input_text: str) -> str:
        """执行工具"""

    @abstractmethod
    async def arun(self, input_text: str) -> str:
        """异步执行工具"""
```

## 📊 向量存储 API

### 基础接口

#### BaseVectorStore
```python
class BaseVectorStore(ABC):
    """向量存储基类"""

    @abstractmethod
    def add_vectors(self, vectors: List[Vector]) -> List[str]:
        """添加向量"""

    @abstractmethod
    def search(self, query: VectorStoreQuery) -> VectorStoreResult:
        """搜索相似向量"""

    @abstractmethod
    def delete_vectors(self, vector_ids: List[str]) -> bool:
        """删除向量"""
```

## 📝 示例用法

### 基础检索
```python
from my_langchain.retrieval import DocumentRetriever, Document, RetrievalConfig

# 创建检索器
retriever = DocumentRetriever()

# 添加文档
documents = [
    Document(content="Python是一种编程语言"),
    Document(content="Java也是一种编程语言")
]
retriever.add_documents(documents)

# 配置检索参数
config = RetrievalConfig(top_k=3, search_type="tfidf")
retriever_with_config = DocumentRetriever(config=config)
retriever_with_config.add_documents(documents)

# 执行检索
result = retriever_with_config.retrieve("编程语言")
for doc in result.documents:
    print(f"Score: {doc.relevance_score:.3f}")
    print(f"Content: {doc.content}")
```

### 向量检索
```python
from my_langchain.retrieval import VectorRetriever
from my_langchain.embeddings import MockEmbedding
from my_langchain.vectorstores import InMemoryVectorStore
from my_langchain.vectorstores.types import VectorStoreConfig

# 创建组件
embedding_model = MockEmbedding(embedding_dimension=384)
vector_config = VectorStoreConfig(dimension=384)
vector_store = InMemoryVectorStore(config=vector_config)

# 创建向量检索器
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

# 执行检索并比较结果
result = ensemble.retrieve("Python")
comparison = ensemble.compare_retrievers("Python")
```

## 🔍 错误处理

### 异常类型
```python
from my_langchain.retrieval.types import (
    RetrievalError,              # 基础检索异常
    RetrievalValidationError,     # 验证异常
    RetrievalProcessingError      # 处理异常
)

try:
    result = retriever.retrieve("query")
except RetrievalValidationError as e:
    print(f"验证错误: {e}")
except RetrievalProcessingError as e:
    print(f"处理错误: {e}")
except RetrievalError as e:
    print(f"检索错误: {e}")
```

## 📈 性能监控

### 统计信息
```python
# 获取检索器统计
stats = retriever.get_retriever_info()
print(f"文档数量: {stats['document_count']}")
print(f"检索器类型: {stats['retriever_type']}")

# 向量检索器统计
if isinstance(retriever, VectorRetriever):
    embedding_stats = retriever.get_embedding_stats()
    print(f"向量数量: {embedding_stats['vector_count']}")

# 集成检索器统计
if isinstance(retriever, EnsembleRetriever):
    ensemble_stats = retriever.get_ensemble_stats()
    print(f"检索器数量: {ensemble_stats['num_retrievers']}")
```

---

更多详细的API文档和示例，请参考各模块的具体文档：
- [检索系统详细文档](retrieval.md)
- [检索模块索引](retrieval_index.md)
- [安装指南](installation.md)