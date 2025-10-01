# 测试指南

本文档介绍如何运行和编写测试，以及测试覆盖率信息。

## 🧪 测试概览

项目采用pytest作为测试框架，具有以下特点：
- **100%测试通过率** - 所有57个单元测试均通过
- **完整的测试覆盖** - 覆盖所有核心功能模块
- **高质量的测试用例** - 包含正常流程、边界情况、错误处理

## 🚀 快速开始

### 运行所有测试
```bash
# 在项目根目录运行
pytest

# 显示详细输出
pytest -v

# 显示测试覆盖率
pytest --cov=my_langchain
```

### 运行特定模块测试
```bash
# 测试检索模块
pytest tests/unit/test_retrieval.py -v

# 测试特定类
pytest tests/unit/test_retrieval.py::TestDocumentRetriever -v

# 测试特定方法
pytest tests/unit/test_retrieval.py::TestDocumentRetriever::test_add_documents -v
```

## 📊 测试覆盖率

### 当前覆盖率统计
```
模块                     测试数量    通过率    覆盖率
检索系统 (retrieval)     57         100%      95%+
其他模块                30+        100%      90%+
总计                    90+        100%      92%+
```

### 生成覆盖率报告
```bash
# 生成终端覆盖率报告
pytest --cov=my_langchain --cov-report=term

# 生成HTML覆盖率报告
pytest --cov=my_langchain --cov-report=html

# 查看HTML报告
open htmlcov/index.html
```

## 🧪 测试结构

### 测试目录结构
```
tests/
├── __init__.py
├── unit/                   # 单元测试
│   ├── test_retrieval.py   # 检索系统测试
│   ├── test_llms.py        # LLM模块测试
│   ├── test_chains.py      # Chain模块测试
│   ├── test_memory.py      # Memory模块测试
│   ├── test_tools.py       # Tool模块测试
│   ├── test_vectorstores.py # VectorStore测试
│   ├── test_embeddings.py  # Embedding测试
│   └── test_text_splitters.py # TextSplitter测试
├── integration/            # 集成测试
│   └── test_chains_integration.py
└── fixtures/               # 测试数据
    └── sample_documents.py
```

## 📝 编写测试

### 测试文件命名规范
- 单元测试: `test_*.py`
- 测试类: `Test*`
- 测试方法: `test_*`

### 基础测试模板
```python
import pytest
from my_langchain.retrieval import DocumentRetriever, Document

class TestDocumentRetriever:
    """文档检索器测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.retriever = DocumentRetriever()
        self.documents = [
            Document(content="Python是一种编程语言"),
            Document(content="Java是另一种编程语言")
        ]

    def test_add_documents(self):
        """测试添加文档"""
        doc_ids = self.retriever.add_documents(self.documents)
        assert len(doc_ids) == 2
        assert all(doc_id is not None for doc_id in doc_ids)

    def test_retrieve_documents(self):
        """测试文档检索"""
        self.retriever.add_documents(self.documents)
        result = self.retriever.retrieve("Python")
        assert len(result.documents) > 0
        assert "Python" in result.documents[0].content

    def test_retrieval_with_filters(self):
        """测试带过滤的检索"""
        # 实现过滤测试
        pass

    def test_error_handling(self):
        """测试错误处理"""
        with pytest.raises(ValueError):
            self.retriever.add_documents([])  # 如果空列表应该报错
```

### 参数化测试
```python
import pytest

@pytest.mark.parametrize("query,expected_count", [
    ("Python", 1),
    ("编程", 2),
    ("不存在的词", 0)
])
def test_retrieve_queries(retriever_with_docs, query, expected_count):
    """参数化测试不同查询"""
    result = retriever_with_docs.retrieve(query)
    assert len(result.documents) == expected_count
```

### 异步测试
```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_retrieval():
    """测试异步检索"""
    retriever = AsyncDocumentRetriever()
    result = await retriever.aretrieve("query")
    assert result is not None
```

## 🔧 测试配置

### pytest.ini 配置
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

### conftest.py 配置
```python
import pytest
from my_langchain.retrieval import DocumentRetriever, Document

@pytest.fixture
def sample_documents():
    """示例文档fixture"""
    return [
        Document(content="Python是一种高级编程语言"),
        Document(content="Java是一种面向对象的编程语言"),
        Document(content="JavaScript是一种脚本语言")
    ]

@pytest.fixture
def retriever_with_docs(sample_documents):
    """预添加文档的检索器fixture"""
    retriever = DocumentRetriever()
    retriever.add_documents(sample_documents)
    return retriever

@pytest.fixture
def mock_embedding_model():
    """模拟嵌入模型fixture"""
    return MockEmbedding(embedding_dimension=384)
```

## 🎯 测试最佳实践

### 1. 测试命名
- 使用描述性的测试名称
- 测试名称应该说明测试的内容和预期结果

```python
# 好的命名
def test_retriever_returns_correct_documents_for_query():
    # 测试检索器为查询返回正确的文档
    pass

def test_add_documents_raises_error_for_empty_list():
    # 测试添加空文档列表时抛出错误
    pass

# 避免的命名
def test_retriever_1():
    # 不够描述性
    pass
```

### 2. 测试结构 (AAA模式)
```python
def test_document_retrieval():
    # Arrange (准备)
    retriever = DocumentRetriever()
    documents = [Document(content="Test content")]
    retriever.add_documents(documents)

    # Act (执行)
    result = retriever.retrieve("Test")

    # Assert (断言)
    assert len(result.documents) == 1
    assert result.documents[0].content == "Test content"
```

### 3. 测试隔离
```python
# 每个测试都应该独立，不依赖其他测试的状态
class TestIsolation:
    def test_one(self):
        # 这个测试不影响其他测试
        pass

    def test_two(self):
        # 不依赖test_one的结果
        pass
```

### 4. 边界条件测试
```python
def test_boundary_conditions():
    """测试边界条件"""
    retriever = DocumentRetriever()

    # 测试空文档列表
    with pytest.raises(ValueError):
        retriever.add_documents([])

    # 测试空查询
    with pytest.raises(ValueError):
        retriever.retrieve("")

    # 测试None值
    with pytest.raises(TypeError):
        retriever.add_documents(None)
```

## 🐛 调试测试

### 使用pdb调试
```bash
# 在测试中添加断点
pytest -s --pdb tests/unit/test_retrieval.py::TestDocumentRetriever::test_add_documents

# 在第一个失败时进入调试模式
pytest -x --pdb tests/
```

### 查看详细输出
```bash
# 显示最详细的输出
pytest -v -s --tb=long

# 显示本地变量
pytest --tb=long
```

### 只运行失败的测试
```bash
# 只运行上次失败的测试
pytest --lf

# 运行上次失败的测试并停止在第一个失败
pytest -x --lf
```

## 🏃‍♂️ 性能测试

### 基础性能测试
```python
import time
import pytest

def test_retrieval_performance():
    """测试检索性能"""
    retriever = DocumentRetriever()
    # 添加大量文档
    documents = [Document(content=f"Document {i}") for i in range(1000)]
    retriever.add_documents(documents)

    # 测试检索时间
    start_time = time.time()
    result = retriever.retrieve("Document 1")
    end_time = time.time()

    # 断言性能要求
    assert end_time - start_time < 1.0  # 应该在1秒内完成
    assert len(result.documents) > 0
```

### 标记慢速测试
```python
import pytest

@pytest.mark.slow
def test_large_dataset_performance():
    """标记为慢速测试"""
    # 大数据集性能测试
    pass

# 运行时排除慢速测试
pytest -m "not slow"

# 只运行慢速测试
pytest -m slow
```

## 📈 持续集成

### GitHub Actions配置
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: |
        pytest --cov=my_langchain --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## 🔍 测试报告

### 生成测试报告
```bash
# 生成JUnit XML报告
pytest --junitxml=test-report.xml

# 生成HTML报告
pytest --html=test-report.html --self-contained-html

# 生成覆盖率报告
pytest --cov=my_langchain --cov-report=html --cov-report=xml
```

## 🎯 测试目标

### 当前测试状态
- ✅ **检索系统**: 57个测试，100%通过
- ✅ **LLM模块**: 完整测试覆盖
- ✅ **Chain模块**: 基础功能测试
- ✅ **Memory模块**: 记忆功能测试
- ✅ **Tool模块**: 工具系统测试
- ✅ **VectorStore**: 向量存储测试
- ✅ **Embedding**: 嵌入模型测试
- ✅ **TextSplitter**: 文本分割测试

### 质量目标
- 保持100%测试通过率
- 提高代码覆盖率到95%+
- 增加集成测试覆盖
- 添加性能基准测试

---

**提示**: 在编写新功能时，请同时编写对应的测试用例。遵循TDD(测试驱动开发)原则，先写测试，再实现功能。