#!/usr/bin/env python3
"""
Retrieval演示测试脚本
验证演示代码的正确性
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from my_langchain.retrieval import (
    DocumentRetriever,
    VectorRetriever,
    EnsembleRetriever,
    Document,
    RetrievalConfig
)
from my_langchain.embeddings import MockEmbedding
from my_langchain.vectorstores import InMemoryVectorStore
from my_langchain.vectorstores.types import VectorStoreConfig

def test_document_retriever():
    """测试文档检索器"""
    print("🧪 测试 DocumentRetriever...")

    # 创建检索器和文档
    retriever = DocumentRetriever()
    documents = [
        Document(content="Python programming language"),
        Document(content="Machine learning algorithms"),
        Document(content="Java programming language")
    ]

    # 添加文档
    doc_ids = retriever.add_documents(documents)
    assert len(doc_ids) == 3

    # 测试检索
    result = retriever.retrieve("Python")
    assert len(result.documents) > 0
    assert "Python" in result.documents[0].content

    print("   ✅ DocumentRetriever 测试通过")

def test_vector_retriever():
    """测试向量检索器"""
    print("🧪 测试 VectorRetriever...")

    # 创建组件
    embedding_model = MockEmbedding(embedding_dimension=384)
    vector_config = VectorStoreConfig(dimension=384)
    vector_store = InMemoryVectorStore(config=vector_config)

    # 创建检索器
    retriever = VectorRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    # 添加文档
    documents = [
        Document(content="Deep learning"),
        Document(content="Neural networks"),
        Document(content="Natural language processing")
    ]
    doc_ids = retriever.add_documents(documents)
    assert len(doc_ids) == 3

    # 测试检索
    result = retriever.retrieve("AI技术")
    assert len(result.documents) > 0
    assert result.retrieval_method.startswith("vector_")

    print("   ✅ VectorRetriever 测试通过")

def test_ensemble_retriever():
    """测试集成检索器"""
    print("🧪 测试 EnsembleRetriever...")

    # 创建基础检索器
    doc_retriever = DocumentRetriever()

    embedding_model = MockEmbedding(embedding_dimension=384)
    vector_config = VectorStoreConfig(dimension=384)
    vector_store = InMemoryVectorStore(config=vector_config)
    vector_retriever = VectorRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    # 添加文档
    documents = [
        Document(content="Data science"),
        Document(content="Statistical analysis"),
        Document(content="Machine learning")
    ]
    doc_retriever.add_documents(documents)
    vector_retriever.add_documents(documents)

    # 创建集成检索器
    ensemble = EnsembleRetriever(
        retrievers=[doc_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    # 测试检索
    result = ensemble.retrieve("数据分析")
    assert len(result.documents) > 0
    assert result.retrieval_method.startswith("ensemble_")

    print("   ✅ EnsembleRetriever 测试通过")

def test_configuration():
    """测试配置功能"""
    print("🧪 测试配置功能...")

    retriever = DocumentRetriever()
    documents = [
        Document(content="Python programming", metadata={"type": "programming"}),
        Document(content="Java programming", metadata={"type": "programming"}),
        Document(content="Artificial Intelligence", metadata={"type": "technology"})
    ]
    retriever.add_documents(documents)

    # 创建带配置的检索器
    config = RetrievalConfig(
        filter_dict={"type": "programming"},
        top_k=2
    )
    retriever_with_config = DocumentRetriever(config=config)
    retriever_with_config.add_documents(documents)
    result = retriever_with_config.retrieve("language")
    assert len(result.documents) <= 2
    assert all(doc.metadata.get("type") == "programming" for doc in result.documents)

    print("   ✅ 配置功能测试通过")

def test_statistics():
    """测试统计功能"""
    print("🧪 测试统计功能...")

    retriever = DocumentRetriever()
    documents = [
        Document(content="Python programming language"),
        Document(content="Java programming language"),
        Document(content="Web development frameworks")
    ]
    retriever.add_documents(documents)

    # 测试统计信息
    stats = retriever.get_term_statistics()
    assert stats["total_documents"] == 3
    assert stats["total_terms"] > 0
    assert stats["vocabulary_size"] > 0

    print("   ✅ 统计功能测试通过")

def main():
    """运行所有测试"""
    print("🚀 Retrieval模块演示测试")
    print("=" * 50)

    try:
        test_document_retriever()
        test_vector_retriever()
        test_ensemble_retriever()
        test_configuration()
        test_statistics()

        print("\n" + "=" * 50)
        print("✅ 所有测试通过!")
        print("💡 可以运行 retrieval_demo.py 查看完整演示")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())