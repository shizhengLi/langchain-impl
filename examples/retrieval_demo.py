#!/usr/bin/env python3
"""
Retrieval模块使用示例
演示了三种检索器的使用方法
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
import time

def create_sample_documents():
    """创建示例文档"""
    return [
        Document(
            content="Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。",
            metadata={"language": "Python", "type": "programming", "difficulty": "beginner"}
        ),
        Document(
            content="机器学习是人工智能的一个分支，专注于开发能够从数据中学习的算法和统计模型。机器学习算法通过识别数据中的模式来进行预测和决策。",
            metadata={"field": "AI", "type": "concept", "difficulty": "intermediate"}
        ),
        Document(
            content="深度学习是机器学习的一个子领域，基于人工神经网络。深度学习在计算机视觉、自然语言处理和语音识别等领域取得了突破性进展。",
            metadata={"field": "AI", "type": "advanced", "difficulty": "advanced"}
        ),
        Document(
            content="Java是一种面向对象的编程语言，由Sun Microsystems开发。Java的设计理念是'一次编写，到处运行'，具有平台无关性。",
            metadata={"language": "Java", "type": "programming", "difficulty": "intermediate"}
        ),
        Document(
            content="自然语言处理(NLP)是人工智能和语言学的交叉领域，专注于计算机与人类语言之间的交互。NLP技术包括机器翻译、情感分析、文本摘要等。",
            metadata={"field": "AI", "type": "application", "difficulty": "intermediate"}
        ),
        Document(
            content="数据科学是一个跨学科领域，使用科学方法、过程、算法和系统从数据中提取知识和见解。数据科学结合了统计学、数学和计算机科学。",
            metadata={"field": "Data Science", "type": "concept", "difficulty": "intermediate"}
        ),
        Document(
            content="算法是解决特定问题的一系列明确指令。好的算法应该具有正确性、可读性、健壮性和高效性等特点。时间和空间复杂度是评估算法性能的重要指标。",
            metadata={"field": "Computer Science", "type": "concept", "difficulty": "beginner"}
        ),
        Document(
            content="云计算是通过互联网提供计算服务的模式，包括服务器、存储、数据库、网络、软件等。云计算提供了弹性、可扩展性和成本效益等优势。",
            metadata={"field": "Cloud Computing", "type": "technology", "difficulty": "intermediate"}
        )
    ]

def demo_document_retriever():
    """演示文档检索器的使用"""
    print("=" * 50)
    print("📄 DocumentRetriever 演示")
    print("=" * 50)

    # 创建检索器
    retriever = DocumentRetriever()

    # 添加文档
    documents = create_sample_documents()
    retriever.add_documents(documents)

    # 获取统计信息
    stats = retriever.get_term_statistics()
    print(f"📊 文档统计:")
    print(f"   总文档数: {stats['total_documents']}")
    print(f"   总词数: {stats['total_terms']}")
    print(f"   唯一词汇数: {stats['vocabulary_size']}")
    print(f"   平均文档长度: {stats['avg_document_length']:.1f}")

    # 执行不同类型的检索
    queries = [
        ("Python编程", "similarity"),
        ("机器学习算法", "tfidf"),
        ("AI技术", "bm25")
    ]

    for query, search_type in queries:
        print(f"\n🔍 查询: '{query}' (搜索类型: {search_type})")
        # 创建特定配置的检索器
        config_retriever = DocumentRetriever(config=RetrievalConfig(search_type=search_type))
        config_retriever.add_documents(documents)
        result = config_retriever.retrieve(query)

        print(f"   检索方法: {result.retrieval_method}")
        print(f"   检索时间: {result.search_time:.4f}秒")
        print(f"   结果数量: {len(result.documents)}")

        for i, doc in enumerate(result.documents[:3]):
            print(f"   {i+1}. Score: {doc.relevance_score:.3f} | {doc.get_text_snippet(60)}")

def demo_vector_retriever():
    """演示向量检索器的使用"""
    print("\n" + "=" * 50)
    print("🔢 VectorRetriever 演示")
    print("=" * 50)

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
    documents = create_sample_documents()
    doc_ids = retriever.add_documents(documents)
    print(f"📝 已添加 {len(doc_ids)} 个文档")

    # 获取统计信息
    embedding_stats = retriever.get_embedding_stats()
    print(f"📊 向量统计:")
    print(f"   向量数量: {embedding_stats['vector_count']}")
    print(f"   嵌入维度: {embedding_stats['embedding_dimension']}")
    print(f"   缓存大小: {embedding_stats['cache_size']}")

    # 执行不同类型的检索
    queries = [
        ("深度学习和神经网络", "similarity"),
        ("编程语言比较", "mmr"),
        ("数据科学技术", "similarity")
    ]

    for query, search_type in queries:
        print(f"\n🧠 查询: '{query}' (搜索类型: {search_type})")

        config = RetrievalConfig(
            search_type=search_type,
            top_k=3,
            score_threshold=0.3
        )

        # 创建特定配置的检索器
        config_retriever = VectorRetriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
            config=config
        )
        config_retriever.add_documents(documents)
        result = config_retriever.retrieve(query)

        print(f"   检索方法: {result.retrieval_method}")
        print(f"   检索时间: {result.search_time:.4f}秒")
        print(f"   结果数量: {len(result.documents)}")

        for i, doc in enumerate(result.documents):
            print(f"   {i+1}. Score: {doc.relevance_score:.3f} | {doc.get_text_snippet(60)}")
            if doc.additional_info:
                vector_score = doc.additional_info.get("vector_score", 0)
                print(f"       原始向量分数: {vector_score:.3f}")

def demo_ensemble_retriever():
    """演示集成检索器的使用"""
    print("\n" + "=" * 50)
    print("🎭 EnsembleRetriever 演示")
    print("=" * 50)

    # 创建基础检索器
    doc_retriever = DocumentRetriever()

    # 创建向量检索器
    embedding_model = MockEmbedding(embedding_dimension=384)
    vector_config = VectorStoreConfig(dimension=384)
    vector_store = InMemoryVectorStore(config=vector_config)
    vector_retriever = VectorRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    # 添加文档
    documents = create_sample_documents()
    doc_retriever.add_documents(documents)
    vector_retriever.add_documents(documents)

    # 创建集成检索器
    ensemble = EnsembleRetriever(
        retrievers=[doc_retriever, vector_retriever],
        weights=[0.4, 0.6],  # 向量检索权重更高
        fusion_strategy="weighted_score"
    )

    # 获取集成统计信息
    ensemble_stats = ensemble.get_ensemble_stats()
    print(f"📊 集成统计:")
    print(f"   检索器数量: {ensemble_stats['num_retrievers']}")
    print(f"   融合策略: {ensemble_stats['fusion_strategy']}")
    print(f"   权重分配: {ensemble_stats['weights']}")
    print(f"   总文档数: {ensemble_stats['total_documents']}")

    # 执行检索
    queries = ["人工智能应用", "编程语言学习", "数据分析方法"]

    for query in queries:
        print(f"\n🎯 查询: '{query}'")

        # 创建带配置的集成检索器
        ensemble_with_config = EnsembleRetriever(
            retrievers=[doc_retriever, vector_retriever],
            weights=[0.4, 0.6],  # 向量检索权重更高
            fusion_strategy="weighted_score",
            config=RetrievalConfig(top_k=3)
        )

        # 执行集成检索
        start_time = time.time()
        result = ensemble_with_config.retrieve(query)
        total_time = time.time() - start_time

        print(f"   检索方法: {result.retrieval_method}")
        print(f"   检索时间: {total_time:.4f}秒")
        print(f"   结果数量: {len(result.documents)}")

        for i, doc in enumerate(result.documents):
            print(f"   {i+1}. Score: {doc.relevance_score:.3f} | {doc.get_text_snippet(60)}")
            if doc.additional_info:
                source_retrievers = doc.additional_info.get("source_retrievers", [])
                print(f"       来源检索器: {source_retrievers}")

        # 比较各个检索器的结果
        print(f"\n   🔄 检索器结果对比:")
        comparison = ensemble_with_config.compare_retrievers(query)
        for name, comp_result in comparison.items():
            print(f"     {name}: {len(comp_result.documents)} 个结果, "
                  f"平均分数: {comp_result.get_average_score():.3f}")

def demo_performance_comparison():
    """演示性能对比"""
    print("\n" + "=" * 50)
    print("⚡ 性能对比测试")
    print("=" * 50)

    # 准备测试数据
    documents = create_sample_documents()
    test_query = "机器学习和人工智能"

    # 测试DocumentRetriever
    print("\n📄 DocumentRetriever 性能:")
    doc_retriever = DocumentRetriever()
    doc_retriever.add_documents(documents)

    start_time = time.time()
    doc_result = doc_retriever.retrieve(test_query)
    doc_time = time.time() - start_time

    print(f"   检索时间: {doc_time:.4f}秒")
    print(f"   结果数量: {len(doc_result.documents)}")

    # 测试VectorRetriever
    print("\n🔢 VectorRetriever 性能:")
    embedding_model = MockEmbedding(embedding_dimension=384)
    vector_config = VectorStoreConfig(dimension=384)
    vector_store = InMemoryVectorStore(config=vector_config)
    vector_retriever = VectorRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store
    )
    vector_retriever.add_documents(documents)

    start_time = time.time()
    vector_result = vector_retriever.retrieve(test_query)
    vector_time = time.time() - start_time

    print(f"   检索时间: {vector_time:.4f}秒")
    print(f"   结果数量: {len(vector_result.documents)}")

    # 测试EnsembleRetriever
    print("\n🎭 EnsembleRetriever 性能:")
    ensemble = EnsembleRetriever(
        retrievers=[doc_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    start_time = time.time()
    ensemble_result = ensemble.retrieve(test_query)
    ensemble_time = time.time() - start_time

    print(f"   检索时间: {ensemble_time:.4f}秒")
    print(f"   结果数量: {len(ensemble_result.documents)}")

    # 性能总结
    print(f"\n📊 性能总结:")
    print(f"   DocumentRetriever:  {doc_time:.4f}秒 (最简单，最快)")
    print(f"   VectorRetriever:    {vector_time:.4f}秒 (语义理解，中等)")
    print(f"   EnsembleRetriever:   {ensemble_time:.4f}秒 (综合效果，稍慢)")

def demo_filtering_and_config():
    """演示过滤和配置功能"""
    print("\n" + "=" * 50)
    print("🔧 过滤和配置演示")
    print("=" * 50)

    # 创建检索器
    retriever = DocumentRetriever()
    documents = create_sample_documents()
    retriever.add_documents(documents)

    # 演示元数据过滤
    print("📋 元数据过滤:")

    # 过滤编程语言相关的文档
    config = RetrievalConfig(
        filter_dict={"type": "programming"},
        top_k=5
    )
    filter_retriever = DocumentRetriever(config=config)
    filter_retriever.add_documents(documents)
    result = filter_retriever.retrieve("语言")
    print(f"   编程类型文档: {len(result.documents)} 个")
    for doc in result.documents:
        print(f"     - {doc.get_text_snippet(40)} ({doc.metadata.get('language', 'Unknown')})")

    # 过滤AI相关的文档
    config = RetrievalConfig(
        filter_dict={"field": "AI"},
        top_k=5
    )
    filter_retriever2 = DocumentRetriever(config=config)
    filter_retriever2.add_documents(documents)
    result = filter_retriever2.retrieve("技术")
    print(f"   AI领域文档: {len(result.documents)} 个")
    for doc in result.documents:
        print(f"     - {doc.get_text_snippet(40)} ({doc.metadata.get('type', 'Unknown')})")

    # 演示分数阈值过滤
    print("\n🎯 分数阈值过滤:")
    config = RetrievalConfig(
        score_threshold=0.3,  # 只返回分数大于0.3的结果
        top_k=10
    )
    filter_retriever3 = DocumentRetriever(config=config)
    filter_retriever3.add_documents(documents)
    result = filter_retriever3.retrieve("Python 编程")
    print(f"   高质量结果: {len(result.documents)} 个")
    for doc in result.documents:
        print(f"     - Score: {doc.relevance_score:.3f} | {doc.get_text_snippet(40)}")

def main():
    """主函数"""
    print("🚀 Retrieval模块功能演示")
    print("=" * 80)

    try:
        # 演示各种检索器
        demo_document_retriever()
        demo_vector_retriever()
        demo_ensemble_retriever()

        # 性能对比
        demo_performance_comparison()

        # 高级功能
        demo_filtering_and_config()

        print("\n" + "=" * 80)
        print("✅ 演示完成!")
        print("\n💡 提示:")
        print("   - DocumentRetriever: 适合小数据集，无需向量化")
        print("   - VectorRetriever: 适合语义检索，理解查询意图")
        print("   - EnsembleRetriever: 结合多种策略，提高检索质量")
        print("   - 所有检索器都支持配置、过滤和性能优化")

    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()