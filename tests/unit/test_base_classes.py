"""
测试基础抽象类
"""
import pytest
from abc import ABC
from pydantic import Field
from my_langchain.base.base import (
    BaseComponent,
    BaseLLM,
    BasePromptTemplate,
    BaseChain,
    BaseMemory,
    BaseAgent,
    BaseTool,
    BaseEmbedding,
    BaseVectorStore,
    BaseTextSplitter,
    BaseRetriever
)


class TestBaseComponent:
    """测试 BaseComponent 基类"""

    def test_base_component_is_abstract(self):
        """测试 BaseComponent 是抽象类"""
        assert issubclass(BaseComponent, ABC)
        with pytest.raises(TypeError):
            BaseComponent()

    def test_concrete_component_implementation(self):
        """测试具体组件实现"""

        class ConcreteComponent(BaseComponent):
            def run(self, *args, **kwargs):
                return "test_result"

        component = ConcreteComponent()
        assert component.run() == "test_result"

        # 测试异步方法
        import asyncio
        result = asyncio.run(component.arun())
        assert result == "test_result"


class TestBaseLLM:
    """测试 BaseLLM 抽象类"""

    def test_base_llm_is_abstract(self):
        """测试 BaseLLM 是抽象类"""
        assert issubclass(BaseLLM, ABC)
        with pytest.raises(TypeError):
            BaseLLM(model_name="test")

    def test_concrete_llm_implementation(self):
        """测试具体 LLM 实现"""

        class ConcreteLLM(BaseLLM):
            def generate(self, prompt: str, **kwargs) -> str:
                return f"Response to: {prompt}"

            async def agenerate(self, prompt: str, **kwargs) -> str:
                return f"Async response to: {prompt}"

            def generate_batch(self, prompts: list, **kwargs) -> list:
                return [f"Response to: {p}" for p in prompts]

        llm = ConcreteLLM(model_name="test-model", temperature=0.5, max_tokens=100)

        # 测试属性
        assert llm.model_name == "test-model"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 100

        # 测试方法
        assert llm.generate("Hello") == "Response to: Hello"
        assert llm.run("Hello") == "Response to: Hello"
        assert llm.generate_batch(["Hello", "World"]) == [
            "Response to: Hello",
            "Response to: World"
        ]


class TestBasePromptTemplate:
    """测试 BasePromptTemplate 抽象类"""

    def test_base_prompt_template_is_abstract(self):
        """测试 BasePromptTemplate 是抽象类"""
        assert issubclass(BasePromptTemplate, ABC)
        with pytest.raises(TypeError):
            BasePromptTemplate(template="test")

    def test_concrete_prompt_template_implementation(self):
        """测试具体提示词模板实现"""

        class ConcretePromptTemplate(BasePromptTemplate):
            def format(self, **kwargs) -> str:
                result = self.template
                for key, value in kwargs.items():
                    result = result.replace(f"{{{key}}}", str(value))
                return result

            def save(self, file_path: str) -> None:
                with open(file_path, 'w') as f:
                    f.write(self.template)

            @classmethod
            def load(cls, file_path: str):
                with open(file_path, 'r') as f:
                    template = f.read()
                return cls(template=template)

        template = ConcretePromptTemplate(
            template="Hello {name}, you are {age} years old.",
            input_variables=["name", "age"]
        )

        # 测试格式化
        result = template.format(name="Alice", age=25)
        assert result == "Hello Alice, you are 25 years old."

        # 测试run方法
        assert template.run(name="Bob", age=30) == "Hello Bob, you are 30 years old."


class TestBaseChain:
    """测试 BaseChain 抽象类"""

    def test_base_chain_is_abstract(self):
        """测试 BaseChain 是抽象类"""
        assert issubclass(BaseChain, ABC)
        with pytest.raises(TypeError):
            BaseChain()

    def test_concrete_chain_implementation(self):
        """测试具体链实现"""

        class ConcreteChain(BaseChain):
            def __call__(self, inputs: dict) -> dict:
                return {"output": f"Processed: {inputs.get('input', '')}"}

            async def acall(self, inputs: dict) -> dict:
                return {"output": f"Async processed: {inputs.get('input', '')}"}

        chain = ConcreteChain()

        # 测试调用
        result = chain({"input": "test"})
        assert result == {"output": "Processed: test"}

        # 测试run方法
        result = chain.run(input="test")
        assert result == {"output": "Processed: test"}


class TestBaseMemory:
    """测试 BaseMemory 抽象类"""

    def test_base_memory_is_abstract(self):
        """测试 BaseMemory 是抽象类"""
        assert issubclass(BaseMemory, ABC)
        with pytest.raises(TypeError):
            BaseMemory()

    def test_concrete_memory_implementation(self):
        """测试具体记忆实现"""

        class ConcreteMemory(BaseMemory):
            memory: list = Field(default_factory=list)

            def save_context(self, inputs: dict, outputs: dict) -> None:
                self.memory.append({"inputs": inputs, "outputs": outputs})

            def load_memory(self) -> dict:
                return {"history": self.memory}

            def clear(self) -> None:
                self.memory = []

        memory = ConcreteMemory()

        # 测试保存和加载
        memory.save_context({"question": "Hello"}, {"answer": "Hi"})
        memory.save_context({"question": "How are you?"}, {"answer": "Fine"})

        loaded = memory.load_memory()
        assert len(loaded["history"]) == 2
        assert loaded["history"][0]["inputs"]["question"] == "Hello"

        # 测试清空
        memory.clear()
        loaded = memory.load_memory()
        assert len(loaded["history"]) == 0


class TestBaseAgent:
    """测试 BaseAgent 抽象类"""

    def test_base_agent_is_abstract(self):
        """测试 BaseAgent 是抽象类"""
        assert issubclass(BaseAgent, ABC)
        with pytest.raises(TypeError):
            BaseAgent()

    def test_concrete_agent_implementation(self):
        """测试具体智能体实现"""

        class ConcreteAgent(BaseAgent):
            def plan(self, task: str, **kwargs) -> list:
                return [{"step": 1, "action": "analyze"}, {"step": 2, "action": "execute"}]

            def execute_step(self, step: dict) -> str:
                return f"Executed step {step['step']}: {step['action']}"

            def should_continue(self, result: str) -> bool:
                return "step 1" in result.lower()

        agent = ConcreteAgent()

        # 测试规划
        plan = agent.plan("test task")
        assert len(plan) == 2
        assert plan[0]["action"] == "analyze"

        # 测试执行步骤
        result = agent.execute_step({"step": 1, "action": "test"})
        assert result == "Executed step 1: test"


class TestBaseTool:
    """测试 BaseTool 抽象类"""

    def test_base_tool_is_abstract(self):
        """测试 BaseTool 是抽象类"""
        assert issubclass(BaseTool, ABC)
        with pytest.raises(TypeError):
            BaseTool(name="test", description="test tool")

    def test_concrete_tool_implementation(self):
        """测试具体工具实现"""

        class ConcreteTool(BaseTool):
            name: str = "calculator"
            description: str = "A simple calculator"

            def _run(self, a: int, b: int, operation: str = "add") -> int:
                if operation == "add":
                    return a + b
                elif operation == "multiply":
                    return a * b
                else:
                    raise ValueError(f"Unknown operation: {operation}")

            async def _arun(self, a: int, b: int, operation: str = "add") -> int:
                return self._run(a, b, operation)

        tool = ConcreteTool()

        # 测试属性
        assert tool.name == "calculator"
        assert tool.description == "A simple calculator"

        # 测试执行
        result = tool.run(2, 3, operation="add")
        assert result == 5

        result = tool.run(2, 3, operation="multiply")
        assert result == 6


class TestBaseEmbedding:
    """测试 BaseEmbedding 抽象类"""

    def test_base_embedding_is_abstract(self):
        """测试 BaseEmbedding 是抽象类"""
        assert issubclass(BaseEmbedding, ABC)
        with pytest.raises(TypeError):
            BaseEmbedding()

    def test_concrete_embedding_implementation(self):
        """测试具体嵌入实现"""

        class ConcreteEmbedding(BaseEmbedding):
            def embed_text(self, text: str) -> list:
                # 简单的伪嵌入：使用字符的ASCII码
                return [ord(c) / 255.0 for c in text[:10]]  # 限制长度

            def embed_texts(self, texts: list) -> list:
                return [self.embed_text(text) for text in texts]

            async def aembed_text(self, text: str) -> list:
                return self.embed_text(text)

        embedding = ConcreteEmbedding()

        # 测试单个文本嵌入
        result = embedding.embed_text("hello")
        assert len(result) == min(5, 10)  # "hello" 有5个字符
        assert all(0 <= val <= 1 for val in result)

        # 测试批量嵌入
        results = embedding.embed_texts(["hello", "world"])
        assert len(results) == 2
        assert len(results[0]) == 5
        assert len(results[1]) == 5


class TestBaseVectorStore:
    """测试 BaseVectorStore 抽象类"""

    def test_base_vector_store_is_abstract(self):
        """测试 BaseVectorStore 是抽象类"""
        assert issubclass(BaseVectorStore, ABC)
        with pytest.raises(TypeError):
            BaseVectorStore()

    def test_concrete_vector_store_implementation(self):
        """测试具体向量存储实现"""

        class ConcreteVectorStore(BaseVectorStore):
            vectors: list = Field(default_factory=list)
            texts: list = Field(default_factory=list)

            def add_vectors(self, vectors: list, texts: list) -> list:
                ids = [f"id_{len(self.vectors) + i}" for i in range(len(vectors))]
                self.vectors.extend(vectors)
                self.texts.extend(texts)
                return ids

            def similarity_search(self, query_vector: list, k: int = 4) -> list:
                # 简单的余弦相似度
                import math

                def cosine_similarity(v1, v2):
                    dot_product = sum(a * b for a, b in zip(v1, v2))
                    norm1 = math.sqrt(sum(a * a for a in v1))
                    norm2 = math.sqrt(sum(b * b for b in v2))
                    return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0

                similarities = []
                for i, vector in enumerate(self.vectors):
                    if len(vector) == len(query_vector):
                        sim = cosine_similarity(vector, query_vector)
                        similarities.append({"text": self.texts[i], "score": sim, "id": f"id_{i}"})

                similarities.sort(key=lambda x: x["score"], reverse=True)
                return similarities[:k]

            async def asimilarity_search(self, query_vector: list, k: int = 4) -> list:
                return self.similarity_search(query_vector, k)

        store = ConcreteVectorStore()

        # 测试添加向量
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        texts = ["text1", "text2"]
        ids = store.add_vectors(vectors, texts)

        assert len(ids) == 2
        assert ids[0] == "id_0"
        assert ids[1] == "id_1"

        # 测试相似度搜索
        results = store.similarity_search([0.1, 0.2, 0.3], k=2)
        assert len(results) <= 2
        assert "text" in results[0]
        assert "score" in results[0]


class TestBaseTextSplitter:
    """测试 BaseTextSplitter 抽象类"""

    def test_base_text_splitter_is_abstract(self):
        """测试 BaseTextSplitter 是抽象类"""
        assert issubclass(BaseTextSplitter, ABC)
        with pytest.raises(TypeError):
            BaseTextSplitter()

    def test_concrete_text_splitter_implementation(self):
        """测试具体文本分割器实现"""

        class ConcreteTextSplitter(BaseTextSplitter):
            chunk_size: int = Field(default=100)

            def split_text(self, text: str) -> list:
                chunks = []
                for i in range(0, len(text), self.chunk_size):
                    chunks.append(text[i:i + self.chunk_size])
                return chunks

            def split_texts(self, texts: list) -> list:
                all_chunks = []
                for text in texts:
                    all_chunks.extend(self.split_text(text))
                return all_chunks

        splitter = ConcreteTextSplitter(chunk_size=5)

        # 测试文本分割
        text = "Hello world! This is a test."
        chunks = splitter.split_text(text)
        assert len(chunks) == 6  # "Hello", " worl", "d! Th", "is is", " a te", "st."
        assert chunks[0] == "Hello"

        # 测试批量分割
        texts = ["Hello world", "Foo bar baz"]
        chunks = splitter.split_texts(texts)
        assert len(chunks) > 0


class TestBaseRetriever:
    """测试 BaseRetriever 抽象类"""

    def test_base_retriever_is_abstract(self):
        """测试 BaseRetriever 是抽象类"""
        assert issubclass(BaseRetriever, ABC)
        with pytest.raises(TypeError):
            BaseRetriever()

    def test_concrete_retriever_implementation(self):
        """测试具体检索器实现"""

        class ConcreteRetriever(BaseRetriever):
            documents: list = Field(default_factory=lambda: [
                {"text": "Python is a programming language", "id": "doc1"},
                {"text": "JavaScript is also a programming language", "id": "doc2"},
                {"text": "Machine learning is a subset of AI", "id": "doc3"}
            ])

            def retrieve(self, query: str, **kwargs) -> list:
                # 简单的关键词匹配
                results = []
                query_lower = query.lower()

                for doc in self.documents:
                    if any(word in doc["text"].lower() for word in query_lower.split()):
                        results.append(doc.copy())

                return results

            async def aretrieve(self, query: str, **kwargs) -> list:
                return self.retrieve(query, **kwargs)

        retriever = ConcreteRetriever()

        # 测试检索
        results = retriever.retrieve("programming")
        assert len(results) == 2  # 包含 "programming" 的文档

        results = retriever.retrieve("Python")
        assert len(results) == 1  # 只有一个包含 "Python" 的文档
        assert results[0]["id"] == "doc1"