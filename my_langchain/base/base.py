"""
核心基础抽象类

定义了 LangChain 框架的所有基础接口。
这些接口确保了组件之间的互操作性和可扩展性。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from pydantic import BaseModel, Field


class BaseComponent(ABC, BaseModel):
    """
    所有组件的基础类

    提供通用的组件功能，如序列化、配置管理等。
    """

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        执行组件的主要功能

        Returns:
            组件的执行结果
        """
        pass

    async def arun(self, *args, **kwargs) -> Any:
        """
        异步执行组件的主要功能

        默认实现是简单的同步包装，子类可以重写以提供真正的异步实现。

        Returns:
            组件的执行结果
        """
        return self.run(*args, **kwargs)


class BaseLLM(BaseComponent):
    """
    大语言模型的基础抽象类

    定义了所有 LLM 实现必须遵循的接口。
    """

    model_name: str = Field(..., description="模型名称")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成随机性")
    max_tokens: Optional[int] = Field(default=None, description="最大生成令牌数")

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本回复

        Args:
            prompt: 输入提示词
            **kwargs: 其他生成参数

        Returns:
            生成的文本回复
        """
        pass

    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """
        异步生成文本回复

        Args:
            prompt: 输入提示词
            **kwargs: 其他生成参数

        Returns:
            生成的文本回复
        """
        pass

    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        批量生成文本回复

        Args:
            prompts: 输入提示词列表
            **kwargs: 其他生成参数

        Returns:
            生成的文本回复列表
        """
        pass

    def run(self, prompt: str, **kwargs) -> str:
        """实现 BaseComponent 的 run 方法"""
        return self.generate(prompt, **kwargs)

    async def arun(self, prompt: str, **kwargs) -> str:
        """实现 BaseComponent 的 arun 方法"""
        return await self.agenerate(prompt, **kwargs)


class BasePromptTemplate(BaseComponent):
    """
    提示词模板的基础抽象类

    定义了动态生成提示词的接口。
    """

    input_variables: List[str] = Field(default_factory=list, description="输入变量名列表")
    template: str = Field(..., description="模板字符串")

    @abstractmethod
    def format(self, **kwargs) -> str:
        """
        格式化模板

        Args:
            **kwargs: 模板变量的值

        Returns:
            格式化后的提示词
        """
        pass

    @abstractmethod
    def save(self, file_path: str) -> None:
        """
        保存模板到文件

        Args:
            file_path: 保存路径
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, file_path: str) -> "BasePromptTemplate":
        """
        从文件加载模板

        Args:
            file_path: 文件路径

        Returns:
            加载的模板实例
        """
        pass

    def run(self, **kwargs) -> str:
        """实现 BaseComponent 的 run 方法"""
        return self.format(**kwargs)


class BaseChain(BaseComponent):
    """
    链式调用的基础抽象类

    定义了将多个组件连接成处理链的接口。
    """

    @abstractmethod
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行链式调用

        Args:
            inputs: 输入数据字典

        Returns:
            输出数据字典
        """
        pass

    @abstractmethod
    async def acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        异步执行链式调用

        Args:
            inputs: 输入数据字典

        Returns:
            输出数据字典
        """
        pass

    def run(self, **kwargs) -> Dict[str, Any]:
        """实现 BaseComponent 的 run 方法"""
        return self(kwargs)


class BaseMemory(BaseComponent):
    """
    记忆系统的基础抽象类

    定义了存储和检索对话历史的接口。
    """

    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        保存对话上下文

        Args:
            inputs: 输入数据
            outputs: 输出数据
        """
        pass

    @abstractmethod
    def load_memory(self) -> Dict[str, Any]:
        """
        加载记忆内容

        Returns:
            记忆数据字典
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空记忆"""
        pass

    def run(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """实现 BaseComponent 的 run 方法"""
        self.save_context(inputs, outputs)


class BaseAgent(BaseComponent):
    """
    智能体的基础抽象类

    定义了自主决策和执行任务的接口。
    """

    @abstractmethod
    def plan(self, task: str, **kwargs) -> List[Dict[str, Any]]:
        """
        制定执行计划

        Args:
            task: 任务描述
            **kwargs: 其他参数

        Returns:
            执行步骤列表
        """
        pass

    @abstractmethod
    def execute_step(self, step: Dict[str, Any]) -> Any:
        """
        执行单个步骤

        Args:
            step: 步骤描述

        Returns:
            执行结果
        """
        pass

    @abstractmethod
    def should_continue(self, result: Any) -> bool:
        """
        判断是否应该继续执行

        Args:
            result: 当前执行结果

        Returns:
            是否继续
        """
        pass

    def run(self, task: str, **kwargs) -> Any:
        """实现 BaseComponent 的 run 方法"""
        plan = self.plan(task, **kwargs)
        results = []

        for step in plan:
            result = self.execute_step(step)
            results.append(result)

            if not self.should_continue(result):
                break

        return results


class BaseTool(BaseComponent):
    """
    工具的基础抽象类

    定义了可执行工具的接口。
    """

    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")

    @abstractmethod
    def _run(self, *args, **kwargs) -> Any:
        """
        工具的具体实现

        Returns:
            工具执行结果
        """
        pass

    @abstractmethod
    async def _arun(self, *args, **kwargs) -> Any:
        """
        工具的异步实现

        Returns:
            工具执行结果
        """
        pass

    def run(self, *args, **kwargs) -> Any:
        """实现 BaseComponent 的 run 方法"""
        return self._run(*args, **kwargs)

    async def arun(self, *args, **kwargs) -> Any:
        """实现 BaseComponent 的 arun 方法"""
        return await self._arun(*args, **kwargs)


class BaseEmbedding(BaseComponent):
    """
    嵌入模型的基础抽象类

    定义了文本向量化的接口。
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        将单个文本转换为向量

        Args:
            text: 输入文本

        Returns:
            文本的向量表示
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文本转换为向量

        Args:
            texts: 输入文本列表

        Returns:
            文本向量的列表
        """
        pass

    @abstractmethod
    async def aembed_text(self, text: str) -> List[float]:
        """
        异步将单个文本转换为向量

        Args:
            text: 输入文本

        Returns:
            文本的向量表示
        """
        pass

    def run(self, text: str) -> List[float]:
        """实现 BaseComponent 的 run 方法"""
        return self.embed_text(text)


class BaseVectorStore(BaseComponent):
    """
    向量存储的基础抽象类

    定义了向量数据库的接口。
    """

    @abstractmethod
    def add_vectors(self, vectors: List[List[float]], texts: List[str]) -> List[str]:
        """
        添加向量和文本

        Args:
            vectors: 向量列表
            texts: 对应的文本列表

        Returns:
            添加的文档ID列表
        """
        pass

    @abstractmethod
    def similarity_search(self, query_vector: List[float], k: int = 4) -> List[Dict[str, Any]]:
        """
        相似度搜索

        Args:
            query_vector: 查询向量
            k: 返回结果数量

        Returns:
            相似文档列表，包含文档和相似度分数
        """
        pass

    @abstractmethod
    async def asimilarity_search(self, query_vector: List[float], k: int = 4) -> List[Dict[str, Any]]:
        """
        异步相似度搜索

        Args:
            query_vector: 查询向量
            k: 返回结果数量

        Returns:
            相似文档列表，包含文档和相似度分数
        """
        pass


class BaseTextSplitter(BaseComponent):
    """
    文本分割器的基础抽象类

    定义了文本分割的接口。
    """

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        分割单个文本

        Args:
            text: 输入文本

        Returns:
            分割后的文本块列表
        """
        pass

    @abstractmethod
    def split_texts(self, texts: List[str]) -> List[str]:
        """
        批量分割文本

        Args:
            texts: 输入文本列表

        Returns:
            分割后的文本块列表
        """
        pass

    def run(self, text: str) -> List[str]:
        """实现 BaseComponent 的 run 方法"""
        return self.split_text(text)


class BaseRetriever(BaseComponent):
    """
    检索器的基础抽象类

    定义了文档检索的接口。
    """

    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        检索相关文档

        Args:
            query: 查询文本
            **kwargs: 其他检索参数

        Returns:
            相关文档列表
        """
        pass

    @abstractmethod
    async def aretrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        异步检索相关文档

        Args:
            query: 查询文本
            **kwargs: 其他检索参数

        Returns:
            相关文档列表
        """
        pass

    def run(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """实现 BaseComponent 的 run 方法"""
        return self.retrieve(query, **kwargs)