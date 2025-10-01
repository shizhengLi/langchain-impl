# 安装指南

本文档将指导您如何安装和配置LangChain实现项目。

## 系统要求

- Python 3.8 或更高版本
- pip (Python包管理器)
- Git (可选，用于克隆项目)

## 🚀 快速安装

### 1. 克隆项目
```bash
git clone https://github.com/your-username/langchain-impl.git
cd langchain-impl
```

### 2. 创建虚拟环境（推荐）
```bash
# 使用venv创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 验证安装
```bash
# 运行测试确保安装成功
pytest

# 运行示例验证功能
python examples/basic_usage.py
python examples/retrieval_demo.py
```

## 📋 依赖说明

### 核心依赖
```txt
# requirements.txt 主要内容
pydantic>=2.0.0          # 数据验证和类型安全
numpy>=1.21.0            # 数值计算
typing-extensions>=4.0.0  # 类型注解扩展
pytest>=7.0.0            # 测试框架
```

### 可选依赖
```txt
# 用于特定功能的可选依赖
scikit-learn>=1.0.0      # 机器学习算法（用于TF-IDF等）
faiss-cpu>=1.7.0         # 向量检索（可选GPU版本）
openai>=1.0.0            # OpenAI API（可选）
```

## 🔧 开发环境设置

### 1. 开发模式安装
```bash
# 以开发模式安装项目
pip install -e .

# 安装开发依赖
pip install -r requirements-dev.txt
```

### 2. 代码格式化工具
```bash
# 安装代码格式化工具
pip install black isort flake8

# 格式化代码
black .
isort .

# 代码检查
flake8 .
```

### 3. 类型检查
```bash
# 安装类型检查工具
pip install mypy

# 运行类型检查
mypy my_langchain/
```

## 🧪 测试环境

### 运行所有测试
```bash
# 运行完整测试套件
pytest

# 显示测试覆盖率
pytest --cov=my_langchain

# 生成HTML覆盖率报告
pytest --cov=my_langchain --cov-report=html
```

### 运行特定模块测试
```bash
# 测试检索模块
pytest tests/unit/test_retrieval.py -v

# 测试特定功能
pytest tests/unit/test_retrieval.py::TestDocumentRetriever -v
```

## 🐳 Docker安装（可选）

### 1. 创建Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["pytest"]
```

### 2. 构建和运行
```bash
# 构建镜像
docker build -t langchain-impl .

# 运行容器
docker run langchain-impl
```

## 🔍 故障排除

### 常见问题

#### 1. Python版本不兼容
**错误**: `Python 3.7 is not supported`
**解决**: 升级到Python 3.8或更高版本

#### 2. 依赖安装失败
**错误**: `ERROR: Could not install packages`
**解决**:
```bash
# 升级pip
pip install --upgrade pip

# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 3. 测试失败
**错误**: `ImportError: No module named 'my_langchain'`
**解决**:
```bash
# 确保在项目根目录
cd /path/to/langchain-impl

# 添加当前目录到Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 4. Pydantic版本问题
**错误**: `PydanticUserError: If you use @root_validator with pre=False`
**解决**: 确保使用Pydantic v2.0+
```bash
pip install "pydantic>=2.0.0"
```

### 权限问题
```bash
# 如果遇到权限问题，使用用户安装
pip install --user -r requirements.txt
```

### 虚拟环境问题
```bash
# 如果虚拟环境无法激活，重新创建
rm -rf venv
python -m venv venv
source venv/bin/activate  # 或 venv\Scripts\activate (Windows)
```

## 📦 发布安装

### 从PyPI安装（如果已发布）
```bash
pip install langchain-impl
```

### 从源码安装
```bash
pip install git+https://github.com/your-username/langchain-impl.git
```

## 🔧 配置选项

### 环境变量
```bash
# 可选的环境变量配置
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LANGCHAIN_IMPL_LOG_LEVEL=INFO
export LANGCHAIN_IMPL_CACHE_DIR=./cache
```

### 配置文件
```python
# config.py 示例
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get('LANGCHAIN_IMPL_CACHE_DIR', os.path.join(BASE_DIR, 'cache'))
LOG_LEVEL = os.environ.get('LANGCHAIN_IMPL_LOG_LEVEL', 'INFO')
```

## 📚 验证安装

### 1. 基础功能测试
```python
# test_installation.py
from my_langchain.retrieval import DocumentRetriever, Document
from my_langchain.llms import MockLLM

def test_basic_functionality():
    # 测试检索功能
    retriever = DocumentRetriever()
    doc = Document(content="Hello world")
    retriever.add_documents([doc])
    result = retriever.retrieve("hello")
    assert len(result.documents) > 0

    # 测试LLM功能
    llm = MockLLM()
    response = llm.generate("Test prompt")
    assert response.text is not None

    print("✅ 安装验证成功！")

if __name__ == "__main__":
    test_basic_functionality()
```

### 2. 运行验证脚本
```bash
python test_installation.py
```

## 🎯 下一步

安装完成后，您可以：

1. 📖 阅读[检索系统文档](retrieval.md)了解核心功能
2. 🚀 运行[示例代码](../examples/)体验功能
3. 🧪 运行[测试套件](../tests/)验证环境
4. 📚 查看[教程目录](tutorials/)学习用法

## 🤝 获取帮助

如果遇到安装问题：

1. 查看[故障排除指南](tutorials/troubleshooting.md)
2. 提交[Issue](https://github.com/your-username/langchain-impl/issues)
3. 查看[常见问题解答](faq.md)

---

**提示**: 建议在虚拟环境中进行开发，以避免依赖冲突。如果您是新手，请先阅读Python虚拟环境的官方文档。