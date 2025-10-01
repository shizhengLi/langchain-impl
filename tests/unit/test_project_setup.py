"""
测试项目设置是否正确
"""
import pytest
from pathlib import Path


def test_project_structure_exists():
    """测试项目结构是否存在"""
    base_path = Path(__file__).parent.parent.parent

    # 检查主要目录
    expected_dirs = [
        "my_langchain",
        "my_langchain/base",
        "my_langchain/llms",
        "my_langchain/prompts",
        "my_langchain/chains",
        "tests/unit",
        "tests/integration",
        "examples",
        "docs"
    ]

    for dir_path in expected_dirs:
        full_path = base_path / dir_path
        assert full_path.exists(), f"目录 {dir_path} 不存在"
        assert full_path.is_dir(), f"{dir_path} 不是目录"


def test_init_files_exist():
    """测试所有 __init__.py 文件是否存在"""
    base_path = Path(__file__).parent.parent.parent

    # 检查 Python 包的 __init__.py 文件
    init_files = [
        "my_langchain/__init__.py",
        "my_langchain/base/__init__.py",
        "my_langchain/llms/__init__.py",
        "my_langchain/prompts/__init__.py",
        "my_langchain/chains/__init__.py",
        "tests/__init__.py",
        "tests/unit/__init__.py",
        "tests/integration/__init__.py"
    ]

    for init_file in init_files:
        full_path = base_path / init_file
        assert full_path.exists(), f"文件 {init_file} 不存在"


def test_config_files_exist():
    """测试配置文件是否存在"""
    base_path = Path(__file__).parent.parent.parent

    config_files = [
        "requirements.txt",
        "pyproject.toml",
        "pytest.ini",
        ".gitignore",
        "README.md"
    ]

    for config_file in config_files:
        full_path = base_path / config_file
        assert full_path.exists(), f"配置文件 {config_file} 不存在"


def test_can_import_base_package():
    """测试可以导入基础包"""
    try:
        import my_langchain
        assert my_langchain is not None
    except ImportError as e:
        pytest.fail(f"无法导入 my_langchain 包: {e}")