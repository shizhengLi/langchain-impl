# -*- coding: utf-8 -*-
"""
Text splitter module for document chunking and segmentation
"""

from .base import BaseTextSplitter, TextSplitterConfig
from .character_splitter import CharacterTextSplitter
from .recursive_splitter import RecursiveCharacterTextSplitter
from .types import Document, Chunk, SplitStrategy, SplitResult

__all__ = [
    "BaseTextSplitter",
    "TextSplitterConfig",
    "CharacterTextSplitter",
    "RecursiveCharacterTextSplitter",
    "Document",
    "Chunk",
    "SplitStrategy",
    "SplitResult"
]