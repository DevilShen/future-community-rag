"""
RAG 核心包
==========

对外统一出口，把七个模块的公共接口聚合成一条可读的流水线：

    解析 loader → 分块 chunker → 筛选 screener → 向量化 embedder
                                                     ↓
                    问答 llm ← 检索 retriever ← 存储 store

外部（ingest.py / api.py）只从这里 import，不直接碰子模块，方便日后替换实现。

导出采用惰性加载（PEP 562 __getattr__），而不是在包初始化时 import 全部子模块：
    1. 只用到 loader/chunker 时不必连带加载 chromadb、openai 这些重依赖，启动更快；
    2. 避免 `python -m rag.xxx` 触发
       "found in sys.modules after import of package" 的 RuntimeWarning。

用法：
    from rag import VectorStore, Retriever, get_llm      # 惰性解析并在首次访问后缓存
"""

from typing import TYPE_CHECKING, Any

# 符号名 -> 所在子模块。新增导出只需在这里加一行。
_EXPORTS: dict[str, str] = {
    # 数据结构
    "Chunk": "rag.chunker",
    "LoadedDoc": "rag.loader",
    "RetrievedChunk": "rag.store",
    "Answer": "rag.llm",
    "ScreenResult": "rag.screener",
    # 解析 / 分块
    "load_file": "rag.loader",
    "load_dir": "rag.loader",
    "normalize_text": "rag.loader",
    "split_text": "rag.chunker",
    "chunk_document": "rag.chunker",
    # 筛选
    "desensitize": "rag.screener",
    "screen_chunks": "rag.screener",
    # 向量化 / 存储 / 检索
    "Embedder": "rag.embedder",
    "get_embedder": "rag.embedder",
    "VectorStore": "rag.store",
    "Retriever": "rag.retriever",
    "format_context": "rag.retriever",
    # 作答
    "LLMClient": "rag.llm",
    "get_llm": "rag.llm",
    "BM25Index": "rag.bm25",
    "HybridRetriever": "rag.hybrid",
}

__all__ = sorted(_EXPORTS)

if TYPE_CHECKING:
    # 仅给类型检查器和 IDE 用：真实导入交给下面的 __getattr__，
    # 这样静态分析能看到真实符号，运行时又不产生 eager import。
    from rag.chunker import Chunk, chunk_document, split_text
    from rag.embedder import Embedder, get_embedder
    from rag.llm import Answer, LLMClient, get_llm
    from rag.loader import LoadedDoc, load_dir, load_file, normalize_text
    from rag.retriever import Retriever, format_context
    from rag.screener import ScreenResult, desensitize, screen_chunks
    from rag.store import RetrievedChunk, VectorStore


def __getattr__(name: str) -> Any:
    """
    按需导入子模块符号。

    输入：属性名；输出：对应对象；名字未登记时抛 AttributeError（符合模块协议）。
    首次访问后写入 globals() 缓存，后续访问不再经过本函数。
    """
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """让 dir(rag) 列出全部导出符号。输入：无；输出：排序后的符号名列表。"""
    return __all__
