"""
检索模块
========

职责：把「用户问题」变成「带来源的候选片段」，并负责把它们编号排版成
      可送进大模型的上下文字符串。

这一层的价值在于把三件事解耦：
    - 问题向量化（用与入库完全相同的 Embedder，保证同向量空间）
    - 相似度检索与阈值过滤（委托 VectorStore）
    - 上下文编排（编号、元数据标注、长度截断）

对外接口：
    Retriever.search(question, top_k, category) -> list[RetrievedChunk]
    format_context(chunks, max_chars)           -> (上下文文本, 实际入选的片段)
"""

import config
from rag.embedder import Embedder, get_embedder
from rag.store import RetrievedChunk, VectorStore


class Retriever:
    """
    检索器。

    输入（构造参数，均可省略）：
        store    : VectorStore 实例，缺省新建
        embedder : Embedder 实例，缺省取全局单例（务必与入库时同一模型）
    输出：通过 search 返回命中片段。
    """

    def __init__(self, store: VectorStore | None = None, embedder: Embedder | None = None) -> None:
        self.store = store or VectorStore()
        self.embedder = embedder or get_embedder()

    def search(
        self,
        question: str,
        top_k: int | None = None,
        category: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """
        检索与问题最相关的片段。

        输入：
            question        : 用户自然语言问题
            top_k           : 召回条数，缺省 config.TOP_K
            category        : 分类 id 过滤；None 或 "all" 为全库
            score_threshold : 相似度下限，缺省按向量模式自动选择
                              （见 config.effective_score_threshold）
        输出：RetrievedChunk 列表（按相似度降序）；问题为空或无命中时返回空列表。
        """
        question = (question or "").strip()
        if not question:
            return []

        query_vector = self.embedder.embed_query(question)
        return self.store.query(
            query_embedding=query_vector,
            top_k=top_k,
            category=category,
            score_threshold=score_threshold,
        )


def format_context(
    chunks: list[RetrievedChunk],
    max_chars: int | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    """
    把命中片段编号排版成提示词上下文。

    输入：
        chunks    : 检索结果（已按相似度降序）
        max_chars : 正文总字符上限，缺省 config.CONTEXT_MAX_CHARS，超出则按排名截断
    输出：(上下文文本, 实际写入上下文的那部分片段)

    为什么要返回第二个值：调用方需要用「真正进了上下文」的片段来构造来源列表，
    否则会出现答案引用了 [3]、但来源列表里只有 [1][2] 的错位。

    格式示例：
        [1] 来源：02_交付流程/承接查验.md｜分类：交付流程｜标题：承接查验｜相似度：0.83
        正文……
    """
    max_chars = max_chars or config.CONTEXT_MAX_CHARS

    blocks: list[str] = []
    used: list[RetrievedChunk] = []
    total = 0

    for index, hit in enumerate(chunks, start=1):
        header = (
            f"[{index}] 来源：{hit.source}｜分类：{hit.category_name or hit.category}"
            f"｜标题：{hit.title}｜相似度：{hit.score:.2f}"
        )
        block = f"{header}\n{hit.text.strip()}"
        # 至少保留第一条：即便单条就超限，也不能给模型一个空上下文
        if used and total + len(block) > max_chars:
            break
        blocks.append(block)
        used.append(hit)
        total += len(block)

    return "\n\n".join(blocks), used


if __name__ == "__main__":
    # 自检：python -m rag.retriever "承接查验包括哪些内容"
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "承接查验包括哪些内容"
    retriever = Retriever()
    hits = retriever.search(question)

    print(f"问题：{question}")
    print(f"命中 {len(hits)} 条（top_k={config.TOP_K}，阈值={config.effective_score_threshold()}）")
    for i, hit in enumerate(hits, 1):
        print(f"  [{i}] {hit.score:.4f} {hit.category_name} {hit.source}")
        print(f"      {hit.text[:60]}")

    context, used = format_context(hits)
    print(f"\n上下文 {len(context)} 字，实际入选 {len(used)} 条")
    print(context[:300])
