"""混合检索模块：稀疏(BM25) + 稠密(向量) 双路召回 + RRF 融合
============================================================

纯向量检索的短板是「低频专名、精确词」召回差；纯 BM25 的短板是不会
同义改写。混合检索把两条链路的排名用 RRF 融合，取长补短。

RRF（Reciprocal Rank Fusion）只看排名、不看分数绝对值：
    RRF(doc) = 1/(k + rank_vector) + 1/(k + rank_bm25)   k=60
某个文档在两条链路里名次都靠前，RRF 分就高。

对外接口：
    HybridRetriever.search(question, top_k, category, score_threshold)
        返回 list[RetrievedChunk]，与 Retriever.search 完全同构，
        下游（format_context / LLM）无需任何改动。
"""

from dataclasses import replace

import config
from rag.bm25 import BM25Index
from rag.embedder import Embedder, get_embedder
from rag.store import RetrievedChunk, VectorStore

# RRF 平滑常数：k 越大融合越「平均」；60 是业界常用值
_RRF_K = 60.0


class HybridRetriever:
    """混合检索器。

    输入（构造参数均可省略，省略则读 config）：
        store    : VectorStore 实例
        embedder : Embedder 实例（与入库时同一模型）
    输出：通过 search 返回命中片段。
    """

    def __init__(self, store: VectorStore | None = None, embedder: Embedder | None = None) -> None:
        self.store = store or VectorStore()
        self.embedder = embedder or get_embedder()
        self._bm25: BM25Index | None = None
        self._chunks_by_id: dict[str, RetrievedChunk] = {}

    def _ensure_index(self) -> None:
        """懒加载：首次检索时从向量库读全部块，构建 BM25 索引 + id 映射。

        千级语料构建不到 1 秒，故不单独持久化；语料上十万级再考虑落盘。
        """
        if self._bm25 is not None:
            return
        chunks = self.store.all_chunks()
        records = [(c.chunk_id, c.text, c.category) for c in chunks]
        self._chunks_by_id = {c.chunk_id: c for c in chunks}
        self._bm25 = BM25Index()
        self._bm25.build(records)

    def search(
        self,
        question: str,
        top_k: int | None = None,
        category: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """混合检索。

        输入：同 Retriever.search。
        输出：按 RRF 融合分降序的 RetrievedChunk 列表（score 字段保留向量相似度）。
        """
        question = (question or "").strip()
        if not question:
            return []
        top_k = top_k or config.TOP_K
        if score_threshold is None:
            score_threshold = config.effective_score_threshold()
        self._ensure_index()

        # 1) 向量路：语义召回（带相似度阈值过滤）
        query_vector = self.embedder.embed_query(question)
        vector_hits = self.store.query(
            query_embedding=query_vector,
            top_k=top_k,
            category=category,
            score_threshold=score_threshold,
        )
        vector_rank = {c.chunk_id: i for i, c in enumerate(vector_hits)}   # 0-based

        # 2) BM25 路：字面召回
        bm25_hits = self._bm25.search(question, top_k, category)
        bm25_rank = {cid: i for i, (cid, _s) in enumerate(bm25_hits)}

        # 3) RRF 融合：两条链路的排名取倒数相加
        fused: list[tuple[str, float]] = []
        for cid in set(vector_rank) | set(bm25_rank):
            rrf = 0.0
            if cid in vector_rank:
                rrf += 1.0 / (_RRF_K + vector_rank[cid] + 1)
            if cid in bm25_rank:
                rrf += 1.0 / (_RRF_K + bm25_rank[cid] + 1)
            fused.append((cid, rrf))
        fused.sort(key=lambda x: -x[1])

        # 4) 映射回 RetrievedChunk，按 RRF 取前 top_k
        result: list[RetrievedChunk] = []
        for cid, _rrf in fused[:top_k]:
            chunk = self._chunks_by_id.get(cid)
            if chunk is None:
                continue
            if cid in vector_rank:
                # 保留向量相似度，便于前端展示匹配度
                chunk = replace(chunk, score=vector_hits[vector_rank[cid]].score)
            else:
                # 纯 BM25 命中的块没有向量相似度，score 记 0
                chunk = replace(chunk, score=0.0)
            result.append(chunk)
        return result


if __name__ == "__main__":
    # 自检：python -m rag.hybrid "示例e家"
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "示例e家"
    hits = HybridRetriever().search(q)
    print(f"问题：{q}")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h.source}  {h.text[:40]}")