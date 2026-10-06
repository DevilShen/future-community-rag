"""
向量库存储模块
==============

职责：封装 ChromaDB 的本地持久化读写，对上只暴露业务语义的方法。

设计要点：
    1. 向量由本项目自己算好再传入（add 时给 embeddings、query 时给 query_embeddings），
       不用 Chroma 内置的默认向量函数——否则会绕过我们的 mock 模式且难以离线运行；
    2. 距离空间固定为 cosine，检索分数统一换算成「相似度 = 1 - 距离」，
       与 config.SCORE_THRESHOLD 的语义对齐，前端展示也直观；
    3. 用 upsert 而非 add：同一文档重复入库不会因主键冲突报错；
       但内容变动会产生新 chunk_id、留下旧块，所以入库脚本要配合 delete_by_source 使用。

对外接口：
    VectorStore.add(chunks, embeddings)                      写库
    VectorStore.query(query_embedding, top_k, category)      检索（支持分类过滤）
    VectorStore.get_by_source(source)                        取某文档的全部片段（引用定位用）
    VectorStore.count() / category_counts() / list_sources() 统计
    VectorStore.delete_by_source(source) / reset()            维护
"""

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

import config
from rag.chunker import Chunk

# 单次写入的最大条数：Chroma 对单批体积有上限，分批写既能规避也能让进度可见
_WRITE_BATCH = 256


@dataclass
class RetrievedChunk:
    """
    一条检索命中结果。

    输入：由 VectorStore.query 构造。
    输出字段：
        chunk_id      : 块 id
        text          : 块正文（用于拼 prompt 和前端展示引用）
        score         : 相似度，0~1，越大越相关
        category      : 分类 id
        category_name : 分类中文名
        source        : 来源文件相对路径
        title         : 文档标题
        chunk_index   : 块序号
        char_start    : 块在原文中的起始字符偏移（前端定位用不到的中间量，保留以完整反映 schema）
        page          : 块起始位置的 PDF 页码（1 起），0 表示该格式无页码
        heading_path  : 块起始位置的 docx 章节路径，空串表示无
        para_index    : 块起始位置在所属章节内的段落序号（1 起），0 表示无
        mask_hits     : 该块入库前的脱敏命中次数
    """

    chunk_id: str
    text: str
    score: float
    category: str = "uncategorized"
    category_name: str = ""
    source: str = ""
    title: str = ""
    chunk_index: int = 0
    char_start: int = 0
    page: int = 0
    heading_path: str = ""
    para_index: int = 0
    mask_hits: int = 0


def _row_to_chunk(
    chunk_id: str,
    text: str,
    meta: dict,
    score: float = 0.0,
) -> RetrievedChunk:
    """
    把 Chroma 的一行（id + 正文 + 元数据）映射成 RetrievedChunk。

    输入：块 id、块正文、元数据字典、相似度（非检索场景传 0.0）；
    输出：RetrievedChunk。
    抽成独立函数的原因：query（检索）与 get_by_source（引用定位）用的是同一套字段映射，
    写两份迟早会漏改其中一份。所有 meta 取值都带默认值兜底 —— 旧块（新增字段前入库的）
    缺键时退化成「无位置信息」，而不是抛异常。
    """
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        category=str(meta.get("category", "uncategorized")),
        category_name=str(meta.get("category_name", "")),
        source=str(meta.get("source", "")),
        title=str(meta.get("title", "")),
        chunk_index=int(meta.get("chunk_index", 0)),
        char_start=int(meta.get("char_start", 0)),
        page=int(meta.get("page", 0)),
        heading_path=str(meta.get("heading_path", "") or ""),
        para_index=int(meta.get("para_index", 0)),
        mask_hits=int(meta.get("mask_hits", 0)),
    )


class VectorStore:
    """
    ChromaDB 持久化向量库。

    输入（构造参数，均可省略，省略则读 config）：
        persist_dir     : 持久化目录
        collection_name : 集合名
    输出：通过 add / query 等方法读写。
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.persist_dir = str(persist_dir or config.CHROMA_DIR)
        self.collection_name = collection_name or config.CHROMA_COLLECTION

        # anonymized_telemetry=False：私有知识库场景，不发遥测；也避免内网环境超时
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},   # 距离空间：余弦
            embedding_function=None,             # 向量全部由外部传入
        )

    # -------------------------------------------------------- 写入
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """
        写入向量库。

        输入：
            chunks     : Chunk 列表
            embeddings : 与 chunks 一一对应的向量列表
        输出：成功写入的条数；chunks 为空时返回 0。
        说明：用 upsert，重复 chunk_id 覆盖而非报错，支持反复重跑入库。
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(f"块数({len(chunks)})与向量数({len(embeddings)})不一致")

        for start in range(0, len(chunks), _WRITE_BATCH):
            batch_chunks = chunks[start:start + _WRITE_BATCH]
            batch_vectors = embeddings[start:start + _WRITE_BATCH]
            self.collection.upsert(
                ids=[c.chunk_id for c in batch_chunks],
                documents=[c.text for c in batch_chunks],
                embeddings=batch_vectors,
                metadatas=[c.to_metadata() for c in batch_chunks],
            )
        return len(chunks)

    # -------------------------------------------------------- 检索
    def query(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        category: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """
        相似度检索。

        输入：
            query_embedding : 问题向量（必须与入库向量同模型同维度）
            top_k           : 召回条数，缺省 config.TOP_K
            category        : 分类 id 过滤；None 或 "all" 表示全库检索
            score_threshold : 相似度下限，缺省按当前向量模式取
                              （真实模型 config.SCORE_THRESHOLD，mock 模式用更低的
                              config.SCORE_THRESHOLD_MOCK），低于则丢弃
        输出：RetrievedChunk 列表，按相似度降序。
        """
        top_k = top_k or config.TOP_K
        if score_threshold is None:
            score_threshold = config.effective_score_threshold()

        total = self.count()
        if total == 0:
            return []

        # n_results 不能超过库里实际条数，否则部分版本会直接抛错
        n_results = min(top_k, total)
        where = {"category": {"$eq": category}} if category and category != "all" else None

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma 的返回是「按查询分组」的嵌套结构，这里只有一个查询，取第 0 组
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        hits: list[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            score = 1.0 - float(distance)        # 余弦距离 → 相似度
            if score < score_threshold:
                continue
            hits.append(_row_to_chunk(chunk_id, text, meta, round(score, 4)))
        return hits

    def get_by_source(self, source: str) -> list[RetrievedChunk]:
        """
        取某个来源文档的全部片段，按 chunk_index 升序。

        输入：来源相对路径（如 "02_交付流程/xxx.pdf"）；输出：RetrievedChunk 列表，无该文档时返回 []。
        用途：前端「查看原文」浮层 —— 用这些片段拼出该文档的阅读视图并定位高亮命中块。

        两点说明：
            1. 走 Chroma 而非读原始文件：既不碰文件系统（免路径穿越），
               也保证展示的就是「入库后（已脱敏、已去重）」的内容，与模型所见一致；
            2. score 一律 0.0 —— 这不是检索结果，没有相似度语义，前端不应展示匹配度。
        """
        if not source:
            return []
        records = self.collection.get(
            where={"source": {"$eq": source}},
            include=["documents", "metadatas"],
        )
        ids = records.get("ids") or []
        documents = records.get("documents") or []
        metadatas = records.get("metadatas") or []
        chunks = [
            _row_to_chunk(chunk_id, text, meta or {})
            for chunk_id, text, meta in zip(ids, documents, metadatas)
        ]
        # 库里写入顺序不保证，按 chunk_index 还原文档内顺序
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    # -------------------------------------------------------- 统计
    def count(self) -> int:
        """库内总块数。输入：无；输出：int。"""
        return self.collection.count()

    def category_counts(self) -> dict[str, int]:
        """
        各分类的块数。

        输入：无；输出：{分类id: 块数}，只统计有数据的分类。
        实现说明：取全部元数据在本地聚合。演示语料规模（千级）下开销可忽略；
                  若上到十万级，应在入库时单独维护计数表。
        """
        records = self.collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in records.get("metadatas") or []:
            key = str(meta.get("category", "uncategorized"))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def list_sources(self) -> list[str]:
        """已入库的来源文件列表（去重排序）。输入：无；输出：来源路径列表。"""
        records = self.collection.get(include=["metadatas"])
        sources = {str(meta.get("source", "")) for meta in (records.get("metadatas") or [])}
        return sorted(s for s in sources if s)

    # -------------------------------------------------------- 维护
    def all_chunks(self) -> list[RetrievedChunk]:
        """返回库内全部片段，用于构建 BM25 索引等全量扫描场景。

        输入：无；输出：RetrievedChunk 列表（顺序不保证）。
        说明：千级语料下全量拉取开销可忽略；十万级需改分页或持久化索引。
        """
        records = self.collection.get(include=["documents", "metadatas"])
        ids = records.get("ids") or []
        documents = records.get("documents") or []
        metadatas = records.get("metadatas") or []
        return [
            _row_to_chunk(cid, text, meta or {})
            for cid, text, meta in zip(ids, documents, metadatas)
        ]

    def delete_by_source(self, source: str) -> int:
        """
        按来源文件删除全部块。

        输入：来源相对路径；输出：删除前的匹配条数。
        用途：文档更新后重新入库前调用，清掉旧版本残留的块（chunk_id 变了，upsert 覆盖不掉）。
        """
        existing = self.collection.get(where={"source": {"$eq": source}})
        ids = existing.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def reset(self) -> None:
        """
        清空整个集合（慎用）。

        输入：无；输出：无。删除后按相同配置重建，保证后续写入可用。
        """
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )


if __name__ == "__main__":
    # 自检：python -m rag.store
    store = VectorStore()
    print(f"集合={store.collection_name} 目录={store.persist_dir}")
    print(f"当前块数={store.count()}")
    print(f"分类分布={store.category_counts()}")
    sources = store.list_sources()
    print(f"已入库来源（{len(sources)}）={sources}")

    # 引用定位自检：抽查第一个来源，确认能按 chunk_index 还原文档顺序
    if sources:
        chunks = store.get_by_source(sources[0])
        print(f"\nget_by_source 自检：{sources[0]}")
        print(f"  片段数={len(chunks)} 首块 chunk_index={chunks[0].chunk_index if chunks else '-'}")
        for c in chunks[:2]:
            loc = f"第{c.page}页" if c.page else (c.heading_path or "无位置")
            print(f"  #{c.chunk_index} {loc} {c.text[:40]}")
