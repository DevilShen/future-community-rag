"""
向量化模块
==========

职责：把文本转成定长向量，屏蔽不同向量供应商的差异。

两条实现路径：
    1. provider = openai（默认含义：任何 OpenAI 兼容的向量服务）
       硅基流动 bge-m3、DeepSeek、OpenAI 官方都走这条分支——换供应商只改
       config 里的 EMBED_BASE_URL / EMBED_MODEL / EMBED_DIM，代码零改动。
    2. provider = mock
       不联网，用「字符 3-gram 哈希 + L2 归一化」生成确定性假向量。
       虽是假向量，但它保留了词面相似性：同样包含「承接查验」的两段文本向量
       仍然靠近，因此离线状态下检索链路（召回→排序→拼 prompt）能真实跑通，
       便于无网络演示和 CI。

对外接口：
    Embedder.embed_texts(texts) -> list[list[float]]   批量向量化文档
    Embedder.embed_query(text)  -> list[float]         向量化单条查询
    get_embedder()              -> Embedder            进程内单例
"""

import hashlib
import math
import re
import time

import config

# mock 向量使用的 n-gram 长度，与 screener 的 shingle 保持一致，便于对照理解
_MOCK_NGRAM = 3
# 真实接口失败时的最大重试次数与初始退避秒数（指数退避）
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.0


def _normalize_vector(vector: list[float]) -> list[float]:
    """
    L2 归一化。

    输入：任意向量；输出：模长为 1 的向量（零向量原样返回）。
    作用：归一化后余弦相似度等于点积，检索分数区间稳定在 [0,1]，阈值才好设。
    """
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def _mock_embed(text: str, dim: int) -> list[float]:
    """
    生成确定性假向量（离线模式）。

    输入：文本、目标维度；输出：归一化后的 dim 维向量。
    算法：对去空白小写文本取所有 3-gram，用 md5 映射到某一维并累加计数，
          最后 L2 归一化——本质是哈希版词袋，词面越接近向量越相似。

    注意：必须用 md5 而不是内置 hash()，后者每个进程随机加盐，会导致同一段
    文本跨进程得到不同向量，入库和查询对不上。
    """
    vector = [0.0] * dim
    compact = re.sub(r"\s+", "", text).lower()
    if not compact:
        return vector

    for i in range(max(1, len(compact) - _MOCK_NGRAM + 1)):
        gram = compact[i:i + _MOCK_NGRAM]
        digest = hashlib.md5(gram.encode("utf-8")).hexdigest()
        vector[int(digest[:8], 16) % dim] += 1.0

    return _normalize_vector(vector)


class Embedder:
    """
    向量化器。

    输入（构造参数，均可省略，省略则读 config）：
        provider   : "mock" 或 "openai"（含所有 OpenAI 兼容服务）
        api_key    : 供应商密钥
        base_url   : 兼容端点
        model      : 模型名
        dim        : 期望维度，用于校验接口返回是否符合配置
        batch_size : 单次请求的文本条数
    输出：通过 embed_texts / embed_query 返回向量。
    """

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dim: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.provider = (provider or config.EMBED_PROVIDER).lower()
        self.api_key = api_key if api_key is not None else config.EMBED_API_KEY
        self.base_url = base_url or config.EMBED_BASE_URL
        self.model = model or config.EMBED_MODEL
        self.dim = dim or config.EMBED_DIM
        self.batch_size = batch_size or config.EMBED_BATCH_SIZE
        self._client = None       # 懒加载，mock 模式下永不创建

    # -------------------------------------------------------- 属性
    @property
    def is_mock(self) -> bool:
        """是否离线 mock 模式。输出：bool。"""
        return self.provider == "mock"

    @property
    def client(self):
        """
        懒加载 openai 客户端。

        输入：无；输出：openai.OpenAI 实例（同 base_url 只建一次）。
        说明：密钥缺失时立刻报错，而不是等调用接口才失败，便于早发现问题。
        """
        if self._client is None:
            from openai import OpenAI

            if not self.api_key:
                raise RuntimeError(
                    f"向量供应商 {self.provider} 需要 API Key，请在 .env 中配置 EMBED_API_KEY；"
                    f"或改用离线模式 EMBED_PROVIDER=mock"
                )
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    # -------------------------------------------------------- 真实接口
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """
        调用一次向量接口（带指数退避重试）。

        输入：一批文本；输出：与输入等长的向量列表。
        """
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self.client.embeddings.create(model=self.model, input=batch)
                # 接口不保证返回顺序，按 index 排序后再取，防止向量与文本错位
                items = sorted(resp.data, key=lambda d: d.index)
                return [item.embedding for item in items]
            except Exception as exc:                      # noqa: BLE001 - 统一重试
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    sleep_for = _RETRY_BACKOFF * (2 ** attempt)
                    print(f"[embedder] 第 {attempt + 1} 次调用失败：{exc}，{sleep_for:.0f}s 后重试")
                    time.sleep(sleep_for)
        raise RuntimeError(f"向量接口连续 {_MAX_RETRIES} 次失败：{last_error}")

    def _validate_dim(self, vector: list[float]) -> None:
        """
        校验返回维度。

        输入：任一向量；输出：无（不一致则抛异常）。
        作用：维度与配置不符会在写 ChromaDB 时才炸且报错晦涩，这里提前拦住并说清原因。
        """
        if len(vector) != self.dim:
            raise ValueError(
                f"向量维度不匹配：模型 {self.model} 返回 {len(vector)} 维，"
                f"配置 EMBED_DIM={self.dim}。请修改 .env 中的 EMBED_DIM。"
            )

    # -------------------------------------------------------- 对外接口
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        批量向量化（入库用）。

        输入：文本列表；输出：与输入顺序一致的向量列表。
        """
        if not texts:
            return []
        if self.is_mock:
            return [_mock_embed(t, self.dim) for t in texts]

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        self._validate_dim(vectors[0])
        # 统一归一化，让不同供应商的分数区间一致
        return [_normalize_vector(v) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """
        向量化单条查询（检索用）。

        输入：查询文本；输出：单个向量。
        说明：与 embed_texts 走同一路径，保证「问题」和「文档」在同一向量空间。
              必须用同一个模型，否则检索结果没有意义。
        """
        return self.embed_texts([text])[0]


# 进程内单例：避免每次请求都重建客户端与重复读配置
_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    """获取全局向量化器。输入：无；输出：Embedder 单例。"""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER


if __name__ == "__main__":
    # 自检：python -m rag.embedder
    embedder = get_embedder()
    print(f"provider={embedder.provider} model={embedder.model} dim={embedder.dim} mock={embedder.is_mock}")

    a = embedder.embed_query("承接查验包括哪些内容")
    b = embedder.embed_query("承接查验的内容有哪些")
    c = embedder.embed_query("今天天气不错")
    dot = lambda x, y: sum(i * j for i, j in zip(x, y))       # noqa: E731 - 已归一化，点积即余弦
    print(f"向量长度={len(a)}")
    print(f"相似问法余弦相似度：{dot(a, b):.4f}  （应明显高于）")
    print(f"无关问法余弦相似度：{dot(a, c):.4f}")
