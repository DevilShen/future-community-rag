"""BM25 稀疏检索模块（手写，零外部依赖）
======================================

BM25 是经典的「词频 + 逆文档频率」排序算法，按字面匹配打分，与向量检索互补。

为什么需要它：向量检索擅长语义相似，但对「低频专名、精确词」不敏感——
例如「示例e家」这种只出现一次的专名，向量模型没学出强语义，召回会偏；
而 BM25 里这类词 IDF 极高，一搜就能锁定。

中文分词问题：BM25 是词频算法，英文天然按空格切词，中文没有空格。
这里用「字符 bigram（2-gram）」当词：
    "示例e家" -> 示例 / 例e / e家
查询和文档共享这些 bigram，所以专名、精确词能直接命中。
后续若需更语义化的词级匹配，可把 tokenize 换成 jieba 分词。

对外接口：
    tokenize(text) -> list[str]                      文本 -> bigram 词序列
    BM25Index.build(records)                         从 (doc_id, text, category) 建索引
    BM25Index.search(query, top_k, category) -> [(doc_id, score)]
"""

import math
import re
from collections import defaultdict


def tokenize(text: str) -> list[str]:
    """把文本切成字符 bigram 作为 BM25 的「词」。

    先去掉所有空白、统一小写，再按步长 1 切相邻两字符。
    单字符文本退化为单字符词，尽量不丢信息。
    """
    compact = re.sub(r"\s+", "", text).lower()
    if not compact:
        return []
    if len(compact) == 1:
        return [compact]
    return [compact[i : i + 2] for i in range(len(compact) - 1)]


class BM25Index:
    """BM25 索引。

    参数：
        k1 : 词频饱和系数，控制「出现越多分越高」的上限（默认 1.5）
        b  : 文档长度归一化强度，0 不归一化、1 完全归一化（默认 0.75）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[tuple[str, str, str]] = []          # (doc_id, text, category)
        self._doc_tf: list[dict[str, int]] = []              # 每篇文档的 {词: 词频}
        self._doc_len: list[int] = []                        # 每篇文档的词数
        self._df: defaultdict[str, int] = defaultdict(int)   # 词 -> 出现在多少篇文档
        self._avg_len: float = 0.0

    def build(self, records) -> None:
        """从 (doc_id, text, category) 三元组列表建索引。"""
        self._docs = list(records)
        total = 0
        for _doc_id, text, _category in self._docs:
            tokens = tokenize(text)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._doc_tf.append(tf)
            self._doc_len.append(len(tokens))
            total += len(tokens)
            for t in set(tokens):          # 每篇文档每个词只算一次文档频率
                self._df[t] += 1
        self._avg_len = total / len(self._docs) if self._docs else 0.0

    def _idf(self, term: str) -> float:
        """BM25 的 IDF：词越稀有（含它的文档越少），权重越高。"""
        df = self._df.get(term, 0)
        n = len(self._docs)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def _score(self, query_terms: set[str], doc_idx: int) -> float:
        """计算一篇文档对查询的 BM25 分数。"""
        tf = self._doc_tf[doc_idx]
        doc_len = self._doc_len[doc_idx]
        # 文档长度归一化因子：长文档不会被不公平地抬高分数
        norm = 1.0 - self.b + self.b * (doc_len / self._avg_len)
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            # BM25 词频分量：k1 让词频贡献趋于饱和，避免刷词刷分
            tf_part = f * (self.k1 + 1.0) / (f + self.k1 * norm)
            score += self._idf(term) * tf_part
        return score

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[tuple[str, float]]:
        """检索，返回 [(doc_id, score)]，按分数降序。"""
        query_terms = set(tokenize(query))    # 查询词去重，忽略查询内重复
        if not query_terms or not self._docs:
            return []
        scored: list[tuple[str, float]] = []
        for i, (doc_id, _text, cat) in enumerate(self._docs):
            if category and category != "all" and cat != category:
                continue                       # 分类过滤
            s = self._score(query_terms, i)
            if s > 0:
                scored.append((doc_id, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


if __name__ == "__main__":
    # 自检：python -m rag.bm25
    idx = BM25Index()
    idx.build([
        ("d1", "示例e家小程序操作手册 业主认证流程", "operation_manual"),
        ("d2", "未来社区验收办法 总分640分 新建类500分", "industry_standard"),
    ])
    for q in ["示例e家", "小程序名称", "验收总分"]:
        print(q, "->", idx.search(q, top_k=3))