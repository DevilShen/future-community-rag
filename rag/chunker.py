"""
中文分块模块
============

职责：把整篇正文切成带重叠的语义块，并给每块挂上分类/来源/标题等元数据。

为什么不用「按固定字数硬切」：
    中文没有空格分词边界，硬切极易把一句话、一个条款切断，检索时既破坏语义
    又让引用片段难以阅读。这里改成「先切句子、再贪心装箱、尾部回取重叠」三步。

    1. 切句  ：按中文标点（。！？；）与换行切成原子句，标点跟随前句；
    2. 装箱  ：顺序累加句子，超 max_chars 就封箱，保证不切断句子；
    3. 重叠  ：封箱后从尾部回取整句，凑够 overlap 作为下一块的开头，
               让答案跨边界时仍能被完整召回。

对外接口：
    split_text(text, max_chars, overlap)            -> list[tuple[str, int]]
    chunk_document(doc, max_chars, overlap, ...)    -> list[Chunk]
"""

import hashlib
import re
from bisect import bisect_right
from dataclasses import dataclass, field

import config
from rag.loader import LoadedDoc, Locator, normalize_text

# 句子终止符：中文标点 + 英文标点 + 换行
_TERMINATORS = "。！？；!?;"

# 匹配「若干前导换行 + 一段非终止符文本 + 一个可选终止符」。
# 前导换行必须纳入模式：段落间的 \n\n 自身构不成 token，若不在下一句的前缀里捕获，
# 换行信息就丢了，相邻段落会被拼成「……住宅项目。第二条……」，破坏原文结构。
_TOKEN_RE = re.compile(rf"\n*[^\n{_TERMINATORS}]+[\n{_TERMINATORS}]?")


@dataclass
class Chunk:
    """
    一个待入库的文本块。

    输入：由 chunk_document 构造。
    输出字段：
        chunk_id     : 块唯一 id（含内容指纹），同一文档改一个字 id 就变，避免旧块残留
        text         : 块正文（已筛选，用于展示与引用）
        embed_text   : 送向量模型的文本 = 元数据前缀 + 正文，提升召回的语义锚点
        category     : 分类 id
        category_name: 分类中文名，前端直接展示
        source       : 来源文件相对路径
        title        : 所属文档标题
        chunk_index  : 块在文档内的序号，从 0 开始
        char_start   : 块在规整后正文中的起始字符偏移
        page         : 块的起始位置命中的 PDF 页码（1 起），0 表示无
        heading_path : 块的起始位置命中的 docx 章节路径，空串表示无
        para_index   : 块的起始位置命中的章节内段落序号（1 起），0 表示无
    """

    chunk_id: str
    text: str
    category: str
    category_name: str
    source: str
    title: str
    chunk_index: int
    char_start: int = 0
    page: int = 0
    heading_path: str = ""
    para_index: int = 0
    embed_text: str = field(default="")
    mask_hits: int = 0        # 本块被脱敏命中的次数，由 screener 回填

    def __post_init__(self) -> None:
        # 元数据前缀参与向量化：把「行业标准 / 交付流程」这类分类信号喂给模型，
        # 让「交付流程有哪些环节」这种问法更容易命中同分类的块
        if not self.embed_text:
            self.embed_text = f"【{self.category_name}】{self.title}\n{self.text}"

    def to_metadata(self) -> dict[str, str | int]:
        """
        转成 ChromaDB 可存的元数据（只允许 str/int/float/bool）。

        输入：自身；输出：扁平字典，检索时随命中结果一起返回用于展示引用来源。
        """
        return {
            "category": self.category,
            "category_name": self.category_name,
            "source": self.source,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "page": self.page,
            "heading_path": self.heading_path,
            "para_index": self.para_index,
            "mask_hits": self.mask_hits,
        }


# ------------------------------------------------------------------ 切句
def _split_units(text: str) -> list[tuple[str, str]]:
    """
    切成原子句。

    输入：规整后的文本；输出：[(句子, 尾部换行), ...]，尾部换行为 "\\n" 表示
          此处是段落边界，用于拼接时还原原文结构。

    换行归属规则：token 内部的前导换行属于「前一句」的结尾（说明前一句之后有
    段落断开），token 末尾的换行属于「本句」的结尾。两侧都要判，否则
    "……标准\\n第一条……" 和 "……项目。\\n\\n第二条……" 这两种换行都会被丢掉。
    """
    units: list[list[str]] = []
    for token in _TOKEN_RE.findall(text):
        stripped = token.strip()
        if not stripped:
            continue
        # 本 token 前的换行 → 记到上一句尾部
        if token.startswith("\n") and units:
            units[-1][1] = "\n"
        sep = "\n" if token.endswith("\n") else ""
        units.append([stripped, sep])
    return [(sentence, sep) for sentence, sep in units]


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """
    超长单句兜底硬切。

    输入：长度超过 max_chars 的单句、块上限；输出：等长切片列表。
    场景：PDF 抽出的文本常整页无标点，或固化成表格一行，必须先切开。
    """
    return [sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars)]


# ------------------------------------------------------------------ 主流程
def split_text(
    text: str,
    max_chars: int | None = None,
    overlap: int | None = None,
) -> list[tuple[str, int]]:
    """
    把长文本切成带重叠的块。

    输入：
        text      : 原始文本
        max_chars : 单块最大字符数，缺省取 config.CHUNK_SIZE
        overlap   : 相邻块重叠字符数，缺省取 config.CHUNK_OVERLAP
    输出：[(块文本, 起始字符偏移), ...]，偏移基于规整后的文本，可回溯定位原文。
    """
    max_chars = max_chars or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    # 防御：重叠必须小于块上限，否则装箱无法推进（每轮都吞回整块）
    overlap = max(0, min(overlap, max_chars - 1))

    text = normalize_text(text)
    if not text:
        return []

    # 1) 切句 + 超长句硬切，得到 (句子, 尾部换行, 原文偏移)
    items: list[tuple[str, str, int]] = []
    cursor = 0
    for sentence, sep in _split_units(text):
        # 用 find 定位原始偏移，找不到（理论上不会）则退回当前位置
        pos = text.find(sentence, cursor)
        if pos < 0:
            pos = cursor
        cursor = pos + len(sentence)
        if len(sentence) <= max_chars:
            items.append((sentence, sep, pos))
        else:
            pieces = _hard_split(sentence, max_chars)
            for i, piece in enumerate(pieces):
                # 最后一片继承原句尾换行，前面几片无换行
                items.append((piece, sep if i == len(pieces) - 1 else "", pos + i * max_chars))

    if not items:
        return []

    # 2) 贪心装箱 + 3) 尾部回取重叠
    chunks: list[tuple[str, int]] = []
    bucket: list[tuple[str, str, int]] = []
    bucket_len = 0

    def flush() -> list[tuple[str, str, int]]:
        """封箱：输出当前桶，并返回供下一块复用的重叠尾巴。"""
        nonlocal bucket, bucket_len
        body = "".join(s + sep for s, sep, _ in bucket).strip()
        if body:
            chunks.append((body, bucket[0][2]))

        # 从尾部回取整句，凑够 overlap；但若尾巴本身已超过一整块上限则放弃重叠，
        # 否则下一块开头就超限，装箱会原地打转
        tail: list[tuple[str, str, int]] = []
        tail_len = 0
        for item in reversed(bucket):
            tail.insert(0, item)
            tail_len += len(item[0])
            if tail_len >= overlap:
                break
        if tail_len + 1 >= max_chars:
            tail, tail_len = [], 0

        bucket = list(tail)
        bucket_len = tail_len
        return bucket

    for item in items:
        sentence, sep, _ = item
        cost = len(sentence) + len(sep)
        if bucket and bucket_len + cost > max_chars:
            flush()
        bucket.append(item)
        bucket_len += cost

    # 收尾：最后一批未封箱的内容
    if bucket:
        body = "".join(s + sep for s, sep, _ in bucket).strip()
        if body:
            chunks.append((body, bucket[0][2]))

    # 去重：重叠可能让小段落被前一块完整吞掉，产生完全相同的块
    seen: set[str] = set()
    unique: list[tuple[str, int]] = []
    for body, start in chunks:
        if body not in seen:
            seen.add(body)
            unique.append((body, start))
    return unique


def _make_chunk_id(source: str, index: int, text: str) -> str:
    """
    生成块 id。

    输入：来源、序号、正文；输出：形如 "手册-a1b2c3d4-0007" 的稳定 id。
    带上内容指纹：文档改动后旧块 id 不同，重新入库不会与历史残留混淆。
    """
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    stem = source.rsplit("/", 1)[-1].rsplit(".", 1)[0][:20]
    safe_stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]", "_", stem) or "doc"
    return f"{safe_stem}-{digest}-{index:04d}"


def chunk_document(
    doc: LoadedDoc,
    max_chars: int | None = None,
    overlap: int | None = None,
    min_length: int | None = None,
) -> list[Chunk]:
    """
    把一篇文档切成分块列表。

    输入：
        doc        : loader 产出的 LoadedDoc
        max_chars  : 单块最大字符数，缺省 config.CHUNK_SIZE
        overlap    : 重叠字符数，缺省 config.CHUNK_OVERLAP
        min_length : 碎块下限，低于此长度丢弃，缺省 config.CHUNK_MIN_LENGTH
    输出：Chunk 列表（保留文档内顺序）。
    """
    min_length = config.CHUNK_MIN_LENGTH if min_length is None else min_length
    pieces = split_text(doc.text, max_chars=max_chars, overlap=overlap)

    # 位置反查：locator 按 start 升序且互不重叠，二分即可找到块起点所在的片。
    # 块跨片时不拆分，以「起点所在片」为准 —— 引用定位只需要给出一个可读的出处。
    locators = doc.locators or []
    loc_starts = [loc.start for loc in locators]

    def locate(pos: int) -> Locator | None:
        if not loc_starts:
            return None
        idx = bisect_right(loc_starts, pos) - 1
        return locators[max(idx, 0)]

    chunks: list[Chunk] = []
    dropped = 0
    for body, start in pieces:
        # 丢弃页眉页脚残渣：过短的块没有检索价值，反而稀释 top-k
        if len(body) < min_length:
            dropped += 1
            continue
        index = len(chunks)
        loc = locate(start)
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(doc.source, index, body),
                text=body,
                category=doc.category,
                category_name=config.CATEGORY_NAME.get(doc.category, doc.category),
                source=doc.source,
                title=doc.title,
                chunk_index=index,
                char_start=start,
                page=loc.page if loc else 0,
                heading_path=loc.heading_path if loc else "",
                para_index=loc.para_index if loc else 0,
            )
        )
    if dropped:
        print(f"[chunker] {doc.source}：丢弃 {dropped} 个过短碎块（<{min_length} 字）")
    return chunks


if __name__ == "__main__":
    # 自检：python -m rag.chunker
    demo = (
        "第一条 为规范未来社区物业服务，制定本标准。本标准适用于新建住宅项目。\n"
        "第二条 物业服务企业应当建立 24 小时值班制度，接到报修后 30 分钟内响应。\n"
        "第三条 交付前应当完成承接查验，查验内容包括共用部位、共用设施设备。"
    ) * 6
    for i, (body, start) in enumerate(split_text(demo, max_chars=200, overlap=40)):
        print(f"--- 块{i} 偏移{start} 长度{len(body)} ---")
        print(body[:60] + ("..." if len(body) > 60 else ""))
