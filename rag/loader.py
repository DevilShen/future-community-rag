"""
文档解析模块
============

职责：把 docx / pdf / md / txt 四种格式统一解析成纯文本，并补上「分类 / 来源 / 标题」
      三项元数据，供下游 chunker 和 screener 使用。
      同时产出「位置锚点」（Locator），让检索命中的块能反查出它在原文的哪一页、哪一章、哪一段。

设计要点：
    1. 每个解析器只负责「取文本 + 标出片边界」，元数据组装统一在 load_file 里做，避免四份重复逻辑；
    2. 分类不靠人工标注，而是由文件所在目录反查 config.CATEGORY_BY_DIR 自动打标；
    3. txt 编码做多级回退，兼容 Windows 下常见的 GBK 中文文档；
    4. 位置锚点的前提是「逐片规整后的拼接」与「整串规整」逐字符一致 —— 见 _build_anchored。

对外接口：
    load_file(path)      -> LoadedDoc    解析单个文件
    load_dir(corpus_dir) -> list[LoadedDoc]  递归解析整个语料目录，自动跳过不支持的后缀
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import config

# txt 解码回退顺序：utf-8（含 BOM）→ gb18030（GBK/GB2312 的超集，覆盖老中文文档）
_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")

# 标题候选中，超过该长度的首行视为正文而非标题
_TITLE_MAX_LEN = 40


@dataclass
class Locator:
    """
    位置锚点：正文里「一片」文本所对应的原文位置。

    输入：由 _build_anchored 构造，调用方不直接 new。
    输出字段：
        start        : 片在 doc.text 中的起始偏移
        end          : 结束偏移（不含）
        page         : PDF 页码（从 1 起）；0 表示该格式无页码
        heading_path : docx 章节路径，如 "3 承接查验 > 3.2 查验内容"；空串表示无
        para_index   : docx 所属章节下的第 N 段（从 1 起，遇标题归零）；0 表示无
    """

    start: int
    end: int
    page: int = 0
    heading_path: str = ""
    para_index: int = 0


@dataclass
class LoadedDoc:
    """
    解析结果。

    输入：由 load_file 构造，调用方不直接 new。
    输出字段：
        text      : 解析后的纯文本正文（已规整换行）
        source    : 来源标识，语料目录内的相对路径，如 "01_行业标准/xxx.docx"
        title     : 文档标题，取 md 首个一级标题 / 首个短行 / 文件名
        category  : 分类 id，如 "industry_standard"；不在语料目录内则为 "uncategorized"
        ext       : 小写后缀，如 ".docx"
        char_count: 正文字符数，便于入库时输出统计
        locators  : 位置锚点列表（按 start 升序且互不重叠）；md/txt 等无语义位置的格式为空列表
    """

    text: str
    source: str
    title: str
    category: str = "uncategorized"
    ext: str = ""
    char_count: int = field(default=0)
    locators: list[Locator] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 统一在这里算，避免每个解析分支各写一遍
        self.char_count = len(self.text)

    @property
    def is_empty(self) -> bool:
        """正文是否为空（扫描版 PDF 抽不出文字时会出现）。"""
        return not self.text.strip()


# ------------------------------------------------------------------ 文本规整
def normalize_text(text: str) -> str:
    """
    规整换行与空白。

    输入：任意原始文本；输出：\\n 换行、无行尾空格、连续空行压成一个的文本。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ").replace("\xa0", " ")   # 全角空格 / 不换行空格
    lines = [line.rstrip() for line in text.split("\n")]
    # 连续 3 个以上换行压成 2 个（保留段落间隔，不保留大片空白）
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# ------------------------------------------------------------------ 位置锚点
def _normalize_piece(piece: str, is_first: bool) -> str:
    """
    单片规整化 —— 与 normalize_text 对「整串拼接」的效果逐字符等价。

    输入：原始片文本、是否为首片；输出：规整后的片文本。

    三条规则缺一不可（靠 _build_anchored 的断言兜底，改了要重跑模糊测试）：
        1. 首片 strip()：整串的 .strip() 只作用于全文首尾，故只有首片能去掉首部空白；
        2. 非首片只 lstrip("\\n")：整串规整**不会**动中间片首部的空格（那可能是段首缩进），
           只能吃掉片间叠加出来的换行；
        3. 所有片 rstrip()：否则「片尾换行 + 拼接符 \\n\\n」会叠成 4 个换行，
           而整串规整会把 3 个以上压成 2 个。
    """
    piece = piece.replace("\r\n", "\n").replace("\r", "\n")
    piece = piece.replace("\u3000", " ").replace("\xa0", " ")
    lines = [line.rstrip() for line in piece.split("\n")]
    piece = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return piece.strip() if is_first else piece.lstrip("\n").rstrip()


def _build_anchored(
    raws: list[str],
    metas: list[dict],
    source: str,
) -> tuple[str, list[Locator]]:
    """
    把「逐片文本 + 逐片位置」拼成完整正文，并给出每片的字符区间。

    输入：
        raws   : 逐片原始文本（PDF 一片一页 / docx 一片一段）
        metas  : 与 raws 一一对应的位置字典，键为 page / heading_path / para_index
        source : 来源标识，仅用于降级告警
    输出：(完整正文, Locator 列表)。

    为什么要这么绕：下游的 chunker 只知道块在正文里的字符偏移（char_start），
    要在不改变正文内容的前提下把这个偏移反查成「第几页第几段」，就必须保证
    「逐片规整后拼接」与「整串规整」逐字符一致 —— 偏移才成立。
    一旦两者不一致（例如 normalize_text 日后被改动），这里降级为「无位置信息」：
    text 本身仍然正确，问答不受影响，只是引用面板显示不出位置。
    """
    kept_raw: list[str] = []
    kept_meta: list[dict] = []
    for raw, meta in zip(raws, metas):
        # 先过滤空片，再决定谁是首片：否则「首片为空白被丢弃」时，
        # 真正的首片会按原索引被当成非首片，首部空白就多留了
        if raw and raw.strip():
            kept_raw.append(raw)
            kept_meta.append(meta or {})

    pieces: list[str] = []
    locators: list[Locator] = []
    cursor = 0
    for i, (raw, meta) in enumerate(zip(kept_raw, kept_meta)):
        piece = _normalize_piece(raw, is_first=(i == 0))
        if not piece:
            continue
        if pieces:
            cursor += 2                       # 片间以 "\n\n" 连接
        start = cursor
        cursor += len(piece)
        pieces.append(piece)
        locators.append(
            Locator(
                start=start,
                end=cursor,
                page=int(meta.get("page", 0)),
                heading_path=str(meta.get("heading_path", "") or ""),
                para_index=int(meta.get("para_index", 0)),
            )
        )

    text = "\n\n".join(pieces)

    if text != normalize_text("\n\n".join(kept_raw)):
        # 静默丢锚点会让整库丧失定位能力却无人察觉，必须打印
        print(f"[loader] 规整化不一致，降级为无位置信息：{source}")
        return text, []

    return text, locators


def _read_text_file(path: Path) -> str:
    """
    读取纯文本文件，编码多级回退。

    输入：文件路径；输出：解码后的文本（全部编码失败则用替换符兜底，不抛异常）。
    """
    raw = path.read_bytes()
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# ------------------------------------------------------------------ 各格式解析器
# 标题样式名：真实文档用英文 "Heading 1"~"Heading 9"，中文 Word 可能写成 "标题 1"。
# 不要依赖 outlineLvl —— 实测真实语料里它全为 None。
_HEAD_RE = re.compile(r"^(?:Heading|标题)\s*([1-9])$", re.I)


def _heading_level(style_name: str) -> int:
    """
    从段落样式名判断标题层级。

    输入：python-docx 的 style.name；输出：1~9 的层级，非标题返回 0。
    """
    match = _HEAD_RE.match((style_name or "").strip())
    return int(match.group(1)) if match else 0


def _parse_docx(path: Path) -> tuple[list[str], list[dict]]:
    """
    解析 .docx。输入：文件路径；输出：(逐片文本, 逐片位置)。

    覆盖段落与表格：交付流程类文档大量信息在表格里，只读段落会丢内容。
    按「文档流顺序」遍历而非「先所有段落、后所有表格」—— 后者会把表格全部堆到文末，
    表格内容因此脱离它真正所属的章节，既让章节归属错乱，也让引用位置无从谈起。
    表格按「一行一片」处理，便于把命中块定位到具体行。

    位置字段：heading_path 用栈维护的章节路径（如 "3 承接查验 > 3.2 查验内容"），
    para_index 是「所属章节下的第 N 段」（遇标题归零），比「文档第 127 段」可读得多。
    """
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    raws: list[str] = []
    metas: list[dict] = []
    stack: list[str] = []
    para = 0

    def push(text: str, para_index: int) -> None:
        raws.append(text)
        metas.append({"heading_path": " > ".join(stack), "para_index": para_index})

    for child in doc.element.body.iterchildren():
        tag = child.tag
        if tag.endswith("}p"):
            paragraph = Paragraph(child, doc)
            text = paragraph.text
            if not text.strip():
                continue
            level = _heading_level(paragraph.style.name if paragraph.style is not None else "")
            if level:
                # 同级或更深层级出现时，先把栈截到父级再压入自身
                stack = stack[: level - 1]
                stack.append(text.strip())
                para = 0
                push(text, 0)
            else:
                para += 1
                push(text, para)
        elif tag.endswith("}tbl"):
            para += 1
            for row in Table(child, doc).rows:
                cells = [cell.text.strip() for cell in row.cells]
                # 同一行单元格用 | 分隔，保留表格的行列语义
                if any(cells):
                    push(" | ".join(cells), para)

    return raws, metas


def _parse_pdf(path: Path) -> tuple[list[str], list[dict]]:
    """
    解析 .pdf。输入：文件路径；输出：(逐页文本, 逐页位置)。

    注：pypdf 只能抽取文本层，扫描件（图片型 PDF）会返回空，需另做 OCR，本项目不处理。
    页码必须取 enumerate 的**真实页号**，不能用「过滤掉空页后的序号」，
    否则一旦中间有空白页，后续所有页码都会前移。
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    raws: list[str] = []
    metas: list[dict] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            raws.append(text)
            metas.append({"page": page_no})

    return raws, metas


def _parse_markdown(path: Path) -> str:
    """
    解析 .md / .txt。输入：文件路径；输出：原文文本（Markdown 语法保留）。

    不做 Markdown → 纯文本转换：标题层级、列表符号对 LLM 理解结构有帮助，
    分块阶段按换行切分即可，去标记反而丢信息。
    """
    return _read_text_file(path)


# ------------------------------------------------------------------ 标题推断
def _guess_title(text: str, path: Path) -> str:
    """
    推断文档标题。

    输入：正文、文件路径；输出：标题字符串。
    策略：md 一级标题 > 首个二级标题 > 首个不超过 40 字的短行 > 文件名主干。
    """
    for line in text.split("\n")[:20]:        # 只看前 20 行，避免全文扫描
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    for line in text.split("\n")[:20]:
        stripped = line.strip()
        if stripped and len(stripped) <= _TITLE_MAX_LEN:
            return stripped
    return path.stem


# ------------------------------------------------------------------ 对外接口
def load_file(path: str | Path, corpus_dir: Path | None = None) -> LoadedDoc:
    """
    解析单个文档。

    输入：
        path       : 文件路径
        corpus_dir : 语料根目录，用于计算 source 相对路径与反查分类；缺省用 config.CORPUS_DIR
    输出：LoadedDoc；后缀不支持时抛 ValueError。
    """
    path = Path(path)
    corpus_dir = Path(corpus_dir) if corpus_dir else config.CORPUS_DIR
    ext = path.suffix.lower()

    if ext not in config.SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型：{ext}（支持 {sorted(config.SUPPORTED_EXTENSIONS)}）")

    # source 用相对语料根的路径，既唯一又能反推分类目录
    try:
        source = path.relative_to(corpus_dir).as_posix()
    except ValueError:
        source = path.name

    # docx / pdf 逐片解析并保留片位置；md / txt 无页与章节概念，退化为无位置信息
    if ext == ".docx":
        raws, metas = _parse_docx(path)
        text, locators = _build_anchored(raws, metas, source)
    elif ext == ".pdf":
        raws, metas = _parse_pdf(path)
        text, locators = _build_anchored(raws, metas, source)
    else:                                      # .md / .txt
        text, locators = normalize_text(_parse_markdown(path)), []

    # 分类取「相对语料根的第一层目录」：corpus/04_交付文档/子目录/x.docx → delivery_doc
    # 不能只看 path.parent.name —— 语料常按项目阶段再分子目录，只看直接父目录
    # 会把嵌套文件全判成未分类（实测 04 目录下 1260 块全部中招，分类筛选查不到）
    parts = Path(source).parts
    top_dir = parts[0] if len(parts) > 1 else ""      # 只有文件名说明不在语料目录内
    category = config.CATEGORY_BY_DIR.get(top_dir, "uncategorized")

    return LoadedDoc(
        text=text,
        source=source,
        title=_guess_title(text, path),
        category=category,
        ext=ext,
        locators=locators,
    )


def load_dir(corpus_dir: str | Path | None = None) -> list[LoadedDoc]:
    """
    递归解析语料目录下的全部受支持文档。

    输入：语料根目录，缺省用 config.CORPUS_DIR。
    输出：LoadedDoc 列表（按路径排序保证结果稳定）；单个文件解析失败会被跳过并打印告警，
          不中断整批入库。
    """
    corpus_dir = Path(corpus_dir) if corpus_dir else config.CORPUS_DIR
    if not corpus_dir.exists():
        raise FileNotFoundError(f"语料目录不存在：{corpus_dir}")

    docs: list[LoadedDoc] = []
    # 排序让每次运行的入库顺序一致，便于复现和对比
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            continue
        try:
            doc = load_file(path, corpus_dir)
        except Exception as exc:                       # noqa: BLE001 - 单文件失败不影响整批
            print(f"[loader] 跳过 {path.name}：{exc}")
            continue
        if doc.is_empty:
            print(f"[loader] 跳过 {path.name}：正文为空（可能是扫描版 PDF）")
            continue
        docs.append(doc)

    return docs


if __name__ == "__main__":
    # 自检：python -m rag.loader
    for d in load_dir():
        print(f"{d.category:18} {d.char_count:>7} 字  {d.title[:24]:24} <- {d.source}")
