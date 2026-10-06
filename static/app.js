/* ============================================================
   未来社区 · 智慧物业知识库 —— H5 前端逻辑
   职责：
     1. 侧栏工作台：知识库统计、分类筛选、历史会话（localStorage 持久化）
     2. 问答主区：SSE 流式问答、Markdown 渲染、来源引用与匹配度
     3. 移动端：侧栏抽屉开合
   ============================================================ */

"use strict";

// ---------------- 常量 ----------------
const RECOMMENDED = [
  "承接查验包括哪些内容？",
  "交付流程有哪些关键环节？",
  "报修响应的时限要求是什么？",
  "如何办理房屋交付手续？",
  "物业费收取的标准是什么？",
  "过往的成功案例有哪些？",
];

const STORAGE_KEY = "fc_kb_sessions_v1";   // 历史会话在 localStorage 中的键名
const MAX_SESSIONS = 30;                   // 最多保留的会话条数，超出丢弃最旧的
const TITLE_MAX = 14;                      // 会话标题取首条问题的前 N 个字

// ---------------- 状态 ----------------
let sessions = [];             // 全部历史会话（最新在前）
let activeId = null;           // 当前会话 id，null 表示「尚未落库的空会话」
let currentCategory = null;    // 当前选中的分类 id，null 表示全库
let categoriesData = null;     // 最近一次 /api/categories 的返回，切分类时复用
let sending = false;           // 是否正在等待回答（防止连点）
let viewerHitEl = null;        // 阅读器里当前被高亮的片段，供「回到命中处」使用
let viewerToken = 0;           // 阅读器的请求令牌：连点不同来源时只认最后一次的结果
const docCache = new Map();    // source -> /api/doc 返回体；同一文档重复点开不重复请求

// ---------------- DOM ----------------
const appEl = document.getElementById("app");
const backdrop = document.getElementById("backdrop");
const menuBtn = document.getElementById("menuBtn");
const chatEl = document.getElementById("chat");
const chatTitle = document.getElementById("chatTitle");
const chatSub = document.getElementById("chatSub");
const statusBadge = document.getElementById("statusBadge");
const modelInfo = document.getElementById("modelInfo");
const statTotal = document.getElementById("statTotal");
const distBar = document.getElementById("distBar");
const categoryNav = document.getElementById("categoryNav");
const historyList = document.getElementById("historyList");
const newSessionBtn = document.getElementById("newSessionBtn");
const recommendedEl = document.getElementById("recommended");
const emptyBanner = document.getElementById("emptyBanner");
const askForm = document.getElementById("askForm");
const inputEl = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
// 引用原文阅读器（浮层）
const viewerEl = document.getElementById("docViewer");
const viewerSheet = viewerEl.querySelector(".viewer__sheet");
const viewerTitleEl = document.getElementById("viewerTitle");
const viewerMetaEl = document.getElementById("viewerMeta");
const viewerBodyEl = document.getElementById("viewerBody");
const viewerHitInfoEl = document.getElementById("viewerHitInfo");
const viewerLocateBtn = document.getElementById("viewerLocate");
const viewerCloseBtn = document.getElementById("viewerClose");

// ================================================================
// 通用工具
// ================================================================

/** HTML 转义：所有动态文本进 innerHTML 前必须经过它，防 XSS。 */
function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/** 生成会话 id。输入：无；输出：时间戳 + 随机后缀的短字符串。 */
function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

/** 相对时间展示。输入：毫秒时间戳；输出：今天显示 HH:MM，其余显示 M/D。 */
function fmtTime(ts) {
  const d = new Date(ts || Date.now());
  const hm = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  if (d.toDateString() === new Date().toDateString()) return hm;
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

/** 滚动聊天区到底部。输入：无；输出：无。 */
function scrollBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

/** 解析一段 SSE 事件文本（以空行分隔）为 {event, data}。 */
function parseSSE(block) {
  let event = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (!dataLines.length) return null;
  let data;
  try {
    data = JSON.parse(dataLines.join("\n"));
  } catch {
    data = { text: dataLines.join("\n") };
  }
  return { event, data };
}

// ================================================================
// 极简 Markdown 渲染
// ================================================================
/**
 * 把模型回答渲染成 HTML。
 *
 * 只覆盖大模型实际会用到的语法：标题、有序/无序列表、表格、加粗、
 * 行内代码、围栏代码块、引用标注 [1]。不引第三方库。
 *
 * 安全约定：入参一律先 escapeHtml 再做语法替换，替换规则只生成白名单标签，
 * 不会把文本里的尖括号还原成可执行标签，因此可以安全地塞进 innerHTML。
 *
 * 输入：原始回答文本；输出：HTML 字符串。
 */
function renderMarkdown(text) {
  const lines = escapeHtml(text).split("\n");
  const out = [];
  let listTag = null;      // 当前打开的列表标签："ul" | "ol" | null
  let inCode = false;
  let codeBuf = [];

  const closeList = () => {
    if (listTag) { out.push(`</${listTag}>`); listTag = null; }
  };
  const openList = (tag) => {
    if (listTag !== tag) { closeList(); out.push(`<${tag}>`); listTag = tag; }
  };
  const flushCode = () => {
    out.push(`<pre class="md-code"><code>${codeBuf.join("\n")}</code></pre>`);
    codeBuf = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // 围栏代码块：``` 起止，内部原样保留
    if (/^\s*```/.test(line)) {
      if (inCode) { flushCode(); inCode = false; }
      else { closeList(); inCode = true; }
      continue;
    }
    if (inCode) { codeBuf.push(line); continue; }

    // 空行：结束当前列表
    if (!line.trim()) { closeList(); continue; }

    // 表格：表头行 + 分隔行（|---|:--:|）才成立，避免把普通含 | 的句子误判
    if (isTableRow(line) && isTableSep(lines[i + 1] || "")) {
      closeList();
      const head = splitRow(line);
      const body = [];
      i += 2;                                   // 跳过表头与分隔行
      while (i < lines.length && isTableRow(lines[i])) {
        body.push(splitRow(lines[i]));
        i++;
      }
      i--;                                      // 外层 for 还会 +1，这里补回来
      out.push(tableHtml(head, body));
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      closeList();
      const level = Math.min(heading[1].length + 2, 6);   // # → h3，不抢页面标题层级
      out.push(`<h${level} class="md-h">${mdInline(heading[2])}</h${level}>`);
      continue;
    }

    const ul = line.match(/^\s*[-*+]\s+(.+)$/);
    if (ul) { openList("ul"); out.push(`<li>${mdInline(ul[1])}</li>`); continue; }

    const ol = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (ol) { openList("ol"); out.push(`<li>${mdInline(ol[1])}</li>`); continue; }

    closeList();
    out.push(`<p>${mdInline(line)}</p>`);
  }

  if (inCode) flushCode();   // 流式输出时代码块可能尚未闭合，兜底输出已到达的部分
  closeList();
  return out.join("");
}

/** 行内语法：行内代码、加粗、引用标注。输入：已转义的整行；输出：含标签的 HTML。 */
function mdInline(s) {
  return s
    .replace(/`([^`]+)`/g, '<code class="md-inline">$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d+)\]/g, '<span class="cite" data-ref="$1">[$1]</span>');
}

/** 判断是否是表格数据行。输入：一行文本；输出：布尔。 */
function isTableRow(line) {
  return /^\s*\|.*\|\s*$/.test(line);
}

/** 判断是否是表格分隔行（如 |---|---|）。输入：一行文本；输出：布尔。 */
function isTableSep(line) {
  return /^\s*\|[\s:|-]+\|\s*$/.test(line) && line.includes("-");
}

/** 拆一行为单元格数组。输入：表格行；输出：去空白后的单元格文本数组。 */
function splitRow(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

/** 生成表格 HTML。输入：表头单元格、数据行；输出：可横向滚动的表格 HTML。 */
function tableHtml(head, body) {
  const th = head.map((c) => `<th>${mdInline(c)}</th>`).join("");
  const rows = body
    .map((r) => `<tr>${r.map((c) => `<td>${mdInline(c)}</td>`).join("")}</tr>`)
    .join("");
  return `<div class="md-table-wrap"><table class="md-table">`
    + `<thead><tr>${th}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

// ================================================================
// 会话持久化（localStorage）
// ================================================================
/** 判断一条本地数据是否是形状合法的会话。输入：任意值；输出：布尔。 */
function isValidSession(s) {
  return !!s && typeof s.id === "string" && Array.isArray(s.messages);
}

/** 从 localStorage 读取历史会话。输入：无；输出：无（写入 sessions）。 */
function loadSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    sessions = Array.isArray(parsed) ? parsed.filter(isValidSession) : [];
  } catch (err) {
    console.warn("[history] 本地会话读取失败，按空历史处理：", err);
    sessions = [];
  }
}

/**
 * 把会话写回 localStorage。
 * 来源片段文本较长，多次问答后可能撞上浏览器 5MB 配额，
 * 因此失败时裁掉一半最旧会话再重试一次，仍失败则放弃本次持久化（不影响当前使用）。
 */
function saveSessions() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch (err) {
    console.warn("[history] 会话写入失败，裁剪后重试：", err);
    sessions = sessions.slice(0, Math.max(1, Math.floor(sessions.length / 2)));
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    } catch (err2) {
      console.warn("[history] 二次写入仍失败，本次不做持久化：", err2);
    }
  }
}

/** 取当前会话；没有则新建一个空会话并置于列表首位。输入：无；输出：会话对象。 */
function getOrCreateSession() {
  let s = sessions.find((x) => x.id === activeId);
  if (!s) {
    s = {
      id: uid(),
      title: "新会话",
      category: null,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
    };
    sessions.unshift(s);
    activeId = s.id;
    if (sessions.length > MAX_SESSIONS) sessions.length = MAX_SESSIONS;   // 丢弃最旧的
  }
  return s;
}

/** 让会话冒泡到列表最前（最近活跃）。输入：会话对象；输出：无。 */
function promoteSession(session) {
  const idx = sessions.indexOf(session);
  if (idx > 0) {
    sessions.splice(idx, 1);
    sessions.unshift(session);
  }
}

// ================================================================
// 渲染：侧栏
// ================================================================
/**
 * 健康状态徽标。
 * 输入：/api/health 返回体；输出：无。
 * retrieval_only：能检索但没配对话模型密钥，问答降级为返回检索原文，不能谎报「在线」。
 */
function renderStatus(health) {
  if (health.count === 0) {
    statusBadge.className = "badge badge--empty";
    statusBadge.textContent = "知识库为空";
    emptyBanner.classList.remove("banner--hidden");
  } else if (health.mock_llm || health.mock_embed) {
    statusBadge.className = "badge badge--mock";
    statusBadge.textContent = "离线 mock 模式";
  } else if (health.retrieval_only) {
    statusBadge.className = "badge badge--mock";
    statusBadge.textContent = "仅检索模式";
    statusBadge.title = "未配置 LLM_API_KEY，回答为检索到的原文片段";
  } else {
    statusBadge.className = "badge badge--online";
    statusBadge.textContent = "在线模式";
  }
  // mock 模式下模型名是占位值，显示出来反而误导，改为明确标注
  modelInfo.textContent = health.mock_llm ? "mock 作答" : (health.llm_model || "");
  modelInfo.title = modelInfo.textContent;
}

/**
 * 知识库统计卡：总片段数 + 五类分布条。
 * 输入：/api/categories 返回体；输出：无。
 * 注意 total 是「向量片段数」而非文档篇数 —— ChromaDB 按块存储，一个文档会切出多块。
 */
function renderStats(data) {
  statTotal.textContent = data.total;
  const max = Math.max(1, ...data.categories.map((c) => c.count));   // 以最多的一类为满格
  distBar.innerHTML = data.categories.map((c) => `
    <div class="dist__row">
      <span class="dist__name">${escapeHtml(c.name)}</span>
      <span class="dist__bar"><i style="width:${((c.count / max) * 100).toFixed(1)}%"></i></span>
      <span class="dist__num">${c.count}</span>
    </div>`).join("");
}

/**
 * 分类筛选树。传 null 表示「沿用上次的数据，只刷新选中态」。
 * 输入：/api/categories 返回体或 null；输出：无。
 */
function renderCategories(data) {
  if (data) categoriesData = data;
  if (!categoriesData) return;

  const items = [{ id: null, name: "全部", count: categoriesData.total }]
    .concat(categoriesData.categories);

  categoryNav.innerHTML = items.map((c) => `
    <button class="cat-item ${c.id === currentCategory ? "cat-item--active" : ""}"
            type="button" data-category="${c.id === null ? "" : escapeHtml(c.id)}">
      <span class="cat-item__name">${escapeHtml(c.name)}</span>
      <span class="cat-item__count">${c.count}</span>
    </button>`).join("");
}

/** 刷新顶栏副标题，让它跟当前分类一致。输入：无；输出：无。 */
function refreshTopbarSub() {
  const cat = categoriesData
    ? categoriesData.categories.find((c) => c.id === currentCategory)
    : null;
  chatTitle.textContent = "智能问答";
  chatSub.textContent = cat ? `筛选：${cat.name}` : "全库检索";
}

/** 渲染历史会话列表。输入：无；输出：无。 */
function renderHistory() {
  if (!sessions.length) {
    historyList.innerHTML = '<li class="history__empty">还没有会话，问一个问题即开始记录。</li>';
    return;
  }
  historyList.innerHTML = sessions.map((s) => `
    <li class="history__item ${s.id === activeId ? "history__item--active" : ""}"
        data-id="${escapeHtml(s.id)}">
      <button class="history__btn" type="button">
        <span class="history__title">${escapeHtml(s.title || "新会话")}</span>
        <span class="history__meta">${s.messages.length} 条 · ${fmtTime(s.updatedAt)}</span>
      </button>
      <button class="history__del" type="button" title="删除会话" aria-label="删除会话">✕</button>
    </li>`).join("");
}

// ================================================================
// 渲染：消息气泡
// ================================================================
/**
 * 建一个空消息骨架（不含内容）。
 * 拆成「骨架 + 填充」两步的原因：流式回答需要先把容器挂进 DOM 再逐字填；
 * 而切换历史会话时又要从头渲染同一条消息，两者复用同一套 DOM 结构。
 *
 * 输入："user" | "ai"；输出：{wrap, bubble, answerEl?, sourcesEl?}。
 */
function createBubble(role) {
  const wrap = document.createElement("div");
  wrap.className = `msg msg--${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  wrap.appendChild(bubble);

  if (role === "user") return { wrap, bubble };

  const answerEl = document.createElement("div");
  answerEl.className = "answer-text";
  const sourcesEl = document.createElement("div");
  sourcesEl.className = "sources";
  bubble.appendChild(answerEl);
  bubble.appendChild(sourcesEl);
  return { wrap, bubble, answerEl, sourcesEl };
}

/**
 * 把来源的位置信息拼成一句可读的「定位串」。
 *
 * 输入：来源对象（含 page / heading_path / para_index）；
 * 输出：如「第 12 页 · 3 承接查验 > 3.2 查验内容 · 第 4 段」；三者皆无时返回「该格式无位置信息」。
 * 旧会话（本次变更前存储的）缺这三个字段，也会落到「无位置信息」，不会显示成「第 0 页」。
 */
function formatLocator(s) {
  const parts = [];
  if (s.page > 0) parts.push(`第 ${s.page} 页`);
  if (s.heading_path) parts.push(s.heading_path);
  if (s.para_index > 0) parts.push(`第 ${s.para_index} 段`);
  return parts.length ? parts.join(" · ") : "该格式无位置信息";
}

/**
 * 填充来源引用列表（折叠态 = 标题 + 匹配度 + 展开提示；展开态 = 定位串 + 命中全文）。
 *
 * 输入：来源容器、来源数组；输出：无。
 * 所有动态文本一律经 escapeHtml，防止文档标题/正文里的尖括号注入。
 *
 * 定位信息（source / chunk_id / page …）挂在卡片根节点上，两个「原文」入口共用：
 *   折叠态元信息行的「原文 ›」按钮 —— 一眼可见，一步打开，手机上尤其重要；
 *   展开态的定位串本身 —— 已经展开看清了「第 12 页 · 3.2」，就地能跳。
 * 挂在根节点而不是各按钮上，是为了不把同一组属性写两遍（写两遍迟早漏改一处）。
 */
function fillSources(container, sources) {
  if (!sources || !sources.length) {
    container.innerHTML = "";
    return;
  }
  container.innerHTML = `
    <div class="sources__head">引用来源（${sources.length}）</div>
    ${sources.map((s) => {
      const loc = formatLocator(s);
      // 复制的内容带上文件路径，单独一条定位串脱离上下文后无法定位
      const copyText = s.source ? `${s.source} · ${loc}` : loc;
      return `
      <div class="source-item" data-ref="${s.index}"
           data-source="${escapeHtml(s.source || "")}"
           data-title="${escapeHtml(s.title || "")}"
           data-chunk-id="${escapeHtml(s.chunk_id || "")}"
           data-chunk-index="${s.chunk_index != null ? s.chunk_index : ""}"
           data-page="${s.page || 0}"
           data-heading="${escapeHtml(s.heading_path || "")}"
           data-para="${s.para_index || 0}">
        <div class="source-item__row">
          <span class="source-item__title">[${s.index}] ${escapeHtml(s.title || s.source || "未命名")}</span>
          <span class="source-item__score">匹配 ${Math.round((s.score || 0) * 100)}%</span>
        </div>
        <div class="source-item__meta">
          <span class="source-item__metatext">${escapeHtml(s.category || "未分类")} · ${escapeHtml(s.source || "")}</span>
          <button class="source-item__opendoc source-item__opendoc--mini" type="button"
                  title="查看该片段在文档中的原文">原文 ›</button>
          <span class="source-item__hint">片段 ›</span>
        </div>
        <div class="source-item__panel">
          <div class="source-item__loc">
            <button class="source-item__loctext source-item__opendoc" type="button"
                    title="打开文档并定位到该片段">${escapeHtml(loc)}</button>
            <button class="source-item__copy" type="button" data-loc="${escapeHtml(copyText)}">复制路径</button>
          </div>
          <div class="source-item__full">${escapeHtml(s.text || s.snippet || "")}</div>
        </div>
      </div>`;
    }).join("")}`;
}

/** 把一条已存储的消息渲染成 DOM。输入：{role,text,sources,error}；输出：元素。 */
function renderMessage(msg) {
  if (msg.role === "user") {
    const b = createBubble("user");
    b.bubble.textContent = msg.text;     // textContent 天然转义
    return b.wrap;
  }
  const b = createBubble("ai");
  fillSources(b.sourcesEl, msg.sources);
  b.answerEl.innerHTML = msg.error
    ? "⚠️ " + escapeHtml(msg.text)
    : renderMarkdown(msg.text || "");
  return b.wrap;
}

/** 渲染推荐问题（只在启动时构建一次，之后靠显隐切换）。输入：无；输出：无。 */
function renderRecommended() {
  recommendedEl.innerHTML = `
    <div class="recommended__title">可以这样问我：</div>
    <div class="recommended__grid">
      ${RECOMMENDED.map((q) => `<button class="suggestion" type="button">${escapeHtml(q)}</button>`).join("")}
    </div>`;
  recommendedEl.querySelectorAll(".suggestion").forEach((btn) => {
    btn.addEventListener("click", () => sendQuestion(btn.textContent));
  });
}

/** 渲染整个会话的记录。输入：会话对象或 null；输出：无。 */
function renderSession(session) {
  chatEl.querySelectorAll(".msg").forEach((el) => el.remove());
  const msgs = session ? session.messages : [];
  if (!msgs.length) {
    recommendedEl.classList.remove("recommended--hidden");
    return;
  }
  recommendedEl.classList.add("recommended--hidden");
  msgs.forEach((m) => chatEl.appendChild(renderMessage(m)));
  scrollBottom();
}

// ================================================================
// 引用原文阅读器（浮层）
// ================================================================
/**
 * 打开阅读器并定位到指定片段。
 *
 * 输入：定位信息 {source, title, chunkId, chunkIndex, page, headingPath, paraIndex}；
 *       只有 source 必需，其余用于在文档里定位高亮；
 * 输出：Promise，无返回值（失败信息就地渲染在浮层里，不弹窗打断）。
 *
 * 为什么要令牌 viewerToken：连点两条不同来源会有两个请求同时在飞，先发的未必先回；
 * 没有令牌时，后到的旧响应会把新打开的文档覆盖掉，界面显示的就是错的文档。
 */
async function openDocViewer(ref) {
  const source = (ref && ref.source) || "";
  if (!source) return;

  const token = ++viewerToken;
  viewerHitEl = null;
  viewerEl.classList.remove("viewer--hidden");
  document.body.classList.add("viewer-open");     // 锁住底层页面滚动
  viewerTitleEl.textContent = ref.title || source;
  viewerMetaEl.textContent = "读取中…";
  viewerBodyEl.innerHTML = '<p class="viewer__note">正在读取文档…</p>';
  viewerHitInfoEl.textContent = "";
  viewerLocateBtn.disabled = true;

  let data = docCache.get(source);
  if (!data) {
    try {
      const resp = await fetch(`/api/doc?source=${encodeURIComponent(source)}`, { cache: "no-store" });
      if (!resp.ok) {
        let message = "文档读取失败";
        try {
          message = (await resp.json()).detail || message;
        } catch { /* 非 JSON 错误体，用默认文案 */ }
        throw new Error(message);
      }
      data = await resp.json();
      docCache.set(source, data);
    } catch (err) {
      if (token !== viewerToken) return;            // 已被更晚的打开动作取代，丢弃本次结果
      viewerMetaEl.textContent = "";
      viewerBodyEl.innerHTML = `<p class="viewer__error">⚠️ ${escapeHtml(err.message || "文档读取失败")}</p>`;
      return;
    }
  }
  if (token !== viewerToken) return;                // 读取期间用户已关闭浮层或另开了一篇

  viewerMetaEl.textContent = docMetaText(data);
  renderDocBody(data);

  const hit = locateChunk(ref);
  viewerHitEl = hit;
  viewerLocateBtn.disabled = !hit;
  viewerHitInfoEl.textContent = hit
    ? "定位：" + formatLocator({
        page: Number(ref.page) || 0,
        heading_path: ref.headingPath || "",
        para_index: Number(ref.paraIndex) || 0,
      })
    : "未在库内片段中定位到该处，已显示文档开头";
}

/** 组装浮层副标题。输入：/api/doc 返回体；输出：一行描述文本。 */
function docMetaText(data) {
  const parts = [data.category_name || "未分类", `共 ${data.chunk_count} 个片段`];
  // 脱敏是加分项，为 0 时不该占位；命中过才值得说一句
  if (data.mask_hits > 0) parts.push(`入库时脱敏 ${data.mask_hits} 处`);
  return parts.join(" · ");
}

/**
 * 渲染阅读视图正文。输入：/api/doc 返回体；输出：无（写进 viewerBodyEl）。
 *
 * 分组规则：PDF 按「页」，其余格式按「章节路径」。命中块与它的分组标题由 locateChunk 标记。
 */
function renderDocBody(data) {
  const isPdf = (data.ext || "") === ".pdf";
  // 明示「这不是原始版式」：去重会删块、脱敏会改写，用户看到的本就不等于原件
  const parts = ['<p class="viewer__note">以下按入库片段拼成（入库时已脱敏、已去重），非原始文件版式。</p>'];
  let prevGroup = null;
  let prevIndex = null;

  for (const c of data.chunks) {
    // 去重是在库里整块删掉的，chunk_index 因此出现跳空；不提示会被误读成文档缺页
    if (prevIndex !== null && c.chunk_index > prevIndex + 1) {
      parts.push(`<p class="doc-gap">（此处 ${c.chunk_index - prevIndex - 1} 个片段与库内其他内容重复，去重未入库）</p>`);
    }
    prevIndex = c.chunk_index;

    const group = isPdf
      ? (c.page > 0 ? `第 ${c.page} 页` : "（无页码）")
      : (c.heading_path || "（未标注章节）");
    if (group !== prevGroup) {
      // title 存完整路径：章节路径可能嵌套 6 层，标签最多显示两行，截断后靠悬浮看全
      parts.push(`<div class="doc-group__label" title="${escapeHtml(group)}">${escapeHtml(group)}</div>`);
      prevGroup = group;
    }

    parts.push(
      `<p class="doc-block" data-chunk-id="${escapeHtml(c.chunk_id || "")}"`
      + ` data-chunk-index="${c.chunk_index}" data-page="${c.page || 0}"`
      + ` data-heading="${escapeHtml(c.heading_path || "")}" data-para="${c.para_index || 0}"`
      + `>${escapeHtml(c.text)}</p>`
    );
  }
  viewerBodyEl.innerHTML = parts.join("");
}

/**
 * 在已渲染的正文里标出命中片段并滚动到位。
 *
 * 输入：定位信息；输出：命中的片段元素，定位不到时返回 null。
 * 三级回退：块 id 最准 → 块序号 → 「页码 + 章节 + 段号」全等。
 * 历史会话（本次改动之前存下的来源）没有 chunk_id，只能落到第三级。
 */
function locateChunk(ref) {
  const blocks = Array.from(viewerBodyEl.querySelectorAll(".doc-block"));

  let target = ref.chunkId ? blocks.find((b) => b.dataset.chunkId === ref.chunkId) : null;
  if (!target && ref.chunkIndex !== "" && ref.chunkIndex != null) {
    target = blocks.find((b) => Number(b.dataset.chunkIndex) === Number(ref.chunkIndex));
  }
  if (!target) {
    target = blocks.find((b) =>
      Number(b.dataset.page) === (Number(ref.page) || 0)
      && (b.dataset.heading || "") === (ref.headingPath || "")
      && Number(b.dataset.para) === (Number(ref.paraIndex) || 0));
  }
  if (!target) return null;

  target.classList.add("doc-block--hit");
  // 命中块的所属分组标题也标上：滚过去才知道自己停在第几页 / 哪一章
  const label = previousLabel(target);
  if (label) label.classList.add("doc-group__label--hit");

  // 用 rAF 等一帧再滚：刚写完 innerHTML，滚动容器高度尚未结算，
  // 立刻 scrollIntoView 在移动端会滚不到位
  requestAnimationFrame(() => scrollToBlock(target));
  return target;
}

/**
 * 把某个片段滚到容器中线。
 *
 * 输入：片段元素；输出：无。
 * 近距离平滑滚动，远距离直接跳 —— 大文档实测正文高 20 万像素（483 个片段），
 * 平滑滚动要飞约 1.5 秒、中途掠过几百个片段，看着晕还没到；两屏以内才值得平滑。
 * 距离用 rect 差值算而不是 offsetTop：后者的参照系是 offsetParent，与滚动容器不是同一套坐标。
 */
function scrollToBlock(el) {
  const distance = Math.abs(
    el.getBoundingClientRect().top - viewerBodyEl.getBoundingClientRect().top - viewerBodyEl.clientHeight / 2
  );
  el.scrollIntoView({
    behavior: distance > viewerBodyEl.clientHeight * 2 ? "auto" : "smooth",
    block: "center",
  });
}

/** 向上找最近的兄弟分组标题。输入：任意元素；输出：标题元素或 null。 */
function previousLabel(el) {
  for (let node = el.previousElementSibling; node; node = node.previousElementSibling) {
    if (node.classList.contains("doc-group__label")) return node;
  }
  return null;
}

/** 关闭阅读器。输入：无；输出：无。令牌自增让在途请求的结果作废。 */
function closeDocViewer() {
  viewerToken++;
  viewerEl.classList.add("viewer--hidden");
  document.body.classList.remove("viewer-open");
  viewerHitEl = null;
}

// ---------------- 阅读器事件 ----------------
viewerCloseBtn.addEventListener("click", closeDocViewer);
// 只在点到弹层自身（sheet 之外的半透明区域）时关闭；
// 用 target === viewerEl 而非 !sheet.contains(target)，避免拖选文字松手在框外时误关
viewerEl.addEventListener("click", (e) => {
  if (e.target === viewerEl) closeDocViewer();
});
/** 「回到命中处」：翻远之后一键滚回命中片段，并重放一次高亮动画。 */
viewerLocateBtn.addEventListener("click", () => {
  if (!viewerHitEl) return;
  viewerHitEl.classList.remove("doc-block--hit");
  void viewerHitEl.offsetWidth;                     // 强制回流，让动画可重复触发
  viewerHitEl.classList.add("doc-block--hit");
  scrollToBlock(viewerHitEl);
});

// ================================================================
// 会话操作
// ================================================================
/** 新建会话：只重置状态，空会话不落库，避免历史里堆一堆空条目。输入：无；输出：无。 */
function newSession() {
  activeId = null;
  currentCategory = null;
  renderCategories(null);
  refreshTopbarSub();
  renderSession(null);
  renderHistory();
  closeDrawer();
  inputEl.focus();
}

/** 打开某个历史会话。输入：会话 id；输出：无。 */
function openSession(id) {
  const s = sessions.find((x) => x.id === id);
  if (!s) return;
  activeId = id;
  currentCategory = s.category || null;   // 恢复该会话建库时用的分类筛选
  renderCategories(null);
  refreshTopbarSub();
  renderSession(s);
  renderHistory();
  closeDrawer();
}

/** 删除某个历史会话；若删的是当前会话则回到空态。输入：会话 id；输出：无。 */
function deleteSession(id) {
  const idx = sessions.findIndex((x) => x.id === id);
  if (idx < 0) return;
  sessions.splice(idx, 1);
  saveSessions();

  if (activeId === id) {
    activeId = null;
    currentCategory = null;
    renderCategories(null);
    refreshTopbarSub();
    renderSession(null);
  }
  renderHistory();
}

/** 追加一条消息到指定会话并落库。输入：会话对象、消息对象；输出：无。 */
function pushMessage(session, msg) {
  session.messages.push(msg);
  session.updatedAt = Date.now();
  promoteSession(session);
  saveSessions();
  renderHistory();
}

// ================================================================
// 核心：SSE 流式问答
// ================================================================
/**
 * 发起一次问答并把流式结果画到界面上。
 *
 * 输入：问题、分类过滤、所属会话（在发送时就固定下来 —— 流式过程中用户可能
 *       切到别的会话，若结束时才去找「当前会话」，答案会被写错地方）；
 * 输出：Promise，异常时抛出（由调用方兜底渲染错误气泡）。
 */
async function ask(question, category, session) {
  const resp = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, category, top_k: null }),
  });

  if (!resp.ok) {
    let message = "请求失败，请稍后重试";
    try {
      const j = await resp.json();
      message = j.detail || message;
    } catch { /* 忽略非 JSON 错误体 */ }
    throw new Error(message);
  }

  const el = createBubble("ai");
  el.answerEl.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';
  chatEl.appendChild(el.wrap);
  scrollBottom();

  let answerRaw = "";
  let sources = [];
  let errorMsg = null;

  // 事件分发：sources 先渲染引用骨架，delta 逐步填充正文
  const handle = (event, data) => {
    if (event === "sources") {
      sources = data.sources || [];
      fillSources(el.sourcesEl, sources);
    } else if (event === "delta") {
      answerRaw += data.text || "";
      el.answerEl.innerHTML = renderMarkdown(answerRaw) + '<span class="cursor"></span>';
      scrollBottom();
    } else if (event === "error") {
      errorMsg = data.message || "服务出错了";
      el.answerEl.innerHTML = "⚠️ " + escapeHtml(errorMsg);
    }
    // done 事件无需额外处理，收尾在循环外
  };

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const ev = parseSSE(block);
      if (ev) handle(ev.event, ev.data);
    }
  }

  if (!errorMsg) {
    el.answerEl.innerHTML = renderMarkdown(answerRaw) || "（无回答内容）";
    scrollBottom();
  }

  pushMessage(session, {
    role: "ai",
    text: errorMsg || answerRaw || "（无回答内容）",
    sources,
    error: !!errorMsg,
  });
}

// ================================================================
// 交互入口
// ================================================================
/** 发送一条问题。输入：问题文本；输出：Promise。 */
async function sendQuestion(text) {
  const question = (text || "").trim();
  if (!question || sending) return;

  const session = getOrCreateSession();
  // 首条用户消息决定会话标题与所属分类
  if (!session.messages.length) {
    session.title = question.slice(0, TITLE_MAX) + (question.length > TITLE_MAX ? "…" : "");
    session.category = currentCategory;
  }

  recommendedEl.classList.add("recommended--hidden");
  const userEl = createBubble("user");
  userEl.bubble.textContent = question;
  chatEl.appendChild(userEl.wrap);

  session.messages.push({ role: "user", text: question });
  session.updatedAt = Date.now();
  promoteSession(session);
  saveSessions();
  renderHistory();
  scrollBottom();

  inputEl.value = "";
  sending = true;
  sendBtn.disabled = true;

  try {
    await ask(question, currentCategory, session);
  } catch (err) {
    const el = createBubble("ai");
    el.answerEl.innerHTML = "⚠️ " + escapeHtml(err.message || "网络异常，请重试");
    chatEl.appendChild(el.wrap);
    pushMessage(session, {
      role: "ai",
      text: err.message || "网络异常，请重试",
      sources: [],
      error: true,
    });
    scrollBottom();
  } finally {
    sending = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

// ---------------- 抽屉（移动端） ----------------
/** 打开侧栏抽屉。输入：无；输出：无。 */
function openDrawer() { appEl.classList.add("app--sidebar-open"); }

/** 关闭侧栏抽屉。输入：无；输出：无。 */
function closeDrawer() { appEl.classList.remove("app--sidebar-open"); }

menuBtn.addEventListener("click", openDrawer);
backdrop.addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  // 阅读器浮层盖在抽屉之上，Esc 先关最上面那一层
  if (!viewerEl.classList.contains("viewer--hidden")) closeDocViewer();
  else closeDrawer();
});

// ---------------- 分类点击 ----------------
categoryNav.addEventListener("click", (e) => {
  const btn = e.target.closest(".cat-item");
  if (!btn) return;
  currentCategory = btn.dataset.category || null;
  renderCategories(null);
  refreshTopbarSub();
  closeDrawer();
});

// ---------------- 历史会话点击 / 删除 ----------------
historyList.addEventListener("click", (e) => {
  const item = e.target.closest(".history__item");
  if (!item) return;
  if (e.target.closest(".history__del")) {
    deleteSession(item.dataset.id);
    return;
  }
  if (e.target.closest(".history__btn")) openSession(item.dataset.id);
});

newSessionBtn.addEventListener("click", newSession);

// ---------------- 发送表单 ----------------
askForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendQuestion(inputEl.value);
});

// ---------------- 引用标注点击 / 来源展开 / 复制定位 / 查看原文 ----------------
/** 复制成功后就地反馈。输入：按钮元素；输出：无。 */
function flashCopied(btn) {
  const original = btn.textContent;
  btn.textContent = "已复制";
  btn.classList.add("source-item__copy--done");
  setTimeout(() => {
    btn.textContent = original;
    btn.classList.remove("source-item__copy--done");
  }, 1200);
}

chatEl.addEventListener("click", (e) => {
  // 复制按钮不在 .source-item__row 内，先把分支拦下来，避免落到下面的折叠切换
  const copyBtn = e.target.closest(".source-item__copy");
  if (copyBtn) {
    const text = copyBtn.dataset.loc || "";
    // clipboard 在非 HTTPS 下可能不存在，带可选链降级：取不到就静默不复制，不报错
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(() => flashCopied(copyBtn)).catch(() => {});
    }
    return;
  }

  // 「原文」入口（折叠态的迷你按钮 / 展开态的定位串）→ 打开阅读器。
  // 两者也都在 .source-item__row 之外，同样要在折叠切换之前拦截
  const openBtn = e.target.closest(".source-item__opendoc");
  if (openBtn) {
    const card = openBtn.closest(".source-item");
    if (card) {
      openDocViewer({
        source: card.dataset.source || "",
        title: card.dataset.title || "",
        chunkId: card.dataset.chunkId || "",
        chunkIndex: card.dataset.chunkIndex,
        page: card.dataset.page,
        headingPath: card.dataset.heading || "",
        paraIndex: card.dataset.para,
      });
    }
    return;
  }

  const cite = e.target.closest(".cite");
  if (cite) {
    // 只在当前这条回答内找来源，避免多条回答里都有 [1] 时跳错消息
    const msg = cite.closest(".msg");
    const target = msg && msg.querySelector(`.source-item[data-ref="${cite.dataset.ref}"]`);
    if (target) {
      target.classList.add("source-item--open");
      target.classList.remove("source-item--flash");
      void target.offsetWidth;                 // 强制回流，让动画可重复触发
      target.classList.add("source-item--flash");
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    return;
  }
  const row = e.target.closest(".source-item__row");
  if (row) row.parentElement.classList.toggle("source-item--open");
});

// ================================================================
// 初始化
// ================================================================
/** 拉取 JSON，非 2xx 直接抛错，便于上层统一重试。 */
async function fetchJSON(url) {
  const resp = await fetch(url, { cache: "no-store" });
  if (!resp.ok) throw new Error(`${url} → HTTP ${resp.status}`);
  return resp.json();
}

/**
 * 初始化：恢复历史会话，拉取健康状态与分类列表。
 *
 * 带递增退避重试 —— 后端还没起来 / 刚重启时，用户刷新页面很容易撞上瞬时失败，
 * 只试一次会让徽标永久停在「无法连接后端」，误导成后端挂了。
 */
async function init(retries = 5) {
  loadSessions();
  renderHistory();
  renderRecommended();
  refreshTopbarSub();

  let lastError = null;

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const [health, cat] = await Promise.all([
        fetchJSON("/api/health"),
        fetchJSON("/api/categories"),
      ]);
      renderStatus(health);
      renderStats(cat);
      renderCategories(cat);
      refreshTopbarSub();
      return;
    } catch (err) {
      lastError = err;
      console.warn(`[init] 第 ${attempt}/${retries} 次拉取后端状态失败：`, err);
      if (attempt < retries) {
        await new Promise((r) => setTimeout(r, attempt * 600));   // 600/1200/1800/2400ms
      }
    }
  }

  // 全部重试都失败：把真实原因留在界面上，而不是一句无从下手的「无法连接后端」
  statusBadge.className = "badge badge--empty";
  statusBadge.textContent = "无法连接后端";
  statusBadge.title =
    `已重试 ${retries} 次仍失败：${lastError && lastError.message}\n` +
    `当前页面来源：${location.origin}\n` +
    `请确认后端已启动（python api.py）。若用 localhost 打不开，改试 http://127.0.0.1:8000 —— ` +
    `Windows 下 localhost 可能解析到 IPv6 ::1，而服务只监听了 IPv4。`;
  console.error("[init] 无法连接后端，最后错误：", lastError);
}

init();
