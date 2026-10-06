"""生成评测脚本（Generation Evaluation）
======================================

读取 questions.jsonl，复用项目的 Retriever + LLMClient，量化「生成层」质量：

    1. Correctness（正确性）：答案是否命中 gold_facts 关键事实点（规则匹配）
    2. 负样本拒绝率：negative=true 的题，系统是否回答「未收录」而非编造
    3. 脱敏专项：检索文本里手机号是否已被打码（无原文泄露）

依赖：需要配置真实 LLM（LLM_PROVIDER 非 mock、LLM_API_KEY 非空）才有意义。
mock / 仅检索模式下答案非模型生成，脚本会打印提示但照常跑（结果仅作结构自检）。

用法（在项目 eval/ 目录下运行，自动定位项目根目录）：
    python generation_eval.py
    python generation_eval.py --project ..
    python generation_eval.py --list
"""

import argparse
import json
import re
import sys

# 强制 stdout 用 UTF-8 输出：避免 Windows 默认 GBK 控制台打印中文/特殊符号时报
# UnicodeEncodeError；errors="replace" 保证遇到生僻字符也不会中断。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pathlib import Path


def parse_args():
    """解析命令行参数。输入：无（读 sys.argv）；输出：argparse.Namespace。"""
    p = argparse.ArgumentParser(description="生成评测")
    p.add_argument("--questions", default=None, help="测评集 JSONL 路径，默认同目录 questions.jsonl")
    p.add_argument("--project", default=None, help="项目根目录，默认脚本所在目录的上一级")
    p.add_argument("--hybrid", action="store_true", help="用混合检索（BM25+向量+RRF）对比")
    p.add_argument("--list", action="store_true", help="只列出题目与分类，不调用模型/向量库")
    return p.parse_args()


def resolve_paths(args):
    """确定测评集路径与项目根目录（脚本放 eval/ 下，项目根 = 上一级）。"""
    script_dir = Path(__file__).resolve().parent
    questions = Path(args.questions) if args.questions else script_dir / "questions.jsonl"
    project = Path(args.project) if args.project else script_dir.parent
    return questions, project


def load_questions(path):
    """逐行读 JSONL 测评集。坏行只告警并跳过，不中断整批。"""
    items = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[警告] 第 {lineno} 行 JSON 解析失败，已跳过：{e}")
    return items


def classify(q):
    """把一道题归类，决定它走哪条评测路径（eval / negative / mask / no_gold）。"""
    if q.get("negative"):
        return "negative"
    if q.get("q_type") == "脱敏专项":
        return "mask"
    if not q.get("gold_sources"):
        return "no_gold"
    return "eval"


def _norm(s):
    """去掉所有空白字符，用于做「忽略空格/换行」的宽松子串匹配。"""
    return re.sub(r"\s+", "", s)


def _fact_hit(answer, fact):
    """判断一个关键事实点 fact 是否出现在答案 answer 里（忽略空白差异）。"""
    return _norm(fact) in _norm(answer)


def _eval_normal(q, retriever, llm):
    """正常题：检索 → 生成答案 → 逐条核对 gold_facts 是否被答案覆盖。"""
    hits = retriever.search(q["question"])
    answer = (llm.answer(q["question"], hits).text or "")
    facts = q.get("gold_facts") or []
    hit_facts = [f for f in facts if _fact_hit(answer, f)]
    return {
        "id": q["id"], "kind": "eval", "category": q.get("category", ""),
        "question": q["question"], "answer": answer,
        "total_facts": len(facts), "hit_facts": len(hit_facts),
        # 正确 = 全部 gold_facts 都命中（严格口径）
        "correct": len(facts) > 0 and len(hit_facts) == len(facts),
    }


def _eval_negative(q, retriever, llm):
    """负样本：检索 → 生成答案 → 判断系统是否「诚实拒绝」而非编造。

    判定用启发式：答案里出现「未收录 / 未提及」即视为正确拒绝。
    （更严格的判定可换 LLM-as-judge，属后续演进方向。）
    """
    hits = retriever.search(q["question"])
    answer = (llm.answer(q["question"], hits).text or "")
    rejected = ("未收录" in answer) or ("未提及" in answer)
    return {
        "id": q["id"], "kind": "negative", "category": q.get("category", ""),
        "question": q["question"], "answer": answer, "rejected": rejected,
    }


def _eval_mask(q, retriever, mask_token):
    """脱敏专项：直接读该文档入库后的全部片段，检查原文手机号是否泄漏。

    不从原始文件读（和项目 /api/doc 一致，只走向量库），检查两项：
        - raw_leaked：原文号码是否还残留在片段文本里（应为 False）
        - masked_seen：是否出现了打码占位符【已脱敏】（应为 True）
    """
    source = (q.get("gold_sources") or [None])[0]
    if not source:
        return {"id": q["id"], "kind": "mask", "question": q["question"], "status": "skip", "note": "无 gold_source"}

    chunks = retriever.store.get_by_source(source)   # 取该文档入库后的全部片段
    texts = " ".join(c.text for c in chunks)
    m = re.search(r"1[3-9]\d{9}", q.get("question", ""))  # 从题目里抠出那个假手机号
    raw = m.group(0) if m else "13800138000"

    return {
        "id": q["id"], "kind": "mask", "question": q["question"], "status": "ok", "raw": raw,
        "raw_leaked": raw in texts,
        "masked_seen": mask_token in texts,
        "chunk_count": len(chunks),
    }


def evaluate(questions, retriever, llm, mask_token):
    """按题目类型分发到对应的评测函数，返回结果列表。"""
    rows = []
    for q in questions:
        kind = classify(q)
        if kind == "mask":
            rows.append(_eval_mask(q, retriever, mask_token))
        elif kind == "negative":
            rows.append(_eval_negative(q, retriever, llm))
        elif kind == "eval":
            rows.append(_eval_normal(q, retriever, llm))
        # no_gold 直接跳过
    return rows


def report(rows):
    """汇总并打印生成评测报告（正确率 / 负样本拒绝率 / 脱敏 + 逐题）。"""
    evals = [r for r in rows if r["kind"] == "eval"]
    negs = [r for r in rows if r["kind"] == "negative"]
    masks = [r for r in rows if r["kind"] == "mask"]

    print("\n" + "=" * 64)
    print("生成评测结果")
    print("=" * 64)

    # 正确率（facts 全命中）+ 事实点命中率（粒度为单个 fact）
    if evals:
        correct = sum(1 for r in evals if r["correct"])
        fact_hit = sum(r["hit_facts"] for r in evals)
        fact_total = sum(r["total_facts"] for r in evals)
        print(f"  正确率（facts 全命中）：{correct}/{len(evals)} = {correct/len(evals):.0%}")
        print(f"  事实点命中：{fact_hit}/{fact_total} = {fact_hit/fact_total:.0%}")

    # 负样本拒绝率：回答里出现「未收录/未提及」即算正确拒绝
    if negs:
        rej = sum(1 for r in negs if r["rejected"])
        print(f"  负样本拒绝率：{rej}/{len(negs)} = {rej/len(negs):.0%}")

    # 脱敏专项：原文未泄露 + 打码占位可见 = 通过
    if masks:
        for m in masks:
            if m.get("status") == "skip":
                print(f"  脱敏专项：跳过（{m.get('note')}）")
            else:
                ok = (not m["raw_leaked"]) and m["masked_seen"]
                print(f"  脱敏专项：原文泄露={m['raw_leaked']} 打码可见={m['masked_seen']} → {'通过' if ok else '未通过'}")

    # 逐题明细
    print("-" * 64)
    print("逐题明细：")
    for r in rows:
        if r["kind"] == "eval":
            mark = "[OK]" if r["correct"] else "[X ]"
            print(f"  [{mark}] {r['id']} 事实 {r['hit_facts']}/{r['total_facts']}  {r['question'][:30]}")
        elif r["kind"] == "negative":
            mark = "[OK]" if r["rejected"] else "[X ]"
            print(f"  [{mark}] {r['id']} 负样本 拒绝={r['rejected']}  {r['question'][:30]}")
        elif r["kind"] == "mask":
            ok = (not r.get("raw_leaked", True)) and r.get("masked_seen", False)
            print(f"  [{'[OK]' if ok else '[X ]'}] {r['id']} 脱敏专项  {r['question'][:30]}")

    # 未通过题单独列出（附答案片段，便于人工复核）
    bad = [r for r in rows if (r["kind"] == "eval" and not r["correct"]) or (r["kind"] == "negative" and not r["rejected"])]
    if bad:
        print("-" * 64)
        print(f"未通过题目（{len(bad)} 条，重点排查）：")
        for r in bad:
            print(f"  - {r['id']} 问：{r['question']}")
            print(f"    答：{r['answer'][:120]}")


def main():
    """脚本入口：解析参数 → 读题 → 定位项目 → 检查 LLM 模式 → 跑评测 → 出报告。"""
    args = parse_args()
    questions_path, project = resolve_paths(args)
    questions = load_questions(questions_path)

    if args.list:
        tags = {"eval": "生成评测", "negative": "负样本", "mask": "脱敏专项", "no_gold": "缺gold_sources(跳过)"}
        print(f"共 {len(questions)} 题：")
        for q in questions:
            print(f"  {q['id']:<6} [{tags[classify(q)]}] {q.get('q_type','')}  {q['question'][:38]}")
        return 0

    if not (project / "rag").exists():
        print(f"[错误] 未找到项目目录：{project}。请用 --project 指定。")
        return 1

    sys.path.insert(0, str(project))     # 把项目根加入 sys.path，才能 import rag
    try:
        import config
        from rag import HybridRetriever, Retriever, get_llm
    except Exception as e:
        print(f"[错误] 无法导入项目 rag 包：{e}")
        return 1

    # 提示当前生成模式：mock / 仅检索 时答案非模型生成，结果只作结构自检
    if config.is_mock_llm():
        print("[提示] 当前 LLM_PROVIDER=mock，答案为模板，生成评测结果仅作结构自检。")
    elif not config.LLM_API_KEY:
        print("[提示] 未配置 LLM_API_KEY，当前为「仅检索」模式，答案非模型生成。")

    retriever = HybridRetriever() if args.hybrid else Retriever()
    rows = evaluate(questions, retriever, get_llm(), config.MASK_TOKEN)
    report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())