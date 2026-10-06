"""检索评测脚本（Retrieval Evaluation）
======================================

读取测评集 questions.jsonl，复用项目的 Retriever，量化「检索层」质量：

    - Hit Rate@k（命中率）：该题 gold_sources 是否出现在 top-k 召回里
    - MRR（Mean Reciprocal Rank）：第一个 gold 来源排名的倒数均值

为什么先测检索：RAG 是两段式（先检索后生成），检索漏了，生成再好也答不对。
检索评测不需要 LLM key，只依赖向量模型，能最快暴露「分块 / 向量 / 阈值」问题。

用法（在项目 eval/ 目录下运行，自动定位项目根目录）：
    python retrieval_eval.py                          # 读同目录 questions.jsonl
    python retrieval_eval.py --project ..             # 显式指定项目根目录
    python retrieval_eval.py --category past_case     # 只评测某分类
    python retrieval_eval.py --list                   # 只列题目清单，不调用向量库

说明：
    - 负样本（negative=true）测的是「生成层是否诚实说未收录」，本脚本跳过；
    - 脱敏专项（q_type=脱敏专项）测的是「脱敏是否生效」，本脚本跳过；
    - 本脚本按 score_threshold=0.0 检索，评估「纯召回/排序质量」；
      线上 /api/ask 还会再套一层相似度阈值过滤（见 config.SCORE_THRESHOLD）。
"""

import argparse
import json
import sys

# 强制 stdout 用 UTF-8 输出：避免 Windows 默认 GBK 控制台打印中文/特殊符号时报
# UnicodeEncodeError；errors="replace" 保证遇到生僻字符也不会中断。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from collections import defaultdict
from pathlib import Path

# 统计三个档位的命中率：Hit@1 / Hit@3 / Hit@5。
# k 越大越容易命中（看的名额越多），所以恒有 Hit@1 <= Hit@3 <= Hit@5。
K_VALUES = (1, 3, 5)


def parse_args():
    """解析命令行参数。输入：无（读 sys.argv）；输出：argparse.Namespace。"""
    p = argparse.ArgumentParser(description="检索评测")
    p.add_argument("--questions", default=None, help="测评集 JSONL 路径，默认同目录 questions.jsonl")
    p.add_argument("--project", default=None, help="项目根目录，默认脚本所在目录的上一级")
    p.add_argument("--top-k", type=int, default=5, help="召回条数，默认 5")
    p.add_argument("--category", default=None, help="只评测某分类（英文 id）")
    p.add_argument("--hybrid", action="store_true", help="用混合检索（BM25+向量+RRF）对比")
    p.add_argument("--list", action="store_true", help="只列出题目与分类，不调用向量库")
    return p.parse_args()


def resolve_paths(args):
    """确定测评集路径与项目根目录。

    默认约定：本脚本放在项目 eval/ 下，所以「项目根 = 脚本所在目录的上一级」。
    """
    script_dir = Path(__file__).resolve().parent          # 脚本所在目录 = eval/
    questions = Path(args.questions) if args.questions else script_dir / "questions.jsonl"
    project = Path(args.project) if args.project else script_dir.parent   # 上一级 = 项目根
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
    """把一道题归类，决定它走哪条评测路径。

    返回：
        "eval"     —— 正常题，参与检索评测
        "negative" —— 负样本，测生成层（本脚本跳过）
        "mask"     —— 脱敏专项（本脚本跳过）
        "no_gold"  —— 没标 gold_sources，无法评测
    """
    if q.get("negative"):
        return "negative"
    if q.get("q_type") == "脱敏专项":
        return "mask"
    if not q.get("gold_sources"):
        return "no_gold"
    return "eval"


def list_questions(questions):
    """--list 时只打印题目清单和归类，不碰向量库。"""
    tags = {"eval": "检索评测", "negative": "负样本(跳过)", "mask": "脱敏专项(跳过)", "no_gold": "缺gold_sources(跳过)"}
    print(f"共 {len(questions)} 题：")
    for q in questions:
        print(f"  {q['id']:<6} [{tags[classify(q)]}] {q.get('category','-'):<16} {q.get('q_type','')}  {q['question'][:38]}")


def evaluate(questions, retriever, top_k, category_filter):
    """跑检索并计算每题的排名与 hit@k。

    核心逻辑：
        1) 对每道「正常题」，用 retriever.search 检索 top_k 条；
        2) 关闭阈值（score_threshold=0.0），纯看排序能力，不被阈值过滤干扰；
        3) 在召回结果里找第一个命中的 gold_source，记下它的名次 rank；
        4) rank <= k 即为 Hit@k 命中。
    """
    rows = []
    for q in questions:
        if classify(q) != "eval":          # 只处理正常题
            continue
        if category_filter and q.get("category") != category_filter:   # 可选按分类筛题
            continue

        gold = set(q["gold_sources"])      # 该题「应当命中」的来源文件集合
        hits = retriever.search(
            q["question"],
            top_k=top_k,
            category=None,                 # 全库检索：测「不靠分类过滤也能召回」
            score_threshold=0.0,           # 关闭阈值：测纯排序质量
        )
        sources = [h.source for h in hits]  # 召回结果的来源路径，按相似度降序

        # 找第一个命中 gold 的位置（1 起）。没命中则 rank=None。
        rank = None
        for i, s in enumerate(sources, start=1):
            if s in gold:
                rank = i
                break

        rows.append({
            "id": q["id"],
            "question": q["question"],
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
            "gold": sorted(gold),
            "retrieved": sources,
            "rank": rank,
            # 对每个 k 判断「rank 是否落在前 k 名内」
            "hit": {k: rank is not None and rank <= k for k in K_VALUES},
        })
    return rows


def report(rows):
    """汇总并打印评测报告（总体指标 + 分层 + 逐题 + 未命中诊断）。"""
    n = len(rows)
    if n == 0:
        print("没有可评测的题目。")
        return

    # Hit@k = 命中题数 / 总题数（对每个 k 分别统计）
    hit = {k: sum(1 for r in rows if r["hit"][k]) for k in K_VALUES}
    # MRR = 各题「1 / 首个正确来源的排名」的平均值；rank=None 的题计 0
    mrr = sum(1.0 / r["rank"] for r in rows if r["rank"]) / n

    print("\n" + "=" * 64)
    print("检索评测结果")
    print("=" * 64)
    for k in K_VALUES:
        print(f"  Hit@{k:<2} : {hit[k]}/{n} = {hit[k]/n:.1%}")
    print(f"  MRR     : {mrr:.3f}")
    print("-" * 64)

    # 按分类分层：定位是「某一类语料」检索差，还是整体问题
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"] or "未分类"].append(r)
    print("按分类 hit@5：")
    for cat, rs in sorted(by_cat.items()):
        c = len(rs)
        h5 = sum(1 for r in rs if r["hit"][5])
        print(f"  {cat:<18} {h5}/{c} = {h5/c:.0%}")

    # 按难度分层：定位是「简单题都挂」还是「只有难题挂」
    by_diff = defaultdict(list)
    for r in rows:
        by_diff[r["difficulty"] or "未知"].append(r)
    print("按难度 hit@5：")
    for diff, rs in sorted(by_diff.items()):
        c = len(rs)
        h5 = sum(1 for r in rs if r["hit"][5])
        print(f"  {diff:<10} {h5}/{c} = {h5/c:.0%}")

    # 逐题明细
    print("-" * 64)
    print("逐题明细（hit@5）：")
    for r in rows:
        mark = "[OK]" if r["hit"][5] else "[X ]"
        rank_s = f"rank={r['rank']}" if r["rank"] else "未命中"
        print(f"  [{mark}] {r['id']}  {rank_s:<8} {r['question'][:38]}")

    # 未命中题单独列出，并给出「应命中 vs 实际召回」，便于定位原因
    bad = [r for r in rows if not r["hit"][5]]
    if bad:
        print("-" * 64)
        print(f"未命中题目（{len(bad)} 条，重点排查）：")
        for r in bad:
            print(f"  - {r['id']} 分类={r['category']} 难度={r['difficulty']}")
            print(f"    问：{r['question']}")
            print(f"    应命中：{r['gold']}")
            print(f"    实际召回前3：{r['retrieved'][:3]}")


def main():
    """脚本入口：解析参数 → 读题 → 定位项目 → 跑评测 → 出报告。"""
    args = parse_args()
    questions_path, project = resolve_paths(args)
    questions = load_questions(questions_path)

    if args.list:
        list_questions(questions)
        return 0

    if not (project / "rag").exists():
        print(f"[错误] 未找到项目目录：{project}（应包含 rag/ 子目录）。请用 --project 指定。")
        return 1

    sys.path.insert(0, str(project))     # 把项目根加入 sys.path，才能 import rag
    try:
        if args.hybrid:
            from rag import HybridRetriever
            retriever = HybridRetriever()
        else:
            from rag import Retriever
            retriever = Retriever()
    except Exception as e:
        print(f"[错误] 无法导入项目 rag 包：{e}\n请确认 --project 指向项目根目录，且依赖已安装。")
        return 1

    rows = evaluate(questions, retriever, args.top_k, args.category)
    report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())