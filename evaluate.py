"""
MealMind Evaluation Suite
1. Retrieval Benchmark (no API key needed): BM25 vs Dense vs Hybrid (RRF)
   Metrics: Recall@1, Recall@3, MRR (overall + per query category)
2. End-to-End QA Benchmark (requires GROQ_API_KEY or Ollama):
   Baseline (No RAG) vs Dense RAG vs Hybrid RAG
   Metrics: Token F1, ROUGE-L, Semantic Similarity, Avg Latency
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from rag_engine import MealMindRAG, tokenize

DATA_PATH = Path(__file__).parent / "data" / "eval_qa.json"
RESULTS_DIR = Path(__file__).parent / "results"


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = sum(1 for t in pred_tokens if t in common) / len(pred_tokens)
    recall = sum(1 for t in ref_tokens if t in common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 via Longest Common Subsequence (no external NLTK dependency)."""
    p = tokenize(prediction)
    r = tokenize(reference)
    if not p or not r:
        return 0.0
    m, n = len(p), len(r)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[i - 1] == r[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    return 2 * prec * rec / (prec + rec)


def evaluate_retrieval(rag: MealMindRAG, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    modes = ["bm25", "dense", "hybrid"]
    summary: Dict[str, Any] = {}

    for mode in modes:
        r1_list, r3_list, mrr_list = [], [], []
        by_cat: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"r1": [], "r3": [], "mrr": []})

        for item in qa_pairs:
            expected = set(item["expected_recipe_ids"])
            retrieved = rag.retrieve(item["question"], top_k=3, mode=mode)
            ret_ids = [x.recipe["id"] for x in retrieved]

            hit1 = 1.0 if (ret_ids and ret_ids[0] in expected) else 0.0
            hit3 = 1.0 if any(rid in expected for rid in ret_ids) else 0.0
            rr = 0.0
            for rank, rid in enumerate(ret_ids, 1):
                if rid in expected:
                    rr = 1.0 / rank
                    break

            r1_list.append(hit1)
            r3_list.append(hit3)
            mrr_list.append(rr)
            cat = item["category"]
            by_cat[cat]["r1"].append(hit1)
            by_cat[cat]["r3"].append(hit3)
            by_cat[cat]["mrr"].append(rr)

        summary[mode] = {
            "recall@1": round(float(np.mean(r1_list)), 4),
            "recall@3": round(float(np.mean(r3_list)), 4),
            "mrr": round(float(np.mean(mrr_list)), 4),
            "by_category": {
                cat: {
                    "recall@1": round(float(np.mean(vals["r1"])), 4),
                    "recall@3": round(float(np.mean(vals["r3"])), 4),
                    "mrr": round(float(np.mean(vals["mrr"])), 4),
                }
                for cat, vals in by_cat.items()
            },
        }

    return summary


def evaluate_generation(
    rag: MealMindRAG, qa_pairs: List[Dict[str, Any]], max_samples: int = 36
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    modes = ["none", "dense", "hybrid"]
    gen_summary: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    subset = qa_pairs[:max_samples]

    for mode in modes:
        f1s, rouges, sims, lats = [], [], [], []
        for item in subset:
            out = rag.answer(item["question"], mode=mode, top_k=3)
            pred = out["answer"]
            ref = item["reference_answer"]

            f1 = token_f1(pred, ref)
            rl = rouge_l_f1(pred, ref)
            embs = rag.embedder.encode([pred, ref], convert_to_numpy=True, normalize_embeddings=True)
            sim = float(np.dot(embs[0], embs[1]))

            f1s.append(f1)
            rouges.append(rl)
            sims.append(sim)
            lats.append(out["latency_sec"])

            rows.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "mode": mode,
                    "provider": out["provider"],
                    "token_f1": round(f1, 4),
                    "rouge_l": round(rl, 4),
                    "semantic_similarity": round(sim, 4),
                    "latency_sec": out["latency_sec"],
                }
            )

        gen_summary[mode] = {
            "avg_token_f1": round(float(np.mean(f1s)), 4),
            "avg_rouge_l": round(float(np.mean(rouges)), 4),
            "avg_semantic_similarity": round(float(np.mean(sims)), 4),
            "avg_latency_sec": round(float(np.mean(lats)), 4),
        }

    return gen_summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MealMind Hybrid RAG")
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="Also run end-to-end LLM generation comparison (requires GROQ_API_KEY or Ollama)",
    )
    parser.add_argument("--samples", type=int, default=36, help="Number of QA pairs for generation eval")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print("Loading MealMind RAG engine...")
    rag = MealMindRAG()

    print(f"\nRunning Retrieval Benchmark on {len(qa_pairs)} QA pairs...")
    retrieval_results = evaluate_retrieval(rag, qa_pairs)

    print("\n=== Retrieval Results (Overall) ===")
    for mode, metrics in retrieval_results.items():
        print(
            f"  {mode.upper():7s} | Recall@1: {metrics['recall@1']:.3f} | "
            f"Recall@3: {metrics['recall@3']:.3f} | MRR: {metrics['mrr']:.3f}"
        )

    print("\n=== Retrieval Recall@3 by Category ===")
    categories = sorted(next(iter(retrieval_results.values()))["by_category"].keys())
    for cat in categories:
        b = retrieval_results["bm25"]["by_category"][cat]["recall@3"]
        d = retrieval_results["dense"]["by_category"][cat]["recall@3"]
        h = retrieval_results["hybrid"]["by_category"][cat]["recall@3"]
        print(f"  {cat:16s} | BM25: {b:.3f} | Dense: {d:.3f} | Hybrid: {h:.3f}")

    report: Dict[str, Any] = {
        "total_questions": len(qa_pairs),
        "retrieval": retrieval_results,
    }

    if args.with_generation:
        print(f"\nRunning End-to-End Generation Benchmark ({args.samples} questions)...")
        gen_results, detail_rows = evaluate_generation(rag, qa_pairs, max_samples=args.samples)
        report["generation"] = gen_results
        pd.DataFrame(detail_rows).to_csv(RESULTS_DIR / "eval_details.csv", index=False)

        print("\n=== End-to-End Generation Results ===")
        for mode, m in gen_results.items():
            print(
                f"  {mode.upper():7s} | F1: {m['avg_token_f1']:.3f} | ROUGE-L: {m['avg_rouge_l']:.3f} | "
                f"SemSim: {m['avg_semantic_similarity']:.3f} | Latency: {m['avg_latency_sec']:.2f}s"
            )

    out_file = RESULTS_DIR / "eval_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {out_file}")


if __name__ == "__main__":
    main()
