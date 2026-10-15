#!/usr/bin/env python3
"""Convert BFCL v4 single-turn Python categories into the harness task file.

Draws a stratified sample (equal count per category, fixed seed) and attaches
the BFCL ground truth from possible_answer/. The output keeps the original
BFCL function documents untouched; the harness converts them to Anthropic tool
schemas at run time and grades against these originals.

Usage:
    python convert_bfcl.py --bfcl-data <gorilla>/berkeley-function-call-leaderboard/bfcl_eval/data \
        --commit <gorilla commit sha> --per-category 30 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

# BFCL v4 single-turn, non-live Python categories graded by the AST checker.
# irrelevance has no possible_answer file: success means the model makes no call.
CATEGORIES = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfcl-data", required=True, type=Path)
    ap.add_argument("--commit", required=True, help="gorilla repo commit the data came from")
    ap.add_argument("--per-category", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=HERE / "bfcl_v4.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_rows, mix, pool_sizes = [], {}, {}
    for cat in CATEGORIES:
        rows = read_jsonl(args.bfcl_data / f"BFCL_v4_{cat}.json")
        answers = {}
        ans_path = args.bfcl_data / "possible_answer" / f"BFCL_v4_{cat}.json"
        if ans_path.exists():
            answers = {a["id"]: a["ground_truth"] for a in read_jsonl(ans_path)}
        # single-turn only: one turn in "question"
        rows = [r for r in rows if len(r["question"]) == 1]
        pool_sizes[cat] = len(rows)
        picked = rng.sample(rows, min(args.per_category, len(rows)))
        for r in sorted(picked, key=lambda r: r["id"]):
            if cat != "irrelevance" and r["id"] not in answers:
                raise SystemExit(f"missing ground truth for {r['id']}")
            out_rows.append({
                "id": r["id"],
                "category": cat,
                "question": r["question"][0],
                "function": r["function"],
                "ground_truth": answers.get(r["id"], []),
            })
        mix[cat] = len(picked)

    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    manifest = {
        "benchmark": "Berkeley Function-Calling Leaderboard (BFCL) v4",
        "source_repo": "https://github.com/ShishirPatil/gorilla",
        "source_path": "berkeley-function-call-leaderboard/bfcl_eval/data",
        "commit": args.commit,
        "categories": CATEGORIES,
        "pool_sizes": pool_sizes,
        "category_mix": mix,
        "per_category": args.per_category,
        "seed": args.seed,
        "n_tasks": len(out_rows),
        "license": "Apache-2.0 (BFCL data)",
    }
    (args.out.with_name(args.out.stem + "_manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
