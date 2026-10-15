#!/usr/bin/env python3
"""Sample BFCL v4 multi_turn_base tasks into the multi-turn harness file.

Keeps each entry's turns, initial simulator state, involved classes, excluded
functions, and the per-turn ground-truth call strings. Function documents are
attached at run time from bfcl_mt/func_doc/.

Usage:
    python convert_bfcl_multi_turn.py --bfcl-data <gorilla>/berkeley-function-call-leaderboard/bfcl_eval/data \
        --commit <sha> --n 50 --seed 42
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATEGORY = "multi_turn_base"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfcl-data", required=True, type=Path)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=HERE / "bfcl_v4_multi_turn.jsonl")
    args = ap.parse_args()

    rows = read_jsonl(args.bfcl_data / f"BFCL_v4_{CATEGORY}.json")
    answers = {a["id"]: a["ground_truth"] for a in
               read_jsonl(args.bfcl_data / "possible_answer" / f"BFCL_v4_{CATEGORY}.json")}
    picked = sorted(random.Random(args.seed).sample(rows, args.n), key=lambda r: int(r["id"].rsplit("_", 1)[1]))
    with open(args.out, "w") as f:
        for r in picked:
            f.write(json.dumps({
                "id": r["id"], "category": CATEGORY,
                "turns": r["question"], "initial_config": r["initial_config"],
                "involved_classes": r["involved_classes"],
                "excluded_function": r.get("excluded_function", []),
                "ground_truth": answers[r["id"]],
            }) + "\n")
    manifest = {
        "benchmark": "Berkeley Function-Calling Leaderboard (BFCL) v4, multi_turn_base",
        "source_repo": "https://github.com/ShishirPatil/gorilla",
        "commit": args.commit, "pool_size": len(rows), "n_tasks": len(picked), "seed": args.seed,
        "turns_histogram": dict(collections.Counter(len(r["question"]) for r in picked)),
        "class_mix": dict(collections.Counter(c for r in picked for c in r["involved_classes"])),
        "ground_truth_calls_mean": sum(sum(len(t) for t in answers[r["id"]]) for r in picked) / len(picked),
        "license": "Apache-2.0 (BFCL data and simulators)",
    }
    args.out.with_name(args.out.stem + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
