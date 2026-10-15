#!/usr/bin/env python3
"""Recompute every number in the article from the run logs in harness/.

No API calls. Writes results/article_numbers.json and prints a summary.
Run from the repo root:  python analysis/reproduce.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from receipts import PRICES, Receipt, break_even, load_run  # noqa: E402

ST_RUNS = sorted((ROOT / "harness" / "runs").glob("2026*"))
MT_RUN = ROOT / "harness" / "runs_mt" / "20261014-172808"
RECEIPT_TASK = "multi_turn_base_56"
SONNET, HAIKU = "claude-sonnet-5-5", "claude-haiku-5-5"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b: int, c: int) -> float:
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def single_turn() -> dict:
    """Pool every full single-turn run per configuration."""
    pooled: dict = {}
    for run in ST_RUNS:
        for f in run.glob("tasks_*.jsonl"):
            cfg = f.stem.removeprefix("tasks_")
            rows = [json.loads(x) for x in open(f)]
            p = pooled.setdefault(cfg, {"runs": [], "tasks": []})
            p["runs"].append(statistics.mean(r["cost_usd"] for r in rows))
            p["tasks"] += rows
    out = {}
    for cfg, p in pooled.items():
        t = p["tasks"]
        out[cfg] = {
            "runs": len(p["runs"]),
            "success": sum(r["success"] for r in t) / len(t),
            "cost_per_task": statistics.mean(r["cost_usd"] for r in t),
            "cost_cv_across_runs": (statistics.stdev(p["runs"]) / statistics.mean(p["runs"])) if len(p["runs"]) > 1 else None,
        }
    return out


def mt_tasks(cfg: str) -> dict:
    return {t["task_id"]: t for t in (json.loads(x) for x in open(MT_RUN / f"tasks_{cfg}.jsonl"))}


def reprice(receipts: dict, mode: str) -> float:
    """Mean cost per task if the same token traffic were billed differently.

    mode="uncached": every prompt token at the fresh-input rate (a validity check
    against the measured uncached bill). mode="broken_cache": every prompt token
    written to the cache and never read (a cache that is marked but always misses).
    """
    totals = []
    for r in receipts.values():
        c = 0.0
        for l in r.lines:
            card = PRICES[l.model] if "tier_low" not in PRICES[l.model] else PRICES[l.model]["tier_low"]
            rate = card["input_per_mtok"] if mode == "uncached" else card["cache_write_5min_per_mtok"]
            c += (l.prompt_tokens * rate + l.usage["output_tokens"] * card["output_per_mtok"]) / 1e6
        totals.append(c)
    return statistics.mean(totals)


def main() -> None:
    N: dict = {}

    st = single_turn()
    N["single_turn"] = st
    N["single_turn_small_saving_vs_frontier"] = 1 - st["small"]["cost_per_task"] / st["frontier"]["cost_per_task"]
    N["single_turn_low_effort_cost_change_frontier"] = st["frontier_low"]["cost_per_task"] / st["frontier"]["cost_per_task"] - 1

    cfgs = ["frontier", "frontier_cache", "small", "small_cache_budget", "frontier_low", "small_low"]
    T = {c: mt_tasks(c) for c in cfgs}
    R = {c: load_run(MT_RUN, c) for c in ["frontier", "frontier_cache", "small", "small_cache_budget"]}
    ids = sorted(set.intersection(*(set(T[c]) for c in cfgs)))
    N["multi_turn_n_tasks"] = len(ids)

    table = {}
    for c in cfgs:
        k = sum(T[c][i]["success"] for i in ids)
        cost = sum(T[c][i]["cost_usd"] for i in ids)
        table[c] = {"successes": k, "success": k / len(ids), "wilson95": wilson(k, len(ids)),
                    "cost_per_task": cost / len(ids), "cost_per_success": cost / k,
                    "steps_per_task": statistics.mean(T[c][i]["llm_steps"] for i in ids)}
    N["multi_turn"] = table

    def disc(a, b):
        return (sum(1 for i in ids if T[a][i]["success"] and not T[b][i]["success"]),
                sum(1 for i in ids if T[b][i]["success"] and not T[a][i]["success"]))
    for a, b in [("frontier", "small"), ("frontier_cache", "small_cache_budget")]:
        x, y = disc(a, b)
        N[f"mcnemar_{a}_vs_{b}"] = {"frontier_only": x, "small_only": y, "p_exact": mcnemar_exact(x, y)}

    for c in ["frontier", "frontier_cache", "small", "small_cache_budget"]:
        cats = {"fresh_input": 0.0, "cache_write": 0.0, "cache_read": 0.0, "output": 0.0}
        for r in R[c].values():
            for k, v in r.by_category().items():
                cats[k] += v
        tot = sum(cats.values())
        N[f"composition_{c}"] = {k: {"usd_per_task": v / len(R[c]), "share": v / tot} for k, v in cats.items()}

    N["caching_saving_frontier"] = 1 - table["frontier_cache"]["cost_per_task"] / table["frontier"]["cost_per_task"]
    N["caching_saving_small"] = 1 - table["small_cache_budget"]["cost_per_task"] / table["small"]["cost_per_task"]
    N["multi_vs_single_turn_cost_ratio_frontier"] = table["frontier"]["cost_per_task"] / st["frontier"]["cost_per_task"]
    N["multi_turn_low_effort_cost_change_frontier"] = table["frontier_low"]["cost_per_task"] / table["frontier"]["cost_per_task"] - 1
    N["multi_turn_low_effort_success_change_frontier_pts"] = 100 * (table["frontier_low"]["success"] - table["frontier"]["success"])

    be = {}
    for name, f, s in [("uncached", "frontier", "small"), ("cached", "frontier_cache", "small_cache_budget"),
                       ("cached_frontier_vs_uncached_small", "frontier_cache", "small")]:
        v, ci = break_even(R[f], R[s])
        be[name] = {"v_star_usd": v, "ci95_usd": ci}
    N["break_even"] = be
    N["cost_gap_per_task"] = {
        "uncached": table["frontier"]["cost_per_task"] - table["small"]["cost_per_task"],
        "cached": table["frontier_cache"]["cost_per_task"] - table["small_cache_budget"]["cost_per_task"],
    }

    measured = table["frontier"]["cost_per_task"]
    N["reprice_check_uncached_frontier"] = {"repriced": reprice(R["frontier"], "uncached"), "measured": measured}
    broken = reprice(R["frontier"], "broken_cache")
    N["broken_cache_frontier"] = {
        "cost_per_task": broken,
        "x_working_cache": broken / table["frontier_cache"]["cost_per_task"],
        "vs_no_cache": broken / measured - 1,
        "v_star_vs_cached_small": (broken - table["small_cache_budget"]["cost_per_task"])
        / (table["frontier_cache"]["success"] - table["small_cache_budget"]["success"]),
    }

    ctx, after = [], []
    for r in R["small"].values():
        for idx, l in enumerate(r.lines):
            if l.kind == "failed":
                ctx.append(l.prompt_tokens)
                after.append(len(r.lines) - idx - 1)
                break
    write_rate = PRICES[SONNET]["cache_write_5min_per_mtok"]
    handoff = statistics.median(ctx) * write_rate / 1e6
    N["handoff_tax"] = {
        "small_tasks_with_tool_error": len(ctx),
        "median_context_tokens_at_first_error": statistics.median(ctx),
        "median_handoff_write_usd": handoff,
        "share_of_cached_frontier_task": handoff / table["frontier_cache"]["cost_per_task"],
        "median_steps_remaining": statistics.median(after),
        "note": "small-model token counts used for the frontier model (same model family)",
    }

    rec = R["frontier_cache"][RECEIPT_TASK]
    failed = sum(l.cost() for l in rec.lines if l.kind == "failed")
    N["figure1_receipt"] = {
        "task_id": RECEIPT_TASK, "success": rec.success, "total_usd": rec.total(),
        "failed_steps_usd": failed, "failed_share": failed / rec.total(),
        "same_task_uncached_frontier_usd": R["frontier"][RECEIPT_TASK].total(),
        "same_task_small_usd": R["small"][RECEIPT_TASK].total(),
        "same_task_small_cached_usd": R["small_cache_budget"][RECEIPT_TASK].total(),
        "small_success": R["small"][RECEIPT_TASK].success,
        "lines": [{"step": l.step, "turn": l.turn, "kind": l.kind, **l.usage, "usd": l.cost()} for l in rec.lines],
    }

    out = ROOT / "results" / "article_numbers.json"
    out.write_text(json.dumps(N, indent=2) + "\n")

    p = lambda x: f"{100 * x:.1f}%"
    print("single-turn  success/cost:", {c: (p(v["success"]), round(v["cost_per_task"], 5)) for c, v in st.items()})
    for c, v in table.items():
        lo, hi = v["wilson95"]
        print(f"multi-turn {c:<20} {p(v['success'])} [{100*lo:.0f}, {100*hi:.0f}]  ${v['cost_per_task']:.4f}/task  ${v['cost_per_success']:.4f}/success")
    for k, v in be.items():
        print(f"break-even {k:<36} ${v['v_star_usd']:.3f}  95% CI ${v['ci95_usd'][0]:.3f} to ${v['ci95_usd'][1]:.3f}")
    print("reprice check:", N["reprice_check_uncached_frontier"])
    print("broken cache:", N["broken_cache_frontier"])
    print("handoff tax:", N["handoff_tax"])
    print(rec.render())
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
