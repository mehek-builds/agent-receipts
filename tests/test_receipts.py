"""The receipt ledger must reproduce the harness's own per-step bill exactly."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from receipts import Receipt, break_even, cache_misses, load_run  # noqa: E402

MT_RUN = ROOT / "harness" / "runs_mt" / "20261014-172808"
CONFIGS = ["frontier", "frontier_cache", "small", "small_cache_budget"]


def test_every_logged_step_reprices_to_the_harness_cost():
    checked = 0
    for cfg in CONFIGS:
        receipts = load_run(MT_RUN, cfg)
        logged = {}
        for raw in open(MT_RUN / f"steps_{cfg}.jsonl"):
            s = json.loads(raw)
            logged[(s["task_id"], s["step"])] = s["cost_usd"]
        for r in receipts.values():
            for line in r.lines:
                assert abs(line.cost() - logged[(r.task_id, line.step)]) < 1e-9
                checked += 1
    assert checked > 2000


def test_task_totals_match_task_records():
    for cfg in CONFIGS:
        receipts = load_run(MT_RUN, cfg)
        for raw in open(MT_RUN / f"tasks_{cfg}.jsonl"):
            t = json.loads(raw)
            assert abs(receipts[t["task_id"]].total() - t["cost_usd"]) < 1e-6


def test_provider_field_names_are_accepted():
    r = Receipt("t")
    line = r.add(0, "act", "claude-sonnet-5-5", {"input_tokens": 1_000_000, "cache_creation_input_tokens": 0,
                                                 "cache_read_input_tokens": 0, "output_tokens": 0})
    assert abs(line.cost() - 2.00) < 1e-12


def _toy(success, cost_tokens):
    r = Receipt("x", success=success)
    r.add(0, "act", "claude-sonnet-5-5", {"input_tokens": cost_tokens, "output_tokens": 0})
    return r


def test_break_even_formula():
    # frontier: +0.5 success, +$0.10 per task -> V* = $0.20
    frontier = {"a": _toy(True, 100_000), "b": _toy(True, 100_000)}
    small = {"a": _toy(True, 50_000), "b": _toy(False, 50_000)}
    v, _ = break_even(frontier, small, n_boot=200)
    assert abs(v - 0.20) < 1e-9


def test_break_even_infinite_when_frontier_not_better():
    frontier = {"a": _toy(True, 100_000)}
    small = {"a": _toy(True, 10_000)}
    v, _ = break_even(frontier, small, n_boot=50)
    assert v == float("inf")


def test_cache_check_is_quiet_on_working_cache_and_flags_a_miss():
    cached = load_run(MT_RUN, "frontier_cache")
    assert all(cache_misses(r) == [] for r in cached.values())
    uncached = load_run(MT_RUN, "frontier")
    assert any(cache_misses(r) for r in uncached.values())
