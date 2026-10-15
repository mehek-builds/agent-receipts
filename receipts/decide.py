"""Decisions computed from receipts: break-even task value and cache health."""
from __future__ import annotations

import random

from .ledger import Receipt


def break_even(frontier: dict[str, Receipt], small: dict[str, Receipt],
               n_boot: int = 10_000, seed: int = 1) -> tuple[float, tuple[float, float]]:
    """Value of one completed task above which `frontier` beats `small`.

    V* = (extra cost per task) / (extra success rate), on tasks both configurations ran.
    Returns the point estimate and a 95% interval from a paired bootstrap over tasks.
    Infinite when the frontier configuration is not more successful.
    """
    ids = sorted(frontier.keys() & small.keys())
    if not ids:
        raise ValueError("no shared task_ids")
    cost_f = {i: frontier[i].total() for i in ids}
    cost_s = {i: small[i].total() for i in ids}

    def v_star(sample: list[str]) -> float:
        extra_cost = sum(cost_f[i] - cost_s[i] for i in sample) / len(sample)
        extra_success = sum(frontier[i].success - small[i].success for i in sample) / len(sample)
        return extra_cost / extra_success if extra_success > 0 else float("inf")

    rng = random.Random(seed)
    boots = sorted(v_star([rng.choice(ids) for _ in ids]) for _ in range(n_boot))
    return v_star(ids), (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)])


def cache_misses(receipt: Receipt, min_prompt_tokens: int = 2_048) -> list[int]:
    """Steps after the first that read nothing from the cache despite a long prompt.

    In a working cached agent loop every later step re-reads the prefix, so a zero
    here means the prefix changed or the cache marker never took effect.
    """
    return [line.step for line in receipt.lines[1:]
            if line.prompt_tokens >= min_prompt_tokens and line.usage["cache_read_tokens"] == 0]
