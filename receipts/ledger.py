"""Per-step receipts for LLM agent tasks.

A receipt has one line per model call. Each line is priced from the provider's
own usage block (fresh input, cache write, cache read, output) at dated list
prices, so a task's cost is the sum of its lines and every dollar stays
attributable to the step, and the kind of step, that spent it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

PRICING_FILE = Path(__file__).resolve().parent.parent / "harness" / "pricing_oct2026.json"
HAIKU_TIER_THRESHOLD = 100_000  # prompt tokens; Haiku 5.5 bills longer prompts at tier_high


def load_prices(path: Path = PRICING_FILE) -> dict:
    """USD per million tokens, per model, as verified on the provider's pricing page."""
    with open(path) as f:
        return json.load(f)["models"]


PRICES = load_prices()


def rate_card(model: str, prompt_tokens: int, prices: dict = PRICES) -> dict:
    card = prices[model]
    if "tier_low" in card:
        card = card["tier_high"] if prompt_tokens > HAIKU_TIER_THRESHOLD else card["tier_low"]
    return card


def normalize_usage(usage: dict) -> dict:
    """Accept the provider's field names or the harness log's, return one shape."""
    return {
        "input_tokens": usage.get("input_tokens", 0) or 0,
        "cache_write_tokens": usage.get("cache_creation_input_tokens", usage.get("cache_write_tokens", 0)) or 0,
        "cache_read_tokens": usage.get("cache_read_input_tokens", usage.get("cache_read_tokens", 0)) or 0,
        "output_tokens": usage.get("output_tokens", 0) or 0,
    }


@dataclass
class Line:
    step: int
    kind: str  # "act" (tool call), "final" (reply to user), "failed" (tool call errored), "handoff"
    model: str
    usage: dict
    turn: int = 0

    def __post_init__(self) -> None:
        self.usage = normalize_usage(self.usage)

    @property
    def prompt_tokens(self) -> int:
        u = self.usage
        return u["input_tokens"] + u["cache_write_tokens"] + u["cache_read_tokens"]

    def cost_by_category(self, prices: dict = PRICES) -> dict:
        r, u = rate_card(self.model, self.prompt_tokens, prices), self.usage
        return {
            "fresh_input": u["input_tokens"] * r["input_per_mtok"] / 1e6,
            "cache_write": u["cache_write_tokens"] * r["cache_write_5min_per_mtok"] / 1e6,
            "cache_read": u["cache_read_tokens"] * r["cache_read_per_mtok"] / 1e6,
            "output": u["output_tokens"] * r["output_per_mtok"] / 1e6,
        }

    def cost(self, prices: dict = PRICES) -> float:
        return sum(self.cost_by_category(prices).values())


@dataclass
class Receipt:
    task_id: str
    config: str = ""
    lines: list[Line] = field(default_factory=list)
    success: bool = False

    def add(self, step: int, kind: str, model: str, usage: dict, turn: int = 0) -> Line:
        """`usage` is the provider's usage block, e.g. response.usage.model_dump()."""
        line = Line(step, kind, model, usage, turn)
        self.lines.append(line)
        return line

    def total(self, prices: dict = PRICES) -> float:
        return sum(line.cost(prices) for line in self.lines)

    def by_kind(self, prices: dict = PRICES) -> dict:
        out: dict = {}
        for line in self.lines:
            out[line.kind] = out.get(line.kind, 0.0) + line.cost(prices)
        return out

    def by_category(self, prices: dict = PRICES) -> dict:
        out = {"fresh_input": 0.0, "cache_write": 0.0, "cache_read": 0.0, "output": 0.0}
        for line in self.lines:
            for k, v in line.cost_by_category(prices).items():
                out[k] += v
        return out

    def render(self, prices: dict = PRICES) -> str:
        rows = ["step turn kind     fresh_in cache_read cache_write output      cost"]
        for l in self.lines:
            u = l.usage
            rows.append(f"{l.step:>4} {l.turn:>4} {l.kind:<8} {u['input_tokens']:>8} {u['cache_read_tokens']:>10} "
                        f"{u['cache_write_tokens']:>11} {u['output_tokens']:>6} {l.cost(prices):>9.5f}")
        rows.append(f"total {self.total(prices):.5f} USD, success={self.success}")
        return "\n".join(rows)


def load_run(run_dir: Path, config: str) -> dict[str, Receipt]:
    """Receipts for one configuration of a harness run, keyed by task_id."""
    receipts: dict[str, Receipt] = {}
    tasks = run_dir / f"tasks_{config}.jsonl"
    for raw in open(tasks):
        t = json.loads(raw)
        receipts[t["task_id"]] = Receipt(t["task_id"], config, success=bool(t["success"]))
    steps = run_dir / f"steps_{config}.jsonl"
    if steps.exists():
        rows = [json.loads(raw) for raw in open(steps)]
        for s in sorted(rows, key=lambda s: (s["task_id"], s["step"])):
            if s["task_id"] in receipts:
                receipts[s["task_id"]].add(s["step"], s["kind"], s["model"], s["usage"], s.get("turn", 0))
    return receipts
