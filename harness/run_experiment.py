#!/usr/bin/env python3
"""Single-turn harness for the agent-receipts study.

Runs BFCL v4 single-turn function-calling tasks under three model/prompt
configurations and records per-step monetary cost attribution (token spend by
billing category, tool charges, and the cost of failed attempts that were
retried). Every step is written as one receipt line (see ../README.md).

Protocol per task: the model sees the BFCL question and the BFCL functions as
Anthropic tools. Tools are never executed. If the emitted tool calls fail
schema validation (unknown function, missing required or unexpected
parameter, wrong type), the harness returns tool_result blocks with
is_error=true and the model may retry, up to --max-attempts LLM calls and the
configuration's tool-call budget. The final attempt's calls are graded with
BFCL's own AST checker (vendored in bfcl_ast_checker.py) against the BFCL
possible_answer ground truth. For the irrelevance category, success means the
model made no tool call.

Usage:
    python run_experiment.py --all --dry-run                       # zero spend, mock model
    python run_experiment.py --all --max-tasks 3                    # real smoke test
    python run_experiment.py --all --max-tasks 150 --max-spend-usd 80
    python run_experiment.py --analyze runs/<run-id>                # table + figure from logs

Guardrails: --max-spend-usd is a HARD stop checked before every API call.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bfcl_ast_checker import Language, ast_checker  # noqa: E402

# ---------------------------------------------------------------------------
# Configurations: frontier; frontier+cache; small; small+cache+budget; low-effort variants
# ---------------------------------------------------------------------------
CONFIGS = {
    "frontier": {
        "model": "claude-sonnet-5-5",
        "provider": "anthropic",
        "prompt_caching": False,
        "max_tool_calls": 12,
        "pricing_key": "claude-sonnet-5-5",
    },
    "small": {
        "model": "claude-haiku-5-5",
        "provider": "anthropic",
        "prompt_caching": False,
        "max_tool_calls": 12,
        "pricing_key": "claude-haiku-5-5",
    },
    "small_cache_budget": {
        "model": "claude-haiku-5-5",
        "provider": "anthropic",
        "prompt_caching": True,
        "max_tool_calls": 5,  # tool-call budget for the small+cache+budget config
        "pricing_key": "claude-haiku-5-5",
    },
    # Effort ablation: same as (a) and (b) but at reasoning effort "low".
    "frontier_low": {
        "model": "claude-sonnet-5-5",
        "provider": "anthropic",
        "prompt_caching": False,
        "max_tool_calls": 12,
        "pricing_key": "claude-sonnet-5-5",
        "effort": "low",
    },
    # Multi-turn only: the frontier model with prompt caching (no tool-call budget).
    "frontier_cache": {
        "model": "claude-sonnet-5-5",
        "provider": "anthropic",
        "prompt_caching": True,
        "max_tool_calls": 12,
        "pricing_key": "claude-sonnet-5-5",
    },
    "small_low": {
        "model": "claude-haiku-5-5",
        "provider": "anthropic",
        "prompt_caching": False,
        "max_tool_calls": 12,
        "pricing_key": "claude-haiku-5-5",
        "effort": "low",
    },
}
MAIN_CONFIGS = ["frontier", "small", "small_cache_budget"]
# Default effort for every configuration unless the config sets "effort" (ablation only),
# so the main comparison isolates model, caching and budget.
EFFORT = "medium"
MAX_TOKENS = 16000  # thinking counts toward max_tokens; leave room
PROMPT_TOKEN_LIMIT = 100_000  # Haiku 5.5 changes rate card above this prompt length

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
def load_pricing() -> dict:
    with open(HERE / "pricing_oct2026.json") as f:
        return json.load(f)


def resolve_rate(pricing: dict, key: str, prompt_tokens: int) -> dict:
    """Per-request rate card. Tiered models are priced by that request's prompt length."""
    entry = pricing["models"][key]
    if "tier_low" in entry:
        return entry["tier_high"] if prompt_tokens > PROMPT_TOKEN_LIMIT else entry["tier_low"]
    return entry


def step_cost(usage: dict, rate: dict, tool_charge: float = 0.0) -> float:
    """Per-step cost (one receipt line): token spend by billing category + tool charge."""
    mtok = 1_000_000
    return (
        usage.get("input_tokens", 0) / mtok * rate["input_per_mtok"]
        + usage.get("output_tokens", 0) / mtok * rate["output_per_mtok"]
        + usage.get("cache_read_tokens", 0) / mtok * rate.get("cache_read_per_mtok", 0.0)
        + usage.get("cache_write_tokens", 0) / mtok * rate.get("cache_write_5min_per_mtok", 0.0)
        + tool_charge
    )


def prompt_tokens(usage: dict) -> int:
    return (usage.get("input_tokens", 0) + usage.get("cache_read_tokens", 0)
            + usage.get("cache_write_tokens", 0))


def tool_charge(pricing: dict, tool_name: str) -> float:
    charges = pricing.get("tool_charges", {})
    return float(charges.get(tool_name, charges.get("default_per_call_usd", 0.0)))

# ---------------------------------------------------------------------------
# BFCL function documents -> Anthropic tools
# ---------------------------------------------------------------------------
_TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "any": "string",
             "integer": "integer", "string": "string", "boolean": "boolean",
             "array": "array", "number": "number", "object": "object"}
_DROP_KEYS = {"optional"}


def to_json_schema(node):
    """Recursively map BFCL parameter schemas to JSON Schema."""
    if isinstance(node, list):
        return [to_json_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k in _DROP_KEYS:
            continue
        if k == "type" and isinstance(v, str):
            out[k] = _TYPE_MAP.get(v, "string")
        elif k == "properties" and isinstance(v, dict):
            out[k] = {pk: to_json_schema(pv) for pk, pv in v.items()}
        else:
            out[k] = to_json_schema(v)
    return out


def sanitize_names(functions: list[dict]) -> dict[str, str]:
    """Map original BFCL names to names matching ^[a-zA-Z0-9_-]{1,64}$ (unique)."""
    mapping, used = {}, set()
    for fn in functions:
        base = re.sub(r"[^a-zA-Z0-9_-]", "_", fn["name"])[:64] or "fn"
        name, k = base, 1
        while name in used:
            suffix = f"_{k}"
            name, k = base[: 64 - len(suffix)] + suffix, k + 1
        used.add(name)
        mapping[fn["name"]] = name
    return mapping


def build_tools(task: dict) -> tuple[list[dict], dict[str, str]]:
    fwd = sanitize_names(task["function"])
    tools = []
    for fn in task["function"]:
        schema = to_json_schema(fn.get("parameters", {"type": "dict", "properties": {}}))
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        tools.append({"name": fwd[fn["name"]],
                      "description": fn.get("description", ""),
                      "input_schema": schema})
    return tools, {v: k for k, v in fwd.items()}

# ---------------------------------------------------------------------------
# Schema validation of emitted calls (decides whether a retry happens)
# ---------------------------------------------------------------------------
def _type_ok(value, bfcl_type: str) -> bool:
    if bfcl_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if bfcl_type in ("float", "number"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if bfcl_type == "string":
        return isinstance(value, str)
    if bfcl_type == "boolean":
        return isinstance(value, bool)
    if bfcl_type in ("array", "tuple"):
        return isinstance(value, list)
    if bfcl_type in ("dict", "object"):
        return isinstance(value, dict)
    return True  # "any" and unknown types


def validate_call(call: dict, functions_by_name: dict[str, dict]) -> str | None:
    """Return an error message, or None when the call is schema-valid."""
    fn = functions_by_name.get(call["name"])
    if fn is None:
        return f"unknown function {call['name']!r}"
    params = fn.get("parameters", {})
    props = params.get("properties", {})
    args = call.get("arguments") or {}
    for req in params.get("required", []):
        if req not in args:
            return f"missing required parameter {req!r}"
    for k, v in args.items():
        if k not in props:
            return f"unexpected parameter {k!r}"
        if not _type_ok(v, props[k].get("type", "any")):
            return f"parameter {k!r} has wrong type (expected {props[k].get('type')})"
    return None

# ---------------------------------------------------------------------------
# Grading: BFCL AST checker on the final attempt's calls
# ---------------------------------------------------------------------------
def grade(task: dict, calls: list[dict]) -> tuple[bool, str]:
    if task["category"] == "irrelevance":
        return (len(calls) == 0, "" if not calls else "called a function on an irrelevant query")
    if not calls:
        return False, "no function call"
    model_output = [{c["name"]: c.get("arguments") or {}} for c in calls]
    res = ast_checker(task["function"], model_output, task["ground_truth"],
                      Language.PYTHON, task["category"], "harness")
    return bool(res["valid"]), "" if res["valid"] else str(res.get("error_type", "invalid"))

# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------
class MockClient:
    """Dry-run client. Emits the ground-truth calls (first allowed value per
    parameter), with a deliberately invalid first attempt on every 7th task to
    exercise the retry path. Synthetic usage; zero spend."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def call(self, system, messages, tools, task, reverse) -> tuple[dict, dict]:
        attempt = sum(1 for m in messages if m["role"] == "assistant")
        base = 600 + 40 * len(tools) + 120 * attempt
        cached = self.cfg["prompt_caching"]
        usage = {
            "input_tokens": 60 if cached else base,
            "output_tokens": 150,
            "cache_read_tokens": (base - 60) if cached and attempt > 0 else 0,
            "cache_write_tokens": (base - 60) if cached and attempt == 0 else 0,
        }
        fwd = {v: k for k, v in reverse.items()}
        calls = []
        for gt in task["ground_truth"]:
            (fname, params), = gt.items()
            args = {k: v[0] for k, v in params.items() if v and v[0] != ""}
            calls.append({"name": fwd.get(fname, fname), "arguments": args,
                          "id": f"toolu_mock_{attempt}_{len(calls)}"})
        if calls and attempt == 0 and sum(map(ord, task["id"])) % 7 == 0:
            calls[0] = {**calls[0], "arguments": {}}  # missing required params
        return {"tool_calls": calls, "text": "", "stop_reason": "tool_use" if calls else "end_turn",
                "content": [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                            for c in calls] or [{"type": "text", "text": "No suitable function."}]}, usage

    def count_tokens(self, system, messages, tools) -> int | None:
        return None


class AnthropicClient:
    def __init__(self, cfg: dict):
        import anthropic  # pip install anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.cfg = cfg

    def _system(self, system: str):
        # Breakpoint on the system block caches the whole tools + system prefix
        # (render order is tools -> system -> messages).
        block = {"type": "text", "text": system}
        if self.cfg["prompt_caching"]:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def call(self, system, messages, tools, task, reverse) -> tuple[dict, dict]:
        resp = self.client.messages.create(
            model=self.cfg["model"], max_tokens=MAX_TOKENS,
            system=self._system(system), messages=messages, tools=tools,
            tool_choice={"type": "auto"}, output_config={"effort": self.cfg.get("effort", EFFORT)},
        )
        u = resp.usage
        usage = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
        calls, text = [], ""
        for block in resp.content:
            if block.type == "tool_use":
                calls.append({"name": block.name, "arguments": block.input, "id": block.id})
            elif block.type == "text":
                text += block.text
        # resp.content is passed back unchanged (keeps thinking blocks valid on retry)
        return {"tool_calls": calls, "text": text, "stop_reason": resp.stop_reason,
                "content": resp.content}, usage

    def count_tokens(self, system, messages, tools) -> int | None:
        r = self.client.messages.count_tokens(model=self.cfg["model"], system=self._system(system),
                                              messages=messages, tools=tools)
        return r.input_tokens


def make_client(cfg: dict, dry_run: bool):
    if dry_run:
        return MockClient(cfg)
    if cfg["provider"] == "anthropic":
        return AnthropicClient(cfg)
    raise ValueError(f"unsupported provider: {cfg['provider']}")

# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------
def load_tasks(path: Path, max_tasks: int) -> list[dict]:
    """Load the stratified BFCL file. When max_tasks is below the file size,
    take tasks round-robin across categories so the subset stays stratified."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    out, i = [], 0
    while len(out) < min(max_tasks, len(rows)):
        for cat in by_cat:
            if i < len(by_cat[cat]) and len(out) < max_tasks:
                out.append(by_cat[cat][i])
        i += 1
    return out


SYSTEM_PROMPT = (
    "You are a function-calling assistant. Answer the user's request by calling the "
    "provided functions with correct arguments; call several functions, or the same "
    "function several times, when the request needs it. If none of the functions "
    "fits the request, do not call any function and say so briefly."
)


def first_user_messages(task: dict) -> list[dict]:
    msgs = [{"role": m["role"], "content": m["content"]} for m in task["question"]
            if m["role"] in ("user", "assistant")]
    return msgs or [{"role": "user", "content": ""}]


def estimate_prompt_tokens(task: dict) -> int:
    """Conservative char-based estimate (3 chars/token) plus the documented
    286-token tool-use system prompt overhead."""
    chars = len(SYSTEM_PROMPT) + len(json.dumps(task["question"])) + len(json.dumps(task["function"]))
    return math.ceil(chars / 3) + 286


def prompt_length_check(tasks: list[dict], client, cfgs: list[str]) -> dict:
    est = {t["id"]: estimate_prompt_tokens(t) for t in tasks}
    worst = max(est, key=est.get)
    report = {"estimator": "ceil(chars/3)+286", "max_estimated_tokens": est[worst],
              "max_task": worst, "limit": PROMPT_TOKEN_LIMIT}
    task = next(t for t in tasks if t["id"] == worst)
    for cfg_name in cfgs:
        cfg = CONFIGS[cfg_name]
        try:
            tools, _ = build_tools(task)
            n = make_client(cfg, client).count_tokens(SYSTEM_PROMPT, first_user_messages(task), tools)
        except Exception as e:  # noqa: BLE001 - informational only
            n = f"unavailable: {e}"
        report[f"count_tokens_{cfg['model']}"] = n
    if est[worst] > PROMPT_TOKEN_LIMIT:
        sys.exit(f"prompt for {worst} may exceed {PROMPT_TOKEN_LIMIT} tokens: {report}")
    return report

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
@dataclass
class StepLog:
    task_id: str
    category: str
    config: str
    step: int
    kind: str  # llm | tool
    model: str
    usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    tool_name: str = ""
    ok: bool = True
    error: str = ""
    latency_s: float = 0.0


class SpendCapReached(Exception):
    pass


def run_task(task, cfg, cfg_name, pricing, client, args, step_logs, spend) -> dict:
    tools, reverse = build_tools(task)
    functions_by_sanitized = {n: fn for n, fn in
                              ((n, next(f for f in task["function"] if f["name"] == o))
                               for n, o in reverse.items())}
    messages = first_user_messages(task)
    task_cost = failed_cost = 0.0
    tool_calls_made = attempts = failed_attempts = 0
    final_calls: list[dict] = []
    stop_reason = ""
    step = 0

    while attempts < args.max_attempts:
        if spend["total"] >= args.max_spend_usd:
            raise SpendCapReached
        if task_cost >= args.max_task_spend_usd:
            break
        t0 = time.time()
        out, usage = client.call(SYSTEM_PROMPT, messages, tools, task, reverse)
        rate = resolve_rate(pricing, cfg["pricing_key"], prompt_tokens(usage))
        llm_cost = step_cost(usage, rate)
        attempts += 1
        stop_reason = out.get("stop_reason", "")
        log = StepLog(task["id"], task["category"], cfg_name, step, "llm", cfg["model"],
                      usage, llm_cost, latency_s=time.time() - t0)
        step_logs.append(log)
        task_cost += llm_cost
        spend["total"] += llm_cost
        step += 1

        calls = out.get("tool_calls", [])
        # tool-call budget: calls beyond the remaining budget are not accepted
        remaining = cfg["max_tool_calls"] - tool_calls_made
        accepted, dropped = calls[:max(remaining, 0)], calls[max(remaining, 0):]
        final_calls = [{**c, "name": reverse.get(c["name"], c["name"])} for c in accepted]
        if not calls or stop_reason == "refusal":
            break

        results, any_error = [], False
        for c in calls:
            err = "tool-call budget exhausted" if c in dropped else validate_call(
                {**c, "name": reverse.get(c["name"], c["name"])},
                {reverse[k]: v for k, v in functions_by_sanitized.items()})
            charge = 0.0 if c in dropped else tool_charge(pricing, c["name"])
            step_logs.append(StepLog(task["id"], task["category"], cfg_name, step, "tool",
                                     cfg["model"], {}, charge, c["name"], err is None, err or ""))
            task_cost += charge
            spend["total"] += charge
            step += 1
            if c not in dropped:
                tool_calls_made += 1
            any_error |= err is not None
            results.append({"type": "tool_result", "tool_use_id": c["id"], "is_error": err is not None,
                            "content": err or "Call recorded."})
        if not any_error or dropped or tool_calls_made >= cfg["max_tool_calls"]:
            break
        # failed attempt: its cost is charged to the task as a failed-step line
        failed_attempts += 1
        failed_cost += llm_cost
        log.ok = False
        log.error = "schema-invalid call; retried"
        messages = messages + [{"role": "assistant", "content": out["content"]},
                               {"role": "user", "content": results + [
                                   {"type": "text", "text": "Fix the invalid call(s) and call the function(s) again."}]}]

    success, why = grade(task, final_calls)
    return {
        "task_id": task["id"], "category": task["category"], "config": cfg_name,
        "success": success, "grade_error": why, "stop_reason": stop_reason,
        "cost_usd": round(task_cost, 8), "failed_attempt_cost_usd": round(failed_cost, 8),
        "llm_calls": attempts, "failed_attempts": failed_attempts,
        "tool_calls": tool_calls_made, "steps": step,
        "calls": [{"name": c["name"], "arguments": c.get("arguments")} for c in final_calls],
    }

# ---------------------------------------------------------------------------
# Analysis: results table, selection rule, Pareto figure
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def analyze(run_dir: Path, threshold: float = 0.9) -> dict:
    task_rows, step_rows = [], []
    for p in sorted(run_dir.glob("tasks_*.jsonl")):
        task_rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for p in sorted(run_dir.glob("steps_*.jsonl")):
        step_rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    by_cfg: dict[str, list[dict]] = {}
    for r in task_rows:
        by_cfg.setdefault(r["config"], []).append(r)

    summary = {}
    print(f"\n{'Configuration':<22}{'n':>5}{'Success':>9}{'95% CI':>15}{'Mean $/task':>13}"
          f"{'$/success':>11}{'f':>7}")
    for cfg, rows in sorted(by_cfg.items()):
        n = len(rows)
        succ = sum(r["success"] for r in rows)
        total = sum(r["cost_usd"] for r in rows)
        llm = [s for s in step_rows if s["config"] == cfg and s["kind"] == "llm"]
        failed = sum(r["failed_attempts"] for r in rows)
        tok = {k: sum(s["usage"].get(k, 0) for s in llm)
               for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}
        lo, hi = wilson(succ, n)
        per_cat = {}
        for r in rows:
            c = per_cat.setdefault(r["category"], {"n": 0, "success": 0, "cost_usd": 0.0})
            c["n"] += 1
            c["success"] += int(r["success"])
            c["cost_usd"] += r["cost_usd"]
        summary[cfg] = {
            "n": n, "successes": succ, "success_rate": succ / n if n else 0.0,
            "success_ci95": [lo, hi],
            "total_cost_usd": total,
            "mean_cost_per_task": total / n if n else 0.0,
            "cost_per_success": total / succ if succ else math.inf,
            "llm_calls": len(llm), "failed_attempts": failed,
            "failure_rate_f": failed / len(llm) if llm else 0.0,
            "failed_attempt_cost_usd": sum(r["failed_attempt_cost_usd"] for r in rows),
            "tokens": tok, "per_category": per_cat,
        }
        s = summary[cfg]
        ci = f"[{lo:.2f},{hi:.2f}]"
        cps = f"{s['cost_per_success']:.6f}" if succ else "n/a"
        print(f"{cfg:<22}{n:>5}{s['success_rate']:>9.1%}{ci:>15}"
              f"{s['mean_cost_per_task']:>13.6f}{cps:>11}{s['failure_rate_f']:>7.3f}")

    # Pareto frontier: configs not dominated on (lower cost, higher success)
    for c, s in summary.items():
        s["pareto"] = not any(
            o["mean_cost_per_task"] <= s["mean_cost_per_task"] and o["success_rate"] >= s["success_rate"]
            and (o["mean_cost_per_task"] < s["mean_cost_per_task"] or o["success_rate"] > s["success_rate"])
            for oc, o in summary.items() if oc != c)

    selection = {}
    if summary:
        best_cfg = max(summary, key=lambda c: summary[c]["success_rate"])
        best = summary[best_cfg]["success_rate"]
        eligible = [c for c, s in summary.items() if s["success_rate"] >= threshold * best]
        pick = min(eligible, key=lambda c: summary[c]["mean_cost_per_task"])
        saving = 1 - summary[pick]["mean_cost_per_task"] / max(summary[best_cfg]["mean_cost_per_task"], 1e-12)
        selection = {"threshold": threshold, "best_config": best_cfg, "best_success_rate": best,
                     "pick": pick, "pick_success_rate": summary[pick]["success_rate"],
                     "saving_vs_best_config": saving}
        print(f"\nSelection rule (>= {threshold:.0%} of best success {best:.1%}, best = '{best_cfg}'): "
              f"pick '{pick}' at ${summary[pick]['mean_cost_per_task']:.6f}/task, "
              f"{saving:.1%} cheaper per task than '{best_cfg}'.")

    (run_dir / "results_summary.json").write_text(
        json.dumps({"configs": summary, "selection": selection}, indent=2, default=str))
    with open(run_dir / "results_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["configuration", "n", "success_rate", "ci95_low", "ci95_high",
                    "mean_cost_per_task_usd", "cost_per_success_usd", "failure_rate_f", "pareto"])
        for cfg, s in sorted(summary.items()):
            w.writerow([cfg, s["n"], round(s["success_rate"], 4), round(s["success_ci95"][0], 4),
                        round(s["success_ci95"][1], 4), round(s["mean_cost_per_task"], 8),
                        round(s["cost_per_success"], 8) if s["cost_per_success"] != math.inf else "n/a",
                        round(s["failure_rate_f"], 4), s["pareto"]])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

        labels = {"frontier": "(a) Frontier", "small": "(b) Small",
                  "small_cache_budget": "(c) Small + cache + budget"}
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        pts = sorted(summary.items(), key=lambda kv: kv[1]["mean_cost_per_task"])
        front = [s for _, s in pts if s["pareto"]]
        ax.plot([s["mean_cost_per_task"] for s in front], [s["success_rate"] * 100 for s in front],
                color="#888888", linewidth=1, linestyle="--", zorder=1)
        xs = [s["mean_cost_per_task"] for _, s in pts]
        for idx, (cfg, s) in enumerate(pts):
            lo, hi = s["success_ci95"]
            x, y = s["mean_cost_per_task"], s["success_rate"] * 100
            ax.errorbar(x, y, yerr=[[y - lo * 100], [hi * 100 - y]], fmt="o", color="#1f4e79",
                        markersize=4, capsize=2, linewidth=1, zorder=2)
            right = idx == len(pts) - 1
            ax.annotate(labels.get(cfg, cfg), (x, y), textcoords="offset points",
                        xytext=(-6 if right else 6, 4 + 9 * (idx % 2)), fontsize=6.5,
                        ha="right" if right else "left")
        ax.set_ylim(top=max(s["success_rate"] * 100 for _, s in pts) + 9)
        ax.set_xscale("log")
        ax.set_xlim(min(xs) / 2.5, max(xs) * 2.5)
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
        ax.set_xlabel("Mean cost per task (USD, log scale)", fontsize=7.5)
        ax.set_ylabel("Task success (%)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, which="major", linewidth=0.3, alpha=0.5)
        fig.savefig(run_dir / "pareto.pdf", bbox_inches="tight")
        fig.savefig(run_dir / "pareto.png", dpi=200, bbox_inches="tight")
        print(f"Pareto figure: {run_dir / 'pareto.pdf'}")
    except ImportError:
        print("matplotlib not installed; skipping figure (CSV still written).")
    print(f"Table CSV: {run_dir / 'results_table.csv'}")
    return summary

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="agent-receipts single-turn harness")
    ap.add_argument("--config", choices=list(CONFIGS), help="single configuration to run")
    ap.add_argument("--all", action="store_true", help="run the three main configurations")
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS), help="explicit list of configurations")
    ap.add_argument("--tasks", default=str(HERE / "bfcl_v4.jsonl"), help="task JSONL (convert_bfcl.py)")
    ap.add_argument("--max-tasks", type=int, default=150)
    ap.add_argument("--max-attempts", type=int, default=3, help="LLM calls per task (1 + retries)")
    ap.add_argument("--max-task-spend-usd", type=float, default=2.0, help="per-task spend cap (hard)")
    ap.add_argument("--max-spend-usd", type=float, default=80.0, help="run-wide spend cap (hard stop)")
    ap.add_argument("--dry-run", action="store_true", help="zero-spend pipeline validation")
    ap.add_argument("--analyze", metavar="RUN_DIR", help="analyze an existing run and exit")
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="selection-rule quality bar as fraction of best success rate")
    ap.add_argument("--out", default="runs", help="output directory")
    args = ap.parse_args()

    if args.analyze:
        analyze(Path(args.analyze), args.threshold)
        return

    cfgs = args.configs or (MAIN_CONFIGS if args.all else [args.config or "frontier"])
    pricing = load_pricing()
    tasks = load_tasks(Path(args.tasks), args.max_tasks)
    if not tasks:
        sys.exit("no tasks loaded; check --tasks path")

    run_id = time.strftime("%Y%m%d-%H%M%S") + ("-dry" if args.dry_run else "")
    run_dir = HERE / args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    check = prompt_length_check(tasks, args.dry_run, cfgs)
    print(f"Prompt-length check: {check}")
    manifest_path = Path(args.tasks).with_name(Path(args.tasks).stem + "_manifest.json")
    run_meta = {
        "run_id": run_id, "dry_run": args.dry_run, "configs": cfgs,
        "effort": {c: CONFIGS[c].get("effort", EFFORT) for c in cfgs},
        "max_tokens": MAX_TOKENS, "max_attempts": args.max_attempts,
        "n_tasks": len(tasks), "category_mix": {},
        "prompt_length_check": check, "pricing": pricing,
        "dataset_manifest": json.loads(manifest_path.read_text()) if manifest_path.exists() else None,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for t in tasks:
        run_meta["category_mix"][t["category"]] = run_meta["category_mix"].get(t["category"], 0) + 1
    if not args.dry_run:
        import anthropic
        run_meta["anthropic_sdk_version"] = anthropic.__version__

    spend = {"total": 0.0}
    capped = False
    for cfg_name in cfgs:
        cfg = CONFIGS[cfg_name]
        client = make_client(cfg, args.dry_run)
        step_logs: list[StepLog] = []
        task_out = []
        for i, task in enumerate(tasks):
            try:
                row = run_task(task, cfg, cfg_name, pricing, client, args, step_logs, spend)
            except SpendCapReached:
                print(f"HARD STOP: run-wide cap ${args.max_spend_usd} reached.")
                capped = True
                break
            task_out.append(row)
            print(f"[{cfg_name}] {i+1}/{len(tasks)} {task['id']} success={row['success']} "
                  f"cost=${row['cost_usd']:.6f} attempts={row['llm_calls']}")
        with open(run_dir / f"tasks_{cfg_name}.jsonl", "w") as f:
            for r in task_out:
                f.write(json.dumps(r) + "\n")
        with open(run_dir / f"steps_{cfg_name}.jsonl", "w") as f:
            for s in step_logs:
                f.write(json.dumps(asdict(s)) + "\n")
        with open(run_dir / f"config_{cfg_name}.json", "w") as f:
            json.dump({"config": cfg_name, **cfg, "effort": cfg.get("effort", EFFORT),
                       "rates": pricing["models"][cfg["pricing_key"]]}, f, indent=2)
        if capped:
            break

    run_meta.update({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "total_spend_usd": spend["total"], "spend_cap_hit": capped})
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    print(f"\nRun {run_id}: total spend ${spend['total']:.4f} (cap ${args.max_spend_usd})")
    analyze(run_dir, args.threshold)


if __name__ == "__main__":
    main()
