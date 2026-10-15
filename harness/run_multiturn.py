#!/usr/bin/env python3
"""Multi-turn, tool-executing study with per-step cost attribution.

Runs BFCL v4 multi_turn_base tasks as real agent loops: each user turn is
answered by repeated LLM calls; every emitted tool call is executed on BFCL's
own Python simulators (vendored in bfcl_mt/), and the result, or the error, is
returned as a tool_result block. A step whose tool calls produce an error is a
failed step: its cost is a failed-step line on the task's receipt, and the model's next
step is the retry. Success follows BFCL's multi-turn checker: after every turn
with ground truth, the model's simulator state must equal the state produced by
the ground-truth calls, and the ground-truth execution results must appear among
the model's execution results so far.

Usage:
    python run_multiturn.py --configs frontier small small_cache_budget --dry-run
    python run_multiturn.py --configs frontier small small_cache_budget --max-spend-usd 40
    python run_multiturn.py --analyze runs_mt/<run-id>
"""
from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiment as R  # noqa: E402  (pricing, schema conversion, configs, wilson)
from bfcl_mt.gorilla_file_system import GorillaFileSystem  # noqa: E402
from bfcl_mt.math_api import MathAPI  # noqa: E402
from bfcl_mt.message_api import MessageAPI  # noqa: E402
from bfcl_mt.posting_api import TwitterAPI  # noqa: E402
from bfcl_mt.ticket_api import TicketAPI  # noqa: E402
from bfcl_mt.trading_bot import TradingBot  # noqa: E402
from bfcl_mt.travel_booking import TravelAPI  # noqa: E402
from bfcl_mt.vehicle_control import VehicleControlAPI  # noqa: E402

CLASSES = {c.__name__: c for c in (GorillaFileSystem, MathAPI, MessageAPI, TwitterAPI, TicketAPI,
                                   TradingBot, TravelAPI, VehicleControlAPI)}
FUNC_DOC = {"GorillaFileSystem": "gorilla_file_system.json", "MathAPI": "math_api.json",
            "MessageAPI": "message_api.json", "TwitterAPI": "posting_api.json",
            "TicketAPI": "ticket_api.json", "TradingBot": "trading_bot.json",
            "TravelAPI": "travel_booking.json", "VehicleControlAPI": "vehicle_control.json"}
STATELESS = {"MathAPI"}
MAX_STEPS_PER_TURN = 20  # BFCL's own per-turn step limit

SYSTEM_PROMPT = (
    "You are an assistant that completes the user's requests by calling the provided "
    "functions. Use the functions to act and to look up anything you need. When a request "
    "is done, reply to the user briefly without calling more functions."
)

# ---------------------------------------------------------------------------
# Simulators: instantiate, execute calls exactly as BFCL serializes results
# ---------------------------------------------------------------------------
def make_instances(task: dict) -> dict:
    inst = {}
    for cls in task["involved_classes"]:
        obj = CLASSES[cls]()
        if cls not in STATELESS:
            obj._load_scenario(copy.deepcopy(task["initial_config"].get(cls, {})), long_context=False)
        inst[cls] = obj
    return inst


def method_map(instances: dict) -> dict:
    """Method name -> bound method; later classes win on a name clash, as in BFCL."""
    out = {}
    for obj in instances.values():
        for name, m in inspect.getmembers(obj, predicate=inspect.ismethod):
            if not name.startswith("_"):
                out[name] = m
    return out


def serialize(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)
    return str(result)


def execute(methods: dict, name: str, args: list, kwargs: dict) -> tuple[str, bool]:
    """Run one call. Returns (BFCL-serialized result, is_error)."""
    if name in ("kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"):
        return f"Error during execution: Function call {name} is not allowed.", True
    fn = methods.get(name)
    if fn is None:
        return f"Error during execution: unknown function {name}", True
    try:
        res = fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 - tool errors are data
        return f"Error during execution: {e}", True
    is_err = isinstance(res, dict) and "error" in res
    return serialize(res), is_err


def parse_gt_call(s: str) -> tuple[str, list, dict]:
    """Parse a ground-truth call string like "mv(source='a', destination='b')" without eval."""
    node = ast.parse(s.strip(), mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"unsupported ground-truth call: {s}")
    return (node.func.id, [ast.literal_eval(a) for a in node.args],
            {k.arg: ast.literal_eval(k.value) for k in node.keywords})


def compare_instances(model_obj, gt_obj) -> bool:
    """BFCL state checker: every public attribute of the ground-truth instance must match."""
    for attr in vars(gt_obj):
        if attr.startswith("_"):
            continue
        if getattr(model_obj, attr) != getattr(gt_obj, attr):
            return False
    return True


def unordered_subset(needles: list, haystack: list) -> bool:
    pool = list(haystack)
    for item in needles:
        try:
            pool.remove(item)
        except ValueError:
            return False
    return True


def build_tools(task: dict) -> tuple[list[dict], dict[str, str]]:
    funcs = []
    for cls in task["involved_classes"]:
        doc = HERE / "bfcl_mt" / "func_doc" / FUNC_DOC[cls]
        funcs += [json.loads(l) for l in doc.read_text().splitlines() if l.strip()]
    funcs = [f for f in funcs if f["name"] not in set(task.get("excluded_function", []))]
    fwd = R.sanitize_names(funcs)
    tools = []
    for fn in funcs:
        schema = R.to_json_schema(fn.get("parameters", {"type": "dict", "properties": {}}))
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        tools.append({"name": fwd[fn["name"]], "description": fn.get("description", ""),
                      "input_schema": schema})
    return tools, {v: k for k, v in fwd.items()}

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
class MockClient:
    """Dry run: replays the turn's ground-truth calls, with a broken first attempt
    on some tasks to exercise failed steps. Synthetic usage; zero spend."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def call(self, system, messages, tools, ctx) -> tuple[dict, dict]:
        task, turn, step_in_turn = ctx["task"], ctx["turn"], ctx["step_in_turn"]
        hist = 900 + 350 * len(messages)
        cached = self.cfg["prompt_caching"]
        usage = {"input_tokens": 120 if cached else hist, "output_tokens": 160,
                 "cache_read_tokens": max(hist - 600, 0) if cached and len(messages) > 1 else 0,
                 "cache_write_tokens": 480 if cached else 0}
        gt = task["ground_truth"][turn]
        fwd = {v: k for k, v in ctx["reverse"].items()}
        broken = sum(map(ord, task["id"])) % 5 == 0 and turn == 0
        if step_in_turn == 0 and gt:
            calls = []
            for i, s in enumerate(gt):
                name, args, kwargs = parse_gt_call(s)
                if args:  # map positional ground-truth args onto parameter names
                    params = list(inspect.signature(ctx["methods"][name]).parameters)
                    kwargs = {**dict(zip(params, args)), **kwargs}
                if broken and i == 0:
                    kwargs = {"__bogus__": 1}
                calls.append({"name": fwd.get(name, name), "arguments": kwargs, "id": f"toolu_m{turn}_{i}"})
            return self._out(calls), usage
        if step_in_turn == 1 and broken and gt:
            name, args, kwargs = parse_gt_call(gt[0])
            if args:
                params = list(inspect.signature(ctx["methods"][name]).parameters)
                kwargs = {**dict(zip(params, args)), **kwargs}
            return self._out([{"name": fwd.get(name, name), "arguments": kwargs, "id": f"toolu_r{turn}"}]), usage
        return self._out([], "Done."), usage

    @staticmethod
    def _out(calls, text=""):
        content = [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]} for c in calls]
        if not calls:
            content = [{"type": "text", "text": text}]
        return {"tool_calls": calls, "text": text, "stop_reason": "tool_use" if calls else "end_turn",
                "content": content}


class AnthropicClient(R.AnthropicClient):
    def call(self, system, messages, tools, ctx) -> tuple[dict, dict]:
        msgs = messages
        if self.cfg["prompt_caching"]:
            # Moving breakpoint on the newest block: each step reads the cached
            # history and writes only what the last step added.
            msgs = copy.copy(messages)
            last = dict(msgs[-1])
            content = last["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            content = list(content)
            content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
            last["content"] = content
            msgs[-1] = last
        return super().call(system, msgs, tools, None, None)

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
@dataclass
class StepLog:
    task_id: str
    config: str
    turn: int
    step: int          # step index within the task (LLM steps only)
    step_in_turn: int
    model: str
    usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    kind: str = ""     # act | failed | final | limit
    n_calls: int = 0
    n_errors: int = 0
    n_budget_denied: int = 0
    latency_s: float = 0.0


def run_task(task, cfg, cfg_name, pricing, client, args, step_logs, spend) -> dict:
    tools, reverse = build_tools(task)
    model_inst = make_instances(task)
    gt_inst = make_instances(task)
    model_methods, gt_methods = method_map(model_inst), method_map(gt_inst)
    messages: list[dict] = []
    all_model_results: list[str] = []
    task_cost = failed_cost = 0.0
    steps = failed_steps = budget_denied = 0
    valid, why, stop = True, "", ""
    per_turn_cost = []

    for t_idx, turn in enumerate(task["turns"]):
        turn_text = "\n".join(m["content"] for m in turn if m["role"] == "user")
        messages = messages + [{"role": "user", "content": turn_text}]
        turn_calls = 0
        turn_cost = 0.0
        for k in range(MAX_STEPS_PER_TURN):
            if spend["total"] >= args.max_spend_usd:
                raise R.SpendCapReached
            t0 = time.time()
            out, usage = client.call(SYSTEM_PROMPT, messages, tools,
                                     {"task": task, "turn": t_idx, "step_in_turn": k,
                                      "reverse": reverse, "methods": model_methods})
            rate = R.resolve_rate(pricing, cfg["pricing_key"], R.prompt_tokens(usage))
            cost = R.step_cost(usage, rate)
            log = StepLog(task["id"], cfg_name, t_idx, steps, k, cfg["model"], usage, cost,
                          latency_s=time.time() - t0)
            step_logs.append(log)
            steps += 1
            task_cost += cost
            turn_cost += cost
            spend["total"] += cost
            stop = out.get("stop_reason", "")
            calls = out.get("tool_calls", [])
            messages = messages + [{"role": "assistant",
                                    "content": out["content"] or [{"type": "text", "text": "(no reply)"}]}]
            if not calls or stop == "refusal":
                log.kind = "final"
                break
            results = []
            for c in calls:
                name = reverse.get(c["name"], c["name"])
                if turn_calls >= cfg["max_tool_calls"]:
                    res, err = ("Tool-call budget for this request is exhausted. "
                                "Reply to the user without calling more functions."), True
                    log.n_budget_denied += 1
                    budget_denied += 1
                else:
                    res, err = execute(model_methods, name, [], dict(c.get("arguments") or {}))
                    all_model_results.append(res)
                    turn_calls += 1
                    log.n_calls += 1
                    log.n_errors += int(err)
                results.append({"type": "tool_result", "tool_use_id": c["id"],
                                "content": res[:4000] or "(empty result)", "is_error": err})
            log.kind = "failed" if log.n_errors else "act"
            if log.n_errors:
                failed_steps += 1
                failed_cost += cost
            messages = messages + [{"role": "user", "content": results}]
        else:
            step_logs[-1].kind = "limit"
        per_turn_cost.append(turn_cost)

        # BFCL checks after every turn that has ground truth
        gt_calls = task["ground_truth"][t_idx]
        gt_results = []
        for s in gt_calls:
            name, a, kw = parse_gt_call(s)
            gt_results.append(execute(gt_methods, name, a, kw)[0])
        if valid and gt_calls:
            if not all(compare_instances(model_inst[c], gt_inst[c]) for c in gt_inst):
                valid, why = False, f"state_mismatch@turn{t_idx}"
            elif not unordered_subset(gt_results, all_model_results):
                valid, why = False, f"response_mismatch@turn{t_idx}"
        if stop == "refusal":
            valid, why = False, "refusal"
            break

    return {"task_id": task["id"], "config": cfg_name, "success": valid, "grade_error": why,
            "turns": len(task["turns"]), "llm_steps": steps, "failed_steps": failed_steps,
            "budget_denied_calls": budget_denied, "cost_usd": round(task_cost, 8),
            "failed_step_cost_usd": round(failed_cost, 8),
            "per_turn_cost_usd": [round(x, 8) for x in per_turn_cost], "stop_reason": stop}

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(run_dirs: list[Path]) -> dict:
    tasks, steps = [], []
    for d in run_dirs:
        for p in sorted(d.glob("tasks_*.jsonl")):
            tasks += [dict(json.loads(l), run=d.name) for l in p.read_text().splitlines() if l.strip()]
        for p in sorted(d.glob("steps_*.jsonl")):
            steps += [dict(json.loads(l), run=d.name) for l in p.read_text().splitlines() if l.strip()]
    summary = {}
    for cfg in sorted({t["config"] for t in tasks}):
        rows = [t for t in tasks if t["config"] == cfg]
        st = [s for s in steps if s["config"] == cfg]
        n, succ = len(rows), sum(t["success"] for t in rows)
        total = sum(t["cost_usd"] for t in rows)
        lo, hi = R.wilson(succ, n)
        by_kind = {}
        for s in st:
            by_kind[s["kind"]] = by_kind.get(s["kind"], 0.0) + s["cost_usd"]
        tok = {k: sum(s["usage"].get(k, 0) for s in st)
               for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}
        rate = R.load_pricing()["models"][R.CONFIGS[cfg]["pricing_key"]]
        rate = rate.get("tier_low", rate)
        cat_cost = {"input": tok["input_tokens"] * rate["input_per_mtok"] / 1e6,
                    "output": tok["output_tokens"] * rate["output_per_mtok"] / 1e6,
                    "cache_read": tok["cache_read_tokens"] * rate["cache_read_per_mtok"] / 1e6,
                    "cache_write": tok["cache_write_tokens"] * rate["cache_write_5min_per_mtok"] / 1e6}
        ok_rows, bad_rows = [t for t in rows if t["success"]], [t for t in rows if not t["success"]]
        runs = sorted({t["run"] for t in rows})
        per_run = []
        for r in runs:
            rr = [t for t in rows if t["run"] == r]
            per_run.append({"run": r, "success_rate": sum(t["success"] for t in rr) / len(rr),
                            "mean_cost_per_task": sum(t["cost_usd"] for t in rr) / len(rr)})
        max_turns = max(t["turns"] for t in rows)
        turn_cost = []
        for i in range(max_turns):
            xs = [t["per_turn_cost_usd"][i] for t in rows if t["per_turn_cost_usd"] and len(t["per_turn_cost_usd"]) > i]
            turn_cost.append(sum(xs) / len(xs) if xs else None)
        summary[cfg] = {
            "n_task_runs": n, "runs": runs, "successes": succ, "success_rate": succ / n,
            "success_ci95": [lo, hi], "total_cost_usd": total, "mean_cost_per_task": total / n,
            "cost_per_success": total / succ if succ else math.inf,
            "llm_steps": sum(t["llm_steps"] for t in rows), "failed_steps": sum(t["failed_steps"] for t in rows),
            "failure_rate_f": sum(t["failed_steps"] for t in rows) / max(sum(t["llm_steps"] for t in rows), 1),
            "failed_step_cost_usd": sum(t["failed_step_cost_usd"] or 0.0 for t in rows),
            "failed_step_cost_share": sum(t["failed_step_cost_usd"] or 0.0 for t in rows) / max(total, 1e-12),
            "tasks_with_failed_step": sum(1 for t in rows if t["failed_steps"]),
            "budget_denied_calls": sum(t["budget_denied_calls"] for t in rows),
            "cost_by_step_kind": by_kind, "tokens": tok, "cost_by_token_category": cat_cost,
            "mean_cost_successful_task": sum(t["cost_usd"] for t in ok_rows) / max(len(ok_rows), 1),
            "mean_cost_failed_task": sum(t["cost_usd"] for t in bad_rows) / max(len(bad_rows), 1),
            "mean_steps_per_task": sum(t["llm_steps"] for t in rows) / n, "mean_cost_by_turn": turn_cost, "per_run": per_run,
            "grade_errors": {e: sum(1 for t in bad_rows if (t["grade_error"] or "unknown").split("@")[0] == e)
                             for e in sorted({(t["grade_error"] or "unknown").split("@")[0] for t in bad_rows})},
            "rows_from_console_log": sum(1 for t in rows if t.get("source")),
        }
        s = summary[cfg]
        print(f"{cfg:<20} n={n:<4} success={s['success_rate']:.1%} [{lo:.2f},{hi:.2f}] "
              f"$/task={s['mean_cost_per_task']:.5f} $/succ={s['cost_per_success']:.5f} "
              f"f={s['failure_rate_f']:.3f} retry_share={s['failed_step_cost_share']:.1%} "
              f"steps/task={s['mean_steps_per_task']:.1f} ok=${s['mean_cost_successful_task']:.5f} "
              f"fail=${s['mean_cost_failed_task']:.5f}")
    for c, s in summary.items():
        s["pareto"] = not any(
            o["mean_cost_per_task"] <= s["mean_cost_per_task"] and o["success_rate"] >= s["success_rate"]
            and (o["mean_cost_per_task"] < s["mean_cost_per_task"] or o["success_rate"] > s["success_rate"])
            for oc, o in summary.items() if oc != c)
    out = run_dirs[0] if len(run_dirs) == 1 else run_dirs[0].parent
    name = "results_summary.json" if len(run_dirs) == 1 else "results_summary_pooled.json"
    (out / name).write_text(json.dumps(summary, indent=2, default=str))
    return summary

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="agent-receipts multi-turn harness")
    ap.add_argument("--configs", nargs="+", choices=list(R.CONFIGS), default=R.MAIN_CONFIGS)
    ap.add_argument("--tasks", default=str(HERE / "bfcl_v4_multi_turn.jsonl"))
    ap.add_argument("--max-tasks", type=int, default=50)
    ap.add_argument("--max-spend-usd", type=float, default=40.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze", nargs="+", metavar="RUN_DIR")
    ap.add_argument("--out", default="runs_mt")
    ap.add_argument("--resume", metavar="RUN_DIR", help="continue a run: skip tasks already logged per config")
    args = ap.parse_args()

    if args.analyze:
        analyze([Path(p) for p in args.analyze])
        return
    pricing = R.load_pricing()
    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()][: args.max_tasks]
    if args.resume:
        run_dir = Path(args.resume)
        run_id = run_dir.name
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S") + ("-dry" if args.dry_run else "")
        run_dir = HERE / args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "run_meta.json"
    prior = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta = {"run_id": run_id, "dry_run": args.dry_run,
            "configs": list(dict.fromkeys(prior.get("configs", []) + args.configs)),
            "effort": {c: R.CONFIGS[c].get("effort", R.EFFORT) for c in R.CONFIGS},
            "max_tokens": R.MAX_TOKENS, "max_steps_per_turn": MAX_STEPS_PER_TURN,
            "n_tasks": len(tasks), "pricing": pricing,
            "started_utc": prior.get("started_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            "sessions": prior.get("sessions", []) + [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())]}
    if not args.dry_run:
        import anthropic
        meta["anthropic_sdk_version"] = anthropic.__version__
    prior_spend = float(prior.get("total_spend_usd", 0.0))
    spend, stopped = {"total": 0.0}, ""
    for cfg_name in args.configs:
        cfg = R.CONFIGS[cfg_name]
        client = MockClient(cfg) if args.dry_run else AnthropicClient(cfg)
        tpath, spath = run_dir / f"tasks_{cfg_name}.jsonl", run_dir / f"steps_{cfg_name}.jsonl"
        done = {json.loads(l)["task_id"] for l in tpath.read_text().splitlines() if l.strip()} if tpath.exists() else set()
        for i, task in enumerate(tasks):
            if task["id"] in done:
                continue
            step_logs: list[StepLog] = []
            try:
                row = run_task(task, cfg, cfg_name, pricing, client, args, step_logs, spend)
            except R.SpendCapReached:
                stopped = f"run-wide cap ${args.max_spend_usd} reached"
            except Exception as e:  # noqa: BLE001 - e.g. API errors: keep what is already persisted
                if args.dry_run or not hasattr(e, "status_code"):
                    raise
                stopped = f"API error during {cfg_name} {task['id']}: {e}"
            if stopped:
                print(f"STOP: {stopped}")
                break
            # Persist each task as soon as it finishes, so an interrupted run keeps its data.
            with open(tpath, "a") as f:
                f.write(json.dumps(row) + "\n")
            with open(spath, "a") as f:
                f.writelines(json.dumps(asdict(s)) + "\n" for s in step_logs)
            print(f"[{cfg_name}] {i+1}/{len(tasks)} {task['id']} success={row['success']} "
                  f"cost=${row['cost_usd']:.5f} steps={row['llm_steps']} failed={row['failed_steps']}", flush=True)
        if stopped:
            break
    meta.update({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "total_spend_usd": prior_spend + spend["total"], "stopped": stopped or None})
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nRun {run_id}: spend this session ${spend['total']:.4f}, total ${prior_spend + spend['total']:.4f}")
    analyze([run_dir])


if __name__ == "__main__":
    main()
