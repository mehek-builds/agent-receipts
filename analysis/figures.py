#!/usr/bin/env python3
"""Draw the article's figures from results/article_numbers.json.

Run after analysis/reproduce.py:  python analysis/figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N = json.loads((ROOT / "results" / "article_numbers.json").read_text())
OUT = ROOT / "results"

LABEL = {"frontier": "Sonnet 5.5\nno cache", "frontier_cache": "Sonnet 5.5\ncached",
         "small": "Haiku 5.5\nno cache", "small_cache_budget": "Haiku 5.5\ncached"}
PARTS = [("fresh_input", "Fresh input", "#4C6EF5"), ("cache_write", "Cache write", "#F08C00"),
         ("cache_read", "Cached input (read)", "#9BB0FF"), ("output", "Output", "#2B8A3E")]


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def composition():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cfgs = list(LABEL)
    bottom = [0.0] * len(cfgs)
    for key, name, color in PARTS:
        vals = [N[f"composition_{c}"][key]["usd_per_task"] for c in cfgs]
        ax.bar([LABEL[c] for c in cfgs], vals, bottom=bottom, color=color, label=name)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for x, b in enumerate(bottom):
        ax.text(x, b + 0.002, f"${b:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("Mean cost per task (USD)")
    ax.set_title(f"Figure 2. What each receipt is made of ({N['multi_turn_n_tasks']} multi-turn BFCL v4 tasks)")
    ax.legend(frameon=False)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig2-receipt-composition.png", dpi=200)


def break_even():
    be = N["break_even"]
    rows = [("Without prompt caching", be["uncached"], "#868E96"), ("With prompt caching", be["cached"], "#4C6EF5")]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for y, (name, b, color) in enumerate(rows):
        v, (lo, hi) = b["v_star_usd"], b["ci95_usd"]
        ax.barh(name, v, xerr=[[v - lo], [hi - v]], color=color, capsize=6)
        ax.text(hi + 0.02, y, f"{100 * v:.0f} cents (95% CI {100 * lo:.0f} to {100 * hi:.0f})", va="center", fontsize=9)
    ax.set_xlim(0, 1.4)
    ax.set_xlabel("Break-even value of one completed task (USD)")
    ax.set_title("Figure 3. Above this task value, Sonnet 5.5 beats Haiku 5.5")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig3-break-even.png", dpi=200)


if __name__ == "__main__":
    composition()
    break_even()
    print("wrote results/fig2-receipt-composition.png, results/fig3-break-even.png")
