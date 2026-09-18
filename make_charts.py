"""Render shareable charts directly from the published benchmark report.

Run: uv run --group charts make_charts.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "2026-09-18-benchmark.json"
OUTPUT = ROOT / "assets"
BG = "#F5F3ED"
INK = "#172B3A"
MUTED = "#536370"
COLORS = {"jev": "#12847B", "luna": "#C76832"}
MODELS = ("jev", "luna")


def canvas(subtitle):
    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor=BG)
    fig.text(.07, .927, "JEV vs Luna", fontsize=32, weight="bold", color=INK)
    fig.text(.07, .871, subtitle, fontsize=15, color=INK)
    fig.text(.07, .824, "100 synthetic reviews · 3 repeats · 50 cases with 2 contextual variants",
             fontsize=11.5, color=MUTED)
    fig.text(.07, .043, "github.com/mameli/jev-vs-luna", fontsize=11, color=INK)
    fig.text(.93, .043, "18 SEP 2026", ha="right", fontsize=10, color=MUTED)
    return fig


def clean_axis(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=10, colors=MUTED)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#DADDD9", linewidth=.7)


def overview(report):
    summary, derived = report["summary"], report["derived"]
    fig = canvas("Customer-review classification: accuracy, latency and cost")
    metrics = [
        ("Field accuracy", "Higher is better · failures count as wrong",
         [summary[m]["micro_accuracy"] * 100 for m in MODELS], 115,
         [0, 50, 100], ["0", "50", "100%"], lambda v: f"{v:.2f}%"),
        ("Median latency", "Lower is better · valid responses only",
         [summary[m]["latency_successes"]["median"] for m in MODELS], 2.0,
         [0, 1, 2], ["0", "1", "2 s"], lambda v: f"{v:.3f} s"),
        ("Cost per 1,000 calls", "Scaled from mean known cost · USD",
         [derived[m]["cost_per_known_attempt_usd"] * 1000 for m in MODELS], .2,
         [0, .1, .2], ["$0", "$0.10", "$0.20"], lambda v: f"${v:.4f}"),
    ]
    for i, (title, subtitle, values, xmax, ticks, ticklabels, fmt) in enumerate(metrics):
        left = .07 + i * .31
        fig.text(left, .722, title, fontsize=16, weight="bold", color=INK)
        fig.text(left, .678, subtitle, fontsize=9, color=MUTED)
        ax = fig.add_axes([left, .323, .255, .285])
        clean_axis(ax)
        for y, model, value in zip((1, 0), MODELS, values):
            ax.barh(y, value, height=.34, color=COLORS[model], zorder=3)
            ax.text(0, y + .27, "JEV" if model == "jev" else "Luna", color=COLORS[model],
                    fontsize=12, weight="bold")
            ax.text(xmax, y + .27, fmt(value), ha="right", fontsize=12, weight="bold", color=INK)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-.42, 1.65)
        ax.set_yticks([])
        ax.set_xticks(ticks, ticklabels)
    fig.text(.07, .226, "API errors: JEV 1/300 (HTTP 520) · Luna 0/300. Latency excludes the failed call.",
             fontsize=11, color=MUTED)
    fig.text(.07, .180, "Cost coverage including warmup: JEV 300/301 · Luna 301/301; failed-call cost unknown.",
             fontsize=11, color=MUTED)
    fig.text(.07, .122, "One synthetic test with ambiguous labels. These results do not establish a general model ranking.",
             fontsize=10.5, color=MUTED)
    fig.savefig(OUTPUT / "benchmark-overview.png", facecolor=BG)
    plt.close(fig)


def quality(report):
    summary = report["summary"]
    fig = canvas("Accuracy by field: where the models agreed with the expected labels")
    ax = fig.add_axes([.21, .26, .64, .48])
    clean_axis(ax)
    fields = ("topic", "sentiment", "rating", "needs_reply", "defect")
    labels = ("Topic", "Sentiment", "Inferred stars", "Needs reply", "Defect")
    for index, model in enumerate(MODELS):
        positions = [4 - i + (.18 if index == 0 else -.18) for i in range(5)]
        values = [summary[model]["accuracy_per_field"][field] * 100 for field in fields]
        ax.barh(positions, values, height=.28, color=COLORS[model], label="JEV" if model == "jev" else "Luna", zorder=3)
        for y, value in zip(positions, values):
            ax.text(value + 1, y, f"{value:.2f}%", va="center", color=INK, fontsize=10.5)
    ax.set_yticks(list(range(4, -1, -1)), labels, color=INK, fontsize=12)
    ax.tick_params(axis="y", pad=12)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.legend(loc="lower right", bbox_to_anchor=(1.02, 1.01), frameon=False, ncol=2, fontsize=12)
    fig.text(.07, .171, "300 measured attempts per model. API failures count as incorrect on every field.", fontsize=11, color=MUTED)
    fig.text(.07, .123, "JEV: one API failure. Both models classified defects correctly on all valid responses.", fontsize=11, color=MUTED)
    fig.savefig(OUTPUT / "accuracy-by-field.png", facecolor=BG)
    plt.close(fig)


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    OUTPUT.mkdir(exist_ok=True)
    overview(report)
    quality(report)
    print(f"Wrote two 1800 × 1200 PNG charts to {OUTPUT}")


if __name__ == "__main__":
    main()
