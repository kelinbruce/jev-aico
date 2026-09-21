"""Create bar charts from saved Qwen/Jev evaluation results without rerunning models."""

import argparse
import json
import os
from pathlib import Path

from nimble.paths import PROJECT_ROOT

# Keep font caches inside the workspace, including in sandboxed sessions.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


COLORS = {"qwen": "#2563EB", "teacher": "#D97706"}
LABELS = {"qwen": "Qwen3.5-4B", "teacher": "Jev"}


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DFE5EC", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="both", length=0, labelcolor="#475569", pad=8)
    ax.set_facecolor("white")


def plot_comparison(result, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titleweight": "bold", "axes.titlecolor": "#172554",
                         "svg.fonttype": "none"})
    fig = plt.figure(figsize=(14, 9.5), facecolor="white")
    grid = fig.add_gridspec(2, 4, height_ratios=[1.6, 1], hspace=0.65, wspace=0.5)
    fig.subplots_adjust(left=0.065, right=0.975, bottom=0.17, top=0.81)
    count = len(result["rows"])
    models = sorted({row["reference"].get("model", row["reference"].get("source", "unspecified"))
                     for row in result["rows"]})
    fig.text(0.065, 0.962, "Qwen vs. Jev", fontsize=25, weight="bold", color="#172554")
    fig.text(0.065, 0.926, Path(result["dataset"]).parent.name, fontsize=13, color="#475569")
    fig.text(0.065, 0.896,
             f"{count} examples · all splits · unchanged Qwen, temperature 1.0 · saved Jev answers",
             fontsize=11, color="#475569")

    ax = fig.add_subplot(grid[0, :])
    style_axis(ax)
    ax.set_title("Agreement with reference labels  |  higher is better", loc="left", pad=14)
    kinds = ["all", "choice", "noul", "score"]
    names = ["Overall", "Choice", "Boolean (Noul)", "Ordered score level"]
    for model, shift in [("qwen", -0.19), ("teacher", 0.19)]:
        stats = [result["summary"][model][kind] for kind in kinds]
        values = [s["accuracy"] * 100 for s in stats]
        bars = ax.bar([i + shift for i in range(4)], values, width=0.34,
                      color=COLORS[model], label=LABELS[model], zorder=3)
        for bar, value, stat in zip(bars, values, stats):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.8,
                    f"{value:.1f}%\n{stat['correct']}/{stat['count']}",
                    ha="center", va="bottom", fontsize=10, color="#1E293B", linespacing=1.4)
    ax.set_xticks(range(4), names)
    ax.set_ylim(0, 119)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    fig.legend(*ax.get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.982, 0.971),
               frameon=False, ncol=2)

    metrics = [
        ("Multiclass Brier", "All examples · lower is better", "all", "mean_multiclass_brier"),
        ("Log loss", "All examples · lower is better", "all", "mean_negative_log_likelihood"),
        ("Binary Brier", "Boolean only · lower is better", "noul", "mean_binary_brier"),
        ("Expected-score MAE", "Score only · lower is better", "score", "mean_absolute_score_error"),
    ]
    for index, (title, subtitle, kind, metric) in enumerate(metrics):
        ax = fig.add_subplot(grid[1, index])
        style_axis(ax)
        ax.set_title(title, loc="left", fontsize=12, pad=30)
        ax.text(0, 1.06, subtitle, transform=ax.transAxes, color="#64748B", fontsize=9)
        values = [result["summary"][model][kind][metric] for model in ("qwen", "teacher")]
        bars = ax.bar([0, 1], values, color=[COLORS["qwen"], COLORS["teacher"]], width=0.58, zorder=3)
        ceiling = max(max(values) * 1.28, 0.01)
        ax.set_ylim(0, ceiling)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xticks([0, 1], ["Qwen", "Jev"])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + ceiling * 0.025,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=11, color="#1E293B")

    reviewed = sum(row["reference"].get("human_reviewed", False) for row in result["rows"])
    zeros = sum(row["teacher"]["reference_probability"] == 0 for row in result["rows"])
    fig.text(0.065, 0.09, f"Reference source: {', '.join(models)}. Human-reviewed labels: {reviewed}/{count}.",
             fontsize=10, color="#475569")
    fig.text(0.065, 0.062, "Agreement with provisional labels is not established real-world accuracy. Error panels use separate scales.",
             fontsize=10, color="#475569")
    fig.text(0.065, 0.034, f"Log loss clips probabilities at 1e-15; Jev assigns zero probability to {zeros} reference answers.",
             fontsize=10, color="#475569")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"comparison.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    output_dir = args.output_dir or args.results.parent
    plot_comparison(result, output_dir)
    print(f"Saved comparison.png, comparison.svg, and comparison.pdf in {output_dir}")


if __name__ == "__main__":
    main()
