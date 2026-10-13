"""Generate charts and a Markdown research report from the evaluation outputs.

Usage:
    python -m lobpred.report
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from lobpred.backtest import FEES_BPS, run_backtest  # noqa: E402
from lobpred.config import OOF_DIR, PRIMARY_HORIZON_S, PRIMARY_THRESHOLD, REPORT_DIR  # noqa: E402

# Validated categorical palette (first three slots pass all-pairs CVD checks) and chart ink.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MODEL_SERIES = ["XGBoost", "Logistic (all features)", "OFI only (logistic)"]
BASELINE = "Majority class"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
})


def _end_labels(ax, items, min_gap_px: float = 13) -> None:
    """Direct-label line ends, nudging labels apart vertically so they never overlap."""
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    placed = sorted(((to_px((x, y))[1], x, y, text) for x, y, text in items), key=lambda t: t[0])
    ys = [p[0] for p in placed]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap_px)
    shift = (ys[-1] - placed[-1][0]) / 2 if len(ys) > 1 else 0  # recentre the stack
    for (y_px, x, y, text), y_new in zip(placed, ys, strict=True):
        ax.annotate(text, (x, y), xytext=(6, y_new - shift - y_px), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)


def _legend_below(fig, ax, n: int) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n, fontsize=9, bbox_to_anchor=(0.5, 0.0))


def chart_predictability(results: dict, out: Path) -> None:
    horizons = list(results["metrics"])
    xs = [float(h.rstrip("s")) for h in horizons]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    panels = [("balanced_accuracy", "Balanced accuracy (out of sample)"),
              ("ic", "Information coefficient (signal vs. realised return)")]
    for ax, (metric, title) in zip(axes, panels, strict=True):
        base = [results["metrics"][h][BASELINE][metric]["mean"] for h in horizons]
        ax.plot(xs, base, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), label=BASELINE)
        ends = [(xs[-1], base[-1], BASELINE)]
        for name, color in zip(MODEL_SERIES, SERIES, strict=True):
            mean = [results["metrics"][h][name][metric]["mean"] for h in horizons]
            std = [results["metrics"][h][name][metric]["std"] for h in horizons]
            ax.errorbar(xs, mean, yerr=std, color=color, linewidth=2, marker="o", markersize=5,
                        elinewidth=1, capsize=0, label=name)
            ends.append((xs[-1], mean[-1], name))
        ax.set_xscale("log")
        ax.set_xticks(xs, [f"{x:g}s" for x in xs])
        ax.minorticks_off()
        ax.set_xlabel("Prediction horizon")
        ax.set_title(title)
        ax.set_xlim(xs[0] * 0.8, xs[-1] * 3.2)
        _end_labels(ax, ends)
    _legend_below(fig, axes[0], 4)
    fig.text(0.01, 0.075, "Mean across walk-forward folds; bars show one standard deviation.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def chart_importance(results: dict, horizon_key: str, out: Path) -> None:
    imp = results["importance"][horizon_key]
    names, values = list(imp)[::-1], list(imp.values())[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(names) + 1.2))
    ax.set_axisbelow(True)
    ax.barh(names, values, color=SERIES[0], height=0.6)
    ax.axvline(0, color=AXIS, linewidth=1)
    for name, v in zip(names, values, strict=True):
        ax.annotate(f"{v:+.3f}", (max(v, 0), name), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK_2, length=0)
    ax.set_xlabel("Drop in balanced accuracy when the feature is shuffled")
    ax.set_title(f"Feature importance, XGBoost, {horizon_key} horizon")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def chart_backtest(oof: pd.DataFrame, horizon_s: float, threshold: float, out: Path) -> None:
    trades = run_backtest(oof, horizon_s, threshold)
    if trades.empty:
        return
    hours = (trades["exit_ts"] - oof["ts"].iloc[0]) / 3.6e6
    series = [
        ("Gross (mid to mid)", trades["gross_bps"]),
        ("After crossing the spread", trades["after_spread_bps"]),
        (f"After spread and {FEES_BPS[1]:g} bp fee per side", trades["after_spread_bps"] - 2 * FEES_BPS[1]),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axhline(0, color=AXIS, linewidth=1)
    ends = []
    for (name, pnl), color in zip(series, SERIES, strict=True):
        cum = pnl.cumsum()
        ax.plot(hours, cum, color=color, linewidth=2, label=name)
        ends.append((hours.iloc[-1], cum.iloc[-1], name))
    ax.set_xlim(hours.iloc[0], hours.iloc[-1])
    ax.set_xlabel("Hours into the out-of-sample period")
    ax.set_ylabel("Cumulative return (bps)")
    ax.set_title(f"Backtest of out-of-sample signals: {horizon_s:g}s horizon, threshold {threshold:g}, "
                 f"{len(trades):,} trades")
    _legend_below(fig, ax, 3)
    fig.tight_layout(rect=(0, 0.07, 0.76, 1))
    _end_labels(ax, ends)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _fmt(m: dict) -> str:
    return f"{m['mean']:.3f} ± {m['std']:.3f}"


def write_report(results: dict, summary: dict | None, report_dir: Path, primary: str, charts: list[str]) -> Path:
    lines = ["# Results", "",
             f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `python -m lobpred.report`._", ""]

    if summary:
        lat_b, lat_f = summary["book_update_latency_us"], summary["feature_latency_us"]
        lines += [
            "## Data", "",
            "| | |", "|---|---|",
            f"| Rows | {summary['rows']:,} |",
            f"| Hours of data | {summary['hours_of_data']} |",
            f"| Continuous segments | {summary['segments']} |",
            f"| Book updates per second | {summary['book_updates_per_second']} |",
            f"| Price-level changes per second | {summary['level_changes_per_second']} |",
            f"| Trades | {summary['trades']:,} |",
            f"| Book update latency (p50 / p99) | {lat_b['p50']} / {lat_b['p99']} µs |",
            f"| Feature computation latency (p50 / p99) | {lat_f['p50']} / {lat_f['p99']} µs |",
            "",
        ]

    setup = results["setup"]
    lines += [
        "## Predictive performance", "",
        f"{setup['folds']}-fold expanding walk-forward with purging; mean ± std across folds. "
        "Balanced accuracy for three classes is 0.333 by chance.", "",
        "| Horizon | Model | Accuracy | Balanced accuracy | Macro F1 | Log loss | IC |",
        "|---|---|---|---|---|---|---|",
    ]
    for h, by_model in results["metrics"].items():
        for name, m in by_model.items():
            lines.append(f"| {h} | {name} | {_fmt(m['accuracy'])} | {_fmt(m['balanced_accuracy'])} | "
                         f"{_fmt(m['macro_f1'])} | {_fmt(m['log_loss'])} | {_fmt(m['ic'])} |")
    lines.append("")
    if "predictability_by_horizon.png" in charts:
        lines += ["![Predictability by horizon](predictability_by_horizon.png)", ""]

    if primary in results["importance"]:
        top = list(results["importance"][primary].items())[:5]
        lines += [f"## What drives the {primary} prediction", "",
                  "Permutation importance on the final fold (drop in balanced accuracy when shuffled):", ""]
        lines += [f"- `{f}`: {v:+.4f}" for f, v in top]
        lines.append("")
        if "feature_importance.png" in charts:
            lines += ["![Feature importance](feature_importance.png)", ""]

    lines += [
        "## Is the edge tradeable?", "",
        "Out-of-sample XGBoost signals traded with fixed rules: enter when |P(UP) − P(DOWN)| ≥ threshold, "
        "hold for the horizon, cross the spread both ways. Break-even fee is the per-side cost at which the "
        "average trade makes zero.", "",
        "| Horizon | Threshold | Trades/hour | Hit rate | Gross bps/trade | After spread bps/trade "
        "| Break-even fee (bps/side) |",
        "|---|---|---|---|---|---|---|",
    ]
    for h, by_thr in results["backtest"].items():
        for thr, s in by_thr.items():
            if s.get("trades", 0) == 0:
                lines.append(f"| {h} | {thr} | 0 | | | | |")
                continue
            lines.append(f"| {h} | {thr} | {s['trades_per_hour']} | {s['hit_rate']:.1%} | "
                         f"{s['mean_gross_bps']:+.3f} | {s['mean_after_spread_bps']:+.3f} | "
                         f"{s['breakeven_fee_bps_per_side']:+.3f} |")
    lines.append("")
    if "backtest.png" in charts:
        lines += ["![Backtest](backtest.png)", ""]

    lines += [
        "## Caveats", "",
        "- Results cover the recorded period only; short-horizon relationships change with market regime.",
        "- The backtest assumes fills at the recorded best bid/ask with no queueing, latency or market impact.",
        "- Fee levels are illustrative. Shorts are treated symmetrically, as on a perpetual future.",
        "",
    ]
    path = report_dir / "REPORT.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--oof-dir", type=Path, default=OOF_DIR)
    args = parser.parse_args()

    results = json.loads((args.report_dir / "results.json").read_text())
    summary_path = args.report_dir / "dataset_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None

    horizon_keys = list(results["metrics"])
    primary = f"{PRIMARY_HORIZON_S}s" if f"{PRIMARY_HORIZON_S}s" in horizon_keys else horizon_keys[0]
    primary_s = float(primary.rstrip("s"))

    charts = []
    if len(horizon_keys) > 1:
        chart_predictability(results, args.report_dir / "predictability_by_horizon.png")
        charts.append("predictability_by_horizon.png")
    chart_importance(results, primary, args.report_dir / "feature_importance.png")
    charts.append("feature_importance.png")
    oof_path = args.oof_dir / f"oof_{primary}.parquet"
    if oof_path.exists():
        chart_backtest(pd.read_parquet(oof_path), primary_s, PRIMARY_THRESHOLD, args.report_dir / "backtest.png")
        if (args.report_dir / "backtest.png").exists():
            charts.append("backtest.png")

    path = write_report(results, summary, args.report_dir, primary, charts)
    print(f"Wrote {path} and {len(charts)} charts to {args.report_dir}")


if __name__ == "__main__":
    main()
