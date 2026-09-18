"""Figure 3.3: country-specific feature configurations and stress-feature activation.

The figure is generated only from the aligned Stage 1 outputs for CN, ZA, MX and PT.
It exports publication-ready PNG/TIFF together with SVG/PDF and a traceable source CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "output" / "stages_split_aligned" / "stage1_feature_engineering"
RESULTS = OUT_ROOT / "results"
CONFIGS = OUT_ROOT / "recommendations" / "selected_configs"
FIG_DIR = OUT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = ["CN", "ZA", "MX", "PT"]
STRATEGIES = ["Hardcode", "Qwen", "Qwen–RAG"]
STRATEGY_KEYS = {"Hardcode": "hardcode", "Qwen": "qwen_base", "Qwen–RAG": "qwen_rag"}
COLORS = {
    "Hardcode": "#A9B8C6",
    "Qwen": "#DFA45B",
    "Qwen–RAG": "#3B718C",
}
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#222222",
        "axes.titleweight": "bold",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def load_configs() -> pd.DataFrame:
    rows = []
    for country in COUNTRIES:
        for strategy in STRATEGIES:
            key = STRATEGY_KEYS[strategy]
            path = CONFIGS / f"maize_{country}" / key / "selected_config.json"
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            cfg = payload["config"]
            rows.extend(
                [
                    {"country": country, "strategy": strategy, "parameter": "GDD base", "value": cfg["gdd_base_temp"]["maize"]},
                    {"country": country, "strategy": strategy, "parameter": "GDD upper", "value": cfg["gdd_upper_limit"]["maize"]},
                    {"country": country, "strategy": strategy, "parameter": "Tmin threshold", "value": cfg["stress_thresholds"]["tmin"]["threshold"]},
                    {"country": country, "strategy": strategy, "parameter": "Tmax threshold", "value": cfg["stress_thresholds"]["tmax"]["threshold"]},
                    {"country": country, "strategy": strategy, "parameter": "Precipitation threshold", "value": cfg["stress_thresholds"]["prec"]["threshold"]},
                ]
            )
    return pd.DataFrame(rows)


def load_event_rates() -> pd.DataFrame:
    rows = []
    prefixes = {
        "Low temperature": "tmin<",
        "High temperature": "tmax>",
        "Low precipitation": "prec<",
    }
    for country in COUNTRIES:
        path = RESULTS / f"maize_{country}" / "qwen_rag" / "feature_dataset.csv"
        frame = pd.read_csv(path)
        for label, prefix in prefixes.items():
            cols = [col for col in frame.columns if col.startswith(prefix)]
            values = frame[cols].to_numpy(dtype=float)
            rows.append(
                {
                    "country": country,
                    "event": label,
                    "nonzero_rate": 100.0 * np.count_nonzero(values) / values.size,
                }
            )
    return pd.DataFrame(rows)


def save_source_data(configs: pd.DataFrame, events: pd.DataFrame) -> None:
    config_out = configs.rename(columns={"parameter": "variable", "value": "value"}).copy()
    config_out["panel"] = "A"
    event_out = events.rename(columns={"event": "variable", "nonzero_rate": "value"}).copy()
    event_out["panel"] = "B"
    event_out["strategy"] = "Qwen–RAG"
    for frame in (config_out, event_out):
        for col in ["strategy", "variable", "value", "model", "criterion", "panel"]:
            if col not in frame:
                frame[col] = ""
    combined = pd.concat(
        [
            config_out[["panel", "country", "strategy", "variable", "value", "model", "criterion"]],
            event_out[["panel", "country", "strategy", "variable", "value", "model", "criterion"]],
        ],
        ignore_index=True,
    )
    combined.to_csv(FIG_DIR / "figure_3_3_source_data.csv", index=False, encoding="utf-8-sig")


def style_axis(ax) -> None:
    ax.grid(axis="y", color="#D9E0E0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, pad=2, labelsize=7)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")


def draw_parameter_panel(fig, gs, configs: pd.DataFrame) -> None:
    parameters = ["GDD base", "GDD upper", "Tmin threshold", "Tmax threshold", "Precipitation threshold"]
    titles = ["GDD base (°C)", "GDD upper (°C)", "Tmin threshold (°C)", "Tmax threshold (°C)", "Precipitation threshold (mm)"]
    ylimits = [(0, 17), (0, 53), (-4, 17), (28, 39), (-0.5, 7.5)]
    yticks = [[0, 5, 10, 15], [0, 20, 40], [-2, 5, 12], [28, 32, 36], [0, 2, 4, 6]]
    offsets = {"Hardcode": -0.22, "Qwen": 0.0, "Qwen–RAG": 0.22}
    axes = []
    for j, (parameter, title, ylim, ticks) in enumerate(zip(parameters, titles, ylimits, yticks)):
        slots = [gs[0, 0], gs[0, 1], gs[0, 2], gs[1, 0], gs[1, 1]]
        ax = fig.add_subplot(slots[j])
        axes.append(ax)
        style_axis(ax)
        for strategy in STRATEGIES:
            sub = configs[(configs["parameter"] == parameter) & (configs["strategy"] == strategy)]
            x = np.arange(len(COUNTRIES), dtype=float) + offsets[strategy]
            y = [sub.loc[sub["country"] == c, "value"].iloc[0] for c in COUNTRIES]
            ax.vlines(
                x,
                ylim[0],
                y,
                color=COLORS[strategy],
                linewidth=1.5,
                alpha=0.85,
                zorder=2,
            )
            ax.scatter(
                x,
                y,
                s=34,
                color=COLORS[strategy],
                edgecolor="#243C4A",
                linewidth=0.55,
                zorder=3,
                label=strategy,
            )
        ax.set_xlim(-0.55, len(COUNTRIES) - 0.45)
        ax.set_ylim(*ylim)
        ax.set_yticks(ticks)
        ax.set_xticks(range(len(COUNTRIES)))
        ax.set_xticklabels(COUNTRIES, fontsize=7)
        ax.set_ylabel(title, fontsize=6.8, labelpad=3)
        ax.tick_params(axis="y", labelsize=6.2)
    axes[0].text(-0.28, 1.22, "a", transform=axes[0].transAxes, fontsize=11, fontweight="bold", va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax = fig.add_subplot(gs[1, 2])
    legend_ax.axis("off")
    legend_ax.legend(
        handles[:3],
        labels[:3],
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=6.5,
        handletextpad=0.35,
        labelspacing=0.5,
        borderpad=0.2,
    )


def draw_event_panel(fig, slot, events: pd.DataFrame) -> None:
    ax = fig.add_subplot(slot)
    style_axis(ax)
    event_order = ["Low temperature", "High temperature", "Low precipitation"]
    event_colors = {"Low temperature": "#A9B8C6", "High temperature": "#DFA45B", "Low precipitation": "#3B718C"}
    x = np.arange(len(COUNTRIES), dtype=float)
    width = 0.22
    for i, event in enumerate(event_order):
        sub = events[events["event"] == event].set_index("country").loc[COUNTRIES]
        bars = ax.bar(x + (i - 1) * width, sub["nonzero_rate"], width=width, color=event_colors[event], edgecolor="#1F4B50", linewidth=0.55, label=event, zorder=3)
        for bar, value in zip(bars, sub["nonzero_rate"]):
            if value >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 2.0, f"{value:.1f}", ha="center", va="bottom", fontsize=6.5, color="#24484B")
            elif value > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 2.0, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2, color="#24484B")
    ax.set_title("Qwen–RAG stress-feature activation", fontsize=8.5, pad=5, loc="left")
    ax.set_ylabel("Non-zero entries (%)", fontsize=7.5, labelpad=3)
    ax.set_xticks(x)
    ax.set_xticklabels(COUNTRIES)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=7, handlelength=1.2, columnspacing=1.0)
    ax.text(-0.045, 1.14, "b", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def main() -> None:
    configs = load_configs()
    events = load_event_rates()
    save_source_data(configs, events)

    fig = plt.figure(figsize=(6.05, 4.95), facecolor="white")
    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1.18, 0.95],
        hspace=0.36,
        left=0.10,
        right=0.98,
        top=0.96,
        bottom=0.095,
    )
    parameter_gs = gs[0].subgridspec(2, 3, hspace=0.70, wspace=0.28)
    draw_parameter_panel(fig, parameter_gs, configs)
    draw_event_panel(fig, gs[1], events)

    fig.text(0.105, 0.035, "Panel b shows the proportion of non-zero Qwen–RAG stress-feature entries across country–period observations.", fontsize=6.8, color="#444444", ha="left")

    stem = FIG_DIR / "Figure_3_3_country_specific_feature_configurations"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
