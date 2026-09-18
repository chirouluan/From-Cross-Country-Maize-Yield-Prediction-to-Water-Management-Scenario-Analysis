"""Create the Results-section figures for the CY-Bench manuscript.

Figures 2, 4 and 6 were split into standalone scripts under ``visualizations/``;
this script now produces the remaining two:

    fig_3_rag_feature_engineering_performance   (create_figure_32)
    fig_5_agro_tabpfn_transfer                  (create_figure_34)

All quantitative values are read from the completed Stage 1/2/3 output tree.
The script exports editable SVG, 600-dpi PNG/TIFF, and compact source-data CSVs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, to_rgba
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_RUN = ROOT / "cybench/output/stages_abl_heat_hist"
STAGE1 = PRIMARY_RUN / "stage1_feature_engineering"
STAGE2 = PRIMARY_RUN / "stage2_tabpfn_transfer/evaluation"
STAGE3 = PRIMARY_RUN / "stage3_advisory"
OUT = ROOT / "figures/manuscript_results"
SOURCE = OUT / "source_data"

COUNTRIES = ["CN", "ZA", "MX", "PT"]
COUNTRY_NAMES = {"CN": "China", "ZA": "South Africa", "MX": "Mexico", "PT": "Portugal"}
COUNTRY_COLORS = {"CN": "#376F9E", "ZA": "#4E9A8D", "MX": "#D89B45", "PT": "#8B72A8"}

ADVISORS = ["hardcode", "qwen_base", "qwen_rag"]
ADVISOR_LABELS = {"hardcode": "Hardcode", "qwen_base": "Qwen", "qwen_rag": "Qwen-RAG"}
ADVISOR_COLORS = {"hardcode": "#B8B8B8", "qwen_base": "#8BA8C7", "qwen_rag": "#315F8C"}

MODELS = ["ridge", "svr", "xgboost", "cnn1d", "transformer_flat"]
MODEL_LABELS = {
    "ridge": "Ridge",
    "svr": "SVR",
    "xgboost": "XGBoost",
    "cnn1d": "CNN1D",
    "transformer_flat": "Transformer",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.13, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")


def finish(fig: mpl.figure.Figure, stem: str, transparent: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    facecolor = "none" if transparent else "white"
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor=facecolor, transparent=transparent)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor=facecolor, transparent=transparent)
    fig.savefig(
        OUT / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor=facecolor,
        transparent=transparent,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def stage1_dataset(country: str, advisor: str = "hardcode") -> pd.DataFrame:
    path = STAGE1 / f"results/maize_{country}/{advisor}/feature_dataset.csv"
    return pd.read_csv(path)


def numbered_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda x: int(re.search(r"(\d+)$", x).group(1)))


def create_figure_32() -> None:
    """Grouped five-model bars for R2 and MAPE, with SD and repeat points."""
    df = pd.read_csv(STAGE1 / "results/metrics_all.csv")
    df = df[(df["split"] == "test") & df.country.isin(COUNTRIES) & df.advisor.isin(ADVISORS)].copy()
    expected = len(COUNTRIES) * len(ADVISORS) * len(MODELS) * 5
    if len(df) != expected:
        raise ValueError(f"Expected {expected} Stage 1 test rows, found {len(df)}")
    repeats = df.groupby(["country", "advisor", "model"])["repeat"].nunique()
    if not repeats.eq(5).all():
        raise ValueError("Every country/advisor/model combination must contain five repeats")
    df["mape_percent"] = df["mape"] * 100
    df.to_csv(SOURCE / "fig_3_stage1_test_repeats.csv", index=False)

    model_colors = {
        "ridge": "#4E79A7",
        "svr": "#59A14F",
        "xgboost": "#E39C37",
        "cnn1d": "#A06AA1",
        "transformer_flat": "#7A7A7A",
    }
    model_short = {"ridge": "Ridge", "svr": "SVR", "xgboost": "XGB", "cnn1d": "CNN1D", "transformer_flat": "Trans."}
    advisor_alpha = {"hardcode": 0.32, "qwen_base": 0.62, "qwen_rag": 0.95}

    # Negative R2 values remain in the exported source data for traceability,
    # but are omitted from the plotted points and summary statistics.
    plotted_r2 = df.loc[df.r2 >= 0, "r2"]
    if plotted_r2.empty:
        raise ValueError("No non-negative R2 values are available for Figure 3")
    global_r2_min = 0.0
    global_r2_max = max(0.8, np.ceil((df.r2.max() + 0.03) * 10) / 10)

    def nice_mape_upper(value: float) -> float:
        raw = value * 1.10
        step = 5 if raw <= 30 else 10
        return float(np.ceil(raw / step) * step)

    def add_group_header(ax: mpl.axes.Axes, left: float, right: float, label: str) -> None:
        trans = ax.get_xaxis_transform()
        ax.text(
            (left + right) / 2, 0.965, label, transform=trans, ha="center", va="top",
            fontsize=12.0, fontweight="bold", clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
            zorder=7,
        )

    fig, axes = plt.subplots(4, 1, figsize=(14.0, 13.1), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    bar_width = 0.22
    metric_offset = 0.135
    model_step = 0.50
    group_gap = 0.32
    group_span = (len(MODELS) - 1) * model_step
    group_width = group_span + 0.30
    fixed_jitter = np.linspace(-0.045, 0.045, 5)

    for panel_i, (ax, country) in enumerate(zip(axes, COUNTRIES)):
        ax_mape = ax.twinx()
        country_df = df[df.country == country]
        all_centers = []
        xtick_labels = []

        for advisor_i, advisor in enumerate(ADVISORS):
            start = advisor_i * (group_width + group_gap)
            centers = start + np.arange(len(MODELS)) * model_step
            all_centers.extend(centers)
            xtick_labels.extend([model_short[m] for m in MODELS])
            add_group_header(ax, centers[0] - 0.18, centers[-1] + 0.18, ADVISOR_LABELS[advisor])

            if advisor_i > 0:
                boundary = start - group_gap / 2
                ax.axvline(boundary, color="#D7D7D7", lw=0.6, zorder=0)

            for center, model in zip(centers, MODELS):
                sub = country_df[(country_df.advisor == advisor) & (country_df.model == model)].sort_values("repeat")
                r2_values = sub.loc[sub.r2 >= 0, "r2"].to_numpy()
                mape_values = sub.mape_percent.to_numpy()
                color = model_colors[model]

                if len(r2_values):
                    r2_sd = r2_values.std(ddof=1) if len(r2_values) > 1 else 0.0
                    ax.bar(
                        center - metric_offset, r2_values.mean(), width=bar_width,
                        color=to_rgba(color, advisor_alpha[advisor]), edgecolor="#333333", linewidth=0.75,
                        yerr=r2_sd, capsize=4.0,
                        error_kw={"elinewidth": 1.45, "capthick": 1.45}, zorder=2,
                    )
                ax_mape.bar(
                    center + metric_offset, mape_values.mean(), width=bar_width,
                    color=to_rgba(color, 0.16 + 0.20 * advisor_alpha[advisor]),
                    edgecolor=to_rgba(color, 0.92), linewidth=1.1,
                    yerr=mape_values.std(ddof=1), capsize=4.0,
                    error_kw={"elinewidth": 1.45, "capthick": 1.45}, zorder=2,
                )
                if len(r2_values):
                    r2_jitter = np.linspace(-0.045, 0.045, len(r2_values))
                    ax.scatter(
                        center - metric_offset + r2_jitter, r2_values, s=36,
                        facecolor=to_rgba(color, 0.82), edgecolor="#333333", linewidth=0.7, zorder=4,
                    )
                ax_mape.scatter(
                    center + metric_offset + fixed_jitter, mape_values, s=36,
                    facecolor="white", edgecolor=to_rgba(color, 0.95), linewidth=1.0, zorder=4,
                )

        mape_upper = nice_mape_upper(country_df.mape_percent.max())
        ax.set_xlim(min(all_centers) - 0.28, max(all_centers) + 0.28)
        ax.set_ylim(global_r2_min, global_r2_max)
        ax_mape.set_ylim(0, mape_upper)
        ax.set_xticks(all_centers, xtick_labels, rotation=52, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0, pad=4, labelsize=9.5)
        ax.tick_params(axis="y", direction="out", length=4.0, width=1.0, labelleft=True, labelsize=10.0)
        ax_mape.tick_params(axis="y", direction="out", length=4.0, width=1.0, colors="black", labelsize=10.0)
        ax.axhline(0, color="#555555", lw=0.65, zorder=1)
        ax.set_title(COUNTRY_NAMES[country] + f" ({country})", fontsize=14.0, fontweight="bold", y=1.035, pad=0)
        ax.set_ylabel(r"Test $R^2$", fontsize=11.5, labelpad=7)
        ax_mape.set_ylabel("MAPE (%)", color="black", fontsize=11.5, labelpad=8)
        ax.spines["right"].set_visible(False)
        ax_mape.spines["top"].set_visible(False)
        ax_mape.spines["left"].set_visible(False)
        ax_mape.spines["right"].set_visible(True)
        ax_mape.spines["right"].set_color("black")
        ax_mape.spines["right"].set_linewidth(1.0)
        panel_label(ax, chr(ord("a") + panel_i), x=-0.045, y=1.045)

    fig.subplots_adjust(left=0.065, right=0.935, top=0.97, bottom=0.055, hspace=0.34)
    finish(fig, "fig_3_rag_feature_engineering_performance")


def read_config(country: str, advisor: str) -> dict:
    path = STAGE1 / f"recommendations/selected_configs/maize_{country}/{advisor}/selected_config.json"
    return json.loads(path.read_text(encoding="utf-8"))["config"]


def _create_figure_33_heatmap_legacy() -> None:
    """Numerical feature policies, event-channel activation, and aggregation rules."""
    rows = []
    labels = []
    for advisor, countries in [("hardcode", ["CN"]), ("qwen_base", ["CN"]), ("qwen_rag", COUNTRIES)]:
        for country in countries:
            c = read_config(country, advisor)
            rows.append([
                c["gdd_base_temp"]["maize"], c["gdd_upper_limit"]["maize"],
                c["stress_thresholds"]["tmin"]["threshold"], c["stress_thresholds"]["tmax"]["threshold"],
                c["stress_thresholds"]["prec"]["threshold"],
            ])
            labels.append(ADVISOR_LABELS[advisor] if advisor != "qwen_rag" else f"RAG-{country}")
    config_df = pd.DataFrame(rows, index=labels, columns=["GDD base", "GDD upper", "Cold threshold", "Heat threshold", "Rain threshold"])

    activation_rows = []
    aggregation_rows = []
    for country in COUNTRIES:
        df = stage1_dataset(country, "qwen_rag")
        acts = []
        for prefix in ["tmin<", "tmax>", "prec<"]:
            cols = [c for c in df if c.startswith(prefix)]
            acts.append((df[cols].to_numpy() != 0).mean() * 100)
        activation_rows.append(acts)
        c = read_config(country, "qwen_rag")
        aggregation_rows.append([c["veg_agg_method"], c["soil_moisture_agg_method"]])
    activation = pd.DataFrame(activation_rows, index=COUNTRIES, columns=["Cold", "Heat", "Low rain"])
    aggregation = pd.DataFrame(aggregation_rows, index=COUNTRIES, columns=["NDVI/FPAR", "Soil moisture"])
    config_df.reset_index(names="policy").to_csv(SOURCE / "fig_4_feature_policy_parameters.csv", index=False)
    activation.reset_index(names="country").to_csv(SOURCE / "fig_4_event_activation.csv", index=False)
    aggregation.reset_index(names="country").to_csv(SOURCE / "fig_4_aggregation_rules.csv", index=False)

    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.2, 0.8], hspace=0.48, wspace=0.38)
    ax_a = fig.add_subplot(gs[:, 0]); ax_b = fig.add_subplot(gs[0, 1]); ax_c = fig.add_subplot(gs[1, 1])
    z = config_df.sub(config_df.mean()).div(config_df.std(ddof=0).replace(0, 1))
    im = ax_a.imshow(z.values, cmap="PuBuGn", aspect="auto", vmin=-1.8, vmax=1.8)
    ax_a.set_xticks(np.arange(5), ["GDD base\n(°C)", "GDD upper\n(°C)", "Cold rule\n(°C)", "Heat rule\n(°C)", "Rain rule\n(mm)"], rotation=0)
    ax_a.set_yticks(np.arange(len(labels)), labels)
    for i in range(config_df.shape[0]):
        for j in range(config_df.shape[1]):
            ax_a.text(j, i, f"{config_df.iloc[i, j]:g}", ha="center", va="center", fontsize=6.4,
                      color="white" if z.iloc[i, j] > 0.8 else "#222222")
    ax_a.tick_params(length=0); ax_a.set_title("Country-specific numerical policies", loc="left"); panel_label(ax_a, "a")

    im2 = ax_b.imshow(activation.values, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax_b.set_xticks(np.arange(3), activation.columns); ax_b.set_yticks(np.arange(4), COUNTRIES)
    for i in range(4):
        for j in range(3):
            v = activation.iloc[i, j]
            ax_b.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=6.2, color="white" if v > 55 else "#222222")
    ax_b.tick_params(length=0); ax_b.set_title("Non-zero event cells", loc="left"); panel_label(ax_b, "b")

    agg_codes = {"mean": 0, "median": 1, "max": 2}
    agg_num = aggregation.apply(lambda column: column.map(agg_codes)).astype(float)
    cmap = mpl.colors.ListedColormap(["#DDE7F2", "#AFC6DD", "#5D88B0"])
    ax_c.imshow(agg_num.values, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
    ax_c.set_xticks(np.arange(2), aggregation.columns); ax_c.set_yticks(np.arange(4), COUNTRIES)
    for i in range(4):
        for j in range(2): ax_c.text(j, i, aggregation.iloc[i, j], ha="center", va="center", fontsize=6.3)
    ax_c.tick_params(length=0); ax_c.set_title("Temporal aggregation", loc="left"); panel_label(ax_c, "c")
    finish(fig, "fig_4_country_specific_feature_policies")


def paired_metric(ax: mpl.axes.Axes, df: pd.DataFrame, metric: str, ylabel: str, letter: str) -> None:
    models = ["tabpfn_base", "tabpfn_cn_finetuned"]
    x = np.arange(4)
    for i, country in enumerate(COUNTRIES):
        sub = df[df.country == country]
        for j, model in enumerate(models):
            vals = sub[sub.model == model][metric].to_numpy()
            offset = -0.11 if j == 0 else 0.11
            ax.scatter(np.full(len(vals), i + offset), vals, s=9, color="#8F8F8F" if j == 0 else COUNTRY_COLORS[country], alpha=0.45, lw=0)
        means = [sub[sub.model == m][metric].mean() for m in models]
        ax.plot([i - 0.11, i + 0.11], means, color=COUNTRY_COLORS[country], lw=1.5)
        ax.scatter([i - 0.11, i + 0.11], means, s=25, c=["white", COUNTRY_COLORS[country]], edgecolors=COUNTRY_COLORS[country], lw=1, zorder=3)
    ax.set_xticks(x, COUNTRIES); ax.set_ylabel(ylabel); panel_label(ax, letter)


def _create_figure_34_metric_legacy() -> None:
    """Five-repeat transfer performance for base and agricultural TabPFN."""
    df = pd.read_csv(STAGE2 / "metrics_all.csv")
    df = df[(df.split == "test") & df.country.isin(COUNTRIES)].copy()
    df["mape_percent"] = df.mape * 100
    df.to_csv(SOURCE / "fig_5_tabpfn_test_repeats.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    paired_metric(axes[0], df, "r2", r"Test $R^2$", "a"); axes[0].set_title("Explained variance", loc="left")
    paired_metric(axes[1], df, "mape_percent", "Test MAPE (%)", "b"); axes[1].set_title("Absolute percentage error", loc="left")
    paired_metric(axes[2], df, "normalized_rmse", "Test normalized RMSE (%)", "c"); axes[2].set_title("Normalized error", loc="left")
    handles = [
        mpl.lines.Line2D([], [], marker="o", mfc="white", mec="#666666", color="none", label="TabPFN"),
        mpl.lines.Line2D([], [], marker="o", mfc="#315F8C", mec="#315F8C", color="none", label="AGRO-TabPFN"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.78, wspace=0.46)
    finish(fig, "fig_5_agro_tabpfn_transfer")


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[finite], y_pred[finite]
    ss_total = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - np.sum((y_true - y_pred) ** 2) / ss_total if ss_total > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    return float(r2), rmse, mape


def _create_figure_34_previous() -> None:
    """Pooled five-repeat test regression scatter for AGRO-TabPFN only."""
    frames = []
    for country in COUNTRIES:
        path = STAGE2 / f"maize_{country}/predictions_tabpfn_cn_finetuned.csv"
        country_df = pd.read_csv(path)
        country_df = country_df[(country_df["split"] == "test") & (country_df["model"] == "tabpfn_cn_finetuned")].copy()
        if set(country_df["repeat"].unique()) != set(range(1, 6)):
            raise ValueError(f"{country}: AGRO-TabPFN predictions must contain repeats 1–5")
        frames.append(country_df)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(SOURCE / "fig_5_agro_tabpfn_test_predictions.csv", index=False)

    metric_rows = []
    for (country, repeat), group in predictions.groupby(["country", "repeat"]):
        r2, rmse, mape = _regression_metrics(group["yield"].to_numpy(float), group["prediction"].to_numpy(float))
        metric_rows.append({"country": country, "repeat": repeat, "r2": r2, "rmse": rmse, "mape_percent": mape})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(SOURCE / "fig_5_agro_tabpfn_test_metrics_by_repeat.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2))
    fig.patch.set_alpha(0.0)
    repeat_colors = ["#4E79A7", "#E39C37", "#59A14F", "#A06AA1", "#D1785A"]
    for panel_i, (ax, country) in enumerate(zip(axes.ravel(), COUNTRIES)):
        country_df = predictions[predictions.country == country]
        ax.set_facecolor("none")
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size=0.55, pad=0.06, sharex=ax)
        ax_right = divider.append_axes("right", size=0.55, pad=0.06, sharey=ax)
        ax_top.set_facecolor("none")
        ax_right.set_facecolor("none")

        for repeat, repeat_color in zip(range(1, 6), repeat_colors):
            group = country_df[country_df.repeat == repeat]
            ax.scatter(
                group["yield"], group["prediction"], s=31, marker="o",
                facecolor=to_rgba(repeat_color, 0.58), edgecolor=repeat_color,
                linewidth=0.55, label=f"Repeat {repeat}", zorder=2,
            )

        observed = country_df["yield"].to_numpy(float)
        predicted = country_df["prediction"].to_numpy(float)
        value_min = float(min(observed.min(), predicted.min()))
        value_max = float(max(observed.max(), predicted.max()))
        padding = max((value_max - value_min) * 0.07, 0.1)
        lo, hi = value_min - padding, value_max + padding
        fit_x = np.linspace(lo, hi, 200)
        slope, intercept = np.polyfit(observed, predicted, 1)
        ax.plot([lo, hi], [lo, hi], color="#777777", lw=1.1, ls="--", label="1:1 line", zorder=1)
        ax.plot(fit_x, slope * fit_x + intercept, color="#111111", lw=1.7, label="Pooled fit", zorder=4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#E4E4E4", linewidth=0.65)
        ax.set_xlabel(r"Observed yield (t ha$^{-1}$)", fontsize=9.5)
        ax.set_ylabel(r"Predicted yield (t ha$^{-1}$)", fontsize=9.5)
        ax.tick_params(labelsize=8.5)

        bins = np.linspace(lo, hi, 18)
        ax_top.hist(observed, bins=bins, color="#A9A9A9", alpha=0.22, edgecolor="#555555", linewidth=0.35)
        ax_right.hist(observed, bins=bins, orientation="horizontal", color="#A9A9A9", alpha=0.22,
                      edgecolor="#555555", linewidth=0.35)
        for repeat, repeat_color in zip(range(1, 6), repeat_colors):
            repeat_predicted = country_df.loc[country_df.repeat == repeat, "prediction"].to_numpy(float)
            ax_top.hist(repeat_predicted, bins=bins, color=repeat_color, alpha=0.13, edgecolor="none")
            ax_right.hist(repeat_predicted, bins=bins, orientation="horizontal", color=repeat_color,
                          alpha=0.13, edgecolor="none")
        ax_top.axis("off")
        ax_right.axis("off")

        country_metrics = metrics[metrics.country == country]
        means = country_metrics[["r2", "rmse", "mape_percent"]].mean()
        sds = country_metrics[["r2", "rmse", "mape_percent"]].std(ddof=1)
        ax.text(
            0.04, 0.96,
            f"$R^2$ = {means.r2:.3f} ± {sds.r2:.3f}\n"
            f"MAPE = {means.mape_percent:.1f} ± {sds.mape_percent:.1f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9},
        )
        ax_top.set_title(COUNTRY_NAMES[country] + f" ({country})", fontsize=11.5, fontweight="bold", pad=4)
        panel_label(ax_top, chr(ord("a") + panel_i), x=-0.11, y=1.20)

    repeat_handles = [
        mpl.lines.Line2D([], [], marker="o", color="none", markerfacecolor=repeat_color,
                         markeredgecolor=repeat_color, markersize=6, label=f"Repeat {repeat}")
        for repeat, repeat_color in zip(range(1, 6), repeat_colors)
    ]
    line_handles = [
        mpl.lines.Line2D([], [], color="#777777", ls="--", lw=1.1, label="1:1 line"),
        mpl.lines.Line2D([], [], color="#111111", lw=1.7, label="Pooled fit"),
    ]
    fig.legend(handles=repeat_handles + line_handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=7, fontsize=8.5)
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.09, top=0.985, wspace=0.18, hspace=0.20)
    finish(fig, "fig_5_agro_tabpfn_transfer", transparent=True)


def create_figure_34() -> None:
    """AGRO-TabPFN regression scatter, marginals, and residuals for five repeats."""
    previous_font_family = mpl.rcParams["font.family"]
    previous_font_serif = mpl.rcParams["font.serif"]
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    frames = []
    for country in COUNTRIES:
        path = STAGE2 / f"maize_{country}/predictions_tabpfn_cn_finetuned.csv"
        country_df = pd.read_csv(path)
        country_df = country_df[
            (country_df["split"] == "test")
            & (country_df["model"] == "tabpfn_cn_finetuned")
        ].copy()
        if set(country_df["repeat"].unique()) != set(range(1, 6)):
            raise ValueError(f"{country}: AGRO-TabPFN predictions must contain repeats 1–5")
        frames.append(country_df)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(SOURCE / "fig_5_agro_tabpfn_test_predictions.csv", index=False)

    metric_rows = []
    for (country, repeat), group in predictions.groupby(["country", "repeat"]):
        r2, rmse, mape = _regression_metrics(
            group["yield"].to_numpy(float), group["prediction"].to_numpy(float)
        )
        metric_rows.append(
            {"country": country, "repeat": repeat, "r2": r2, "rmse": rmse, "mape_percent": mape}
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(SOURCE / "fig_5_agro_tabpfn_test_metrics_by_repeat.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 15.0))
    fig.patch.set_facecolor("white")
    repeat_colors = ["#4E79A7", "#E39C37", "#59A14F", "#A06AA1", "#D1785A"]

    for panel_i, (ax, country) in enumerate(zip(axes.ravel(), COUNTRIES)):
        country_df = predictions[predictions.country == country]
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size=0.90, pad=0.08, sharex=ax)
        ax_right = divider.append_axes("right", size=0.90, pad=0.08, sharey=ax)
        ax_res = divider.append_axes("bottom", size="30%", pad=0.25, sharex=ax)

        for repeat, color in zip(range(1, 6), repeat_colors):
            group = country_df[country_df.repeat == repeat]
            observed_repeat = group["yield"].to_numpy(float)
            predicted_repeat = group["prediction"].to_numpy(float)
            ax.scatter(
                observed_repeat, predicted_repeat, s=31, marker="o", color=color,
                alpha=0.62, edgecolor="none", label=f"Repeat {repeat}", zorder=3,
            )
            ax_res.scatter(
                observed_repeat, predicted_repeat - observed_repeat, s=27, marker="o",
                color=color, alpha=0.62, edgecolor="none", zorder=3,
            )

        observed = country_df["yield"].to_numpy(float)
        predicted = country_df["prediction"].to_numpy(float)
        value_min = float(min(observed.min(), predicted.min()))
        value_max = float(max(observed.max(), predicted.max()))
        padding = max((value_max - value_min) * 0.07, 0.1)
        lo, hi = value_min - padding, value_max + padding
        fit_x = np.linspace(lo, hi, 240)
        slope, intercept = np.polyfit(observed, predicted, 1)
        fitted = slope * observed + intercept
        fit_y = slope * fit_x + intercept
        if observed.size > 2:
            x_mean = float(observed.mean())
            sxx = float(np.sum((observed - x_mean) ** 2))
            if sxx > 0:
                residual_se = float(
                    np.sqrt(np.sum((predicted - fitted) ** 2) / (observed.size - 2))
                )
                confidence = 1.96 * residual_se * np.sqrt(
                    1.0 / observed.size + (fit_x - x_mean) ** 2 / sxx
                )
                ax.fill_between(
                    fit_x, fit_y - confidence, fit_y + confidence,
                    color="#BDBDBD", alpha=0.25, linewidth=0, zorder=1,
                )
        ax.plot(fit_x, fit_y, color="#111111", lw=1.35, label="Fitted line", zorder=4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#CFCFCF", linewidth=0.85, alpha=0.82)
        ax.set_xlabel("")
        ax.set_ylabel(r"Predicted yield (t ha$^{-1}$)", fontsize=12.0, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11.0, width=1.1, length=4.5)
        ax.tick_params(axis="x", labelbottom=False)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.15)

        bins = np.linspace(lo, hi, 18)
        density_grid = np.linspace(lo, hi, 240)
        bin_width = float(bins[1] - bins[0])
        ax_top.hist(
            observed, bins=bins, color="#A9A9A9", alpha=0.22,
            edgecolor="#555555", linewidth=0.45,
        )
        if np.unique(observed).size > 1:
            density = gaussian_kde(observed)(density_grid)
            ax_top.plot(
                density_grid, density * observed.size * bin_width,
                color="#666666", lw=2.0,
            )
        for repeat, color in zip(range(1, 6), repeat_colors):
            repeat_predicted = country_df.loc[
                country_df.repeat == repeat, "prediction"
            ].to_numpy(float)
            ax_right.hist(
                repeat_predicted, bins=bins, orientation="horizontal",
                color=color, alpha=0.16, edgecolor=color, linewidth=0.30,
            )
            if np.unique(repeat_predicted).size > 1:
                density = gaussian_kde(repeat_predicted)(density_grid)
                ax_right.plot(
                    density * repeat_predicted.size * bin_width, density_grid,
                    color=color, lw=1.65, alpha=0.95,
                )
        ax_top.axis("off")
        ax_right.axis("off")

        residuals = predicted - observed
        residual_limit = max(float(np.nanmax(np.abs(residuals))) * 1.18, 0.1)
        ax_res.axhline(0, color="#111111", lw=1.15, alpha=0.48, zorder=2)
        ax_res.set_ylim(-residual_limit, residual_limit)
        ax_res.set_xlabel(r"Observed yield (t ha$^{-1}$)", fontsize=12.0, fontweight="bold")
        ax_res.set_ylabel("Residual", fontsize=11.5, fontweight="bold")
        ax_res.tick_params(axis="both", labelsize=10.5, width=1.1, length=4.5)
        ax_res.xaxis.set_major_locator(MaxNLocator(5))
        ax_res.yaxis.set_major_locator(MaxNLocator(4))
        ax_res.grid(True, color="#CFCFCF", linewidth=0.85, alpha=0.82)
        for spine in ax_res.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.15)

        country_metrics = metrics[metrics.country == country]
        means = country_metrics[["r2", "mape_percent"]].mean()
        sds = country_metrics[["r2", "mape_percent"]].std(ddof=1)
        ax.text(
            0.045, 0.955,
            f"$R^2_{{test}}$ = {means.r2:.3f} ± {sds.r2:.3f}\n"
            f"MAPE$_{{test}}$ = {means.mape_percent:.1f} ± {sds.mape_percent:.1f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=11.5, color="black",
        )
        ax.legend(
            loc="lower right", ncol=2, fontsize=8.0, frameon=True,
            facecolor="white", edgecolor="#C8C8C8", framealpha=0.90,
            borderpad=0.55, columnspacing=0.8, handletextpad=0.45,
        )
        ax_top.set_title(
            COUNTRY_NAMES[country] + f" ({country})", fontsize=14.0,
            fontweight="bold", pad=6,
        )
        panel_label(ax_top, chr(ord("a") + panel_i), x=-0.09, y=1.19)

    fig.subplots_adjust(
        left=0.075, right=0.955, bottom=0.055, top=0.985,
        wspace=0.18, hspace=0.08,
    )
    finish(fig, "fig_5_agro_tabpfn_transfer")
    mpl.rcParams["font.family"] = previous_font_family
    mpl.rcParams["font.serif"] = previous_font_serif


def _create_figure_35_bars_legacy() -> None:
    """Stress composition and bounded water-recovery gains."""
    roles = ["water_proxy", "temperature_hazard", "crop_state_indicator", "radiation_hazard", "fixed_context"]
    role_labels = ["Water", "Temperature", "Crop state", "Radiation", "Soil context"]
    role_colors = ["#4C86B6", "#D1785A", "#6E9F72", "#D6B04D", "#8B8B8B"]
    risk_rows = []
    contribution_rows = []
    scenario_rows = []
    for country in COUNTRIES:
        diag = pd.read_csv(STAGE3 / f"maize_{country}/qwen_rag/stress_diagnostics.csv")
        counts = diag.loc[diag.stress_score > 0, "feature_role"].value_counts()
        risk_rows.append([counts.get(r, 0) for r in roles])
        changes = pd.read_csv(STAGE3 / f"maize_{country}/qwen_rag/accepted_proxy_changes.csv")
        def family(feature: str) -> str:
            if "ssm" in feature: return "Soil moisture"
            if "cwb" in feature: return "Water balance"
            return "Precipitation"
        changes["family"] = changes.feature.map(family)
        n_locations = changes.adm_id.nunique()
        contrib = changes.groupby("family").predicted_increment.sum() / n_locations
        contribution_rows.append([contrib.get(x, 0) for x in ["Precipitation", "Water balance", "Soil moisture"]])
        policy = pd.read_csv(STAGE3 / f"maize_{country}/policy_comparison.csv")
        policy = policy[policy.policy == "qwen_rag"]
        base = policy.loc[policy.scenario == "S1_observed_conditions", "mean_predicted_yield"].iloc[0]
        recovered = policy.loc[policy.scenario == "S2_bounded_water_recovery", "mean_predicted_yield"].iloc[0]
        scenario_rows.append([base, recovered, recovered - base, n_locations])

    risk = pd.DataFrame(risk_rows, index=COUNTRIES, columns=role_labels)
    contrib = pd.DataFrame(contribution_rows, index=COUNTRIES, columns=["Precipitation", "Water balance", "Soil moisture"])
    scenario = pd.DataFrame(scenario_rows, index=COUNTRIES, columns=["Observed", "Water recovery", "Gain", "n_locations"])
    risk.reset_index(names="country").to_csv(SOURCE / "fig_6_positive_stress_signals.csv", index=False)
    contrib.reset_index(names="country").to_csv(SOURCE / "fig_6_water_feature_contributions.csv", index=False)
    scenario.reset_index(names="country").to_csv(SOURCE / "fig_6_scenario_yields.csv", index=False)

    fig = plt.figure(figsize=(7.2, 4.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.22, 1.0], hspace=0.72, wspace=0.38)
    ax_a = fig.add_subplot(gs[:, 0]); ax_b = fig.add_subplot(gs[0, 1]); ax_c = fig.add_subplot(gs[1, 1])
    proportions = risk.div(risk.sum(axis=1), axis=0) * 100
    left = np.zeros(4)
    y = np.arange(4)
    for label, color in zip(role_labels, role_colors):
        vals = proportions[label].to_numpy()
        ax_a.barh(y, vals, left=left, color=color, height=0.62, label=label)
        for yi, l, v in zip(y, left, vals):
            if v >= 8: ax_a.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=6, color="white" if label in ["Water", "Temperature"] else "#222222")
        left += vals
    for yi, country in enumerate(COUNTRIES): ax_a.text(101.2, yi, f"n={risk.loc[country].sum():,}", va="center", fontsize=6.3)
    ax_a.set_yticks(y, COUNTRIES); ax_a.invert_yaxis(); ax_a.set_xlim(0, 115); ax_a.set_xlabel("Share of positive stress signals (%)")
    ax_a.set_title("Stress composition", loc="left"); panel_label(ax_a, "a")

    x = np.arange(4)
    for i, country in enumerate(COUNTRIES):
        vals = scenario.loc[country, ["Observed", "Water recovery"]].to_numpy(float)
        ax_b.plot([i - 0.12, i + 0.12], vals, color=COUNTRY_COLORS[country], lw=1.5)
        ax_b.scatter([i - 0.12, i + 0.12], vals, s=25, c=["white", COUNTRY_COLORS[country]], edgecolors=COUNTRY_COLORS[country], lw=1, zorder=3)
        ax_b.text(i + 0.16, vals[1], f"+{scenario.loc[country, 'Gain']:.3f}", va="center", fontsize=6.1, color=COUNTRY_COLORS[country])
    ax_b.set_xticks(x, COUNTRIES); ax_b.set_ylabel(r"Predicted yield (t ha$^{-1}$)")
    ax_b.set_title("Bounded water-recovery scenario", loc="left"); panel_label(ax_b, "b", x=-0.14, y=1.13)

    bottoms = np.zeros(4)
    contribution_colors = ["#79A9CF", "#4C86B6", "#244F73"]
    for label, color in zip(contrib.columns, contribution_colors):
        vals = contrib[label].to_numpy(); ax_c.bar(x, vals, bottom=bottoms, color=color, width=0.62, label=label); bottoms += vals
    ax_c.set_xticks(x, COUNTRIES); ax_c.set_ylabel(r"Mean yield gain (t ha$^{-1}$)")
    ax_c.set_ylim(0, max(0.50, contrib.sum(axis=1).max() * 1.35))
    ax_c.set_title("Accepted water-feature contributions", loc="left", pad=9)
    ax_c.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.99), columnspacing=0.7, handlelength=1.1)
    panel_label(ax_c, "c", x=-0.14, y=1.13)
    risk_handles = [mpl.patches.Patch(color=color, label=label) for label, color in zip(role_labels, role_colors)]
    fig.legend(handles=risk_handles, ncol=5, loc="upper center", bbox_to_anchor=(0.47, 0.995), columnspacing=0.9)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.84, wspace=0.40, hspace=0.72)
    finish(fig, "fig_6_risk_and_water_recovery")


def _stacked_node_intervals(totals: dict[str, float], scale: float = 0.20) -> dict[str, tuple[float, float]]:
    """Lay out one Sankey layer while preserving a common flow-width scale."""
    labels = list(totals)
    used = sum(totals.values()) * scale
    gap = (1.0 - used) / (len(labels) + 1)
    cursor = gap
    intervals = {}
    for label in labels:
        height = totals[label] * scale
        intervals[label] = (cursor, cursor + height)
        cursor += height + gap
    return intervals


def _sankey_ribbon(
    ax: mpl.axes.Axes,
    x0: float,
    x1: float,
    source_interval: tuple[float, float],
    target_interval: tuple[float, float],
    color: str,
    alpha: float,
) -> None:
    y0_low, y0_high = source_interval
    y1_low, y1_high = target_interval
    control_left = x0 + (x1 - x0) * 0.42
    control_right = x1 - (x1 - x0) * 0.42
    vertices = [
        (x0, y0_low),
        (control_left, y0_low), (control_right, y1_low), (x1, y1_low),
        (x1, y1_high),
        (control_right, y1_high), (control_left, y0_high), (x0, y0_high),
        (x0, y0_low),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor="none", alpha=alpha))


def _create_figure_35_previous() -> None:
    """Risk-profile to water-recovery Sankey plus absolute scenario gains."""
    roles = ["water_proxy", "temperature_hazard", "crop_state_indicator", "radiation_hazard", "fixed_context"]
    role_labels = {
        "water_proxy": "Water deficit",
        "temperature_hazard": "Temperature",
        "crop_state_indicator": "Crop state",
        "radiation_hazard": "Radiation",
        "fixed_context": "Soil context",
    }
    role_colors = {
        "water_proxy": "#4C86B6",
        "temperature_hazard": "#D1785A",
        "crop_state_indicator": "#6E9F72",
        "radiation_hazard": "#D6B04D",
        "fixed_context": "#8B8B8B",
    }
    families = ["Precipitation", "Water balance", "Soil moisture"]
    family_colors = {"Precipitation": "#A8CAE3", "Water balance": "#5D97C2", "Soil moisture": "#275D85"}

    risk_rows, contribution_rows, scenario_rows, flow_rows, importance_rows = [], [], [], [], []
    for country in COUNTRIES:
        diagnostics = pd.read_csv(STAGE3 / f"maize_{country}/qwen_rag/stress_diagnostics.csv")
        counts = diagnostics.loc[diagnostics.stress_score > 0, "feature_role"].value_counts()
        risk_row = {role: float(counts.get(role, 0)) for role in roles}
        risk_rows.append({"country": country, **risk_row})
        importance = (
            diagnostics[["feature", "feature_role", "model_importance"]]
            .drop_duplicates("feature")
            .nlargest(5, "model_importance")
        )
        for row in importance.itertuples(index=False):
            importance_rows.append(
                {"country": country, "feature": row.feature, "feature_role": row.feature_role,
                 "model_importance": float(row.model_importance)}
            )

        changes = pd.read_csv(STAGE3 / f"maize_{country}/qwen_rag/accepted_proxy_changes.csv")

        def family(feature: str) -> str:
            if "ssm" in feature:
                return "Soil moisture"
            if "cwb" in feature:
                return "Water balance"
            return "Precipitation"

        changes["family"] = changes.feature.map(family)
        contributions = changes.groupby("family").predicted_increment.sum().clip(lower=0)
        contribution_row = {name: float(contributions.get(name, 0)) for name in families}
        contribution_rows.append({"country": country, **contribution_row})

        policy = pd.read_csv(STAGE3 / f"maize_{country}/policy_comparison.csv")
        policy = policy[policy.policy == "qwen_rag"]
        observed = float(policy.loc[policy.scenario == "S1_observed_conditions", "mean_predicted_yield"].iloc[0])
        recovered = float(policy.loc[policy.scenario == "S2_bounded_water_recovery", "mean_predicted_yield"].iloc[0])
        scenario_rows.append({"country": country, "Observed": observed, "Water recovery": recovered, "Gain": recovered - observed})

    risk = pd.DataFrame(risk_rows).set_index("country")
    contribution = pd.DataFrame(contribution_rows).set_index("country")
    scenario = pd.DataFrame(scenario_rows).set_index("country")
    importance = pd.DataFrame(importance_rows)
    risk_share = risk.div(risk.sum(axis=1), axis=0).fillna(0)
    contribution_share = contribution.div(contribution.sum(axis=1), axis=0).fillna(0)

    for country in COUNTRIES:
        for role in roles:
            flow_rows.append(
                {"stage": "risk_to_country", "source": role_labels[role], "target": country,
                 "country": country, "within_country_percent": risk_share.loc[country, role] * 100}
            )
        for family_name in families:
            flow_rows.append(
                {"stage": "country_to_recovery", "source": country, "target": family_name,
                 "country": country, "within_country_percent": contribution_share.loc[country, family_name] * 100}
            )
    pd.DataFrame(flow_rows).to_csv(SOURCE / "fig_6_sankey_flows.csv", index=False)
    risk.reset_index().to_csv(SOURCE / "fig_6_positive_stress_signals.csv", index=False)
    contribution.reset_index().to_csv(SOURCE / "fig_6_water_feature_contributions.csv", index=False)
    scenario.reset_index().to_csv(SOURCE / "fig_6_scenario_yields.csv", index=False)
    importance.to_csv(SOURCE / "fig_6_top_feature_importance.csv", index=False)

    left_totals = {role: float(risk_share[role].sum()) for role in roles}
    middle_totals = {country: 1.0 for country in COUNTRIES}
    right_totals = {name: float(contribution_share[name].sum()) for name in families}
    left_nodes = _stacked_node_intervals(left_totals)
    middle_nodes = _stacked_node_intervals(middle_totals)
    right_nodes = _stacked_node_intervals(right_totals)
    scale = 0.20

    fig = plt.figure(figsize=(13.0, 15.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.55], hspace=0.22)
    ax_flow = fig.add_subplot(gs[0])
    ax_importance = fig.add_subplot(gs[1])
    x_left, x_middle, x_right = 0.08, 0.50, 0.92

    left_offsets = {role: left_nodes[role][0] for role in roles}
    middle_targets = {country: middle_nodes[country][0] for country in COUNTRIES}
    for role in roles:
        for country in COUNTRIES:
            value = float(risk_share.loc[country, role])
            if value <= 0:
                continue
            source = (left_offsets[role], left_offsets[role] + value * scale)
            target = (middle_targets[country], middle_targets[country] + value * scale)
            _sankey_ribbon(ax_flow, x_left, x_middle, source, target, role_colors[role], 0.38)
            left_offsets[role] = source[1]
            middle_targets[country] = target[1]

    middle_sources = {country: middle_nodes[country][0] for country in COUNTRIES}
    right_offsets = {name: right_nodes[name][0] for name in families}
    for country in COUNTRIES:
        for family_name in families:
            value = float(contribution_share.loc[country, family_name])
            if value <= 0:
                continue
            source = (middle_sources[country], middle_sources[country] + value * scale)
            target = (right_offsets[family_name], right_offsets[family_name] + value * scale)
            _sankey_ribbon(ax_flow, x_middle, x_right, source, target, COUNTRY_COLORS[country], 0.36)
            middle_sources[country] = source[1]
            right_offsets[family_name] = target[1]

    node_width = 0.022
    for role, (bottom, top) in left_nodes.items():
        ax_flow.add_patch(Rectangle((x_left - node_width / 2, bottom), node_width, top - bottom,
                                    facecolor=role_colors[role], edgecolor="#444444", lw=0.6, zorder=5))
        ax_flow.text(x_left - 0.018, (bottom + top) / 2, role_labels[role], ha="right", va="center", fontsize=9)
    for country, (bottom, top) in middle_nodes.items():
        ax_flow.add_patch(Rectangle((x_middle - node_width / 2, bottom), node_width, top - bottom,
                                    facecolor=COUNTRY_COLORS[country], edgecolor="#444444", lw=0.6, zorder=5))
        ax_flow.text(
            x_middle, (bottom + top) / 2,
            f"{country}\nΔ{scenario.loc[country, 'Gain']:+.3f}",
            ha="center", va="center", color="white", fontsize=8.2,
            fontweight="bold", linespacing=1.15, zorder=6,
        )
    for family_name, (bottom, top) in right_nodes.items():
        ax_flow.add_patch(Rectangle((x_right - node_width / 2, bottom), node_width, top - bottom,
                                    facecolor=family_colors[family_name], edgecolor="#444444", lw=0.6, zorder=5))
        ax_flow.text(x_right + 0.018, (bottom + top) / 2, family_name, ha="left", va="center", fontsize=9)

    ax_flow.text(x_left, 1.025, "Positive stress profile", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax_flow.text(x_middle, 1.025, "Country", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax_flow.text(x_right, 1.025, "Accepted recovery mechanism", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax_flow.set_xlim(-0.02, 1.02)
    ax_flow.set_ylim(-0.02, 1.08)
    ax_flow.axis("off")
    ax_flow.set_title("Risk profile to bounded water-recovery pathway", loc="left", fontsize=12, fontweight="bold", pad=14)
    panel_label(ax_flow, "a", x=-0.03, y=1.09)

    def pretty_feature_name(feature: str) -> str:
        label = feature.replace("bulk_density", "Bulk density").replace("awc", "Available water capacity")
        label = label.replace("ndvi", "NDVI").replace("fpar", "FPAR").replace("ssm", "soil moisture")
        label = label.replace("cwb", "water balance").replace("prec", "precipitation")
        label = label.replace("tmax", "Tmax").replace("tmin", "Tmin").replace("tavg", "Tavg")
        label = label.replace("rad", "radiation").replace("cum", "cumulative")
        label = label.replace("median_", "median ").replace("mean_", "mean ").replace("max_", "maximum ")
        return label.replace("_", " ")

    bar_y, bar_labels, bar_values, bar_colors = [], [], [], []
    country_midpoints, separators = {}, []
    cursor = 0
    for country in COUNTRIES:
        subset = importance[importance.country == country].sort_values("model_importance", ascending=False)
        start = cursor
        for row in subset.itertuples(index=False):
            bar_y.append(cursor)
            bar_labels.append(pretty_feature_name(row.feature))
            bar_values.append(row.model_importance * 100)
            bar_colors.append(role_colors.get(row.feature_role, "#8B8B8B"))
            cursor += 1
        country_midpoints[country] = (start + cursor - 1) / 2
        separators.append(cursor - 0.5)
        cursor += 0.9

    ax_importance.barh(bar_y, bar_values, color=bar_colors, height=0.70, edgecolor="white", linewidth=0.4)
    ax_importance.set_yticks(bar_y, bar_labels, fontsize=10.5)
    ax_importance.invert_yaxis()
    ax_importance.set_xlabel("ExtraTrees impurity-based feature importance (%)", fontsize=11.5)
    ax_importance.tick_params(axis="x", labelsize=10)
    ax_importance.grid(axis="x", color="#E3E3E3", lw=0.7)
    for y, value in zip(bar_y, bar_values):
        ax_importance.text(value + 0.25, y, f"{value:.1f}%", va="center", fontsize=10)
    for separator in separators[:-1]:
        ax_importance.axhline(separator + 0.45, color="#D7D7D7", lw=0.8)
    for country, midpoint in country_midpoints.items():
        ax_importance.text(
            1.01, midpoint, country, transform=ax_importance.get_yaxis_transform(),
            ha="left", va="center", fontsize=13, fontweight="bold", color=COUNTRY_COLORS[country],
        )
    importance_handles = [
        mpl.patches.Patch(color=role_colors[role], label=role_labels[role])
        for role in roles
    ]
    ax_importance.legend(handles=importance_handles, loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=5,
                         fontsize=10, columnspacing=1.2)
    ax_importance.set_title("Global model explanation: top five features per country", loc="left", fontsize=13, fontweight="bold")
    ax_importance.text(0.995, 1.01, "Model-native global importance; not SHAP", transform=ax_importance.transAxes,
                       ha="right", va="bottom", fontsize=10, color="#666666")
    panel_label(ax_importance, "b", x=-0.06, y=1.08)
    fig.subplots_adjust(left=0.27, right=0.88, bottom=0.075, top=0.965)
    finish(fig, "fig_6_risk_and_water_recovery")


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    create_figure_32()
    create_figure_34()
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
