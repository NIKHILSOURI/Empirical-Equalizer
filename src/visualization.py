import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy import stats as sp_stats
from typing import Dict, List, Any
from .config import (
    FIGURES_DIR, FIG_DPI, FIG_SINGLE_COL, FIG_DOUBLE_COL, FIG_FULL_PAGE,
    REGIME_COLORS, REGIME_LABELS, ALL_PATHS, COMPANION_PROFILES,
    GAME_THRESHOLDS, PERCENTILES,
)


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def _save(fig, name):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig1_regime_distributions(ripe_data: Dict[str, pd.DataFrame],
                               all_stats: Dict) -> str:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))

    for ax, regime in zip(axes, ["short_haul", "regional", "intercontinental"]):
        paths_in_regime = [pid for pid, info in all_stats.items()
                          if info["path_def"].regime == regime]
        data_list = []
        labels = []
        for pid in sorted(paths_in_regime):
            p = all_stats[pid]["path_def"]
            data_list.append(ripe_data[pid]["rtt_ms"].values)
            labels.append(f"{p.source_city[:3]}-{p.dest_city[:3]}")

        bp = ax.boxplot(data_list, labels=labels, patch_artist=True,
                        showfliers=False, widths=0.6,
                        medianprops={"color": "black", "linewidth": 1.2})
        color = REGIME_COLORS[regime]
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(REGIME_LABELS[regime], fontweight="bold")
        ax.set_ylabel("RTT (ms)" if regime == "short_haul" else "")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("RTT Distributions by Path and Regime", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig1_regime_distributions")


def fig2_cdf_comparison(ripe_data: Dict[str, pd.DataFrame],
                         all_stats: Dict) -> str:
    setup_style()
    fig, ax = plt.subplots(figsize=FIG_DOUBLE_COL)

    for regime in ["short_haul", "regional", "intercontinental"]:
        all_rtt = []
        for pid, info in all_stats.items():
            if info["path_def"].regime == regime:
                all_rtt.append(ripe_data[pid]["rtt_ms"].values)
        combined = np.concatenate(all_rtt)
        sorted_rtt = np.sort(combined)
        cdf = np.arange(1, len(sorted_rtt) + 1) / len(sorted_rtt)
        step = max(1, len(sorted_rtt) // 2000)
        ax.plot(sorted_rtt[::step], cdf[::step],
                color=REGIME_COLORS[regime],
                label=REGIME_LABELS[regime], linewidth=1.5)

    ax.set_xlabel("RTT (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Empirical CDF of RTT by Regime")
    ax.legend()
    ax.set_xlim(0, None)
    fig.tight_layout()
    return _save(fig, "fig2_cdf_comparison")


def fig3_tod_heatmap(ripe_data: Dict[str, pd.DataFrame],
                      all_stats: Dict) -> str:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))

    for ax, regime in zip(axes, ["short_haul", "regional", "intercontinental"]):
        paths_in_regime = sorted([pid for pid, info in all_stats.items()
                                  if info["path_def"].regime == regime])
        matrix = []
        ylabels = []
        for pid in paths_in_regime:
            df = ripe_data[pid]
            hourly_mean = df.groupby(df["hour"].astype(int) % 24)["rtt_ms"].mean()
            normalized = (hourly_mean - hourly_mean.min()) / (hourly_mean.max() - hourly_mean.min() + 1e-9)
            matrix.append(normalized.values)
            p = all_stats[pid]["path_def"]
            ylabels.append(f"{p.source_city[:3]}-{p.dest_city[:3]}")

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                       extent=[0, 24, len(matrix) - 0.5, -0.5])
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=6)
        ax.set_xlabel("Hour of Day")
        ax.set_title(REGIME_LABELS[regime], fontweight="bold")
        ax.set_xticks([0, 6, 12, 18, 24])

    fig.colorbar(im, ax=axes, label="Normalized RTT", shrink=0.8, pad=0.02)
    fig.suptitle("Time-of-Day RTT Variation (Normalized)", fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.35)
    return _save(fig, "fig3_tod_heatmap")


def fig4_diurnal_pattern(regime_hourly: Dict[str, pd.DataFrame]) -> str:
    setup_style()
    fig, ax = plt.subplots(figsize=FIG_DOUBLE_COL)

    for regime in ["short_haul", "regional", "intercontinental"]:
        df = regime_hourly[regime]
        overall_mean = df["mean"].mean()
        pct_change = (df["mean"] - overall_mean) / overall_mean * 100
        ax.plot(df["hour"], pct_change, color=REGIME_COLORS[regime],
                label=REGIME_LABELS[regime], linewidth=1.5, marker="o", markersize=3)
        pct_std = df["std"] / overall_mean * 100
        ax.fill_between(df["hour"], pct_change - pct_std, pct_change + pct_std,
                        color=REGIME_COLORS[regime], alpha=0.15)

    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("RTT Change from Mean (%)")
    ax.set_title("Diurnal RTT Variation by Regime")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3))
    fig.tight_layout()
    return _save(fig, "fig4_diurnal_pattern")


def fig5_dow_pattern(ripe_data: Dict[str, pd.DataFrame],
                      all_stats: Dict) -> str:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=False)
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for ax, regime in zip(axes, ["short_haul", "regional", "intercontinental"]):
        regime_rtt_by_dow = {d: [] for d in range(7)}
        for pid, info in all_stats.items():
            if info["path_def"].regime == regime:
                df = ripe_data[pid]
                for d in range(7):
                    vals = df[df["day_of_week"] == d]["rtt_ms"].values
                    regime_rtt_by_dow[d].extend(vals)

        data = [regime_rtt_by_dow[d] for d in range(7)]
        bp = ax.boxplot(data, labels=dow_labels, patch_artist=True,
                        showfliers=False, widths=0.6)
        for patch in bp["boxes"]:
            patch.set_facecolor(REGIME_COLORS[regime])
            patch.set_alpha(0.6)

        ax.set_title(REGIME_LABELS[regime], fontweight="bold")
        ax.set_ylabel("RTT (ms)" if regime == "short_haul" else "")

    fig.suptitle("Day-of-Week RTT Variation by Regime", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig5_dow_pattern")


def fig6_distribution_fits(ripe_data: Dict[str, pd.DataFrame],
                            all_stats: Dict) -> str:
    setup_style()
    representatives = []
    for regime in ["short_haul", "regional", "intercontinental"]:
        paths = [(pid, info) for pid, info in all_stats.items()
                 if info["path_def"].regime == regime]
        paths.sort(key=lambda x: x[1]["statistics"]["median"])
        representatives.append(paths[len(paths) // 2])

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
    for ax, (pid, info) in zip(axes, representatives):
        rtt = ripe_data[pid]["rtt_ms"].values
        ax.hist(rtt, bins=60, density=True, alpha=0.5,
                color=REGIME_COLORS[info["path_def"].regime],
                edgecolor="white", linewidth=0.3, label="Empirical")

        bf = info["best_distribution"]
        if bf:
            dist = getattr(sp_stats, bf["distribution"])
            x = np.linspace(rtt.min(), np.percentile(rtt, 99.5), 200)
            pdf = dist.pdf(x, *bf["params"])
            ax.plot(x, pdf, "k-", linewidth=1.5,
                    label=f'{bf["distribution"]} fit')

        p = info["path_def"]
        ax.set_title(f"{p.source_city}-{p.dest_city}\n({REGIME_LABELS[p.regime]})",
                     fontsize=7, fontweight="bold")
        ax.set_xlabel("RTT (ms)")
        ax.set_ylabel("Density" if p.regime == "short_haul" else "")
        ax.legend(fontsize=6)

    fig.suptitle("Best-Fit Distribution Overlay", fontweight="bold", y=1.04)
    fig.tight_layout()
    return _save(fig, "fig6_distribution_fits")


def fig7_cross_validation(cross_val: Dict, mawi_rtts: pd.DataFrame,
                           ripe_data: Dict[str, pd.DataFrame]) -> str:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE_COL)

    ax = axes[0]
    all_ripe = pd.concat(ripe_data.values())["rtt_ms"].values
    ripe_sample = np.random.choice(all_ripe, min(5000, len(all_ripe)), replace=False)
    mawi_sample = np.random.choice(mawi_rtts["rtt_ms"].values,
                                    min(5000, len(mawi_rtts)), replace=False)

    bins = np.logspace(np.log10(1), np.log10(500), 80)
    ax.hist(ripe_sample, bins=bins, density=True, alpha=0.5,
            color="#2196F3", label="RIPE Atlas", edgecolor="white", linewidth=0.3)
    ax.hist(mawi_sample, bins=bins, density=True, alpha=0.5,
            color="#FF9800", label="MAWI SYN-ACK", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("RTT (ms)")
    ax.set_ylabel("Density")
    ax.set_title("RTT Distribution Comparison")
    ax.legend()

    ax = axes[1]
    regimes = ["short_haul", "regional", "intercontinental"]
    ripe_means = [cross_val["ripe_by_regime"][r]["mean"] for r in regimes]
    proxy_keys = ["short_haul_proxy", "regional_proxy", "intercon_proxy"]
    mawi_means = [cross_val["mawi_regime_proxy"].get(k, {}).get("mean", 0)
                  for k in proxy_keys]

    x = np.arange(3)
    w = 0.35
    ax.bar(x - w / 2, ripe_means, w, label="RIPE Atlas", color="#2196F3", alpha=0.7)
    ax.bar(x + w / 2, mawi_means, w, label="MAWI Proxy", color="#FF9800", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Short-haul", "Regional", "Intercon."])
    ax.set_ylabel("Mean RTT (ms)")
    ax.set_title("Per-Regime Cross-Validation")
    ax.legend()

    fig.suptitle("RIPE Atlas vs MAWI Cross-Validation", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig7_cross_validation")


def fig8_taxonomy_comparison(taxonomy: Dict, comparison: Dict) -> str:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE_COL)

    categories = ["excellent", "good", "average", "poor", "very_poor"]
    display_names = ["Excellent", "Good", "Average", "Poor", "Very Poor"]

    ax = axes[0]
    emp_rtts = [taxonomy[c]["rtt_typical"] for c in categories]
    comp_rtts = [COMPANION_PROFILES[c]["base_delay_ms"] for c in categories]
    x = np.arange(len(categories))
    w = 0.35
    bars1 = ax.bar(x - w / 2, emp_rtts, w, label="Empirical", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x + w / 2, comp_rtts, w, label="Companion Paper", color="#FF9800", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=7)
    ax.set_ylabel("Typical RTT (ms)")
    ax.set_title("RTT: Empirical vs Companion")
    ax.legend(fontsize=6)

    for bar, val in zip(bars1, emp_rtts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}", ha="center", va="bottom", fontsize=5.5)
    for bar, val in zip(bars2, comp_rtts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}", ha="center", va="bottom", fontsize=5.5)

    ax = axes[1]
    emp_jit = [taxonomy[c]["jitter_typical"] for c in categories]
    comp_jit = [COMPANION_PROFILES[c]["jitter_ms"] for c in categories]
    bars1 = ax.bar(x - w / 2, emp_jit, w, label="Empirical", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x + w / 2, comp_jit, w, label="Companion Paper", color="#FF9800", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=7)
    ax.set_ylabel("Typical Jitter (ms)")
    ax.set_title("Jitter: Empirical vs Companion")
    ax.legend(fontsize=6)

    fig.suptitle("Empirical Profile Taxonomy vs Companion Paper",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig8_taxonomy_comparison")


def fig9_game_relevance(game_map: Dict, taxonomy: Dict) -> str:
    setup_style()
    categories = ["excellent", "good", "average", "poor", "very_poor"]
    genres = list(GAME_THRESHOLDS.keys())

    quality_colors = {
        "Excellent": "#4CAF50",
        "Acceptable": "#FFC107",
        "Degraded": "#FF9800",
        "Unplayable": "#F44336",
    }

    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    cell_w, cell_h = 1.0, 0.6

    for j, genre in enumerate(genres):
        for i, cat in enumerate(categories):
            quality = game_map[genre][cat]["quality"]
            color = quality_colors[quality]
            rect = FancyBboxPatch(
                (j * cell_w + 0.05, (len(categories) - 1 - i) * cell_h + 0.05),
                cell_w * 0.9, cell_h * 0.9,
                boxstyle="round,pad=0.02",
                facecolor=color, alpha=0.7, edgecolor="white", linewidth=1)
            ax.add_patch(rect)
            ax.text(j * cell_w + cell_w / 2,
                    (len(categories) - 1 - i) * cell_h + cell_h / 2,
                    quality[0], ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white")

    ax.set_xlim(-0.3, len(genres) * cell_w + 0.1)
    ax.set_ylim(-0.3, len(categories) * cell_h + 0.1)
    ax.set_xticks([j * cell_w + cell_w / 2 for j in range(len(genres))])
    ax.set_xticklabels(genres, fontsize=6.5, rotation=25, ha="right")
    ax.set_yticks([(len(categories) - 1 - i) * cell_h + cell_h / 2
                   for i in range(len(categories))])
    display_names = ["Excellent", "Good", "Average", "Poor", "Very Poor"]
    rtts = [f"{taxonomy[c]['rtt_typical']:.0f}ms" for c in categories]
    ax.set_yticklabels([f"{n} ({r})" for n, r in zip(display_names, rtts)], fontsize=7)
    ax.set_title("Game Quality by Empirical Delay Profile and Genre",
                 fontweight="bold", pad=10)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.7, label=q)
                       for q, c in quality_colors.items()]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=6,
              bbox_to_anchor=(1.15, 1.0))
    ax.set_aspect("equal")
    fig.tight_layout()
    return _save(fig, "fig9_game_relevance")


def fig10_jitter_characterization(ripe_data: Dict[str, pd.DataFrame],
                                   all_stats: Dict) -> str:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))

    ax = axes[0]
    for regime in ["short_haul", "regional", "intercontinental"]:
        means_rtt = []
        means_jit = []
        for pid, info in all_stats.items():
            if info["path_def"].regime == regime:
                means_rtt.append(info["statistics"]["mean"])
                means_jit.append(info["statistics"]["jitter_mean"])
        ax.scatter(means_rtt, means_jit, color=REGIME_COLORS[regime],
                   label=REGIME_LABELS[regime], s=40, alpha=0.8, edgecolors="white",
                   linewidth=0.5)

    z = np.polyfit(
        [info["statistics"]["mean"] for info in all_stats.values()],
        [info["statistics"]["jitter_mean"] for info in all_stats.values()], 1)
    x_fit = np.linspace(0, max(info["statistics"]["mean"]
                               for info in all_stats.values()) * 1.1, 100)
    ax.plot(x_fit, np.polyval(z, x_fit), "k--", linewidth=0.8, alpha=0.5,
            label=f"Trend (slope={z[0]:.3f})")
    ax.set_xlabel("Mean RTT (ms)")
    ax.set_ylabel("Mean Jitter (ms)")
    ax.set_title("RTT vs Jitter")
    ax.legend(fontsize=5.5)

    ax = axes[1]
    for regime in ["short_haul", "regional", "intercontinental"]:
        cvs = [info["statistics"]["cv"] for pid, info in all_stats.items()
               if info["path_def"].regime == regime]
        ax.hist(cvs, bins=8, alpha=0.5, color=REGIME_COLORS[regime],
                label=REGIME_LABELS[regime], edgecolor="white")
    ax.set_xlabel("Coefficient of Variation")
    ax.set_ylabel("Count")
    ax.set_title("RTT Variability (CV)")
    ax.legend(fontsize=5.5)

    ax = axes[2]
    for regime in ["short_haul", "regional", "intercontinental"]:
        acfs = [info["statistics"]["acf1"] for pid, info in all_stats.items()
                if info["path_def"].regime == regime]
        ax.hist(acfs, bins=8, alpha=0.5, color=REGIME_COLORS[regime],
                label=REGIME_LABELS[regime], edgecolor="white")
    ax.set_xlabel("ACF(1)")
    ax.set_ylabel("Count")
    ax.set_title("Temporal Autocorrelation")
    ax.legend(fontsize=5.5)

    fig.suptitle("Jitter and Variability Characterization",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig10_jitter_characterization")


def fig11_violin_comparison(ripe_data: Dict[str, pd.DataFrame],
                             all_stats: Dict) -> str:
    setup_style()
    fig, ax = plt.subplots(figsize=FIG_DOUBLE_COL)

    data_by_regime = []
    positions = []
    colors = []
    labels = []
    pos = 0

    for regime in ["short_haul", "regional", "intercontinental"]:
        paths = sorted([pid for pid, info in all_stats.items()
                        if info["path_def"].regime == regime])
        for pid in paths:
            rtt = ripe_data[pid]["rtt_ms"].values
            subsample = np.random.choice(rtt, min(500, len(rtt)), replace=False)
            data_by_regime.append(subsample)
            positions.append(pos)
            colors.append(REGIME_COLORS[regime])
            p = all_stats[pid]["path_def"]
            labels.append(f"{p.source_city[:3]}-{p.dest_city[:3]}")
            pos += 1
        pos += 0.5

    vp = ax.violinplot(data_by_regime, positions=positions, showmedians=True,
                        showextrema=False, widths=0.7)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.6)
    vp["cmedians"].set_color("black")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, fontsize=5.5)
    ax.set_ylabel("RTT (ms)")
    ax.set_title("RTT Distribution: All Paths (Violin Plot)")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=REGIME_COLORS[r], alpha=0.6,
                              label=REGIME_LABELS[r])
                       for r in ["short_haul", "regional", "intercontinental"]]
    ax.legend(handles=legend_elements, loc="upper left")
    fig.tight_layout()
    return _save(fig, "fig11_violin_comparison")


def fig12_mawi_traffic_correlation(mawi_volume: pd.DataFrame,
                                    mawi_rtts: pd.DataFrame) -> str:
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.0), sharex=True)

    ax = axes[0]
    hourly_vol = mawi_volume.groupby(mawi_volume["hour_jst"].astype(int) % 24)[
        "packets_per_s"].mean()
    ax.plot(hourly_vol.index, hourly_vol.values / 1e6, "o-",
            color="#2196F3", markersize=3, linewidth=1.5)
    ax.set_ylabel("Traffic (M pkts/s)")
    ax.set_title("MAWI: Diurnal Traffic Volume (JST)")

    ax = axes[1]
    mawi_rtts["hour_jst"] = ((mawi_rtts["timestamp_s"] / 3600 + 9) % 24).astype(int)
    hourly_rtt = mawi_rtts.groupby("hour_jst")["rtt_ms"].agg(["mean", "median", "std"])
    ax.plot(hourly_rtt.index, hourly_rtt["mean"], "o-",
            color="#FF9800", markersize=3, linewidth=1.5, label="Mean")
    ax.plot(hourly_rtt.index, hourly_rtt["median"], "s--",
            color="#4CAF50", markersize=3, linewidth=1, label="Median")
    ax.fill_between(hourly_rtt.index,
                    hourly_rtt["mean"] - hourly_rtt["std"],
                    hourly_rtt["mean"] + hourly_rtt["std"],
                    alpha=0.15, color="#FF9800")
    ax.set_xlabel("Hour of Day (JST)")
    ax.set_ylabel("SYN-ACK RTT (ms)")
    ax.set_title("MAWI: Diurnal SYN-ACK RTT Pattern")
    ax.legend()

    fig.suptitle("MAWI Traffic and RTT Correlation", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig12_mawi_correlation")


def fig13_percentile_profiles(all_stats: Dict, taxonomy: Dict) -> str:
    setup_style()
    fig, ax = plt.subplots(figsize=FIG_DOUBLE_COL)

    categories = ["excellent", "good", "average", "poor", "very_poor"]
    display = ["Excellent", "Good", "Average", "Poor", "Very Poor"]
    cat_colors = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336"]

    for i, cat in enumerate(categories):
        lo, hi = taxonomy[cat]["rtt_range"]
        matching_paths = [pid for pid, info in all_stats.items()
                         if lo <= info["statistics"]["median"] <= hi]
        if not matching_paths:
            continue
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        all_pct_vals = []
        for pid in matching_paths:
            vals = [all_stats[pid]["statistics"][f"p{p}"] for p in pcts]
            all_pct_vals.append(vals)
        mean_pcts = np.mean(all_pct_vals, axis=0)
        ax.plot(pcts, mean_pcts, "o-", color=cat_colors[i],
                label=f"{display[i]} ({taxonomy[cat]['rtt_typical']:.0f}ms)",
                linewidth=1.5, markersize=4)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("RTT (ms)")
    ax.set_title("RTT Percentile Profiles by Empirical Category")
    ax.legend()
    ax.set_xticks(pcts)
    fig.tight_layout()
    return _save(fig, "fig13_percentile_profiles")


def generate_all_figures(ripe_data, all_stats, regime_summary, temporal,
                          regime_hourly, cross_val, taxonomy, comparison,
                          game_map, mawi_volume, mawi_rtts) -> List[str]:
    print("\n  Generating publication figures...")
    figures = []

    figs = [
        ("Fig 1: Regime distributions", lambda: fig1_regime_distributions(ripe_data, all_stats)),
        ("Fig 2: CDF comparison", lambda: fig2_cdf_comparison(ripe_data, all_stats)),
        ("Fig 3: Time-of-day heatmap", lambda: fig3_tod_heatmap(ripe_data, all_stats)),
        ("Fig 4: Diurnal pattern", lambda: fig4_diurnal_pattern(regime_hourly)),
        ("Fig 5: Day-of-week pattern", lambda: fig5_dow_pattern(ripe_data, all_stats)),
        ("Fig 6: Distribution fits", lambda: fig6_distribution_fits(ripe_data, all_stats)),
        ("Fig 7: Cross-validation", lambda: fig7_cross_validation(cross_val, mawi_rtts, ripe_data)),
        ("Fig 8: Taxonomy comparison", lambda: fig8_taxonomy_comparison(taxonomy, comparison)),
        ("Fig 9: Game relevance", lambda: fig9_game_relevance(game_map, taxonomy)),
        ("Fig 10: Jitter characterization", lambda: fig10_jitter_characterization(ripe_data, all_stats)),
        ("Fig 11: Violin comparison", lambda: fig11_violin_comparison(ripe_data, all_stats)),
        ("Fig 12: MAWI correlation", lambda: fig12_mawi_traffic_correlation(mawi_volume, mawi_rtts)),
        ("Fig 13: Percentile profiles", lambda: fig13_percentile_profiles(all_stats, taxonomy)),
    ]

    for name, fn in figs:
        try:
            path = fn()
            print(f"    {name}: {path}")
            figures.append(path)
        except Exception as e:
            print(f"    {name}: FAILED - {e}")

    return figures
