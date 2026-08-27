import warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from typing import Dict, List, Tuple, Any
from .config import (
    ALL_PATHS, DISTRIBUTION_CANDIDATES, SIGNIFICANCE_LEVEL,
    PERCENTILES, REGIME_LABELS, COMPANION_PROFILES, GAME_THRESHOLDS,
)


def compute_path_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    rtt = df["rtt_ms"].values
    diffs = np.diff(rtt)

    pct_dict = {}
    for p in PERCENTILES:
        pct_dict[f"p{p}"] = float(np.percentile(rtt, p))

    acf1 = float(np.corrcoef(rtt[:-1], rtt[1:])[0, 1]) if len(rtt) > 2 else 0.0
    iqr = pct_dict["p75"] - pct_dict["p25"]

    return {
        "n_samples": len(rtt),
        "mean": float(np.mean(rtt)),
        "median": float(np.median(rtt)),
        "std": float(np.std(rtt)),
        "min": float(np.min(rtt)),
        "max": float(np.max(rtt)),
        "skewness": float(sp_stats.skew(rtt)),
        "kurtosis": float(sp_stats.kurtosis(rtt)),
        "cv": float(np.std(rtt) / np.mean(rtt)) if np.mean(rtt) > 0 else 0,
        "iqr": iqr,
        "acf1": acf1,
        "jitter_mean": float(np.mean(np.abs(diffs))),
        "jitter_std": float(np.std(diffs)),
        "jitter_p95": float(np.percentile(np.abs(diffs), 95)),
        **pct_dict,
    }


def fit_distributions(rtt: np.ndarray) -> List[Dict[str, Any]]:
    results = []
    for dist_name in DISTRIBUTION_CANDIDATES:
        dist = getattr(sp_stats, dist_name)
        try:
            params = dist.fit(rtt)
            ks_stat, ks_p = sp_stats.kstest(rtt, dist_name, args=params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                ad_result = sp_stats.anderson(rtt, dist_name) if dist_name in [
                    "norm", "expon", "gumbel_r"] else None

            log_likelihood = np.sum(dist.logpdf(rtt, *params))
            k = len(params)
            n = len(rtt)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood

            results.append({
                "distribution": dist_name,
                "params": params,
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_p),
                "ad_statistic": float(ad_result.statistic) if ad_result else None,
                "aic": float(aic),
                "bic": float(bic),
                "log_likelihood": float(log_likelihood),
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["bic"])
    return results


def analyze_all_paths(ripe_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    path_lookup = {p.path_id: p for p in ALL_PATHS}
    all_stats = {}

    for path_id, df in ripe_data.items():
        path_def = path_lookup[path_id]
        basic = compute_path_statistics(df)
        dist_fits = fit_distributions(df["rtt_ms"].values)
        best_fit = dist_fits[0] if dist_fits else None

        all_stats[path_id] = {
            "path_def": path_def,
            "statistics": basic,
            "distribution_fits": dist_fits,
            "best_distribution": best_fit,
        }

    return all_stats


def compute_regime_statistics(all_stats: Dict[str, Dict]) -> Dict[str, Dict]:
    regime_data = {"short_haul": [], "regional": [], "intercontinental": []}
    for path_id, info in all_stats.items():
        regime = info["path_def"].regime
        regime_data[regime].append(info["statistics"])

    regime_summary = {}
    for regime, stats_list in regime_data.items():
        means = [s["mean"] for s in stats_list]
        stds = [s["std"] for s in stats_list]
        medians = [s["median"] for s in stats_list]
        jitters = [s["jitter_mean"] for s in stats_list]
        cvs = [s["cv"] for s in stats_list]
        skews = [s["skewness"] for s in stats_list]

        regime_summary[regime] = {
            "n_paths": len(stats_list),
            "rtt_mean": {"mean": np.mean(means), "std": np.std(means),
                         "min": np.min(means), "max": np.max(means)},
            "rtt_median": {"mean": np.mean(medians), "std": np.std(medians)},
            "rtt_std": {"mean": np.mean(stds), "std": np.std(stds)},
            "jitter": {"mean": np.mean(jitters), "std": np.std(jitters)},
            "cv": {"mean": np.mean(cvs), "std": np.std(cvs)},
            "skewness": {"mean": np.mean(skews), "std": np.std(skews)},
        }

    return regime_summary


def analyze_temporal_patterns(ripe_data: Dict[str, pd.DataFrame],
                              all_stats: Dict[str, Dict]) -> Dict[str, Dict]:
    path_lookup = {p.path_id: p for p in ALL_PATHS}
    temporal = {}

    for path_id, df in ripe_data.items():
        regime = path_lookup[path_id].regime
        hourly = df.groupby(df["hour"].astype(int) % 24)["rtt_ms"].agg(
            ["mean", "std", "median", "count"]
        ).reset_index()
        hourly.columns = ["hour", "mean", "std", "median", "count"]

        daily = df.groupby("day_of_week")["rtt_ms"].agg(
            ["mean", "std", "median", "count"]
        ).reset_index()
        daily.columns = ["day_of_week", "mean", "std", "median", "count"]

        peak_hour = int(hourly.loc[hourly["mean"].idxmax(), "hour"])
        trough_hour = int(hourly.loc[hourly["mean"].idxmin(), "hour"])
        peak_trough_ratio = float(hourly["mean"].max() / hourly["mean"].min())

        f_stat, f_pvalue = sp_stats.f_oneway(
            *[grp["rtt_ms"].values for _, grp in df.groupby(df["hour"].astype(int) % 24)]
        )

        temporal[path_id] = {
            "regime": regime,
            "hourly": hourly,
            "daily": daily,
            "peak_hour": peak_hour,
            "trough_hour": trough_hour,
            "peak_trough_ratio": peak_trough_ratio,
            "anova_f": float(f_stat),
            "anova_p": float(f_pvalue),
        }

    return temporal


def aggregate_temporal_by_regime(temporal: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
    regime_hourly = {"short_haul": [], "regional": [], "intercontinental": []}

    for path_id, t in temporal.items():
        regime = t["regime"]
        regime_hourly[regime].append(t["hourly"].set_index("hour")["mean"])

    aggregated = {}
    for regime, series_list in regime_hourly.items():
        combined = pd.concat(series_list, axis=1)
        aggregated[regime] = pd.DataFrame({
            "hour": combined.index,
            "mean": combined.mean(axis=1).values,
            "std": combined.std(axis=1).values,
            "min": combined.min(axis=1).values,
            "max": combined.max(axis=1).values,
        })

    return aggregated


def cross_validate_ripe_mawi(ripe_data: Dict[str, pd.DataFrame],
                              mawi_rtts: pd.DataFrame) -> Dict[str, Any]:
    all_ripe = pd.concat(ripe_data.values(), ignore_index=True)

    ripe_regimes = {}
    for regime in ["short_haul", "regional", "intercontinental"]:
        subset = all_ripe[all_ripe["regime"] == regime]["rtt_ms"].values
        ripe_regimes[regime] = {
            "mean": float(np.mean(subset)),
            "median": float(np.median(subset)),
            "std": float(np.std(subset)),
            "p5": float(np.percentile(subset, 5)),
            "p95": float(np.percentile(subset, 95)),
        }

    mawi_rtt = mawi_rtts["rtt_ms"].values
    mawi_overall = {
        "mean": float(np.mean(mawi_rtt)),
        "median": float(np.median(mawi_rtt)),
        "std": float(np.std(mawi_rtt)),
        "p5": float(np.percentile(mawi_rtt, 5)),
        "p95": float(np.percentile(mawi_rtt, 95)),
    }

    mawi_bins = [0, 20, 60, 150, 500]
    mawi_labels = ["short_haul_proxy", "regional_proxy", "intercon_proxy"]
    mawi_rtt_clipped = np.clip(mawi_rtt, mawi_bins[0], mawi_bins[-1])
    bin_indices = np.digitize(mawi_rtt_clipped, mawi_bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(mawi_labels) - 1)

    mawi_regime_proxy = {}
    for i, label in enumerate(mawi_labels):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            subset = mawi_rtt[mask]
            mawi_regime_proxy[label] = {
                "mean": float(np.mean(subset)),
                "median": float(np.median(subset)),
                "std": float(np.std(subset)),
                "count": int(np.sum(mask)),
                "fraction": float(np.sum(mask) / len(mawi_rtt)),
            }

    ripe_all = all_ripe["rtt_ms"].values
    ks_stat, ks_p = sp_stats.ks_2samp(
        np.random.choice(ripe_all, min(5000, len(ripe_all)), replace=False),
        np.random.choice(mawi_rtt, min(5000, len(mawi_rtt)), replace=False),
    )

    return {
        "ripe_by_regime": ripe_regimes,
        "mawi_overall": mawi_overall,
        "mawi_regime_proxy": mawi_regime_proxy,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "note": ("MAWI SYN-ACK RTTs are single-vantage-point proxies, not direct "
                 "end-to-end measurements. Comparison is indicative, not exact."),
    }


def cross_validate_caida(ripe_data: Dict[str, pd.DataFrame],
                         caida_rtts: pd.DataFrame) -> Dict[str, Any]:
    all_ripe = pd.concat(ripe_data.values(), ignore_index=True)
    ripe_rtt = all_ripe["rtt_ms"].values
    caida_rtt = caida_rtts["rtt_ms"].values

    caida_overall = {
        "mean": float(np.mean(caida_rtt)),
        "median": float(np.median(caida_rtt)),
        "std": float(np.std(caida_rtt)),
        "n_samples": len(caida_rtt),
    }

    bins = [0, 20, 60, 150, 500]
    labels = ["domestic", "regional", "intercontinental"]
    clipped = np.clip(caida_rtt, bins[0], bins[-1])
    indices = np.clip(np.digitize(clipped, bins) - 1, 0, len(labels) - 1)
    caida_regime_proxy = {}
    for i, label in enumerate(labels):
        mask = indices == i
        if np.sum(mask) > 0:
            subset = caida_rtt[mask]
            caida_regime_proxy[label] = {
                "mean": float(np.mean(subset)),
                "median": float(np.median(subset)),
                "count": int(np.sum(mask)),
            }

    ks_stat, ks_p = sp_stats.ks_2samp(
        np.random.choice(ripe_rtt, min(5000, len(ripe_rtt)), replace=False),
        np.random.choice(caida_rtt, min(5000, len(caida_rtt)), replace=False),
    )

    return {
        "caida_overall": caida_overall,
        "caida_regime_proxy": caida_regime_proxy,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
    }


def compare_protocols(proto_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
    regime_lookup = {p.path_id: p.regime for p in ALL_PATHS}
    protocol_stats = {}
    for proto, path_dict in proto_data.items():
        regime_means = {"short_haul": [], "regional": [], "intercontinental": []}
        regime_jitters = {"short_haul": [], "regional": [], "intercontinental": []}
        for pid, df in path_dict.items():
            regime = regime_lookup[pid]
            rtt = df["rtt_ms"].values
            regime_means[regime].append(float(np.mean(rtt)))
            diffs = np.diff(rtt)
            regime_jitters[regime].append(float(np.mean(np.abs(diffs))))
        protocol_stats[proto] = {
            r: {"mean_rtt": float(np.mean(v)), "mean_jitter": float(np.mean(regime_jitters[r]))}
            for r, v in regime_means.items()
        }

    comparisons = {}
    for regime in ["short_haul", "regional", "intercontinental"]:
        icmp_mean = protocol_stats["icmp"][regime]["mean_rtt"]
        for proto in ["udp", "tcp"]:
            p_mean = protocol_stats[proto][regime]["mean_rtt"]
            diff_pct = (p_mean - icmp_mean) / icmp_mean * 100
            comparisons[f"{proto}_vs_icmp_{regime}"] = {
                "icmp_mean": icmp_mean,
                f"{proto}_mean": p_mean,
                "diff_pct": float(diff_pct),
            }

    all_icmp = []
    all_udp = []
    all_tcp = []
    for pid in proto_data["icmp"]:
        all_icmp.extend(proto_data["icmp"][pid]["rtt_ms"].values[:500])
        all_udp.extend(proto_data["udp"][pid]["rtt_ms"].values[:500])
        all_tcp.extend(proto_data["tcp"][pid]["rtt_ms"].values[:500])
    ks_udp, p_udp = sp_stats.ks_2samp(
        np.array(all_icmp[:5000]), np.array(all_udp[:5000]))
    ks_tcp, p_tcp = sp_stats.ks_2samp(
        np.array(all_icmp[:5000]), np.array(all_tcp[:5000]))

    return {
        "protocol_stats": protocol_stats,
        "comparisons": comparisons,
        "ks_icmp_udp": {"D": float(ks_udp), "p": float(p_udp)},
        "ks_icmp_tcp": {"D": float(ks_tcp), "p": float(p_tcp)},
        "max_diff_pct": float(max(
            abs(v["diff_pct"]) for v in comparisons.values())),
    }


def derive_empirical_taxonomy(all_stats: Dict[str, Dict]) -> Dict[str, Dict]:
    all_medians = []
    all_jitters = []
    for info in all_stats.values():
        all_medians.append(info["statistics"]["median"])
        all_jitters.append(info["statistics"]["jitter_mean"])

    all_medians = np.array(sorted(all_medians))

    boundaries = np.percentile(all_medians, [15, 35, 60, 80])

    taxonomy = {
        "excellent": {
            "rtt_range": (float(np.min(all_medians)), float(boundaries[0])),
            "rtt_typical": float(np.mean(all_medians[all_medians <= boundaries[0]])),
            "description": "Very low latency, same-city/short-haul paths",
        },
        "good": {
            "rtt_range": (float(boundaries[0]), float(boundaries[1])),
            "rtt_typical": float(np.mean(all_medians[(all_medians > boundaries[0]) &
                                                      (all_medians <= boundaries[1])])),
            "description": "Low latency, short-haul to nearby regional",
        },
        "average": {
            "rtt_range": (float(boundaries[1]), float(boundaries[2])),
            "rtt_typical": float(np.mean(all_medians[(all_medians > boundaries[1]) &
                                                      (all_medians <= boundaries[2])])),
            "description": "Moderate latency, regional paths",
        },
        "poor": {
            "rtt_range": (float(boundaries[2]), float(boundaries[3])),
            "rtt_typical": float(np.mean(all_medians[(all_medians > boundaries[2]) &
                                                      (all_medians <= boundaries[3])])),
            "description": "High latency, long regional or short intercontinental",
        },
        "very_poor": {
            "rtt_range": (float(boundaries[3]), float(np.max(all_medians))),
            "rtt_typical": float(np.mean(all_medians[all_medians > boundaries[3]])),
            "description": "Very high latency, long intercontinental paths",
        },
    }

    for cat_name, cat_info in taxonomy.items():
        lo, hi = cat_info["rtt_range"]
        if cat_name == "excellent":
            mask = [(lo <= info["statistics"]["median"] <= hi) for info in all_stats.values()]
        else:
            mask = [(lo < info["statistics"]["median"] <= hi) for info in all_stats.values()]
        matching_jitters = [info["statistics"]["jitter_mean"]
                           for info, m in zip(all_stats.values(), mask) if m]
        matching_stds = [info["statistics"]["std"]
                        for info, m in zip(all_stats.values(), mask) if m]
        cat_info["jitter_typical"] = float(np.mean(matching_jitters)) if matching_jitters else 0
        cat_info["std_typical"] = float(np.mean(matching_stds)) if matching_stds else 0
        cat_info["n_paths"] = sum(mask)

    return taxonomy


def compare_with_companion(taxonomy: Dict[str, Dict]) -> Dict[str, Dict]:
    comparison = {}
    for cat_name in taxonomy:
        emp = taxonomy[cat_name]
        comp = COMPANION_PROFILES.get(cat_name, {})
        if comp:
            rtt_diff = emp["rtt_typical"] - comp["base_delay_ms"]
            jitter_diff = emp["jitter_typical"] - comp["jitter_ms"]
            comparison[cat_name] = {
                "empirical_rtt": emp["rtt_typical"],
                "companion_rtt": comp["base_delay_ms"],
                "rtt_difference": rtt_diff,
                "rtt_pct_difference": rtt_diff / comp["base_delay_ms"] * 100,
                "empirical_jitter": emp["jitter_typical"],
                "companion_jitter": comp["jitter_ms"],
                "jitter_difference": jitter_diff,
                "jitter_pct_difference": jitter_diff / comp["jitter_ms"] * 100
                if comp["jitter_ms"] > 0 else 0,
                "empirical_range": emp["rtt_range"],
            }
    return comparison


def map_game_relevance(taxonomy: Dict[str, Dict]) -> Dict[str, Dict]:
    mapping = {}
    for genre, thresholds in GAME_THRESHOLDS.items():
        genre_map = {}
        for cat_name, cat_info in taxonomy.items():
            rtt = cat_info["rtt_typical"]
            if rtt <= thresholds["excellent"]:
                quality = "Excellent"
            elif rtt <= thresholds["acceptable"]:
                quality = "Acceptable"
            elif rtt <= thresholds["degraded"]:
                quality = "Degraded"
            else:
                quality = "Unplayable"
            genre_map[cat_name] = {
                "rtt_ms": rtt,
                "quality": quality,
                "threshold_excellent": thresholds["excellent"],
                "threshold_acceptable": thresholds["acceptable"],
            }
        mapping[genre] = genre_map
    return mapping


def compute_inter_regime_tests(ripe_data: Dict[str, pd.DataFrame],
                                all_stats: Dict) -> Dict[str, Any]:
    regime_arrays = {"short_haul": [], "regional": [], "intercontinental": []}
    for pid, info in all_stats.items():
        regime_arrays[info["path_def"].regime].append(
            ripe_data[pid]["rtt_ms"].values)

    regime_combined = {r: np.concatenate(arrs) for r, arrs in regime_arrays.items()}

    kw_stat, kw_p = sp_stats.kruskal(*regime_combined.values())

    pairwise = {}
    regime_list = list(regime_combined.keys())
    for i in range(len(regime_list)):
        for j in range(i + 1, len(regime_list)):
            r1, r2 = regime_list[i], regime_list[j]
            s1, s2 = regime_combined[r1], regime_combined[r2]
            u_stat, u_p = sp_stats.mannwhitneyu(s1, s2, alternative="two-sided")
            n1, n2 = len(s1), len(s2)
            r_effect = 1 - (2 * u_stat) / (n1 * n2)
            pairwise[f"{r1}_vs_{r2}"] = {
                "u_statistic": float(u_stat),
                "p_value": float(u_p),
                "effect_size_r": float(r_effect),
            }

    distances = [info["path_def"].propagation_km for info in all_stats.values()]
    medians = [info["statistics"]["median"] for info in all_stats.values()]
    jitters = [info["statistics"]["jitter_mean"] for info in all_stats.values()]

    r_dist_rtt, p_dist_rtt = sp_stats.pearsonr(distances, medians)
    r_dist_jit, p_dist_jit = sp_stats.pearsonr(distances, jitters)
    rho_dist_rtt, sp_dist_rtt = sp_stats.spearmanr(distances, medians)

    return {
        "kruskal_wallis": {"statistic": float(kw_stat), "p_value": float(kw_p)},
        "pairwise_mannwhitney": pairwise,
        "distance_rtt_pearson": {"r": float(r_dist_rtt), "p": float(p_dist_rtt)},
        "distance_rtt_spearman": {"rho": float(rho_dist_rtt), "p": float(sp_dist_rtt)},
        "distance_jitter_pearson": {"r": float(r_dist_jit), "p": float(p_dist_jit)},
    }


def generate_summary_tables(all_stats: Dict, regime_summary: Dict,
                            temporal: Dict, taxonomy: Dict,
                            comparison: Dict, cross_val: Dict,
                            game_map: Dict,
                            regime_tests: Dict = None) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("  EMPIRICAL DELAY CHARACTERIZATION - RESULTS SUMMARY")
    lines.append("=" * 78)

    lines.append("\n--- Table I: Per-Regime RTT Statistics ---")
    lines.append(f"  {'Regime':<20} {'Mean RTT':>10} {'Median':>10} {'Std':>10} "
                 f"{'Skew':>8} {'CV':>8} {'Jitter':>10}")
    lines.append("  " + "-" * 76)
    for regime in ["short_haul", "regional", "intercontinental"]:
        r = regime_summary[regime]
        lines.append(
            f"  {REGIME_LABELS[regime]:<20} "
            f"{r['rtt_mean']['mean']:>8.1f}ms "
            f"{r['rtt_median']['mean']:>8.1f}ms "
            f"{r['rtt_std']['mean']:>8.1f}ms "
            f"{r['skewness']['mean']:>8.2f} "
            f"{r['cv']['mean']:>8.3f} "
            f"{r['jitter']['mean']:>8.1f}ms"
        )

    lines.append("\n--- Table II: Per-Path Statistics (sorted by median RTT) ---")
    lines.append(f"  {'Path':<8} {'Route':<28} {'Regime':<15} {'Median':>8} "
                 f"{'Mean':>8} {'Std':>8} {'P95':>8} {'Jitter':>8}")
    lines.append("  " + "-" * 97)
    sorted_paths = sorted(all_stats.items(),
                          key=lambda x: x[1]["statistics"]["median"])
    for path_id, info in sorted_paths:
        p = info["path_def"]
        s = info["statistics"]
        route = f"{p.source_city}-{p.dest_city}"
        if len(route) > 26:
            route = route[:26]
        lines.append(
            f"  {path_id:<8} {route:<28} {REGIME_LABELS[p.regime]:<15} "
            f"{s['median']:>6.1f}ms {s['mean']:>6.1f}ms "
            f"{s['std']:>6.1f}ms {s['p95']:>6.1f}ms {s['jitter_mean']:>6.1f}ms"
        )

    lines.append("\n--- Table III: Best-Fit Distribution per Path ---")
    lines.append(f"  {'Path':<8} {'Best Fit':<12} {'KS stat':>10} {'KS p':>10} "
                 f"{'AIC':>12} {'BIC':>12}")
    lines.append("  " + "-" * 64)
    for path_id, info in sorted_paths:
        bf = info["best_distribution"]
        if bf:
            lines.append(
                f"  {path_id:<8} {bf['distribution']:<12} "
                f"{bf['ks_statistic']:>10.4f} {bf['ks_pvalue']:>10.4f} "
                f"{bf['aic']:>12.1f} {bf['bic']:>12.1f}"
            )

    lines.append("\n--- Table IV: Time-of-Day Effect ---")
    lines.append(f"  {'Path':<8} {'Peak Hr':>8} {'Trough Hr':>10} "
                 f"{'Peak/Trough':>12} {'ANOVA F':>10} {'p-value':>10}")
    lines.append("  " + "-" * 60)
    for path_id in sorted(temporal.keys()):
        t = temporal[path_id]
        sig = "*" if t["anova_p"] < SIGNIFICANCE_LEVEL else ""
        lines.append(
            f"  {path_id:<8} {t['peak_hour']:>8d} {t['trough_hour']:>10d} "
            f"{t['peak_trough_ratio']:>12.3f} {t['anova_f']:>10.1f} "
            f"{t['anova_p']:>9.4f}{sig}"
        )

    lines.append("\n--- Table V: Empirical Delay/Jitter Profile Taxonomy ---")
    lines.append(f"  {'Category':<12} {'RTT Range':>18} {'Typical RTT':>12} "
                 f"{'Typical Jitter':>14} {'Paths':>6}")
    lines.append("  " + "-" * 64)
    for cat in ["excellent", "good", "average", "poor", "very_poor"]:
        t = taxonomy[cat]
        lo, hi = t["rtt_range"]
        lines.append(
            f"  {cat:<12} {lo:>6.0f} - {hi:>5.0f} ms "
            f"{t['rtt_typical']:>10.1f}ms "
            f"{t['jitter_typical']:>12.1f}ms {t['n_paths']:>6d}"
        )

    lines.append("\n--- Table VI: Empirical vs Companion Paper Profiles ---")
    lines.append(f"  {'Category':<12} {'Emp RTT':>10} {'Comp RTT':>10} "
                 f"{'Diff':>8} {'Emp Jit':>10} {'Comp Jit':>10} {'Diff':>8}")
    lines.append("  " + "-" * 68)
    for cat in ["excellent", "good", "average", "poor", "very_poor"]:
        c = comparison[cat]
        lines.append(
            f"  {cat:<12} {c['empirical_rtt']:>8.1f}ms {c['companion_rtt']:>8.1f}ms "
            f"{c['rtt_difference']:>+7.1f} {c['empirical_jitter']:>8.1f}ms "
            f"{c['companion_jitter']:>8.1f}ms {c['jitter_difference']:>+7.1f}"
        )

    lines.append("\n--- Table VII: Cross-Validation (RIPE Atlas vs MAWI SYN-ACK) ---")
    lines.append(f"  RIPE Atlas by regime:")
    for regime in ["short_haul", "regional", "intercontinental"]:
        r = cross_val["ripe_by_regime"][regime]
        lines.append(f"    {REGIME_LABELS[regime]:<20} mean={r['mean']:.1f}ms "
                     f"median={r['median']:.1f}ms std={r['std']:.1f}ms")
    lines.append(f"  MAWI overall: mean={cross_val['mawi_overall']['mean']:.1f}ms "
                 f"median={cross_val['mawi_overall']['median']:.1f}ms")
    lines.append(f"  KS test (RIPE vs MAWI): D={cross_val['ks_statistic']:.4f}, "
                 f"p={cross_val['ks_pvalue']:.4f}")
    lines.append(f"  Note: {cross_val['note']}")

    lines.append("\n--- Table VIII: Game Relevance Mapping ---")
    header = f"  {'Profile':<12}"
    for genre in GAME_THRESHOLDS:
        header += f" {genre:>16}"
    lines.append(header)
    lines.append("  " + "-" * (12 + 17 * len(GAME_THRESHOLDS)))
    for cat in ["excellent", "good", "average", "poor", "very_poor"]:
        row = f"  {cat:<12}"
        for genre in GAME_THRESHOLDS:
            quality = game_map[genre][cat]["quality"]
            row += f" {quality:>16}"
        lines.append(row)

    if regime_tests:
        lines.append("\n--- Table IX: Inter-Regime Statistical Tests ---")
        kw = regime_tests["kruskal_wallis"]
        lines.append(f"  Kruskal-Wallis H={kw['statistic']:.1f}, p={kw['p_value']:.2e}")
        lines.append(f"  Pairwise Mann-Whitney U tests:")
        for pair, result in regime_tests["pairwise_mannwhitney"].items():
            lines.append(f"    {pair:<35} U={result['u_statistic']:.0f}, "
                         f"p={result['p_value']:.2e}, r={result['effect_size_r']:.3f}")
        dp = regime_tests["distance_rtt_pearson"]
        ds = regime_tests["distance_rtt_spearman"]
        dj = regime_tests["distance_jitter_pearson"]
        lines.append(f"\n  Distance-RTT correlation:")
        lines.append(f"    Pearson  r={dp['r']:.4f}, p={dp['p']:.2e}")
        lines.append(f"    Spearman rho={ds['rho']:.4f}, p={ds['p']:.2e}")
        lines.append(f"  Distance-Jitter correlation:")
        lines.append(f"    Pearson  r={dj['r']:.4f}, p={dj['p']:.2e}")

    lines.append("\n" + "=" * 78)
    return "\n".join(lines)
