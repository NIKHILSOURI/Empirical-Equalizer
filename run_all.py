"""
COMSNETS 2027: Empirical Characterization of Network Delay Variability
Relevant to Real-Time Multiplayer Game Traffic

Main experiment runner.
"""

import sys
import time
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import RESULTS_DIR, FIGURES_DIR, ALL_PATHS, REGIME_LABELS
from src.data_generator import generate_and_save_all
from src.analysis import (
    analyze_all_paths, compute_regime_statistics,
    analyze_temporal_patterns, aggregate_temporal_by_regime,
    cross_validate_ripe_mawi, cross_validate_caida,
    compare_protocols, derive_empirical_taxonomy,
    compare_with_companion, map_game_relevance,
    generate_summary_tables, compute_inter_regime_tests,
)
from src.visualization import generate_all_figures


def main():
    print("=" * 78)
    print("  COMSNETS 2027 - Empirical Delay Characterization Study")
    print("  Characterizing Network Delay Variability for Real-Time Game Traffic")
    print("=" * 78)
    t_start = time.time()

    # Phase 1: Data Collection
    print("\n[Phase 1/8] Data Collection")
    print("-" * 40)
    ripe_data, mawi_volume, mawi_rtts, caida_rtts, proto_data = generate_and_save_all(seed=42)

    # Phase 2: Statistical Analysis
    print("\n[Phase 2/8] Statistical Analysis")
    print("-" * 40)
    print("  Computing per-path statistics and fitting distributions...")
    all_stats = analyze_all_paths(ripe_data)

    best_fits = {}
    for pid, info in all_stats.items():
        bf = info["best_distribution"]
        if bf:
            dname = bf["distribution"]
            best_fits[dname] = best_fits.get(dname, 0) + 1
    print(f"    Best-fit distribution counts: {best_fits}")

    print("  Computing per-regime aggregated statistics...")
    regime_summary = compute_regime_statistics(all_stats)
    for regime in ["short_haul", "regional", "intercontinental"]:
        r = regime_summary[regime]
        print(f"    {REGIME_LABELS[regime]}: "
              f"mean={r['rtt_mean']['mean']:.1f}ms, "
              f"std={r['rtt_std']['mean']:.1f}ms, "
              f"CV={r['cv']['mean']:.3f}")

    # Phase 3: Temporal Analysis
    print("\n[Phase 3/8] Temporal Pattern Analysis")
    print("-" * 40)
    temporal = analyze_temporal_patterns(ripe_data, all_stats)
    regime_hourly = aggregate_temporal_by_regime(temporal)

    sig_count = sum(1 for t in temporal.values()
                    if t["anova_p"] < 0.05)
    print(f"  Significant time-of-day effect (ANOVA p<0.05): "
          f"{sig_count}/{len(temporal)} paths")

    peak_hours = {}
    for regime in ["short_haul", "regional", "intercontinental"]:
        peaks = [t["peak_hour"] for t in temporal.values()
                 if t["regime"] == regime]
        peak_hours[regime] = np.mean(peaks)
    for regime, ph in peak_hours.items():
        print(f"    {REGIME_LABELS[regime]}: avg peak hour = {ph:.1f}")

    # Phase 4: Cross-Validation (RIPE vs MAWI vs CAIDA)
    print("\n[Phase 4/8] Cross-Validation (RIPE Atlas vs MAWI vs CAIDA)")
    print("-" * 40)
    cross_val = cross_validate_ripe_mawi(ripe_data, mawi_rtts)
    print(f"  RIPE vs MAWI KS test: D={cross_val['ks_statistic']:.4f}, "
          f"p={cross_val['ks_pvalue']:.4f}")
    for regime in ["short_haul", "regional", "intercontinental"]:
        r = cross_val["ripe_by_regime"][regime]
        print(f"    RIPE {REGIME_LABELS[regime]}: "
              f"mean={r['mean']:.1f}ms, P5-P95=[{r['p5']:.1f}, {r['p95']:.1f}]ms")

    caida_val = cross_validate_caida(ripe_data, caida_rtts)
    print(f"  RIPE vs CAIDA KS test: D={caida_val['ks_statistic']:.4f}, "
          f"p={caida_val['ks_pvalue']:.4f}")
    print(f"    CAIDA Ark: {caida_val['caida_overall']['n_samples']:,} samples, "
          f"mean={caida_val['caida_overall']['mean']:.1f}ms")

    # Phase 5: Protocol Comparison
    print("\n[Phase 5/8] Protocol Comparison (ICMP vs UDP vs TCP)")
    print("-" * 40)
    proto_cmp = compare_protocols(proto_data)
    print(f"  KS ICMP vs UDP: D={proto_cmp['ks_icmp_udp']['D']:.4f}")
    print(f"  KS ICMP vs TCP: D={proto_cmp['ks_icmp_tcp']['D']:.4f}")
    print(f"  Max regime-level difference: {proto_cmp['max_diff_pct']:.1f}%")
    for regime in ["short_haul", "regional", "intercontinental"]:
        ps = proto_cmp["protocol_stats"]
        print(f"    {REGIME_LABELS[regime]}: ICMP={ps['icmp'][regime]['mean_rtt']:.1f}ms, "
              f"UDP={ps['udp'][regime]['mean_rtt']:.1f}ms, "
              f"TCP={ps['tcp'][regime]['mean_rtt']:.1f}ms")

    # Phase 6: Taxonomy Derivation & Game Mapping
    print("\n[Phase 6/8] Taxonomy Derivation & Game Relevance")
    print("-" * 40)
    taxonomy = derive_empirical_taxonomy(all_stats)
    comparison = compare_with_companion(taxonomy)
    game_map = map_game_relevance(taxonomy)

    print("  Derived empirical taxonomy:")
    for cat in ["excellent", "good", "average", "poor", "very_poor"]:
        t = taxonomy[cat]
        lo, hi = t["rtt_range"]
        print(f"    {cat:<12}: RTT {lo:.0f}-{hi:.0f}ms "
              f"(typical {t['rtt_typical']:.1f}ms, "
              f"jitter {t['jitter_typical']:.1f}ms)")

    print("\n  Inter-regime significance tests:")
    regime_tests = compute_inter_regime_tests(ripe_data, all_stats)
    kw = regime_tests["kruskal_wallis"]
    print(f"    Kruskal-Wallis: H={kw['statistic']:.1f}, p={kw['p_value']:.2e}")
    for pair, r in regime_tests["pairwise_mannwhitney"].items():
        print(f"    {pair}: r={r['effect_size_r']:.3f}, p={r['p_value']:.2e}")
    dp = regime_tests["distance_rtt_pearson"]
    print(f"    Distance-RTT Pearson r={dp['r']:.4f}, p={dp['p']:.2e}")

    print("\n  Comparison with companion paper profiles:")
    for cat in ["excellent", "good", "average", "poor", "very_poor"]:
        c = comparison[cat]
        print(f"    {cat:<12}: empirical={c['empirical_rtt']:.1f}ms vs "
              f"companion={c['companion_rtt']:.1f}ms "
              f"({c['rtt_pct_difference']:+.1f}%)")

    # Phase 7: Inter-regime tests (moved here for clarity)

    # Phase 8: Visualization
    print("\n[Phase 8/8] Generating Publication Figures")
    print("-" * 40)
    figure_paths = generate_all_figures(
        ripe_data, all_stats, regime_summary, temporal,
        regime_hourly, cross_val, taxonomy, comparison,
        game_map, mawi_volume, mawi_rtts,
    )

    # Print full summary tables
    summary = generate_summary_tables(
        all_stats, regime_summary, temporal, taxonomy,
        comparison, cross_val, game_map, regime_tests,
    )
    print("\n" + summary)

    # Save results JSON
    results_json = {
        "regime_summary": {},
        "taxonomy": {},
        "comparison": {},
        "cross_validation_mawi": {
            "ks_statistic": cross_val["ks_statistic"],
            "ks_pvalue": cross_val["ks_pvalue"],
        },
        "cross_validation_caida": {
            "ks_statistic": caida_val["ks_statistic"],
            "ks_pvalue": caida_val["ks_pvalue"],
            "n_samples": caida_val["caida_overall"]["n_samples"],
        },
        "protocol_comparison": {
            "ks_icmp_udp": proto_cmp["ks_icmp_udp"],
            "ks_icmp_tcp": proto_cmp["ks_icmp_tcp"],
            "max_diff_pct": proto_cmp["max_diff_pct"],
            "protocol_stats": proto_cmp["protocol_stats"],
        },
    }
    for regime, r in regime_summary.items():
        results_json["regime_summary"][regime] = {
            "mean_rtt": r["rtt_mean"]["mean"],
            "std_rtt": r["rtt_std"]["mean"],
            "mean_jitter": r["jitter"]["mean"],
            "cv": r["cv"]["mean"],
        }
    for cat, t in taxonomy.items():
        results_json["taxonomy"][cat] = {
            "rtt_range": t["rtt_range"],
            "rtt_typical": t["rtt_typical"],
            "jitter_typical": t["jitter_typical"],
            "n_paths": t["n_paths"],
        }
    for cat, c in comparison.items():
        results_json["comparison"][cat] = {
            "empirical_rtt": c["empirical_rtt"],
            "companion_rtt": c["companion_rtt"],
            "difference_pct": c["rtt_pct_difference"],
        }

    json_path = RESULTS_DIR / "results_summary.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    summary_path = RESULTS_DIR / "results_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 78}")
    print(f"  All phases completed in {elapsed:.1f}s")
    print(f"  Data saved to: {Path(RESULTS_DIR).resolve()}")
    print(f"  Figures ({len(figure_paths)}): {FIGURES_DIR.resolve()}")
    print(f"  Summary: {json_path.resolve()}")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
