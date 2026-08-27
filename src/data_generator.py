import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .config import (
    PathDefinition, ALL_PATHS, MEASUREMENT_INTERVAL_S,
    DATA_DURATION_DAYS, DATA_DIR,
)

PROTOCOL_OFFSETS = {
    "icmp": 0.0,
    "udp":  0.3,
    "tcp":  0.8,
}
PROTOCOL_JITTER_SCALE = {
    "icmp": 1.0,
    "udp":  1.05,
    "tcp":  1.12,
}


class SyntheticRTTGenerator:
    """Generates realistic RTT time series calibrated to published measurement studies.

    Model: RTT(t) = propagation + overhead(t) + diurnal(t) + jitter(t) + spikes(t)
    - propagation: fixed, from distance / (2/3 * c)
    - overhead: base queuing+processing with AR(1) slow drift
    - diurnal: sinusoidal time-of-day variation (peak ~20:00 local)
    - jitter: AR(1) process with log-normal innovations (right-skewed)
    - spikes: occasional congestion events (Poisson process + exponential magnitude)
    """

    C_FIBER_KM_S = 2.0e5

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_path_data(self, path: PathDefinition,
                           n_days: int = DATA_DURATION_DAYS,
                           interval_s: int = MEASUREMENT_INTERVAL_S,
                           protocol: str = "icmp") -> pd.DataFrame:
        n_samples = int(n_days * 86400 / interval_s)
        timestamps_s = np.arange(n_samples, dtype=np.float64) * interval_s

        prop_delay = path.propagation_km / self.C_FIBER_KM_S * 1000.0 * 2.0
        base_overhead = max(path.expected_rtt_ms - prop_delay, 2.0)

        hours = (timestamps_s / 3600.0) % 24.0
        phase_offset = self.rng.uniform(-2, 2)
        diurnal_amp = path.expected_jitter_ms * 0.4
        diurnal = diurnal_amp * (
            -0.7 * np.cos(2 * np.pi * (hours - 20.0 + phase_offset) / 24.0)
            - 0.3 * np.cos(2 * np.pi * (hours - 13.0 + phase_offset) / 12.0)
        )

        day_of_week = ((timestamps_s / 86400).astype(int) + self.rng.randint(0, 7)) % 7
        weekend_mask = (day_of_week >= 5).astype(float)
        dow_factor = 1.0 + weekend_mask * self.rng.uniform(-0.15, -0.05)

        rho = 0.35
        sigma = path.expected_jitter_ms
        innovations = self.rng.standard_normal(n_samples)
        jitter = np.zeros(n_samples)
        for i in range(1, n_samples):
            jitter[i] = rho * jitter[i - 1] + innovations[i] * sigma * np.sqrt(1 - rho ** 2)

        ln_component = self.rng.lognormal(0, 0.6, n_samples) - 1.0
        jitter = jitter * 0.55 + ln_component * sigma * 0.45

        spike_rate = 0.015
        spike_mask = self.rng.random(n_samples) < spike_rate
        spike_mag = self.rng.exponential(path.expected_jitter_ms * 3.0, n_samples)
        spikes = spike_mask * spike_mag

        rtt = prop_delay + base_overhead * dow_factor + diurnal + jitter + spikes

        proto_offset = PROTOCOL_OFFSETS.get(protocol, 0.0)
        proto_jscale = PROTOCOL_JITTER_SCALE.get(protocol, 1.0)
        rtt = rtt * proto_jscale + proto_offset

        floor = prop_delay * 0.95
        rtt = np.maximum(rtt, floor)

        return pd.DataFrame({
            "timestamp_s": timestamps_s,
            "hour": hours,
            "day_of_week": day_of_week,
            "rtt_ms": rtt,
            "path_id": path.path_id,
            "regime": path.regime,
            "protocol": protocol,
        })

    def generate_all_paths(self) -> Dict[str, pd.DataFrame]:
        data = {}
        for i, path in enumerate(ALL_PATHS):
            self.rng = np.random.RandomState(42 + i * 137)
            data[path.path_id] = self.generate_path_data(path)
        return data

    def generate_all_protocols(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        proto_data = {}
        for proto in ["icmp", "udp", "tcp"]:
            proto_data[proto] = {}
            for i, path in enumerate(ALL_PATHS):
                self.rng = np.random.RandomState(42 + i * 137 + hash(proto) % 1000)
                proto_data[proto][path.path_id] = self.generate_path_data(
                    path, protocol=proto)
        return proto_data


class CAIDAGenerator:
    """Generates synthetic CAIDA Ark traceroute-derived RTT samples."""

    def __init__(self, seed: int = 77):
        self.rng = np.random.RandomState(seed)

    def generate_ark_rtts(self, n_days: int = DATA_DURATION_DAYS,
                          vantage_points: int = 8,
                          traces_per_vp_per_day: int = 50) -> pd.DataFrame:
        all_rows = []
        dest_profiles = [
            ("domestic", 10.0, 0.45, 0.40),
            ("regional", 48.0, 0.38, 0.30),
            ("intercontinental", 165.0, 0.32, 0.20),
            ("satellite", 320.0, 0.28, 0.10),
        ]
        for vp in range(vantage_points):
            vp_bias = self.rng.uniform(0.9, 1.1)
            for day in range(n_days):
                n_traces = traces_per_vp_per_day + self.rng.randint(-10, 10)
                for _ in range(n_traces):
                    hour = self.rng.uniform(0, 24)
                    congestion = 1.0 + 0.12 * np.sin(
                        2 * np.pi * (hour - 15) / 24)
                    category = self.rng.choice(
                        [p[0] for p in dest_profiles],
                        p=[p[3] for p in dest_profiles])
                    prof = next(p for p in dest_profiles if p[0] == category)
                    rtt = self.rng.lognormal(
                        np.log(prof[1] * congestion * vp_bias), prof[2])
                    rtt = max(rtt, 1.0)
                    all_rows.append({
                        "timestamp_s": day * 86400 + hour * 3600,
                        "rtt_ms": rtt,
                        "vantage_point": vp,
                        "day": day,
                        "category": category,
                    })
        return pd.DataFrame(all_rows)


class MAWITrafficGenerator:
    """Generates synthetic MAWI-style traffic data calibrated to published WIDE statistics.

    Models: (a) background traffic volume with Tokyo-timezone diurnal pattern,
            (b) TCP SYN-ACK RTT samples from handshake timing.
    """

    def __init__(self, seed: int = 99):
        self.rng = np.random.RandomState(seed)

    def generate_traffic_volume(self, n_days: int = DATA_DURATION_DAYS,
                                bin_size_s: int = 300) -> pd.DataFrame:
        n_bins = int(n_days * 86400 / bin_size_s)
        timestamps = np.arange(n_bins, dtype=np.float64) * bin_size_s
        hours_jst = (timestamps / 3600.0 + 9) % 24.0

        base_rate = 800000.0
        diurnal = base_rate * 0.4 * np.sin(2 * np.pi * (hours_jst - 14.0) / 24.0)
        noise = self.rng.normal(0, base_rate * 0.05, n_bins)

        day_of_week = (timestamps / 86400).astype(int) % 7
        weekend_factor = np.where((day_of_week >= 5), 0.75, 1.0)

        volume = (base_rate + diurnal + noise) * weekend_factor
        volume = np.maximum(volume, base_rate * 0.3)

        return pd.DataFrame({
            "timestamp_s": timestamps,
            "hour_jst": hours_jst,
            "day_of_week": day_of_week,
            "packets_per_s": volume,
        })

    def generate_synack_rtts(self, n_days: int = DATA_DURATION_DAYS,
                             samples_per_capture: int = 500) -> pd.DataFrame:
        """Simulate TCP SYN-ACK RTT extraction from multiple daily MAWI captures."""
        all_rows = []
        capture_hours = [2, 6, 10, 14, 18, 22]
        for day in range(n_days):
            for cap_hour in capture_hours:
                capture_start = day * 86400 + cap_hour * 3600
                n = samples_per_capture + self.rng.randint(-80, 80)

                congestion = 1.0 + 0.15 * np.sin(2 * np.pi * (cap_hour - 14) / 24)

                domestic = self.rng.lognormal(np.log(12.0 * congestion), 0.4, int(n * 0.35))
                regional = self.rng.lognormal(np.log(55.0 * congestion), 0.35, int(n * 0.30))
                intercon = self.rng.lognormal(np.log(180.0 * congestion), 0.3, int(n * 0.25))
                satellite = self.rng.lognormal(np.log(350.0 * congestion), 0.25, int(n * 0.10))

                rtts = np.concatenate([domestic, regional, intercon, satellite])
                self.rng.shuffle(rtts)
                rtts = np.maximum(rtts, 1.0)

                offsets = self.rng.uniform(0, 900, len(rtts))
                timestamps = capture_start + offsets

                for t, r in zip(timestamps, rtts):
                    all_rows.append({
                        "timestamp_s": t,
                        "rtt_ms": r,
                        "day": day,
                        "day_of_week": day % 7,
                    })

        return pd.DataFrame(all_rows)

    def generate_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.generate_traffic_volume(), self.generate_synack_rtts()


def generate_and_save_all(seed: int = 42):
    n_paths = len(ALL_PATHS)
    n_days = DATA_DURATION_DAYS
    print(f"  Generating RIPE Atlas anchor-mesh RTT data ({n_paths} paths x {n_days} days)...")
    ripe_gen = SyntheticRTTGenerator(seed=seed)
    ripe_data = ripe_gen.generate_all_paths()

    total_samples = sum(len(df) for df in ripe_data.values())
    print(f"    -> {len(ripe_data)} paths, {total_samples:,} total ICMP measurements")

    print("  Generating UDP/TCP protocol variants...")
    proto_data = ripe_gen.generate_all_protocols()
    for proto in ["udp", "tcp"]:
        total_p = sum(len(df) for df in proto_data[proto].values())
        print(f"    -> {proto.upper()}: {total_p:,} measurements")

    print("  Generating MAWI traffic volume and SYN-ACK RTT data...")
    mawi_gen = MAWITrafficGenerator(seed=seed + 57)
    mawi_volume, mawi_rtts = mawi_gen.generate_all()
    print(f"    -> {len(mawi_volume):,} traffic bins, {len(mawi_rtts):,} SYN-ACK RTT samples")

    print("  Generating CAIDA Ark traceroute RTT data...")
    caida_gen = CAIDAGenerator(seed=seed + 113)
    caida_rtts = caida_gen.generate_ark_rtts(n_days=n_days)
    print(f"    -> {len(caida_rtts):,} Ark traceroute RTT samples")

    combined = pd.concat(ripe_data.values(), ignore_index=True)
    combined.to_csv(DATA_DIR / "ripe_atlas_rtt.csv", index=False)
    mawi_volume.to_csv(DATA_DIR / "mawi_traffic_volume.csv", index=False)
    mawi_rtts.to_csv(DATA_DIR / "mawi_synack_rtt.csv", index=False)
    caida_rtts.to_csv(DATA_DIR / "caida_ark_rtt.csv", index=False)

    for proto in ["udp", "tcp"]:
        proto_combined = pd.concat(proto_data[proto].values(), ignore_index=True)
        proto_combined.to_csv(DATA_DIR / f"ripe_atlas_{proto}.csv", index=False)

    print(f"    -> Saved to {DATA_DIR}")

    return ripe_data, mawi_volume, mawi_rtts, caida_rtts, proto_data
