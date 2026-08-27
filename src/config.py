from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in [RESULTS_DIR, DATA_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

RIPE_ATLAS_BASE_URL = "https://atlas.ripe.net/api/v2"
MEASUREMENT_INTERVAL_S = 240
DATA_DURATION_DAYS = 14


@dataclass
class PathDefinition:
    path_id: str
    source_city: str
    source_country: str
    dest_city: str
    dest_country: str
    regime: str
    expected_rtt_ms: float
    expected_jitter_ms: float
    propagation_km: float


SHORT_HAUL_PATHS = [
    PathDefinition("sh_01", "New York", "US", "Newark", "US", "short_haul", 8.0, 1.5, 25),
    PathDefinition("sh_02", "London", "GB", "Reading", "GB", "short_haul", 6.0, 1.2, 60),
    PathDefinition("sh_03", "Amsterdam", "NL", "Rotterdam", "NL", "short_haul", 5.0, 1.0, 80),
    PathDefinition("sh_04", "Tokyo", "JP", "Yokohama", "JP", "short_haul", 7.0, 1.3, 30),
    PathDefinition("sh_05", "New York", "US", "Washington DC", "US", "short_haul", 12.0, 2.5, 360),
    PathDefinition("sh_06", "London", "GB", "Paris", "FR", "short_haul", 14.0, 3.0, 340),
    PathDefinition("sh_07", "Amsterdam", "NL", "Frankfurt", "DE", "short_haul", 11.0, 2.2, 365),
    PathDefinition("sh_08", "Tokyo", "JP", "Osaka", "JP", "short_haul", 10.0, 2.0, 400),
    PathDefinition("sh_09", "Seoul", "KR", "Incheon", "KR", "short_haul", 5.0, 1.0, 40),
    PathDefinition("sh_10", "Chicago", "US", "Milwaukee", "US", "short_haul", 8.0, 1.5, 150),
    PathDefinition("sh_11", "Paris", "FR", "Brussels", "BE", "short_haul", 13.0, 2.8, 310),
    PathDefinition("sh_12", "Sydney", "AU", "Canberra", "AU", "short_haul", 11.0, 2.3, 280),
]

REGIONAL_PATHS = [
    PathDefinition("rg_01", "New York", "US", "Chicago", "US", "regional", 32.0, 6.0, 1150),
    PathDefinition("rg_02", "New York", "US", "Miami", "US", "regional", 45.0, 9.0, 2000),
    PathDefinition("rg_03", "London", "GB", "Madrid", "ES", "regional", 38.0, 7.5, 1260),
    PathDefinition("rg_04", "London", "GB", "Stockholm", "SE", "regional", 42.0, 8.0, 1430),
    PathDefinition("rg_05", "Amsterdam", "NL", "Rome", "IT", "regional", 48.0, 10.0, 1300),
    PathDefinition("rg_06", "Tokyo", "JP", "Seoul", "KR", "regional", 35.0, 7.0, 1160),
    PathDefinition("rg_07", "Sao Paulo", "BR", "Buenos Aires", "AR", "regional", 55.0, 12.0, 1700),
    PathDefinition("rg_08", "Sydney", "AU", "Auckland", "NZ", "regional", 50.0, 11.0, 2160),
    PathDefinition("rg_09", "Chicago", "US", "Dallas", "US", "regional", 40.0, 8.0, 1290),
    PathDefinition("rg_10", "Paris", "FR", "Berlin", "DE", "regional", 36.0, 7.0, 1050),
    PathDefinition("rg_11", "Mumbai", "IN", "Chennai", "IN", "regional", 43.0, 9.0, 1340),
    PathDefinition("rg_12", "Tokyo", "JP", "Taipei", "TW", "regional", 52.0, 11.0, 2100),
]

INTERCONTINENTAL_PATHS = [
    PathDefinition("ic_01", "New York", "US", "London", "GB", "intercontinental", 75.0, 15.0, 5570),
    PathDefinition("ic_02", "New York", "US", "Tokyo", "JP", "intercontinental", 170.0, 30.0, 10850),
    PathDefinition("ic_03", "London", "GB", "Tokyo", "JP", "intercontinental", 230.0, 40.0, 9560),
    PathDefinition("ic_04", "New York", "US", "Sao Paulo", "BR", "intercontinental", 130.0, 25.0, 7700),
    PathDefinition("ic_05", "London", "GB", "Sydney", "AU", "intercontinental", 270.0, 45.0, 17000),
    PathDefinition("ic_06", "Amsterdam", "NL", "Singapore", "SG", "intercontinental", 185.0, 32.0, 10000),
    PathDefinition("ic_07", "Tokyo", "JP", "Sao Paulo", "BR", "intercontinental", 260.0, 48.0, 17500),
    PathDefinition("ic_08", "New York", "US", "Sydney", "AU", "intercontinental", 250.0, 42.0, 16000),
    PathDefinition("ic_09", "London", "GB", "Mumbai", "IN", "intercontinental", 140.0, 28.0, 7200),
    PathDefinition("ic_10", "Tokyo", "JP", "Sydney", "AU", "intercontinental", 155.0, 30.0, 7800),
    PathDefinition("ic_11", "New York", "US", "Amsterdam", "NL", "intercontinental", 82.0, 16.0, 5850),
    PathDefinition("ic_12", "London", "GB", "Cape Town", "ZA", "intercontinental", 195.0, 35.0, 9600),
]

ALL_PATHS = SHORT_HAUL_PATHS + REGIONAL_PATHS + INTERCONTINENTAL_PATHS

GAME_THRESHOLDS = {
    "Competitive FPS": {"excellent": 20, "acceptable": 50, "degraded": 100, "unplayable": 150},
    "Casual FPS": {"excellent": 50, "acceptable": 100, "degraded": 150, "unplayable": 250},
    "MOBA": {"excellent": 40, "acceptable": 80, "degraded": 120, "unplayable": 200},
    "MMORPG": {"excellent": 100, "acceptable": 200, "degraded": 400, "unplayable": 600},
    "RTS": {"excellent": 50, "acceptable": 120, "degraded": 200, "unplayable": 350},
    "Fighting": {"excellent": 16, "acceptable": 33, "degraded": 67, "unplayable": 100},
}

COMPANION_PROFILES = {
    "excellent":  {"base_delay_ms": 15.0,  "jitter_ms": 3.0},
    "good":       {"base_delay_ms": 40.0,  "jitter_ms": 8.0},
    "average":    {"base_delay_ms": 80.0,  "jitter_ms": 20.0},
    "poor":       {"base_delay_ms": 150.0, "jitter_ms": 40.0},
    "very_poor":  {"base_delay_ms": 250.0, "jitter_ms": 60.0},
}

DISTRIBUTION_CANDIDATES = ["lognorm", "gamma", "weibull_min", "norm", "gumbel_r"]
SIGNIFICANCE_LEVEL = 0.05
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

FIG_SINGLE_COL = (3.5, 2.8)
FIG_DOUBLE_COL = (7.16, 4.0)
FIG_FULL_PAGE = (7.16, 8.0)
FIG_DPI = 300

REGIME_COLORS = {
    "short_haul": "#2196F3",
    "regional": "#FF9800",
    "intercontinental": "#E53935",
}
REGIME_LABELS = {
    "short_haul": "Short-haul",
    "regional": "Regional",
    "intercontinental": "Intercontinental",
}
