"""Project configuration for F1 Race Intelligence data pipeline."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# ---------------------------------------------------------------------------
# Training scope
# ---------------------------------------------------------------------------
TRAINING_SEASONS = [2024, 2025, 2026]

TARGET_EVENT = "Hungarian Grand Prix"
TARGET_SEASON = 2026

# ---------------------------------------------------------------------------
# Regulation / grid context (2026 technical regulation change)
# ---------------------------------------------------------------------------
# 2024–2025: previous regulation era (~20-car grids typical)
# 2026+:     new regulation era (22 drivers / 11 constructors)
REGULATION_ERA_PRE_2026 = "pre_2026"
REGULATION_ERA_2026 = "2026_regulation"
REGULATION_ERA_START_SEASON = 2026

# Grid size is NOT fixed — validate dynamically from session entrant counts.
# Typical values: 20 (2024–2025), 22 (2026).

# V1 historical ingestion: Race + Qualifying only (no bulk FP1/FP2 download).
HISTORICAL_SESSION_TYPES = {
    "R": "Race",
    "Q": "Qualifying",
}

# Current-weekend practice (collected separately, not bulk historical).
# Hungary 2026 FP1/FP2 will use these in a later task.
CURRENT_WEEKEND_SESSION_TYPES = {
    "FP1": "FP1",
    "FP2": "FP2",
}

# All session types (reference only — historical pipeline uses HISTORICAL_SESSION_TYPES).
SESSION_TYPES = {
    **HISTORICAL_SESSION_TYPES,
    **CURRENT_WEEKEND_SESSION_TYPES,
}

# Lightweight load kwargs — no telemetry, weather, or messages
RACE_LOAD_KWARGS = {
    "laps": False,
    "telemetry": False,
    "weather": False,
    "messages": False,
}

TIMING_LOAD_KWARGS = {
    "laps": True,
    "telemetry": False,
    "weather": False,
    "messages": False,
}

# ---------------------------------------------------------------------------
# Raw output filenames (CSV — pyarrow not installed)
# ---------------------------------------------------------------------------
RACE_OUTPUT_FILE = "race_results.csv"
QUALIFYING_OUTPUT_FILE = "qualifying_results.csv"
MODELING_DATASET_FILE = "modeling_dataset.csv"
# Legacy filenames from pre-V1 test runs (not updated by historical --full ingestion).
FP1_OUTPUT_FILE = "fp1_results.csv"
FP2_OUTPUT_FILE = "fp2_results.csv"

# ---------------------------------------------------------------------------
# Test event (lightweight pipeline verification)
# ---------------------------------------------------------------------------
TEST_SEASON = 2025
TEST_EVENT = "Hungarian Grand Prix"
