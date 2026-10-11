"""Singapore Grand Prix 2026 experiment configuration (does not replace Hungary TARGET_*)."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.config import PROJECT_ROOT, TARGET_SEASON

SINGAPORE_EVENT_NAME = "Singapore Grand Prix"
SINGAPORE_SEASON = TARGET_SEASON
SINGAPORE_ROUND = 17
SINGAPORE_CIRCUIT = "Marina Bay"
SINGAPORE_EVENT_FORMAT = "sprint_qualifying"

# Training / form cutoff: completed GPs strictly before Singapore (FastF1 R1–R16 in 2026).
SINGAPORE_FORM_CUTOFF_EVENT = SINGAPORE_EVENT_NAME

# Historical Singapore analysis — emphasize post–layout-revision era (2023+).
SINGAPORE_LAYOUT_REVISION_SEASON = 2023
SINGAPORE_HISTORY_START_SEASON = 2019

# Comparable street circuits (documented; not averaged blindly).
STREET_COMPARABLE_EVENTS = (
    "Monaco Grand Prix",
    "Azerbaijan Grand Prix",
)

# Recency weights for historical Singapore finishes (season -> weight).
SINGAPORE_SEASON_WEIGHTS: dict[int, float] = {
    2019: 0.35,
    2020: 0.0,  # cancelled
    2021: 0.35,
    2022: 0.5,
    2023: 1.0,
    2024: 1.0,
    2025: 1.0,
}

IST = ZoneInfo("Asia/Kolkata")

# Supplied project schedule (interpretation / documentation — verify against F1 when possible).
SINGAPORE_WEEKEND_SCHEDULE_IST = {
    "fp1": "2026-10-09 14:00–15:00 IST",
    "sprint_qualifying": "2026-10-09 18:00–18:44 IST",
    "sprint": "2026-10-10 14:30–15:30 IST",
    "grand_prix_qualifying": "2026-10-10 18:30–19:30 IST",
    "race": "2026-10-11 17:30 IST",
}

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
DOCS_DIR = PROJECT_ROOT / "docs"

# Stage output filenames (Singapore-only; Hungary paths unchanged).
STAGE_FILES = {
    "pre_weekend_csv": "singapore_2026_pre_weekend_prediction.csv",
    "pre_weekend_json": "singapore_2026_pre_weekend_prediction.json",
    "post_fp1_csv": "singapore_2026_post_fp1_prediction.csv",
    "post_sq_csv": "singapore_2026_post_sprint_quali_prediction.csv",
    "post_sprint_csv": "singapore_2026_post_sprint_prediction.csv",
    "final_csv": "singapore_2026_final_prediction.csv",
    "model_comparison": "singapore_2026_model_comparison.csv",
    "feature_importance": "singapore_2026_feature_importance.csv",
    "evidence_matrix": "singapore_2026_evidence_matrix.csv",
    "weather_context": "singapore_2026_weather_context.csv",
    "fp1_status": "singapore_2026_fp1_retrieval_status.json",
    "pre_weekend_snapshot": "singapore_2026_pre_weekend_snapshot.md",
    "post_fp1_snapshot": "singapore_2026_post_fp1_snapshot.md",
    "post_sq_snapshot": "singapore_2026_post_sprint_quali_snapshot.md",
    "sq_status": "singapore_2026_sq_retrieval_status.json",
    "sq_evidence": "singapore_2026_sq_session_evidence.csv",
    "evidence_matrix_stage_c": "singapore_2026_evidence_matrix_stage_c.csv",
    "weather_context_sq": "singapore_2026_weather_context_sq.csv",
    "post_sprint_snapshot": "singapore_2026_post_sprint_snapshot.md",
    "sprint_status": "singapore_2026_sprint_retrieval_status.json",
    "sprint_evidence": "singapore_2026_sprint_session_evidence.csv",
    "evidence_matrix_stage_d": "singapore_2026_evidence_matrix_stage_d.csv",
    "weather_context_sprint": "singapore_2026_weather_context_sprint.csv",
    "final_snapshot": "singapore_2026_final_snapshot.md",
    "gp_qual_status": "singapore_2026_gp_qual_retrieval_status.json",
    "gp_qual_evidence": "singapore_2026_gp_qual_session_evidence.csv",
    "evidence_matrix_stage_e": "singapore_2026_evidence_matrix_stage_e.csv",
    "weather_context_gp_qual": "singapore_2026_weather_context_gp_qual.csv",
}


def now_ist() -> datetime:
    return datetime.now(IST)


def today_ist() -> date:
    return now_ist().date()
