"""
2026 Hungarian Grand Prix — PRE-QUALIFYING assessment.

Historical ML (Random Forest, pre-qualifying features only) is kept separate
from current-weekend FP1/FP2/FP3 analytical evidence.

Usage:
    python -m src.predict_hungary
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import MODELING_DATASET_FILE, PROCESSED_DATA_DIR, PROJECT_ROOT, TARGET_EVENT, TARGET_SEASON
from src.data_loader import get_event_round, get_event_schedule
from src.features import (
    add_constructor_form_features,
    add_driver_form_features,
    add_season_context,
    load_raw_results,
    merge_race_qualifying,
    normalize_constructors,
)
from src.practice_analysis import (
    OUTPUT_DIR as PRACTICE_OUTPUT_DIR,
    build_hungary_practice_table,
    save_practice_table,
)
from src.train import PRE_QUALIFYING_FEATURES, RANDOM_STATE, TARGET, build_pipeline

logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"

HUNGARY_SEASON = TARGET_SEASON
HUNGARY_ROUND = 11
HUNGARY_CIRCUIT = "Budapest"
HUNGARY_RACE_NAME = "Hungarian Grand Prix"
MAX_TRAIN_ROUND = 10  # Belgian GP — Hungary excluded

PRE_QUALI_PREDICTION_FILE = "hungary_2026_pre_quali_prediction.csv"
PRE_QUALI_ASSESSMENT_FILE = "hungary_2026_pre_quali_assessment.csv"
CONTEXT_FILE = "hungary_2026_context.md"

VALIDATED_PRE_MAE = 3.651


def get_pre_quali_random_forest() -> RandomForestRegressor:
    """Same configuration as src/train.py — not tuned."""
    return RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        n_jobs=-1,
    )


def build_hungary_feature_rows(
    entry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build PRE-qualifying feature rows for Hungary using history through Belgium R10.

    Appends skeleton Hungary rows to historical data, runs the same rolling
    feature functions as features.py, then extracts Hungary rows only.
    """
    race_raw, quali = load_raw_results()
    race = normalize_constructors(race_raw)
    merged = merge_race_qualifying(race, quali)

    skeleton_rows = []
    for _, row in entry.iterrows():
        skeleton_rows.append(
            {
                "season": HUNGARY_SEASON,
                "round": HUNGARY_ROUND,
                "race_name": HUNGARY_RACE_NAME,
                "circuit": HUNGARY_CIRCUIT,
                "driver": row["driver"],
                "constructor": row["constructor"],
                "race_date": pd.NaT,
                "finishing_position": np.nan,
                "classified_status": "Finished",  # placeholder — Hungary target unknown; shift excludes it from features
                "points": np.nan,
                "grid_position": np.nan,
                "qualifying_position": np.nan,
                "qualifying_best_lap_s": np.nan,
                "gap_to_pole_s": np.nan,
            }
        )
    skeleton = pd.DataFrame(skeleton_rows)
    combined = pd.concat([merged, skeleton], ignore_index=True)

    combined = add_driver_form_features(combined)
    combined = add_constructor_form_features(combined)
    combined = add_season_context(combined)

    hungary = combined[
        (combined["season"] == HUNGARY_SEASON) & (combined["round"] == HUNGARY_ROUND)
    ].copy()
    return hungary


def train_pre_quali_model(train_df: pd.DataFrame) -> Pipeline:
    """Train PRE-qualifying Random Forest on all historical rows through Belgium R10."""
    model = get_pre_quali_random_forest()
    pipeline = build_pipeline("RandomForestRegressor", model)
    X = train_df[PRE_QUALIFYING_FEATURES]
    y = train_df[TARGET]
    pipeline.fit(X, y)
    return pipeline


def generate_ml_predictions(
    entry: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """Train on history and predict Hungary regression scores."""
    hungary_features = build_hungary_feature_rows(entry)
    pipeline = train_pre_quali_model(train_df)

    X_pred = hungary_features[PRE_QUALIFYING_FEATURES]
    hungary_features = hungary_features.copy()
    hungary_features["predicted_score"] = pipeline.predict(X_pred)
    hungary_features["ml_pre_quali_rank"] = (
        hungary_features["predicted_score"].rank(method="min").astype(int)
    )

    out_cols = (
        ["ml_pre_quali_rank", "driver", "constructor", "predicted_score"]
        + PRE_QUALIFYING_FEATURES
    )
    return hungary_features[out_cols].sort_values("ml_pre_quali_rank")


def build_assessment_table(
    ml_pred: pd.DataFrame,
    practice: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ML prediction with practice analysis — no combined score."""
    assess_cols = [
        "driver",
        "constructor",
        "ml_pre_quali_rank",
        "predicted_score",
        "fp1_rank",
        "fp2_rank",
        "fp3_rank",
        "fp2_long_run_rank",
        "fp2_long_run_reliability",
        "fp3_s1_rank",
        "fp3_s2_rank",
        "fp3_s3_rank",
        "fp3_teammate_gap_s",
        "consistently_fast",
        "improving",
        "fp2_long_run_strong",
        "technical_sector_strong",
        "teammate_advantage",
    ]
    merged = ml_pred.merge(practice, on=["driver", "constructor"], how="inner")
    return merged[assess_cols].sort_values("ml_pre_quali_rank")


def _fmt(val, fmt: str = ".1f", suffix: str = "") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    return f"{val:{fmt}}{suffix}"


def write_context_doc(fp3_weather: dict, fp3_conditions_note: str = "") -> Path:
    """Write objective Hungaroring context with actual FP3 weather."""
    path = DOCS_DIR / CONTEXT_FILE
    rain = "Yes" if fp3_weather.get("rainfall_occurred") else "No"
    weather_block = f"""## Actual FP3 Weather (FastF1)

| Measure | Value |
|---------|-------|
| Air temperature (avg) | {_fmt(fp3_weather.get('air_temp_avg_c'))} °C |
| Air temperature (min / max) | {_fmt(fp3_weather.get('air_temp_min_c'))} / {_fmt(fp3_weather.get('air_temp_max_c'))} °C |
| Track temperature (avg) | {_fmt(fp3_weather.get('track_temp_avg_c'))} °C |
| Track temperature (min / max) | {_fmt(fp3_weather.get('track_temp_min_c'))} / {_fmt(fp3_weather.get('track_temp_max_c'))} °C |
| Humidity (avg) | {_fmt(fp3_weather.get('humidity_avg_pct'))} % |
| Wind speed (avg) | {_fmt(fp3_weather.get('wind_speed_avg_ms'), '.2f')} m/s |
| Rainfall during FP3 | {rain} |

**Race forecast context:** No trustworthy race forecast is stored in this project. Race weather must be added from an official or trusted external source before the race — do not invent forecast values here.

### Weather interpretation

FP3 conditions were dry with moderate air and track temperatures. {fp3_conditions_note or 'Conditions appear broadly representative of a normal dry practice session.'} These observations are contextual only and are **not** used as ML model inputs.
"""
    content = f"""# Hungary 2026 — Pre-Qualifying Context

Objective circuit and weekend context for interpreting the V1 pre-qualifying assessment.
This document does **not** contain arbitrary numeric model features.

## Circuit Characteristics

| Attribute | Value | Notes |
|-----------|-------|-------|
| Circuit | Hungaroring | |
| Location | Budapest, Hungary | FastF1: Budapest |
| Length | 4.381 km | Official circuit length |
| Corners | 14 | |
| Character | High-downforce, technical, rhythm-heavy | |
| Overtaking | Relatively limited | Track position historically important |
| 2026 Straight Mode | 4 Straight Mode zones | 2026 regulation context |
| 2026 Overtake Mode | Detection approaching Turn 14, deployment onto main straight | Subject to official 2026 configuration |

## Tyres (weekend allocation)

Pirelli compounds **C3 (Hard) / C4 (Medium) / C5 (Soft)** — confirm against official Pirelli/FIA source before race.

## Pit loss estimate

Approximately **20.56 seconds** total pit-lane time loss vs staying out — useful race strategy context; confirm against official or timing source.

## Historical disruption rates (Hungaroring)

| Event | Approximate rate | Notes |
|-------|------------------|-------|
| Safety Car | ~13% | Historical incidence — context only |
| VSC | ~25% | Historical incidence — context only |

## Same-circuit constants vs driver-varying signals

Circuit length, corner count, pit-loss estimate, and historical SC/VSC rates describe the **race environment** shared by all 22 drivers. They cannot distinguish between drivers within one Hungary prediction.

Driver-varying interpretation in V1 relies on separate evidence layers (ML form, FP pace, sectors, teammate gaps).

## Track-Specific Signals Used (V1)

Rather than arbitrary downforce or Active Aero scores, V1 uses objective driver-varying signals:

1. **FP3 Sector 2 relative pace** — proxy for performance through Hungary's technical medium-speed section (high-downforce/chassis signal).
2. **FP3 teammate gap** — separates driver performance from general car performance.
3. **FP2 long-run pace** — approximate race-run evidence with reliability filtering (HIGH/MEDIUM only for team summaries).
4. **FP1 / FP2 / FP3 progression** — current-weekend performance trend (descriptive flags, not ML inputs).
5. **Qualifying position and gap to pole** — will be added to the POST-qualifying model after qualifying.
6. **Team front-row lockout** — will be calculated after qualifying as contextual information.

### Possible V2 historical feature (not implemented)

`high_downforce_form` — rolling form at comparable high-downforce circuits (Monaco, Singapore, Hungary). Not implemented in V1 because the comparable-race sample is small and would increase complexity.

{weather_block}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def print_section_a_ml(ml_pred: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("A. HISTORICAL ML TOP 10 (PRE-QUALIFYING Random Forest)")
    print("=" * 70)
    for _, r in ml_pred.nsmallest(10, "ml_pre_quali_rank").iterrows():
        print(
            f"  {int(r['ml_pre_quali_rank']):>2}. {r['driver']:<4} {r['constructor']:<18} "
            f"score {r['predicted_score']:.3f}"
        )


def print_section_b_fp3(practice: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("B. FP3 TOP 10 — single-lap pace")
    print("=" * 70)
    for _, r in practice.dropna(subset=["fp3_rank"]).nsmallest(10, "fp3_rank").iterrows():
        print(
            f"  {int(r['fp3_rank']):>2}. {r['driver']:<4} {r['constructor']:<18} "
            f"+{r['fp3_gap_s']:.3f}s"
        )


def print_section_c_longrun(practice: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("C. FP2 LONG-RUN TOP 10 (approximate practice signal)")
    print("=" * 70)
    lr = practice.dropna(subset=["fp2_long_run_rank"]).nsmallest(10, "fp2_long_run_rank")
    for _, r in lr.iterrows():
        print(
            f"  {int(r['fp2_long_run_rank']):>2}. {r['driver']:<4} {r['constructor']:<18} "
            f"+{r['fp2_long_run_gap_s']:.3f}s  [{r['fp2_long_run_reliability']}]"
        )


def print_section_d_sector2(practice: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("D. FP3 SECTOR 2 TOP 10 (technical section proxy)")
    print("=" * 70)
    for _, r in practice.dropna(subset=["fp3_s2_rank"]).nsmallest(10, "fp3_s2_rank").iterrows():
        print(
            f"  {int(r['fp3_s2_rank']):>2}. {r['driver']:<4} {r['constructor']:<18} "
            f"+{r['fp3_s2_gap_to_best_s']:.3f}s"
        )


def print_section_e_teammate(practice: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("E. STRONGEST FP3 TEAMMATE ADVANTAGES (≥0.20s faster)")
    print("=" * 70)
    adv = practice[practice["teammate_advantage"]].sort_values("fp3_teammate_gap_s")
    if adv.empty:
        print("  None ≥0.20s")
        return
    for _, r in adv.iterrows():
        tm = practice[
            (practice["constructor"] == r["constructor"]) & (practice["driver"] != r["driver"])
        ]
        tm_name = tm["driver"].iloc[0] if not tm.empty else "?"
        print(f"  {r['driver']} vs {tm_name}: {r['fp3_teammate_gap_s']:+.3f}s")


def print_section_f_weather(weather: dict) -> None:
    print("\n" + "=" * 70)
    print("F. ACTUAL FP3 WEATHER (not an ML input)")
    print("=" * 70)
    if not weather.get("available"):
        print("  Weather data unavailable")
        return
    print(f"  Air temp (avg):    {weather['air_temp_avg_c']:.1f} °C")
    print(f"  Track temp (avg):  {weather['track_temp_avg_c']:.1f} °C")
    print(f"  Track temp (min/max): {weather['track_temp_min_c']:.1f} / {weather['track_temp_max_c']:.1f} °C")
    print(f"  Humidity (avg):    {weather['humidity_avg_pct']:.1f} %")
    print(f"  Wind speed (avg):  {weather['wind_speed_avg_ms']:.2f} m/s")
    print(f"  Rainfall:          {'Yes' if weather['rainfall_occurred'] else 'No'}")


def print_agreement_analysis(assessment: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("AGREEMENT / DISAGREEMENT ANALYSIS")
    print("=" * 70)

    ml_top10 = set(assessment.nsmallest(10, "ml_pre_quali_rank")["driver"])
    fp3_top10 = set(assessment.dropna(subset=["fp3_rank"]).nsmallest(10, "fp3_rank")["driver"])
    practice_strong = assessment[
        assessment["consistently_fast"]
        | assessment["fp2_long_run_strong"]
        | assessment["technical_sector_strong"]
    ]["driver"]

    strong_agreement = sorted(
        d for d in ml_top10
        if d in set(practice_strong)
    )
    print(f"\nSTRONG AGREEMENT (ML top 10 + strong practice flags): {', '.join(strong_agreement) or 'none'}")

    hist_strong_weekend_weak = sorted(
        d for d in ml_top10
        if d not in fp3_top10
        and d not in set(practice_strong)
    )
    print(
        f"HISTORICAL FORM STRONG / WEEKEND WEAKER: "
        f"{', '.join(hist_strong_weekend_weak) or 'none'}"
    )

    weekend_strong_ml_weaker = sorted(
        assessment.loc[
            assessment["fp3_rank"].notna() & (assessment["fp3_rank"] <= 10)
            & (assessment["ml_pre_quali_rank"] > 15),
            "driver",
        ]
    )
    print(
        f"WEEKEND STRONG / HISTORICAL ML WEAKER: "
        f"{', '.join(weekend_strong_ml_weaker) or 'none'}"
    )

    race_pace = sorted(assessment.loc[assessment["fp2_long_run_strong"], "driver"])
    print(f"RACE-PACE INTERESTING (reliable FP2 long-run top 6): {', '.join(race_pace) or 'none'}")

    quali_pace = sorted(
        set(
            assessment.loc[
                assessment["fp3_rank"].notna() & (assessment["fp3_rank"] <= 10), "driver"
            ]
        )
        | set(assessment.loc[assessment["technical_sector_strong"], "driver"])
    )
    print(f"QUALIFYING-PACE INTERESTING (FP3 top 10 / S2 top 6): {', '.join(quali_pace) or 'none'}")


def validate_assessment(
    practice: pd.DataFrame,
    ml_pred: pd.DataFrame,
    train_df: pd.DataFrame,
    reserve_drivers: list[str],
) -> list[str]:
    """Run pre-qualifying validation checks; return issue list."""
    issues: list[str] = []

    if len(practice) != 22:
        issues.append(f"Expected 22 entrants, got {len(practice)}")
    if len(ml_pred) != 22:
        issues.append(f"Expected 22 ML rows, got {len(ml_pred)}")
    if practice["driver"].duplicated().any():
        issues.append("Duplicate drivers in practice table")
    if ml_pred["driver"].duplicated().any():
        issues.append("Duplicate drivers in ML prediction")

    hungary_in_train = (
        (train_df["season"] == HUNGARY_SEASON) & (train_df["round"] == HUNGARY_ROUND)
    ).any()
    if hungary_in_train:
        issues.append("CRITICAL: Hungary found in training data")

    max_round = train_df.loc[train_df["season"] == HUNGARY_SEASON, "round"].max()
    if max_round != MAX_TRAIN_ROUND:
        issues.append(f"Expected max 2026 train round {MAX_TRAIN_ROUND}, got {max_round}")

    if reserve_drivers:
        reserve_in_table = set(reserve_drivers) & set(practice["driver"])
        if reserve_in_table:
            issues.append(f"Reserve driver(s) in official table: {reserve_in_table}")

    missing_fp3 = practice[practice["fp3_best_lap_s"].isna()]["driver"].tolist()
    if missing_fp3:
        issues.append(f"Missing FP3 lap data: {', '.join(missing_fp3)}")

    schedule = get_event_schedule(HUNGARY_SEASON)
    hungary_round = get_event_round(schedule, TARGET_EVENT)
    if hungary_round != HUNGARY_ROUND:
        issues.append(f"FP3/Hungary round mismatch: schedule says {hungary_round}")

    return issues


def print_methodology() -> None:
    print("\n" + "=" * 70)
    print("METHODOLOGY STATEMENT")
    print("=" * 70)
    print(
        "\nThis pre-qualifying assessment was generated before 2026 Hungarian Grand Prix "
        "qualifying. The Random Forest ranking uses historical Race and Qualifying "
        "information only through the 2026 Belgian Grand Prix. Hungary FP1, FP2, FP3, "
        "sector performance and weather are shown as independent current-weekend analytical "
        "evidence and were not used as ML model inputs."
    )
    print(
        f"\nValidated PRE-qualifying model (temporal holdout 2026 R8–R10, NOT Hungary):\n"
        f"  Random Forest MAE ≈ {VALIDATED_PRE_MAE:.3f} positions"
    )


def run_assessment() -> dict:
    """Full pre-qualifying assessment pipeline."""
    practice, conditions, _team, fp3_weather, reserve_drivers = build_hungary_practice_table()
    save_practice_table(practice)

    train_df = pd.read_csv(PROCESSED_DATA_DIR / MODELING_DATASET_FILE)
    train_df = train_df[
        ~((train_df["season"] == HUNGARY_SEASON) & (train_df["round"] == HUNGARY_ROUND))
    ]

    entry = practice[["driver", "constructor"]]
    ml_pred = generate_ml_predictions(entry, train_df)
    assessment = build_assessment_table(ml_pred, practice)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUTPUT_DIR / PRE_QUALI_PREDICTION_FILE
    assess_path = OUTPUT_DIR / PRE_QUALI_ASSESSMENT_FILE
    ml_pred.to_csv(pred_path, index=False)
    assessment.to_csv(assess_path, index=False)

    fp3_note = ""
    if conditions.get("FP3") and conditions["FP3"].notes:
        fp3_note = f"FP3 session notes: {', '.join(conditions['FP3'].notes)}."
    context_path = write_context_doc(fp3_weather, fp3_note)

    issues = validate_assessment(practice, ml_pred, train_df, reserve_drivers)

    return {
        "practice": practice,
        "ml_pred": ml_pred,
        "assessment": assessment,
        "fp3_weather": fp3_weather,
        "reserve_drivers": reserve_drivers,
        "issues": issues,
        "pred_path": pred_path,
        "assess_path": assess_path,
        "context_path": context_path,
        "conditions": conditions,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    print("=" * 70)
    print("2026 HUNGARIAN GP — PRE-QUALIFYING ASSESSMENT")
    print("=" * 70)
    print("\nLayers: A) Historical ML  B) Practice  C) Weather  D) Interpretation")
    print("Practice data is NOT fed into the Random Forest.\n")

    out = run_assessment()
    practice = out["practice"]
    ml_pred = out["ml_pred"]

    print(f"\nFiles saved:")
    print(f"  {PRACTICE_OUTPUT_DIR / 'hungary_2026_practice_analysis.csv'}")
    print(f"  {out['pred_path']}")
    print(f"  {out['assess_path']}")
    print(f"  {out['context_path']}")

    if out["reserve_drivers"]:
        print(f"\nReserve appearances (excluded from 22-driver table): {', '.join(out['reserve_drivers'])}")

    print_section_a_ml(ml_pred)
    print_section_b_fp3(practice)
    print_section_c_longrun(practice)
    print_section_d_sector2(practice)
    print_section_e_teammate(practice)
    print_section_f_weather(out["fp3_weather"])

    print("\n" + "=" * 70)
    print("FP3 SECTOR RANKINGS (practice layer)")
    print("=" * 70)
    for label, col in [("Sector 1", "fp3_s1_rank"), ("Sector 2", "fp3_s2_rank"), ("Sector 3", "fp3_s3_rank")]:
        print(f"\n  {label} Top 10:")
        top = practice.dropna(subset=[col]).nsmallest(10, col)
        for _, r in top.iterrows():
            gap_col = col.replace("_rank", "_gap_to_best_s")
            gap = r.get(gap_col, np.nan)
            print(f"    {int(r[col]):>2}. {r['driver']:<4} +{gap:.3f}s")

    print_agreement_analysis(out["assessment"])
    print_methodology()

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    if out["issues"]:
        for issue in out["issues"]:
            print(f"  ⚠ {issue}")
    else:
        print("  ✓ All validation checks passed")

    print("\n" + "=" * 70)
    print("Assessment complete — no winner declared, no quali/race loaded.")
    print("=" * 70)


if __name__ == "__main__":
    main()
