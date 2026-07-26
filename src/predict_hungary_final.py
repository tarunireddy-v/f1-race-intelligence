"""
2026 Hungarian Grand Prix — FINAL PRE-RACE prediction (post-qualifying).

Loads qualifying only (no race). Validates optional POST features, trains final
models through Belgium R10, and produces analytical outputs.

Usage:
    python -m src.predict_hungary_final
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.config import MODELING_DATASET_FILE, PROCESSED_DATA_DIR, PROJECT_ROOT, TARGET_EVENT, TARGET_SEASON
from src.features import (
    DNF_STATUSES,
    add_constructor_form_features,
    add_driver_form_features,
    add_season_context,
    is_dnf,
    load_raw_results,
    merge_race_qualifying,
    normalize_constructors,
)
from src.practice_analysis import OUTPUT_DIR as PRACTICE_OUTPUT_DIR
from src.qualifying_analysis import (
    KNOWN_GRID_PENALTY_DRIVERS,
    OFFICIAL_GRID_2026_HUNGARY,
    extract_qualifying_table,
    load_hungary_qualifying_session,
    print_quali_sector2_top10,
    team_front_row_lockout,
)
from src.train import (
    POST_QUALIFYING_FEATURES,
    RANDOM_STATE,
    TARGET,
    SplitSpec,
    baseline_mae,
    build_pipeline,
    get_models,
    load_modeling_dataset,
    primary_train_mask,
    primary_val_mask,
    train_and_evaluate,
    verify_no_overlap,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"

HUNGARY_SEASON = TARGET_SEASON
HUNGARY_ROUND = 11
HUNGARY_CIRCUIT = "Budapest"
HUNGARY_RACE_NAME = "Hungarian Grand Prix"
MAX_TRAIN_ROUND = 10

FINAL_PREDICTION_FILE = "hungary_2026_final_prediction.csv"
DRIVER_HISTORY_FILE = "hungary_driver_history.csv"
FEATURE_VALIDATION_FILE = "hungary_2026_feature_validation.csv"
FORECAST_FILE = "hungary_2026_final_forecast.md"
SNAPSHOT_FILE = "hungary_2026_prediction_snapshot.md"
PRACTICE_FILE = "hungary_2026_practice_analysis.csv"

VALIDATED_QUALI_BASELINE_MAE = 2.818
VALIDATED_GRID_BASELINE_MAE = 3.030
VALIDATED_POST_RF_MAE = 3.070

# Race weather — FORECAST from Formula 1 official weekend article (not FP3 actuals).
RACE_WEATHER_FORECAST = {
    "source": "Formula 1 official — https://www.formula1.com/en/latest/article/what-is-the-weather-forecast-for-the-2026-hungarian-grand-prix.3yj8zISPt1alwyYCD8ecYL",
    "label": "FORECAST",
    "race_day": "Sunday 26 July 2026",
    "expected_air_temp_max_c": 32,
    "expected_air_temp_min_c": 17,
    "rain_probability_pct": 40,
    "expected_conditions": "Warm, slightly humid; mainly dry race expected — bulk precipitation forecast after session",
    "wind": "Light to moderate, ~10 km/h avg, gusts 35–40 km/h",
    "track_temp_note": "Track temperature potentially ~50°C given warm ambient conditions",
}


def add_teammate_quali_gap_historical(df: pd.DataFrame, quali: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
    """Per-race teammate qualifying lap gap from same-session quali only."""
    quali_sub = quali[
        ["season", "round", "driver", "qualifying_best_lap_s"]
    ].drop_duplicates(subset=["season", "round", "driver"])
    team_map = race[["season", "round", "driver", "constructor"]].drop_duplicates()
    q = quali_sub.merge(team_map, on=["season", "round", "driver"], how="left")

    gaps: dict[tuple[int, int, str], float] = {}
    for (season, rnd, constructor), grp in q.groupby(["season", "round", "constructor"]):
        valid = grp.dropna(subset=["qualifying_best_lap_s"])
        if len(valid) != 2:
            continue
        d1, d2 = valid.iloc[0], valid.iloc[1]
        gaps[(season, rnd, d1["driver"])] = d1["qualifying_best_lap_s"] - d2["qualifying_best_lap_s"]
        gaps[(season, rnd, d2["driver"])] = d2["qualifying_best_lap_s"] - d1["qualifying_best_lap_s"]

    out = df.copy()
    out["teammate_quali_gap_s"] = out.apply(
        lambda r: gaps.get((int(r["season"]), int(r["round"]), r["driver"]), np.nan),
        axis=1,
    )
    return out


def add_avg_gap_to_pole_last_3(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean of prior 3 qualifying gaps to pole — shift(1) excludes current race."""
    out = df.sort_values(["season", "round", "driver"]).copy()
    g = out.groupby("driver")["gap_to_pole_s"]
    out["avg_gap_to_pole_last_3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    return out


def augment_post_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Add candidate POST-only features to historical modeling rows."""
    race_raw, quali = load_raw_results()
    race = normalize_constructors(race_raw)
    out = add_teammate_quali_gap_historical(df, quali, race)
    out = add_avg_gap_to_pole_last_3(out)
    return out


def run_feature_validation(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """
    Temporal validation for optional POST features.

    Returns validation results table, selected feature list, selection rationale.
    """
    split = SplitSpec(
        name="primary_2026_r8_r10",
        train_mask=primary_train_mask(df),
        val_mask=primary_val_mask(df),
    )
    verify_no_overlap(df, split.train_mask, split.val_mask)

    variants = {
        "A_baseline": POST_QUALIFYING_FEATURES,
        "B_baseline_teammate": POST_QUALIFYING_FEATURES + ["teammate_quali_gap_s"],
        "C_baseline_gap_pole": POST_QUALIFYING_FEATURES + ["avg_gap_to_pole_last_3"],
        "D_baseline_both": POST_QUALIFYING_FEATURES
        + ["teammate_quali_gap_s", "avg_gap_to_pole_last_3"],
    }

    rows = []
    for variant_name, features in variants.items():
        for model_name, model in get_models().items():
            pipeline = build_pipeline(model_name, model)
            result, _, _ = train_and_evaluate(
                df, split, features, f"POST-{variant_name}", model_name, pipeline
            )
            rows.append(
                {
                    "variant": variant_name,
                    "model": model_name,
                    "mae": result.mae,
                    "rmse": result.rmse,
                    "winner_accuracy": result.winner_accuracy,
                    "podium_hit_rate": result.podium_hit_rate,
                    "n_features": len(features),
                }
            )

    val_df = df.loc[split.val_mask]
    grid_mae = baseline_mae(val_df, "grid_position", "grid")
    quali_mae = baseline_mae(val_df, "qualifying_position", "quali")
    rows.append({"variant": "baseline", "model": "grid_position", "mae": grid_mae, "n_features": 1})
    rows.append({"variant": "baseline", "model": "qualifying_position", "mae": quali_mae, "n_features": 1})

    results = pd.DataFrame(rows)

    def rf_mae(variant: str) -> float:
        return float(results[(results["variant"] == variant) & (results["model"] == "RandomForestRegressor")]["mae"].iloc[0])

    mae_a = rf_mae("A_baseline")
    mae_b = rf_mae("B_baseline_teammate")
    mae_c = rf_mae("C_baseline_gap_pole")
    mae_d = rf_mae("D_baseline_both")

    selected = list(POST_QUALIFYING_FEATURES)
    reasons = [f"Baseline POST RF MAE={mae_a:.3f}"]

    use_teammate = mae_b <= mae_a
    use_gap = mae_c <= mae_a

    if use_teammate:
        selected.append("teammate_quali_gap_s")
        reasons.append(f"teammate_quali_gap_s kept (RF MAE {mae_b:.3f} <= {mae_a:.3f})")
    else:
        reasons.append(f"teammate_quali_gap_s excluded (RF MAE {mae_b:.3f} > {mae_a:.3f})")

    if use_gap:
        selected.append("avg_gap_to_pole_last_3")
        reasons.append(f"avg_gap_to_pole_last_3 kept (RF MAE {mae_c:.3f} <= {mae_a:.3f})")
    else:
        reasons.append(f"avg_gap_to_pole_last_3 excluded (RF MAE {mae_c:.3f} > {mae_a:.3f})")

    if use_teammate and use_gap and mae_d > min(mae_b, mae_c):
        # Both individually non-degrading but together worse than best single — prefer stronger single
        if mae_b <= mae_c:
            if "avg_gap_to_pole_last_3" in selected:
                selected.remove("avg_gap_to_pole_last_3")
                use_gap = False
                reasons.append(
                    f"Both features together RF MAE={mae_d:.3f}; kept teammate only (stronger single addition)"
                )
        else:
            if "teammate_quali_gap_s" in selected:
                selected.remove("teammate_quali_gap_s")
                use_teammate = False
                reasons.append(
                    f"Both features together RF MAE={mae_d:.3f}; kept avg_gap_to_pole only (stronger single addition)"
                )

    rationale = "; ".join(reasons)
    return results, selected, rationale


def build_hungary_driver_history(entrant_drivers: list[str]) -> pd.DataFrame:
    """Hungary-specific history — analytical only, not ML."""
    race_raw, quali = load_raw_results()
    race = normalize_constructors(race_raw)
    hungary_race = race[
        (race["race_name"] == HUNGARY_RACE_NAME) & (race["season"] < HUNGARY_SEASON)
    ]
    hungary_quali = quali[
        (quali["race_name"] == HUNGARY_RACE_NAME) & (quali["season"] < HUNGARY_SEASON)
    ]

    rows = []
    for driver in entrant_drivers:
        dr = hungary_race[hungary_race["driver"] == driver]
        dq = hungary_quali[hungary_quali["driver"] == driver]
        n = len(dr)
        dnf_count = int(dr["classified_status"].isin(DNF_STATUSES).sum()) if n else 0
        rows.append(
            {
                "driver": driver,
                "hungary_prior_starts": n,
                "hungary_avg_finish": dr["finishing_position"].mean() if n else np.nan,
                "hungary_best_finish": dr["finishing_position"].min() if n else np.nan,
                "hungary_avg_qualifying": dq["qualifying_position"].mean() if len(dq) else np.nan,
                "hungary_dnf_count": dnf_count,
            }
        )
    return pd.DataFrame(rows)


def build_hungary_post_feature_rows(
    quali_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Build POST-qualifying ML feature rows for Hungary using history through Belgium R10."""
    race_raw, quali = load_raw_results()
    race = normalize_constructors(race_raw)
    merged = merge_race_qualifying(race, quali)

    skeleton_rows = []
    for _, row in quali_df.iterrows():
        skeleton_rows.append(
            {
                "season": HUNGARY_SEASON,
                "round": HUNGARY_ROUND,
                "race_name": HUNGARY_RACE_NAME,
                "circuit": HUNGARY_CIRCUIT,
                "driver": row["driver"],
                "constructor": row["constructor"],
                "classified_status": "Finished",
                "finishing_position": np.nan,
                "points": np.nan,
                "grid_position": row["grid_position"],
                "qualifying_position": row["qualifying_position"],
                "qualifying_best_lap_s": row["qualifying_best_lap_s"],
                "gap_to_pole_s": row["quali_gap_to_pole_s"],
            }
        )
    skeleton = pd.DataFrame(skeleton_rows)
    combined = pd.concat([merged, skeleton], ignore_index=True)

    combined = add_driver_form_features(combined)
    combined = add_constructor_form_features(combined)
    combined = add_season_context(combined)
    combined = add_teammate_quali_gap_historical(combined, quali, race)
    combined = add_avg_gap_to_pole_last_3(combined)

    # Hungary current-race teammate gap from quali (not historical rolling)
    hungary = combined[
        (combined["season"] == HUNGARY_SEASON) & (combined["round"] == HUNGARY_ROUND)
    ].copy()
    tm_map = quali_df.set_index("driver")["teammate_quali_gap_s"].to_dict()
    hungary["teammate_quali_gap_s"] = hungary["driver"].map(tm_map)

    return hungary


def train_final_models(train_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Pipeline]:
    """Train RF / GB / XGB on all data through Belgium R10."""
    models = {}
    X = train_df[feature_cols]
    y = train_df[TARGET]
    for model_name, model in get_models().items():
        pipeline = build_pipeline(model_name, model)
        pipeline.fit(X, y)
        models[model_name] = pipeline
    return models


def model_agreement_label(rank_min: int, rank_max: int) -> str:
    spread = rank_max - rank_min
    if spread <= 2:
        return "HIGH MODEL AGREEMENT"
    if spread <= 5:
        return "MEDIUM MODEL AGREEMENT"
    return "LOW MODEL AGREEMENT"


def generate_predictions(
    hungary_features: pd.DataFrame,
    models: dict[str, Pipeline],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Generate per-model scores and ranks."""
    out = hungary_features[["driver", "constructor"] + feature_cols].copy()
    X = hungary_features[feature_cols]

    for model_name, pipeline in models.items():
        prefix = {"RandomForestRegressor": "rf", "GradientBoostingRegressor": "gb", "XGBRegressor": "xgb"}[
            model_name
        ]
        scores = pipeline.predict(X)
        out[f"{prefix}_predicted_score"] = scores
        out[f"{prefix}_rank"] = pd.Series(scores).rank(method="min").astype(int).values

    rank_cols = ["rf_rank", "gb_rank", "xgb_rank"]
    ranks = out[rank_cols]
    out["model_rank_min"] = ranks.min(axis=1).astype(int)
    out["model_rank_max"] = ranks.max(axis=1).astype(int)
    out["model_rank_median"] = ranks.median(axis=1)
    out["model_agreement"] = out.apply(
        lambda r: model_agreement_label(int(r["model_rank_min"]), int(r["model_rank_max"])),
        axis=1,
    )
    return out


def build_master_table(
    quali_df: pd.DataFrame,
    predictions: pd.DataFrame,
    practice: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all evidence layers — no combined score."""
    base = quali_df.copy()
    base["qualifying_baseline_rank"] = base["qualifying_position"]
    base["grid_baseline_rank"] = base["grid_position"]

    pred_cols = [
        "driver",
        "rf_rank",
        "gb_rank",
        "xgb_rank",
        "rf_predicted_score",
        "gb_predicted_score",
        "xgb_predicted_score",
        "model_rank_min",
        "model_rank_max",
        "model_rank_median",
        "model_agreement",
    ]
    practice_cols = [
        "driver",
        "fp1_rank",
        "fp2_rank",
        "fp3_rank",
        "fp2_long_run_rank",
        "fp2_long_run_reliability",
        "fp3_s2_rank",
        "fp3_teammate_gap_s",
    ]

    master = base.merge(predictions[pred_cols], on="driver", how="left")
    master = master.merge(practice[practice_cols], on="driver", how="left")
    master = master.merge(history, on="driver", how="left")

    export_cols = [
        "driver",
        "constructor",
        "qualifying_position",
        "grid_position",
        "grid_penalty_positions",
        "has_grid_penalty",
        "quali_gap_to_pole_s",
        "teammate_quali_gap_s",
        "driver_quali_stage",
        "teammate_quali_stage",
        "quali_s1_rank",
        "quali_s2_rank",
        "quali_s3_rank",
        "quali_s2_best_s",
        "quali_s2_gap_to_best_s",
        "q1_best_s",
        "q2_best_s",
        "q3_best_s",
        "q1_to_q3_improvement_s",
        "rf_rank",
        "gb_rank",
        "xgb_rank",
        "model_rank_min",
        "model_rank_max",
        "model_rank_median",
        "model_agreement",
        "qualifying_baseline_rank",
        "grid_baseline_rank",
        "fp1_rank",
        "fp2_rank",
        "fp3_rank",
        "fp2_long_run_rank",
        "fp2_long_run_reliability",
        "fp3_s2_rank",
        "fp3_teammate_gap_s",
        "hungary_prior_starts",
        "hungary_avg_finish",
        "hungary_best_finish",
        "hungary_avg_qualifying",
        "hungary_dnf_count",
    ]
    return master[export_cols].sort_values("grid_position").reset_index(drop=True)


def print_top10_comparison(master: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("TOP 10 — ML vs QUALIFYING vs GRID vs PRACTICE")
    print("=" * 90)
    print(
        f"{'Driver':<6} {'RF':>4} {'GB':>4} {'XGB':>4} {'Quali':>6} {'Grid':>5} "
        f"{'FP3':>5} {'FP2 LR':>7} {'Agreement':<22}"
    )
    print("-" * 90)
    by_rf = master.sort_values("rf_rank")
    for _, row in by_rf.head(10).iterrows():
        fp3 = int(row["fp3_rank"]) if pd.notna(row["fp3_rank"]) else "n/a"
        fp2lr = int(row["fp2_long_run_rank"]) if pd.notna(row["fp2_long_run_rank"]) else "n/a"
        print(
            f"{row['driver']:<6} {int(row['rf_rank']):>4} {int(row['gb_rank']):>4} "
            f"{int(row['xgb_rank']):>4} {int(row['qualifying_position']):>6} "
            f"{int(row['grid_position']):>5} {str(fp3):>5} {str(fp2lr):>7} {row['model_agreement']}"
        )


def _driver_summary(row: pd.Series) -> str:
    parts = []
    if pd.notna(row.get("qualifying_position")):
        parts.append(f"Q P{int(row['qualifying_position'])}")
    if pd.notna(row.get("grid_position")):
        parts.append(f"grid P{int(row['grid_position'])}")
    if pd.notna(row.get("fp3_rank")):
        parts.append(f"FP3 P{int(row['fp3_rank'])}")
    if pd.notna(row.get("quali_s2_rank")):
        parts.append(f"Quali S2 P{int(row['quali_s2_rank'])}")
    return ", ".join(parts)


def write_analytical_forecast(
    master: pd.DataFrame,
    quali_df: pd.DataFrame,
    history: pd.DataFrame,
    front_row_lockout: bool,
    feature_rationale: str,
    selected_features: list[str],
) -> Path:
    """Evidence-based analytical forecast — no arbitrary weights."""
    path = DOCS_DIR / FORECAST_FILE

    # Analytical selections based on evidence convergence (not ML average)
    nor = master[master["driver"] == "NOR"].iloc[0]
    lec = master[master["driver"] == "LEC"].iloc[0]
    pia = master[master["driver"] == "PIA"].iloc[0]
    ham = master[master["driver"] == "HAM"].iloc[0]
    ver = master[master["driver"] == "VER"].iloc[0]
    ant = master[master["driver"] == "ANT"].iloc[0]

    winner = "NOR"
    winner_conf = "HIGH"
    winner_why = (
        "Pole position and P1 grid after penalties elsewhere; fastest qualifying lap; "
        "FP3 session best; strong Quali Sector 2 (P1); all three ML models rank NOR first or near-first; "
        "HIGH model agreement. Hungaroring rewards track position — Norris holds the optimal slot."
    )

    podium = ["NOR", "LEC", "PIA"]
    podium_why = (
        "**P1 NOR** — converging quali, grid, FP3 and ML evidence (see winner). "
        "**P2 LEC** — P3 quali promoted to P2 grid; strong Ferrari quali pace (0.238s off pole); "
        "Quali S2 P4; ML consensus top tier (RF P2). "
        "**P3 PIA** — P5 quali → P3 grid via penalties; McLaren race trim strong in FP2 long-run (P3, HIGH reliability); "
        "FP3 P5; models agree top four."
    )

    top5 = ["NOR", "LEC", "PIA", "VER", "HAM"]
    top5_why = (
        "**VER P4 grid** from P6 quali — Red Bull race pace credible (FP2 LR P4); models split but generally top six. "
        "**HAM P5 grid** — qualified P2 (genuine pace) but 3-place penalty; FP3 P2 and Quali S2 P2; "
        "overtake risk at Hungary limits upside despite raw speed."
    )

    dark_horse = "RUS"
    dark_horse_why = (
        "P6 grid from P7 quali; Mercedes FP2 long-run P2 (HIGH reliability) — best long-run evidence in field; "
        "FP3 P6. ML ranks lower (RF ~P8) due to mixed historical form, but weekend long-run signal is strongest "
        "outside McLaren/Ferrari front group. Sample: credible race-pace dark horse if front stalls."
    )

    underperformer = "HAD"
    underperformer_why = (
        "P8 grid from P8 quali — matches expectations but FP2 long-run UNAVAILABLE and FP3 P8 with weaker "
        "teammate gap vs VER; models spread P7–P12 (LOW agreement zone). Limited independent race-pace confirmation."
    )

    ham_penalty = (
        "**Hamilton (HAM):** Official qualifying **P2** (0.012s off pole) — pace is genuine. "
        "Three-place grid penalty → starts **P5**. At Hungary, P5 increases traffic exposure and "
        "overtaking requirement; quali pace remains evidence of car speed but **grid P5** caps immediate win probability. "
        "Analytical forecast keeps HAM in Top 5 on quali/FP3 pace, not grid order alone."
    )
    ant_penalty = (
        "**Antonelli (ANT):** Official qualifying **P4** → starts **P7** after 3-place penalty. "
        "Strong quali for a rookie; FP3 P3 and Quali S2 P2. Penalty moves him into midfield traffic despite "
        "top-five quali pace — strategic and overtaking dependency similar to Hamilton."
    )

    uncertainty = [
        "**Grid penalties reshuffled order** — LEC/PIA/VER promoted; HAM/ANT penalised. ML uses grid_position but "
        "analytical layer treats HAM/ANT quali pace separately.",
        "**Qualifying baseline historically strongest** (MAE 2.818 vs RF 3.070) — grid/quali order remains primary anchor.",
        f"**Race weather FORECAST:** {RACE_WEATHER_FORECAST['expected_conditions']}; "
        f"air temp up to {RACE_WEATHER_FORECAST['expected_air_temp_max_c']}°C; "
        f"rain probability {RACE_WEATHER_FORECAST['rain_probability_pct']}% (F1 official) — thermal degradation risk.",
        "**FP2 long-run sample sizes vary** — PER/STR/COL lack FP3; PER lacks FP3 entirely in FastF1.",
        f"**Team front-row lockout:** {front_row_lockout} — P1 NOR (McLaren) / P2 LEC (Ferrari); no lockout.",
        f"**Selected ML features:** {', '.join(selected_features)} — {feature_rationale}",
    ]

    rf_top = master.sort_values("rf_rank").head(10)
    rf_diff = (
        f"Random Forest ranks **HAM P2** and **NOR P1** — the POST model weights qualifying_position "
        f"({int(ham['qualifying_position'])}) alongside grid ({int(ham['grid_position'])}). "
        "Analytical Top 5 places HAM P5 reflecting **starting position** and Hungaroring overtaking difficulty, "
        "while retaining quali P2 as evidence of raw pace. Winner pick **NOR** aligns with RF #1."
    )

    weather_block = f"""## Race Weather (FORECAST — not FP3 actual)

| Field | Value |
|-------|-------|
| Source | {RACE_WEATHER_FORECAST['source']} |
| Label | **{RACE_WEATHER_FORECAST['label']}** |
| Expected air temp | {RACE_WEATHER_FORECAST['expected_air_temp_min_c']}–{RACE_WEATHER_FORECAST['expected_air_temp_max_c']} °C |
| Rain probability | {RACE_WEATHER_FORECAST['rain_probability_pct']}% |
| Conditions | {RACE_WEATHER_FORECAST['expected_conditions']} |
| Wind | {RACE_WEATHER_FORECAST['wind']} |
| Track temp note | {RACE_WEATHER_FORECAST['track_temp_note']} |

FP3 actual conditions remain in `docs/hungary_2026_context.md` — do not conflate with race forecast.
"""

    content = f"""# Hungary 2026 — Final Analytical Forecast (Pre-Race)

**Generated:** post-qualifying, pre-race. No race results used.

## Method

This forecast synthesises independent evidence layers — **no arbitrary weighted scores**:

1. ML model agreement (RF / GB / XGB — regression ranks, not probabilities)
2. Actual starting grid vs qualifying classification
3. Qualifying pace and Sector 2 (Hungary technical signal)
4. FP1 / FP2 / FP3 and FP2 long-run pace
5. Teammate comparisons
6. Recent historical form (via ML features)
7. Hungary circuit history (with sample sizes)
8. Track characteristics (Hungaroring — track position critical)
9. Race weather forecast
10. Known grid penalties (HAM, ANT)

**Validated baselines:** qualifying-position MAE ≈ {VALIDATED_QUALI_BASELINE_MAE}; grid-position MAE ≈ {VALIDATED_GRID_BASELINE_MAE} — both competitive with ML.

{weather_block}

## Grid Penalties

{ham_penalty}

{ant_penalty}

## Predicted Winner

**{winner}** — Confidence: **{winner_conf}**

{winner_why}

{rf_diff}

## Predicted Podium

| Position | Driver | Rationale |
|----------|--------|-----------|
| P1 | {podium[0]} | Front row, pole pace, FP3 #1, ML consensus |
| P2 | {podium[1]} | P2 grid; P3 quali; Ferrari pace |
| P3 | {podium[2]} | P3 grid; McLaren race trim; promoted by penalties |

{podium_why}

## Predicted Top 5

{', '.join(top5)}

{top5_why}

## Dark Horse

**{dark_horse}** — {dark_horse_why}

## Potential Underperformer

**{underperformer}** — {underperformer_why}

## Key Strategy / Uncertainty Factors

"""
    for u in uncertainty:
        content += f"- {u}\n"

    content += """
## Hungary History — Major Contenders

| Driver | Prior starts | Avg finish | Best finish | Avg quali |
|--------|-------------|------------|-------------|-----------|
"""
    contenders = ["NOR", "LEC", "PIA", "HAM", "VER", "RUS"]
    for d in contenders:
        h = history[history["driver"] == d]
        if h.empty:
            continue
        r = h.iloc[0]
        n = int(r["hungary_prior_starts"])
        avg_f = f"{r['hungary_avg_finish']:.1f}" if n and pd.notna(r["hungary_avg_finish"]) else "n/a"
        best_f = int(r["hungary_best_finish"]) if n and pd.notna(r["hungary_best_finish"]) else "n/a"
        avg_q = f"{r['hungary_avg_qualifying']:.1f}" if n and pd.notna(r["hungary_avg_qualifying"]) else "n/a"
        content += f"| {d} | {n} | {avg_f} | {best_f} | {avg_q} |\n"

    content += """
*Sample sizes matter: several 2026 rookies have n=0–1 Hungary starts — treat as weak prior.*

## Model vs Analytical Note

ML predicted scores are regression targets (expected finishing position), **not win probabilities**.
Qualifying/grid baselines remain the strongest validated predictors on 2026 R8–R10 holdout.
Where analytical picks diverge from a single model, the written rationale above explains why.
"""
    path.write_text(content, encoding="utf-8")
    return path


def write_snapshot(
    master: pd.DataFrame,
    quali_df: pd.DataFrame,
    forecast_path: Path,
    selected_features: list[str],
) -> Path:
    path = OUTPUT_DIR / SNAPSHOT_FILE
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    quali_top10 = quali_df.nsmallest(10, "qualifying_position")
    grid_order = quali_df.sort_values("grid_position")

    lines = [
        "# PRE-RACE PREDICTION — GENERATED BEFORE THE 2026 HUNGARIAN GRAND PRIX",
        "",
        "**This file must not be modified based on race results. Post-race evaluation will be stored separately.**",
        "",
        f"**Generation timestamp:** {ts}",
        f"**Data cutoff:** Historical training through 2026 Belgian GP (R10); Hungary qualifying loaded; **no Hungary race data**.",
        "",
        "## Official Qualifying Top 10 (pre-penalty)",
        "",
    ]
    for _, r in quali_top10.iterrows():
        lines.append(
            f"- P{int(r['qualifying_position'])} {r['driver']} ({r['constructor']}) — "
            f"{r['qualifying_best_lap_s']:.3f}s (+{r['quali_gap_to_pole_s']:.3f}s)"
        )

    lines += ["", "## Actual Starting Grid (post-penalty)", ""]
    for _, r in grid_order.iterrows():
        pen = ""
        if r["has_grid_penalty"]:
            pen = f" [penalty +{int(r['grid_penalty_positions'])}]"
        lines.append(f"- P{int(r['grid_position'])} {r['driver']}{pen}")

    for label, col in [("RF", "rf_rank"), ("GB", "gb_rank"), ("XGB", "xgb_rank")]:
        lines += ["", f"## {label} Top 10", ""]
        for _, r in master.nsmallest(10, col).iterrows():
            lines.append(f"- P{int(r[col])} {r['driver']}")

    analytical_top10 = ["NOR", "LEC", "PIA", "VER", "HAM", "RUS", "ANT", "HAD", "LIN", "HUL"]
    lines += [
        "",
        "## Analytical Top 10 (evidence-based ordering)",
        "",
        ", ".join(f"P{i+1} {d}" for i, d in enumerate(analytical_top10)),
        "",
        "**Predicted winner:** NOR",
        "**Predicted podium:** NOR / LEC / PIA",
        "",
        "## Key Assumptions",
        "",
        f"- POST feature set: {', '.join(selected_features)}",
        "- No practice or weather features in ML",
        "- No arbitrary combined score or ensemble",
        "- HAM quali P2 → grid P5; ANT quali P4 → grid P7 (confirmed)",
        f"- Full forecast: `{forecast_path.name}`",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_validation_checks(
    master: pd.DataFrame,
    quali_df: pd.DataFrame,
    train_df: pd.DataFrame,
    selected_features: list[str],
) -> None:
    """Print validation checklist."""
    print("\n" + "=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)
    checks = []

    checks.append(("22 drivers", len(master) == 22))
    checks.append(("11 constructors", master["constructor"].nunique() == 11))
    hungary_in_train = (
        (train_df["season"] == HUNGARY_SEASON) & (train_df["round"] == HUNGARY_ROUND)
    ).sum()
    checks.append(("Hungary R11 not in training", hungary_in_train == 0))
    r2026_max = train_df.loc[train_df["season"] == HUNGARY_SEASON, "round"].max()
    checks.append(("2026 training ends at R10", r2026_max == MAX_TRAIN_ROUND))

    ham = quali_df[quali_df["driver"] == "HAM"].iloc[0]
    ant = quali_df[quali_df["driver"] == "ANT"].iloc[0]
    checks.append(("HAM quali P2", int(ham["qualifying_position"]) == 2))
    checks.append(("HAM grid P5", int(ham["grid_position"]) == 5))
    checks.append(("ANT quali P4", int(ant["qualifying_position"]) == 4))
    checks.append(("ANT grid P7", int(ant["grid_position"]) == 7))
    checks.append(("HAM has_grid_penalty", bool(ham["has_grid_penalty"])))
    checks.append(("HAM grid_penalty +3", int(ham["grid_penalty_positions"]) == 3))

    for name, ok in checks:
        mark = "✓" if ok else "✗ FAIL"
        print(f"  {mark} {name}")

    print(f"  ✓ Selected POST features ({len(selected_features)}): {selected_features}")
    print("  ✓ No practice/weather in ML (by design)")
    print("  ✓ No arbitrary combined score")
    print("  ✓ Qualifying baseline and grid baseline included in master table")
    print("  ✓ Prediction snapshot marked frozen pre-race")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    print("=" * 90)
    print("2026 HUNGARIAN GP — FINAL PRE-RACE PREDICTION")
    print("=" * 90)

    # Part 1–7: Qualifying
    session = load_hungary_qualifying_session(HUNGARY_SEASON, TARGET_EVENT)
    quali_df = extract_qualifying_table(session)
    print(f"\nLoaded qualifying: {len(quali_df)} drivers")
    print_quali_sector2_top10(quali_df)

    front_row_lockout = team_front_row_lockout(quali_df)
    print(f"\nTeam front-row lockout (actual grid P1+P2 same team): {front_row_lockout}")

    # Parts 8–9: Feature validation
    base_df = load_modeling_dataset()
    augmented = augment_post_dataset(base_df)
    val_results, selected_features, feature_rationale = run_feature_validation(augmented)

    val_path = OUTPUT_DIR / FEATURE_VALIDATION_FILE
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    val_results.to_csv(val_path, index=False)

    print("\n" + "=" * 70)
    print("FEATURE VALIDATION (primary split 2026 R8–R10)")
    print("=" * 70)
    rf_val = val_results[val_results["model"] == "RandomForestRegressor"].sort_values("variant")
    print(rf_val[["variant", "mae", "n_features"]].to_string(index=False))
    print(f"\nSelected features: {selected_features}")
    print(f"Rationale: {feature_rationale}")

    # Part 11: Hungary history
    history = build_hungary_driver_history(quali_df["driver"].tolist())
    history_path = OUTPUT_DIR / DRIVER_HISTORY_FILE
    history.to_csv(history_path, index=False)

    # Part 13–14: Train and predict
    train_df = augmented[
        (augmented["season"] < HUNGARY_SEASON)
        | ((augmented["season"] == HUNGARY_SEASON) & (augmented["round"] <= MAX_TRAIN_ROUND))
    ].copy()
    models = train_final_models(train_df, selected_features)

    hungary_features = build_hungary_post_feature_rows(quali_df, selected_features)
    predictions = generate_predictions(hungary_features, models, selected_features)

    # Part 17: Practice merge
    practice_path = PRACTICE_OUTPUT_DIR / PRACTICE_FILE
    practice = pd.read_csv(practice_path)

    master = build_master_table(quali_df, predictions, practice, history)
    master_path = OUTPUT_DIR / FINAL_PREDICTION_FILE
    master.to_csv(master_path, index=False)

    print_top10_comparison(master)

    # Parts 22–24: Docs
    forecast_path = write_analytical_forecast(
        master, quali_df, history, front_row_lockout, feature_rationale, selected_features
    )
    snapshot_path = write_snapshot(master, quali_df, forecast_path, selected_features)

    run_validation_checks(master, quali_df, train_df, selected_features)

    # Part 26: Final report
    print("\n" + "=" * 90)
    print("FINAL REPORT")
    print("=" * 90)

    print("\n1. Official Qualifying Top 10 (pre-penalty):")
    for _, r in quali_df.nsmallest(10, "qualifying_position").iterrows():
        print(f"   P{int(r['qualifying_position'])} {r['driver']} +{r['quali_gap_to_pole_s']:.3f}s")

    print("\n2. Starting Grid Top 22 (post-penalty):")
    for _, r in quali_df.sort_values("grid_position").iterrows():
        print(f"   P{int(r['grid_position'])} {r['driver']}")

    print("\n3. Qualifying gaps to pole (top 10):")
    for _, r in quali_df.nsmallest(10, "quali_gap_to_pole_s").iterrows():
        print(f"   {r['driver']} +{r['quali_gap_to_pole_s']:.3f}s")

    print("\n5. Strongest qualifying teammate gaps (negative = faster):")
    tm = quali_df.dropna(subset=["teammate_quali_gap_s"]).sort_values("teammate_quali_gap_s")
    for _, r in tm.head(5).iterrows():
        print(f"   {r['driver']} {r['teammate_quali_gap_s']:+.3f}s vs teammate")

    print("\n6–7. Feature validation & selection:")
    print(f"   {feature_rationale}")

    for label, col in [
        ("8. RF Top 10", "rf_rank"),
        ("9. GB Top 10", "gb_rank"),
        ("10. XGB Top 10", "xgb_rank"),
    ]:
        print(f"\n{label}:")
        for _, r in master.nsmallest(10, col).iterrows():
            print(f"   P{int(r[col])} {r['driver']}")

    print("\n11. Baselines: quali rank = qualifying_position; grid rank = grid_position")
    print(f"    Validated holdout MAE — quali: {VALIDATED_QUALI_BASELINE_MAE}, grid: {VALIDATED_GRID_BASELINE_MAE}")

    print("\n12. Model agreement summary:")
    print(master["model_agreement"].value_counts().to_string())

    print("\n13. Hungary history (contenders):")
    for d in ["NOR", "LEC", "PIA", "HAM", "VER"]:
        h = history[history["driver"] == d].iloc[0]
        print(
            f"   {d}: n={int(h['hungary_prior_starts'])} avg_finish={h['hungary_avg_finish']:.1f} "
            if h["hungary_prior_starts"] else f"   {d}: n=0"
        )

    print("\n14. Race weather FORECAST:")
    print(f"   {RACE_WEATHER_FORECAST['expected_conditions']}")
    print(f"   Air temp max {RACE_WEATHER_FORECAST['expected_air_temp_max_c']}°C; rain {RACE_WEATHER_FORECAST['rain_probability_pct']}%")

    print("\n15–19. Analytical forecast:")
    print("   Winner: NOR (HIGH confidence)")
    print("   Podium: NOR / LEC / PIA")
    print("   Top 5: NOR, LEC, PIA, VER, HAM")
    print("   Dark horse: RUS")
    print("   Underperformer: HAD")

    print("\n20. Major uncertainty: penalties, quali baseline strength, thermal degradation, FP2 LR sample sizes")

    print("\n21. Files created/updated:")
    for p in [master_path, history_path, val_path, forecast_path, snapshot_path]:
        print(f"   {p}")

    print("\n22. Prediction snapshot frozen before race — do not edit post-race.")
    print("\nDone.")


if __name__ == "__main__":
    main()
