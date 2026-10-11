"""
Singapore Grand Prix 2026 — staged prediction pipeline.

Priority implementation:
  Stage A — retrospective pre-weekend baseline (honest timestamp; work begins after FP1)
  Stage B — post-FP1 update when FastF1 data is available

Later stages (Sprint Qualifying, Sprint, GP Qualifying, final) expose hooks only until
sessions complete.

Usage:
    python -m src.extend_season_rounds
    python -m src.predict_singapore --stage pre-weekend
    python -m src.predict_singapore --stage post-fp1
    python -m src.predict_singapore --stage all-available
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.circuit_history import (
    build_constructor_circuit_history,
    build_street_circuit_driver_form,
    singapore_driver_history,
)
from src.event_identity import apply_canonical_venues
from src.extend_season_rounds import extend_raw_season
from src.features import (
    add_constructor_form_features,
    add_driver_form_features,
    add_season_context,
    load_raw_results,
    merge_race_qualifying,
    normalize_constructors,
)
from src.singapore_config import (
    DOCS_DIR,
    IST,
    PREDICTIONS_DIR,
    SINGAPORE_CIRCUIT,
    SINGAPORE_EVENT_NAME,
    SINGAPORE_FORM_CUTOFF_EVENT,
    SINGAPORE_ROUND,
    SINGAPORE_SEASON,
    STAGE_FILES,
    now_ist,
    today_ist,
)
from src.train import (
    PRE_QUALIFYING_FEATURES,
    RANDOM_STATE,
    TARGET,
    build_pipeline,
    get_models,
    load_modeling_dataset,
    primary_train_mask,
    primary_val_mask,
    train_and_evaluate,
    SplitSpec,
    verify_no_overlap,
    POST_QUALIFYING_FEATURES,
    baseline_mae,
)
from src.predict_hungary_final import (
    add_avg_gap_to_pole_last_3,
    add_teammate_quali_gap_historical,
    augment_post_dataset,
    generate_predictions,
    train_final_models,
)
from src.weekend_sessions import (
    build_fp1_analysis_table,
    build_gp_qualifying_table,
    build_sprint_qualifying_table,
    build_sprint_race_table,
    save_retrieval_status,
)

logger = logging.getLogger(__name__)

MODELING_THROUGH_R16 = "modeling_dataset_through_r16.csv"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def championship_table(race: pd.DataFrame, season: int, before_round: int) -> pd.DataFrame:
    """Championship points and position before target round (completed races only)."""
    hist = race[(race["season"] == season) & (race["round"] < before_round)].copy()
    pts = hist.groupby("driver")["points"].sum().sort_values(ascending=False)
    rows = []
    for rank, (driver, points) in enumerate(pts.items(), start=1):
        rows.append(
            {
                "driver": driver,
                "championship_points": float(points),
                "championship_position": rank,
            }
        )
    return pd.DataFrame(rows)


def latest_entry_list(race: pd.DataFrame, season: int, round_number: int) -> pd.DataFrame:
    """Official entrants from the most recent completed race at or before round_number."""
    sub = race[(race["season"] == season) & (race["round"] == round_number)]
    if sub.empty:
        sub = race[(race["season"] == season) & (race["round"] < round_number)]
        if sub.empty:
            return pd.DataFrame()
        last_rnd = sub["round"].max()
        sub = race[(race["season"] == season) & (race["round"] == last_rnd)]
    return sub[["driver", "constructor"]].drop_duplicates()


def _load_race_quali_canonical():
    race_raw, quali = load_raw_results()
    return apply_canonical_venues(race_raw, quali)


def rebuild_modeling_through_r16() -> Path:
    """Build modeling matrix from raw CSVs; exclude Singapore 2026."""
    from src.features import build_modeling_dataset

    race_raw, quali = _load_race_quali_canonical()
    df = build_modeling_dataset(race_frame=race_raw, quali_frame=quali, quiet=True)
    out = df[
        (df["season"] < SINGAPORE_SEASON)
        | ((df["season"] == SINGAPORE_SEASON) & (df["round"] < SINGAPORE_ROUND))
    ].copy()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / MODELING_THROUGH_R16
    out.to_csv(path, index=False)
    logger.info("Wrote %s (%s rows)", path, len(out))
    return path


def ensure_2026_raw_extended() -> dict:
    return extend_raw_season(
        SINGAPORE_SEASON,
        SINGAPORE_FORM_CUTOFF_EVENT,
        as_of=today_ist(),
    )


def build_singapore_feature_skeleton(entry: pd.DataFrame) -> pd.DataFrame:
    """PRE-qualifying feature rows for Singapore using history through R16."""
    race_raw, quali = _load_race_quali_canonical()
    race = normalize_constructors(race_raw)
    merged = merge_race_qualifying(race, quali)

    skeleton = []
    for _, row in entry.iterrows():
        skeleton.append(
            {
                "season": SINGAPORE_SEASON,
                "round": SINGAPORE_ROUND,
                "race_name": SINGAPORE_EVENT_NAME,
                "circuit": SINGAPORE_CIRCUIT,
                "driver": row["driver"],
                "constructor": row["constructor"],
                "classified_status": "Finished",
                "finishing_position": np.nan,
                "points": np.nan,
                "grid_position": np.nan,
                "qualifying_position": np.nan,
                "qualifying_best_lap_s": np.nan,
                "gap_to_pole_s": np.nan,
            }
        )
    sk = pd.DataFrame(skeleton)
    combined = pd.concat([merged, sk], ignore_index=True)
    combined = add_driver_form_features(combined)
    combined = add_constructor_form_features(combined)
    combined = add_season_context(combined)
    return combined[
        (combined["season"] == SINGAPORE_SEASON) & (combined["round"] == SINGAPORE_ROUND)
    ].copy()


def run_model_comparison(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal validation on extended dataset (2026 val R14–R16 when present).

    POST models and qualifying/grid baselines use quali/grid from validation races
    only — not applicable to Singapore Stage A (pre-weekend) predictions.
    """
    val_rounds = train_df.loc[
        (train_df["season"] == SINGAPORE_SEASON) & (train_df["round"] >= 14),
        "round",
    ].unique()
    if len(val_rounds) < 2:
        # Fallback: primary split on R8-10 if R14-16 not in file
        split = SplitSpec(
            name="primary_2026_r8_r10",
            train_mask=primary_train_mask(train_df),
            val_mask=primary_val_mask(train_df),
        )
    else:
        val_round_list = sorted(int(r) for r in val_rounds)
        first_val_round = val_round_list[0]
        val_rounds_used = val_round_list[-3:]
        val_mask = (train_df["season"] == SINGAPORE_SEASON) & (
            train_df["round"].isin(val_rounds_used)
        )
        train_mask = ~val_mask & (
            train_df["season"].isin([2024, 2025])
            | (
                (train_df["season"] == SINGAPORE_SEASON)
                & (train_df["round"] < first_val_round)
            )
        )
        split = SplitSpec("singapore_extended_val", train_mask, val_mask)

    verify_no_overlap(train_df, split.train_mask, split.val_mask)
    rows = []
    for model_name, model in get_models().items():
        for config, feats in [
            ("PRE", PRE_QUALIFYING_FEATURES),
            ("POST", POST_QUALIFYING_FEATURES),
        ]:
            pipeline = build_pipeline(model_name, model)
            result, fitted, _ = train_and_evaluate(
                train_df, split, feats, config, model_name, pipeline
            )
            rows.append(
                {
                    "split": split.name,
                    "configuration": config,
                    "model": model_name,
                    "mae": result.mae,
                    "rmse": result.rmse,
                    "winner_accuracy": result.winner_accuracy,
                    "podium_hit_rate": result.podium_hit_rate,
                    "train_rows": result.train_rows,
                    "val_rows": result.val_rows,
                }
            )
    val_df = train_df.loc[split.val_mask]
    rows.append(
        {
            "split": split.name,
            "configuration": "baseline",
            "model": "qualifying_position",
            "mae": baseline_mae(val_df, "qualifying_position", "quali"),
            "val_rows": len(val_df),
        }
    )
    rows.append(
        {
            "split": split.name,
            "configuration": "baseline",
            "model": "grid_position",
            "mae": baseline_mae(val_df, "grid_position", "grid"),
            "val_rows": len(val_df),
        }
    )
    return pd.DataFrame(rows)


def train_pre_rf(train_df: pd.DataFrame) -> Pipeline:
    model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        n_jobs=-1,
    )
    pipe = build_pipeline("RandomForestRegressor", model)
    pipe.fit(train_df[PRE_QUALIFYING_FEATURES], train_df[TARGET])
    return pipe


def stage_a_pre_weekend(
    *,
    generated_at: str,
    data_cutoff: str,
) -> pd.DataFrame:
    """
    Retrospective pre-weekend baseline.

    Label: generated after FP1 scheduled time; excludes all Singapore 2026 sessions.
    """
    race_raw, quali = _load_race_quali_canonical()
    race = normalize_constructors(race_raw)
    entry = latest_entry_list(race, SINGAPORE_SEASON, SINGAPORE_ROUND - 1)
    if entry.empty:
        raise ValueError("No entry list from 2026 races before Singapore")

    modeling_path = rebuild_modeling_through_r16()
    train_df = pd.read_csv(modeling_path)

    skel = build_singapore_feature_skeleton(entry)
    pipe = train_pre_rf(train_df)
    imp = pd.DataFrame(
        {
            "feature": PRE_QUALIFYING_FEATURES,
            "importance": pipe.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    imp.to_csv(PREDICTIONS_DIR.parent / STAGE_FILES["feature_importance"], index=False)

    skel["predicted_score"] = pipe.predict(skel[PRE_QUALIFYING_FEATURES])
    skel["rf_pre_weekend_rank"] = skel["predicted_score"].rank(method="min").astype(int)

    sg_hist = singapore_driver_history(race, quali, SINGAPORE_SEASON, SINGAPORE_ROUND)
    street = build_street_circuit_driver_form(
        race, before_season=SINGAPORE_SEASON, before_round=SINGAPORE_ROUND
    )
    champ = championship_table(race, SINGAPORE_SEASON, SINGAPORE_ROUND)
    ctor_hist = build_constructor_circuit_history(
        race,
        race_name=SINGAPORE_EVENT_NAME,
        before_season=SINGAPORE_SEASON,
        before_round=SINGAPORE_ROUND,
        min_season=2019,
    )

    out = skel.merge(entry, on=["driver", "constructor"], how="left")
    out = out.merge(sg_hist, on="driver", how="left")
    out = out.merge(street, on="driver", how="left")
    out = out.merge(champ, on="driver", how="left")
    out = out.merge(ctor_hist, on="constructor", how="left")

    out["prediction_stage"] = "A_pre_weekend_retrospective"
    out["prediction_timestamp_ist"] = generated_at
    out["input_data_cutoff"] = data_cutoff
    out["label_note"] = (
        "Retrospective baseline generated after scheduled FP1; "
        "NOT frozen before FP1. No Singapore 2026 session data included."
    )
    return out.sort_values("rf_pre_weekend_rank")


def stage_b_post_fp1(stage_a: pd.DataFrame, fp1: pd.DataFrame) -> pd.DataFrame:
    """
    Merge FP1 analytical layer; ML unchanged (practice not in training features).

    ``rf_pre_weekend_rank`` is intentionally unchanged from Stage A; compare ``fp1_rank``
    for practice-based ordering.
    """
    out = stage_a.copy()
    if fp1.empty:
        out["prediction_stage"] = "B_post_fp1_unavailable"
        out["fp1_data_available"] = False
        return out

    fp1_cols = [c for c in fp1.columns if c not in ("constructor",)]
    out = out.merge(fp1, on="driver", how="left", suffixes=("", "_fp1"))
    out["prediction_stage"] = "B_post_fp1"
    out["fp1_data_available"] = True
    out["prediction_timestamp_ist"] = now_ist().isoformat()
    out["fp1_daytime_note"] = (
        "FP1 is daytime; limited transfer to night qualifying/race at Marina Bay."
    )
    return out


def build_evidence_matrix(pred: pd.DataFrame) -> pd.DataFrame:
    """Explicit evidence availability — no combined score."""
    rows = []
    for _, r in pred.iterrows():
        rows.append(
            {
                "driver": r["driver"],
                "constructor": r["constructor"],
                "historical_singapore": "available"
                if pd.notna(r.get("circuit_starts")) and r.get("circuit_starts", 0) > 0
                else "sparse_or_none",
                "season_2026_form": "available",
                "constructor_form": "available",
                "street_circuit_form": "available"
                if pd.notna(r.get("street_circuit_starts")) and r.get("street_circuit_starts", 0) > 0
                else "sparse",
                "championship_context": "available"
                if pd.notna(r.get("championship_position"))
                else "missing",
                "fp1": "available" if r.get("fp1_data_available") else "unavailable",
                "sprint_qualifying": "not_yet_incorporated",
                "sprint": "not_yet_incorporated",
                "gp_qualifying": "not_yet_incorporated",
                "grid_penalties": "not_yet_incorporated",
                "weather_forecast": "not_yet_incorporated",
                "ml_pre_weekend": "available",
            }
        )
    return pd.DataFrame(rows)


def write_pre_weekend_snapshot(path: Path, pred: pd.DataFrame) -> None:
    top = pred.nsmallest(10, "rf_pre_weekend_rank")
    lines = [
        "# Singapore 2026 — Stage A (retrospective pre-weekend baseline)",
        "",
        f"**Generated (IST):** {pred['prediction_timestamp_ist'].iloc[0]}",
        f"**Data cutoff:** {pred['input_data_cutoff'].iloc[0]}",
        "",
        pred["label_note"].iloc[0],
        "",
        "## RF pre-weekend Top 10",
        "",
    ]
    for _, r in top.iterrows():
        lines.append(f"- P{int(r['rf_pre_weekend_rank'])} {r['driver']} ({r['constructor']})")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_context_doc(path: Path) -> None:
    content = f"""# Singapore 2026 — Circuit and weekend context

## Event

| Field | Value |
|-------|-------|
| Event | {SINGAPORE_EVENT_NAME} |
| Season / Round | {SINGAPORE_SEASON} R{SINGAPORE_ROUND} |
| Format | Sprint qualifying weekend |
| Circuit | {SINGAPORE_CIRCUIT} |

## Night race and climate (interpretive context)

- Permanent night race; track and air temperatures typically fall from daytime practice toward qualifying and the race.
- High humidity and limited overtaking historically — track position matters.
- 2023 Marina Bay layout revision: older Singapore results are down-weighted in historical features.

## Session schedule (supplied project schedule — verify with F1)

See `src/singapore_config.py` `SINGAPORE_WEEKEND_SCHEDULE_IST`.

## Data notes

- 2026 season form uses completed races **before** Singapore (FastF1 schedule: through Bahrain R16).
- User brief referenced Malaysia R16; 2026 calendar has **Bahrain** as round 16.

Weather forecasts and observed session weather are stored separately when available.
"""
    path.write_text(content, encoding="utf-8")


def write_methodology_doc(path: Path) -> None:
    path.write_text(
        """# Singapore 2026 — Methodology

## Stages

| Stage | Information cutoff | ML inputs |
|-------|-------------------|-----------|
| A Pre-weekend (retrospective) | 2026 R1–R16 + historical Singapore | PRE features only |
| B Post-FP1 | + FP1 timing (analytical) | Practice not in ML |
| C Post-Sprint Qualifying | + SQ (planned) | TBD |
| D Post-Sprint | + Sprint (planned) | TBD |
| E Final pre-race | + GP qualifying + grid | POST features |

## Leakage rules

- Rolling features use `shift(1)` before windows (same as Hungary V1).
- Singapore 2026 race and future sessions never enter earlier stages.

## Baselines

Qualifying-position and grid-position baselines are reported alongside ML when qualifying/grid exist.
""",
        encoding="utf-8",
    )


def load_frozen_stage_b() -> pd.DataFrame:
    """Load existing Stage B output — does not regenerate Stages A/B."""
    path = PREDICTIONS_DIR / STAGE_FILES["post_fp1_csv"]
    if not path.exists():
        raise FileNotFoundError(
            f"Stage B file missing: {path}. Run post-FP1 pipeline first without overwriting."
        )
    return pd.read_csv(path)


def _analytical_sq_summary(stage_b: pd.DataFrame, stage_c: pd.DataFrame) -> list[str]:
    """Describe how SQ ordering compares to ML and FP1 (no combined score)."""
    lines: list[str] = []
    if "sq_official_rank" not in stage_c.columns or stage_c["sq_official_rank"].isna().all():
        lines.append("Sprint Qualifying evidence unavailable — analytical view unchanged from Stage B.")
        return lines

    ml_unchanged = (
        "rf_pre_weekend_rank" in stage_c.columns
        and stage_c["rf_pre_weekend_rank"].equals(stage_b["rf_pre_weekend_rank"])
    )
    lines.append(
        f"**ML `rf_pre_weekend_rank`:** unchanged from Stage B ({'yes' if ml_unchanged else 'no'}). "
        "SQ is analytical only, not an ML input."
    )

    valid = stage_c.dropna(subset=["sq_official_rank", "fp1_rank"])
    if not valid.empty:
        movers: list[str] = []
        for _, r in valid.iterrows():
            delta_sq_fp1 = int(r["fp1_rank"]) - int(r["sq_official_rank"])
            if abs(delta_sq_fp1) >= 3:
                movers.append(
                    f"{r['driver']}: SQ P{int(r['sq_official_rank'])} vs FP1 P{int(r['fp1_rank'])} "
                    f"({delta_sq_fp1:+d} positions vs FP1)"
                )
        if movers:
            lines.append("**Largest SQ vs FP1 moves (≥3 positions):** " + "; ".join(movers[:8]))
        else:
            lines.append("**SQ vs FP1:** no driver moved ≥3 positions between FP1 rank and SQ official rank.")

    sq_top5 = stage_c.nsmallest(5, "sq_official_rank")[
        ["driver", "sq_official_rank", "rf_pre_weekend_rank", "fp1_rank"]
    ]
    lines.append("**SQ official Top 5 vs ML / FP1 ranks:**")
    for _, r in sq_top5.iterrows():
        fp1 = int(r["fp1_rank"]) if pd.notna(r.get("fp1_rank")) else "n/a"
        lines.append(
            f"- {r['driver']}: SQ P{int(r['sq_official_rank'])} | ML P{int(r['rf_pre_weekend_rank'])} | FP1 P{fp1}"
        )
    return lines


def build_evidence_matrix_stage_c(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in pred.iterrows():
        rows.append(
            {
                "driver": r["driver"],
                "constructor": r["constructor"],
                "ml_pre_weekend": "available",
                "fp1": "available" if r.get("fp1_data_available") else "unavailable",
                "sprint_qualifying": "available"
                if r.get("sq_data_available")
                else "unavailable",
                "sprint": "not_yet_incorporated",
                "gp_qualifying": "not_yet_incorporated",
                "sunday_grid": "not_yet_incorporated",
                "sq_is_not_sunday_grid": True,
            }
        )
    return pd.DataFrame(rows)


def stage_c_post_sprint_qualifying(
    stage_b: pd.DataFrame,
    sq: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Sprint Qualifying analytical layer on frozen Stage B.

    Does not alter ML scores/ranks. SQ is not Grand Prix qualifying or Sunday grid.
    """
    out = stage_b.copy()
    if sq.empty:
        out["prediction_stage"] = "C_post_sprint_qualifying_unavailable"
        out["sq_data_available"] = False
        out["prediction_timestamp_ist"] = now_ist().isoformat()
        return out

    sq_cols = [c for c in sq.columns if c not in ("constructor",)]
    out = out.merge(sq[sq_cols], on="driver", how="left", suffixes=("", "_sqdup"))
    out["prediction_stage"] = "C_post_sprint_qualifying"
    out["sq_data_available"] = True
    out["prediction_timestamp_ist"] = now_ist().isoformat()
    out["input_data_cutoff"] = (
        str(out["input_data_cutoff"].iloc[0])
        + "; + Singapore 2026 Sprint Qualifying (not GP quali / not Sunday grid)"
    )
    return out


def write_post_sq_snapshot(
    path: Path,
    stage_c: pd.DataFrame,
    status_message: str,
    sq_code: str,
    analytical_lines: list[str],
) -> None:
    ts = stage_c["prediction_timestamp_ist"].iloc[0]
    lines = [
        "# Singapore 2026 — Stage C (post–Sprint Qualifying)",
        "",
        f"**Generated (IST):** {ts}",
        f"**FastF1 session code (from schedule):** `{sq_code}`",
        f"**Retrieval:** {status_message}",
        "",
        "Sprint Qualifying sets the **Sprint** grid only — not Sunday GP qualifying or race grid.",
        "",
        "## Analytical assessment vs Stage A / B",
        "",
    ]
    lines.extend(analytical_lines)
    lines.append("")
    lines.append("## SQ official Top 10")
    lines.append("")
    if stage_c.get("sq_data_available", pd.Series([False])).iloc[0]:
        for _, r in stage_c.nsmallest(10, "sq_official_rank").iterrows():
            gap = r.get("sq_gap_to_fastest_s")
            gap_s = f"+{gap:.3f}s" if pd.notna(gap) else "n/a"
            lines.append(
                f"- P{int(r['sq_official_rank'])} {r['driver']} ({r['constructor']}) — "
                f"best {r.get('sq_best_lap_s', float('nan')):.3f}s ({gap_s})"
            )
    else:
        lines.append("- SQ data not available at generation time.")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stage_c() -> int:
    """Stage C only — reads frozen Stage B; writes new Stage C artifacts."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    stage_b = load_frozen_stage_b()

    sq_table, sq_status, sq_weather, sq_code = build_sprint_qualifying_table()
    save_retrieval_status(PREDICTIONS_DIR.parent / STAGE_FILES["sq_status"], sq_status)

    if not sq_table.empty:
        sq_table.to_csv(PREDICTIONS_DIR.parent / STAGE_FILES["sq_evidence"], index=False)
    if sq_weather:
        pd.DataFrame([sq_weather]).to_csv(
            PREDICTIONS_DIR.parent / STAGE_FILES["weather_context_sq"],
            index=False,
        )

    stage_c = stage_c_post_sprint_qualifying(stage_b, sq_table)
    out_csv = PREDICTIONS_DIR / STAGE_FILES["post_sq_csv"]
    stage_c.to_csv(out_csv, index=False)

    build_evidence_matrix_stage_c(stage_c).to_csv(
        PREDICTIONS_DIR.parent / STAGE_FILES["evidence_matrix_stage_c"],
        index=False,
    )

    analytical = _analytical_sq_summary(stage_b, stage_c)
    write_post_sq_snapshot(
        PREDICTIONS_DIR.parent / STAGE_FILES["post_sq_snapshot"],
        stage_c,
        sq_status.message,
        sq_code,
        analytical,
    )

    print("Stage C complete.")
    print("Output:", out_csv)
    print("SQ success:", sq_status.success, "|", sq_status.message)
    return 0 if sq_status.success else 1


def load_frozen_stage_c() -> pd.DataFrame:
    """Load existing Stage C output — does not regenerate Stages A/B/C."""
    path = PREDICTIONS_DIR / STAGE_FILES["post_sq_csv"]
    if not path.exists():
        raise FileNotFoundError(
            f"Stage C file missing: {path}. Run post-SQ pipeline first without overwriting."
        )
    return pd.read_csv(path)


def _analytical_sprint_summary(stage_c: pd.DataFrame, stage_d: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if "sprint_official_rank" not in stage_d.columns or stage_d["sprint_official_rank"].isna().all():
        lines.append("Sprint race evidence unavailable — analytical view unchanged from Stage C.")
        return lines

    ml_unchanged = (
        "rf_pre_weekend_rank" in stage_d.columns
        and stage_c["rf_pre_weekend_rank"].equals(stage_d["rf_pre_weekend_rank"])
    )
    lines.append(
        f"**ML `rf_pre_weekend_rank`:** unchanged from Stage C ({'yes' if ml_unchanged else 'no'}). "
        "Sprint race is analytical only, not an ML input."
    )

    valid = stage_d.dropna(subset=["sprint_official_rank", "sq_official_rank"])
    if not valid.empty:
        movers: list[str] = []
        for _, r in valid.iterrows():
            delta = int(r["sq_official_rank"]) - int(r["sprint_official_rank"])
            if abs(delta) >= 3:
                movers.append(
                    f"{r['driver']}: Sprint P{int(r['sprint_official_rank'])} vs SQ P{int(r['sq_official_rank'])} "
                    f"({delta:+d} positions vs SQ grid order)"
                )
        if movers:
            lines.append("**Largest Sprint vs SQ moves (≥3 positions):** " + "; ".join(movers[:8]))
        else:
            lines.append("**Sprint vs SQ:** no driver moved ≥3 positions vs SQ starting order.")

    dnfs = stage_d[stage_d.get("sprint_dnf", False) == True]
    if not dnfs.empty:
        lines.append(
            "**Sprint retirements / non-finish (FastF1 status):** "
            + ", ".join(
                f"{r['driver']} ({r.get('sprint_classified_status', 'n/a')})" for _, r in dnfs.iterrows()
            )
        )

    pen = stage_d[stage_d["sprint_race_penalty_notes"].notna()] if "sprint_race_penalty_notes" in stage_d.columns else pd.DataFrame()
    if not pen.empty:
        lines.append("**Race-control penalty messages (retrieved, not reclassified):**")
        for _, r in pen.iterrows():
            lines.append(f"- {r['driver']}: {r['sprint_race_penalty_notes']}")

    top5 = stage_d.nsmallest(5, "sprint_official_rank")[
        ["driver", "sprint_official_rank", "sq_official_rank", "rf_pre_weekend_rank", "fp1_rank"]
    ]
    lines.append("**Sprint official Top 5 vs SQ / ML / FP1:**")
    for _, r in top5.iterrows():
        sq = int(r["sq_official_rank"]) if pd.notna(r.get("sq_official_rank")) else "n/a"
        fp1 = int(r["fp1_rank"]) if pd.notna(r.get("fp1_rank")) else "n/a"
        lines.append(
            f"- {r['driver']}: Sprint P{int(r['sprint_official_rank'])} | SQ P{sq} | "
            f"ML P{int(r['rf_pre_weekend_rank'])} | FP1 P{fp1}"
        )
    return lines


def build_evidence_matrix_stage_d(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in pred.iterrows():
        rows.append(
            {
                "driver": r["driver"],
                "constructor": r["constructor"],
                "ml_pre_weekend": "available",
                "fp1": "available" if r.get("fp1_data_available") else "unavailable",
                "sprint_qualifying": "available" if r.get("sq_data_available") else "unavailable",
                "sprint": "available" if r.get("sprint_data_available") else "unavailable",
                "gp_qualifying": "not_yet_incorporated",
                "sunday_grid": "not_yet_incorporated",
                "sprint_is_not_sunday_gp": True,
            }
        )
    return pd.DataFrame(rows)


def stage_d_post_sprint(
    stage_c: pd.DataFrame,
    sprint: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Sprint race analytical layer on frozen Stage C. Does not alter ML scores/ranks."""
    out = stage_c.copy()
    if sprint.empty:
        out["prediction_stage"] = "D_post_sprint_unavailable"
        out["sprint_data_available"] = False
        out["prediction_timestamp_ist"] = now_ist().isoformat()
        return out

    sprint_cols = [c for c in sprint.columns if c not in ("constructor",)]
    out = out.merge(sprint[sprint_cols], on="driver", how="left", suffixes=("", "_sprintdup"))
    out["prediction_stage"] = "D_post_sprint"
    out["sprint_data_available"] = True
    out["prediction_timestamp_ist"] = now_ist().isoformat()
    out["input_data_cutoff"] = (
        str(out["input_data_cutoff"].iloc[0])
        + "; + Singapore 2026 Sprint race (not GP quali / not Sunday race)"
    )
    return out


def write_post_sprint_snapshot(
    path: Path,
    stage_d: pd.DataFrame,
    status_message: str,
    sprint_code: str,
    analytical_lines: list[str],
    verification: dict,
) -> None:
    ts = stage_d["prediction_timestamp_ist"].iloc[0]
    lines = [
        "# Singapore 2026 — Stage D (post–Sprint)",
        "",
        f"**Generated (IST):** {ts}",
        f"**FastF1 session code (from schedule):** `{sprint_code}`",
        f"**Schedule label:** {verification.get('schedule_session_label', 'n/a')}",
        f"**Verified event:** {verification.get('verified_event_name', 'n/a')} (R{verification.get('verified_round', 'n/a')})",
        f"**Session complete:** {verification.get('session_complete', False)}",
        f"**Retrieval:** {status_message}",
        "",
        "The Sprint is a separate session from Sprint Qualifying and from Sunday’s Grand Prix.",
        "",
        "## Analytical assessment vs Stages A–C",
        "",
    ]
    lines.extend(analytical_lines)
    lines.append("")
    lines.append("## Sprint official Top 10")
    lines.append("")
    if stage_d.get("sprint_data_available", pd.Series([False])).iloc[0]:
        for _, r in stage_d.nsmallest(10, "sprint_official_rank").iterrows():
            status_txt = r.get("sprint_classified_status") or "Classified"
            pts = r.get("sprint_points")
            pts_s = f"{pts:g} pts" if pd.notna(pts) else "points not in FastF1 feed"
            lap_s = ""
            if pd.notna(r.get("sprint_best_lap_s")):
                lap_s = f" | best lap {float(r['sprint_best_lap_s']):.3f}s"
            grid_s = ""
            if pd.notna(r.get("sprint_grid_position")):
                grid_s = f" | from grid P{int(r['sprint_grid_position'])}"
            lines.append(
                f"- P{int(r['sprint_official_rank'])} {r['driver']} ({r['constructor']}) — "
                f"{status_txt}{grid_s} | {pts_s}{lap_s}"
            )
    else:
        lines.append("- Sprint data not available at generation time.")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stage_d() -> int:
    """Stage D only — reads frozen Stage C; writes new Stage D artifacts."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    stage_c = load_frozen_stage_c()

    sprint_table, sprint_status, sprint_weather, sprint_code, verification = build_sprint_race_table()
    save_retrieval_status(
        PREDICTIONS_DIR.parent / STAGE_FILES["sprint_status"],
        sprint_status,
        **verification,
    )

    if not sprint_table.empty:
        sprint_table.to_csv(PREDICTIONS_DIR.parent / STAGE_FILES["sprint_evidence"], index=False)
    if sprint_weather:
        pd.DataFrame([sprint_weather]).to_csv(
            PREDICTIONS_DIR.parent / STAGE_FILES["weather_context_sprint"],
            index=False,
        )

    stage_d = stage_d_post_sprint(stage_c, sprint_table)
    out_csv = PREDICTIONS_DIR / STAGE_FILES["post_sprint_csv"]
    stage_d.to_csv(out_csv, index=False)

    build_evidence_matrix_stage_d(stage_d).to_csv(
        PREDICTIONS_DIR.parent / STAGE_FILES["evidence_matrix_stage_d"],
        index=False,
    )

    analytical = _analytical_sprint_summary(stage_c, stage_d)
    write_post_sprint_snapshot(
        PREDICTIONS_DIR.parent / STAGE_FILES["post_sprint_snapshot"],
        stage_d,
        sprint_status.message,
        sprint_code,
        analytical,
        verification,
    )

    print("Stage D complete.")
    print("Output:", out_csv)
    print("Sprint success:", sprint_status.success, "|", sprint_status.message)
    return 0 if sprint_status.success else 1


def load_frozen_stage_d() -> pd.DataFrame:
    """Load existing Stage D output — does not regenerate Stages A–D."""
    path = PREDICTIONS_DIR / STAGE_FILES["post_sprint_csv"]
    if not path.exists():
        raise FileNotFoundError(
            f"Stage D file missing: {path}. Run post-Sprint pipeline first without overwriting."
        )
    return pd.read_csv(path)


def build_singapore_post_feature_rows(gp_quali: pd.DataFrame) -> pd.DataFrame:
    """POST-qualifying feature rows for Singapore using history through R16."""
    race_raw, quali = _load_race_quali_canonical()
    race = normalize_constructors(race_raw)
    merged = merge_race_qualifying(race, quali)

    skeleton_rows = []
    for _, row in gp_quali.iterrows():
        skeleton_rows.append(
            {
                "season": SINGAPORE_SEASON,
                "round": SINGAPORE_ROUND,
                "race_name": SINGAPORE_EVENT_NAME,
                "circuit": SINGAPORE_CIRCUIT,
                "driver": row["driver"],
                "constructor": row["constructor"],
                "classified_status": "Finished",
                "finishing_position": np.nan,
                "points": np.nan,
                "grid_position": row["gp_grid_position"],
                "qualifying_position": row["gp_qualifying_position"],
                "qualifying_best_lap_s": row["gp_qualifying_best_lap_s"],
                "gap_to_pole_s": row["gp_quali_gap_to_pole_s"],
            }
        )
    skeleton = pd.DataFrame(skeleton_rows)
    combined = pd.concat([merged, skeleton], ignore_index=True)
    combined = add_driver_form_features(combined)
    combined = add_constructor_form_features(combined)
    combined = add_season_context(combined)
    combined = add_teammate_quali_gap_historical(combined, quali, race)
    combined = add_avg_gap_to_pole_last_3(combined)

    sg = combined[
        (combined["season"] == SINGAPORE_SEASON) & (combined["round"] == SINGAPORE_ROUND)
    ].copy()
    tm_map = gp_quali.set_index("driver")["gp_teammate_quali_gap_s"].to_dict()
    sg["teammate_quali_gap_s"] = sg["driver"].map(tm_map)
    return sg


def build_evidence_matrix_stage_e(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in pred.iterrows():
        rows.append(
            {
                "driver": r["driver"],
                "constructor": r["constructor"],
                "ml_pre_weekend": "available",
                "fp1": "available" if r.get("fp1_data_available") else "unavailable",
                "sprint_qualifying": "available" if r.get("sq_data_available") else "unavailable",
                "sprint": "available" if r.get("sprint_data_available") else "unavailable",
                "gp_qualifying": "available" if r.get("gp_qual_data_available") else "unavailable",
                "sunday_grid": "available" if r.get("gp_qual_data_available") else "unavailable",
                "predicted_race_finish": "available" if r.get("gp_qual_data_available") else "unavailable",
            }
        )
    return pd.DataFrame(rows)


def _analytical_final_summary(stage_d: pd.DataFrame, stage_e: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if not bool(stage_e.get("gp_qual_data_available", pd.Series([False])).iloc[0]):
        lines.append("GP qualifying unavailable — no final race prediction produced.")
        return lines

    ml_unchanged = (
        "rf_pre_weekend_rank" in stage_e.columns
        and stage_d["rf_pre_weekend_rank"].equals(stage_e["rf_pre_weekend_rank"])
    )
    lines.append(
        f"**Historical PRE-ML `rf_pre_weekend_rank`:** unchanged from Stage D ({'yes' if ml_unchanged else 'no'})."
    )
    lines.append(
        "**POST-qualifying ML** (`rf_post_quali_rank`, etc.) uses grid/quali features trained on "
        f"{SINGAPORE_SEASON} R1–{SINGAPORE_ROUND - 1} — separate from the retrospective pre-weekend rank."
    )

    if "rf_post_quali_rank" in stage_e.columns and "gp_grid_position" in stage_e.columns:
        diff = stage_e.dropna(subset=["rf_post_quali_rank", "gp_grid_position"])
        movers = []
        for _, r in diff.iterrows():
            delta = int(r["gp_grid_position"]) - int(r["rf_post_quali_rank"])
            if abs(delta) >= 4:
                movers.append(
                    f"{r['driver']}: predicted finish P{int(r['rf_post_quali_rank'])} vs grid P{int(r['gp_grid_position'])}"
                )
        if movers:
            lines.append("**Predicted finish vs Sunday grid (≥4 slots):** " + "; ".join(movers[:8]))
        else:
            lines.append("**Predicted finish vs grid:** no driver ≥4 slots from grid in POST-ML vs grid.")

    pen = stage_e[stage_e.get("gp_has_documented_grid_penalty", False) == True]
    if not pen.empty:
        lines.append("**Documented grid penalties (race-control text in quali session):**")
        for _, r in pen.iterrows():
            lines.append(
                f"- {r['driver']}: quali P{int(r['gp_qualifying_position'])} → grid P{int(r['gp_grid_position'])} "
                f"({r.get('gp_quali_penalty_notes', '')})"
            )

    lines.append("")
    lines.append("**Uncertainty:** Marina Bay safety-car density, tyre deg under night temps, and limited FP1→race transfer "
        "(daytime FP1). POST-ML scores are expected finishing positions, not win probabilities. "
        "Sprint/SQ pace can diverge from Sunday race trim.")
    return lines


def write_final_snapshot(
    path: Path,
    stage_e: pd.DataFrame,
    status_message: str,
    q_code: str,
    analytical_lines: list[str],
    verification: dict,
) -> None:
    ts = stage_e["prediction_timestamp_ist"].iloc[0]
    lines = [
        "# Singapore 2026 — Stage E (final pre-race)",
        "",
        f"**Generated (IST):** {ts}",
        f"**FastF1 GP qualifying code (from schedule):** `{q_code}`",
        f"**Schedule label:** {verification.get('schedule_session_label', 'n/a')}",
        f"**Verified event:** {verification.get('verified_event_name', 'n/a')} (R{verification.get('verified_round', 'n/a')})",
        f"**Session complete:** {verification.get('session_complete', False)}",
        f"**Retrieval:** {status_message}",
        "",
        "Grand Prix qualifying (`Q`) is distinct from Sprint Qualifying (`SQ`) and the Sprint (`S`).",
        "",
        "## Reasoning summary",
        "",
    ]
    lines.extend(analytical_lines)
    lines.append("")
    lines.append("## Sunday starting grid (GP qualifying / grid evidence)")
    lines.append("")
    if bool(stage_e.get("gp_qual_data_available", pd.Series([False])).iloc[0]):
        for _, r in stage_e.sort_values("gp_grid_position").head(10).iterrows():
            pen = ""
            if r.get("gp_has_documented_grid_penalty"):
                pen = f" [documented penalty +{int(r['gp_grid_penalty_positions'])}]"
            elif pd.notna(r.get("gp_grid_penalty_positions")) and int(r["gp_grid_penalty_positions"]) != 0:
                pen = f" [grid Δ vs quali {int(r['gp_grid_penalty_positions']):+d}]"
            lines.append(
                f"- Grid P{int(r['gp_grid_position'])} {r['driver']} — quali P{int(r['gp_qualifying_position'])}{pen}"
            )
    else:
        lines.append("- GP qualifying not available at generation time.")

    lines.append("")
    lines.append("## Predicted Grand Prix finishing order (POST-qualifying ML — not the grid)")
    lines.append("")
    lines.append("Primary sort: `rf_post_quali_rank` (Random Forest POST model). "
        "Grid baseline = `gp_grid_position`; quali baseline = `gp_qualifying_position`.")
    lines.append("")
    if bool(stage_e.get("gp_qual_data_available", pd.Series([False])).iloc[0]):
        for _, r in stage_e.sort_values("rf_post_quali_rank").head(10).iterrows():
            agree = r.get("model_agreement", "")
            lines.append(
                f"- Pred P{int(r['rf_post_quali_rank'])} {r['driver']} — grid P{int(r['gp_grid_position'])} | "
                f"quali P{int(r['gp_qualifying_position'])} | PRE-ML P{int(r['rf_pre_weekend_rank'])} | {agree}"
            )
    else:
        lines.append("- Predictions withheld until GP qualifying is complete.")
    path.write_text("\n".join(lines), encoding="utf-8")


def stage_e_merge_final(
    stage_d: pd.DataFrame,
    gp_quali: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Merge GP quali evidence and POST-ML race predictions onto frozen Stage D."""
    out = stage_d.copy()
    if gp_quali.empty:
        out["prediction_stage"] = "E_final_pre_race_unavailable"
        out["gp_qual_data_available"] = False
        out["prediction_timestamp_ist"] = now_ist().isoformat()
        return out

    gcols = [c for c in gp_quali.columns if c not in ("constructor",)]
    out = out.merge(gp_quali[gcols], on="driver", how="left", suffixes=("", "_gpdup"))

    pred_cols = [
        "driver",
        "rf_predicted_score",
        "gb_predicted_score",
        "xgb_predicted_score",
        "rf_rank",
        "gb_rank",
        "xgb_rank",
        "model_rank_min",
        "model_rank_max",
        "model_rank_median",
        "model_agreement",
    ]
    out = out.merge(predictions[pred_cols], on="driver", how="left")
    out = out.rename(
        columns={
            "rf_rank": "rf_post_quali_rank",
            "gb_rank": "gb_post_quali_rank",
            "xgb_rank": "xgb_post_quali_rank",
            "rf_predicted_score": "rf_post_quali_predicted_finish",
            "gb_predicted_score": "gb_post_quali_predicted_finish",
            "xgb_predicted_score": "xgb_post_quali_predicted_finish",
        }
    )
    out["qualifying_baseline_rank"] = out["gp_qualifying_position"]
    out["grid_baseline_rank"] = out["gp_grid_position"]
    out["prediction_stage"] = "E_final_pre_race"
    out["gp_qual_data_available"] = True
    out["prediction_timestamp_ist"] = now_ist().isoformat()
    out["input_data_cutoff"] = (
        str(out["input_data_cutoff"].iloc[0])
        + "; + Singapore 2026 GP qualifying & Sunday grid (POST-ML race prediction)"
    )
    return out


def run_stage_e() -> int:
    """Stage E only — reads frozen Stage D; writes final pre-race artifacts."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    stage_d = load_frozen_stage_d()

    gp_table, gp_status, gp_weather, q_code, verification = build_gp_qualifying_table()
    save_retrieval_status(
        PREDICTIONS_DIR.parent / STAGE_FILES["gp_qual_status"],
        gp_status,
        **verification,
    )

    if gp_table.empty:
        stage_e = stage_e_merge_final(stage_d, gp_table, pd.DataFrame())
        out_csv = PREDICTIONS_DIR / STAGE_FILES["final_csv"]
        stage_e.to_csv(out_csv, index=False)
        write_final_snapshot(
            PREDICTIONS_DIR.parent / STAGE_FILES["final_snapshot"],
            stage_e,
            gp_status.message,
            q_code,
            _analytical_final_summary(stage_d, stage_e),
            verification,
        )
        print("Stage E incomplete — GP qualifying unavailable.")
        print("Output:", out_csv)
        return 1

    gp_table.to_csv(PREDICTIONS_DIR.parent / STAGE_FILES["gp_qual_evidence"], index=False)
    if gp_weather:
        pd.DataFrame([gp_weather]).to_csv(
            PREDICTIONS_DIR.parent / STAGE_FILES["weather_context_gp_qual"],
            index=False,
        )

    modeling_path = PROCESSED_DIR / MODELING_THROUGH_R16
    if not modeling_path.exists():
        rebuild_modeling_through_r16()
    train_df = augment_post_dataset(pd.read_csv(modeling_path))

    feature_cols = list(POST_QUALIFYING_FEATURES)
    models = train_final_models(train_df, feature_cols)
    sg_features = build_singapore_post_feature_rows(gp_table)
    predictions = generate_predictions(sg_features, models, feature_cols)

    stage_e = stage_e_merge_final(stage_d, gp_table, predictions)
    out_csv = PREDICTIONS_DIR / STAGE_FILES["final_csv"]
    stage_e.to_csv(out_csv, index=False)

    build_evidence_matrix_stage_e(stage_e).to_csv(
        PREDICTIONS_DIR.parent / STAGE_FILES["evidence_matrix_stage_e"],
        index=False,
    )

    write_final_snapshot(
        PREDICTIONS_DIR.parent / STAGE_FILES["final_snapshot"],
        stage_e,
        gp_status.message,
        q_code,
        _analytical_final_summary(stage_d, stage_e),
        verification,
    )

    print("Stage E complete.")
    print("Output:", out_csv)
    print("GP qual success:", gp_status.success, "|", gp_status.message)
    return 0 if gp_status.success else 1


def stage_f_post_race_evaluation() -> None:
    raise NotImplementedError("Run after Sunday race — does not modify frozen snapshots")


def run_pipeline(stages: str) -> int:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    ext = ensure_2026_raw_extended()
    logger.info("Raw extension: %s", ext)

    generated_at = now_ist().isoformat()
    data_cutoff = f"{SINGAPORE_SEASON} R1–{SINGAPORE_ROUND - 1} race+qualifying; historical Singapore 2019+"

    write_context_doc(DOCS_DIR / "singapore_2026_context.md")
    write_methodology_doc(DOCS_DIR / "singapore_2026_methodology.md")

    modeling_path = PROCESSED_DIR / MODELING_THROUGH_R16
    if not modeling_path.exists():
        rebuild_modeling_through_r16()
    train_df = pd.read_csv(modeling_path)
    cmp_df = run_model_comparison(train_df)
    cmp_df.to_csv(PREDICTIONS_DIR.parent / STAGE_FILES["model_comparison"], index=False)

    stage_a = stage_a_pre_weekend(generated_at=generated_at, data_cutoff=data_cutoff)

    pre_csv = PREDICTIONS_DIR / STAGE_FILES["pre_weekend_csv"]
    stage_a.to_csv(pre_csv, index=False)
    pre_json = PREDICTIONS_DIR / STAGE_FILES["pre_weekend_json"]
    pre_json.write_text(stage_a.to_json(orient="records", indent=2), encoding="utf-8")

    write_pre_weekend_snapshot(
        PREDICTIONS_DIR.parent / STAGE_FILES["pre_weekend_snapshot"],
        stage_a,
    )

    build_evidence_matrix(stage_a).to_csv(
        PREDICTIONS_DIR.parent / STAGE_FILES["evidence_matrix"],
        index=False,
    )

    if stages in ("post-fp1", "all-available"):
        fp1_table, fp1_status, weather, conditions = build_fp1_analysis_table()
        save_retrieval_status(
            PREDICTIONS_DIR.parent / STAGE_FILES["fp1_status"],
            fp1_status,
        )
        if weather:
            pd.DataFrame([weather]).to_csv(
                PREDICTIONS_DIR.parent / STAGE_FILES["weather_context"],
                index=False,
            )
        stage_b = stage_b_post_fp1(stage_a, fp1_table)
        stage_b.to_csv(PREDICTIONS_DIR / STAGE_FILES["post_fp1_csv"], index=False)

        snap_lines = [
            "# Singapore 2026 — Stage B (post-FP1)",
            "",
            f"**Generated (IST):** {stage_b['prediction_timestamp_ist'].iloc[0]}",
            f"**FP1 retrieval success:** {fp1_status.success}",
            f"**FP1 message:** {fp1_status.message}",
            "",
        ]
        if fp1_status.success:
            snap_lines.append("FP1 timing merged as analytical evidence (not ML input).")
        else:
            snap_lines.append(
                "FP1 data unavailable in FastF1 at retrieval time; Stage A baseline unchanged for ML."
            )
        (PREDICTIONS_DIR.parent / STAGE_FILES["post_fp1_snapshot"]).write_text(
            "\n".join(snap_lines),
            encoding="utf-8",
        )

    print("Singapore pipeline stage(s) complete:", stages)
    print("Pre-weekend CSV:", pre_csv)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    parser = argparse.ArgumentParser(description="Singapore 2026 prediction stages")
    parser.add_argument(
        "--stage",
        choices=["pre-weekend", "post-fp1", "post-sq", "post-sprint", "final", "all-available"],
        default="all-available",
    )
    args = parser.parse_args(argv)
    if args.stage == "post-sq":
        return run_stage_c()
    if args.stage == "post-sprint":
        return run_stage_d()
    if args.stage == "final":
        return run_stage_e()
    return run_pipeline(args.stage)


if __name__ == "__main__":
    sys.exit(main())
