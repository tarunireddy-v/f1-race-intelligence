"""
Singapore 2026 pre-race audit: grid resolution, Ridge validation, experimental forecasts.

Does not modify frozen Stage A–E artifacts or Hungary outputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.predict_hungary_final import (
    augment_post_dataset,
    generate_predictions,
    train_final_models,
)
from src.predict_singapore import (
    MODELING_THROUGH_R16,
    PROCESSED_DIR,
    build_singapore_post_feature_rows,
    load_frozen_stage_d,
    stage_e_merge_final,
)
from src.singapore_config import IST, OUTPUT_DIR, PREDICTIONS_DIR, SINGAPORE_ROUND, SINGAPORE_SEASON, STAGE_FILES, now_ist
from src.train import (
    POST_QUALIFYING_FEATURES,
    TARGET,
    SplitSpec,
    baseline_mae,
    build_pipeline,
    evaluate_predictions,
    get_models,
    train_and_evaluate,
    verify_no_overlap,
)
from src.weekend_sessions import build_gp_qualifying_table

logger = logging.getLogger(__name__)

IST_TZ = ZoneInfo("Asia/Kolkata")

# FIA Doc 75 — Provisional Starting Grid (11 Oct 2026 03:00). No separate "Final Starting Grid" on FIA index at audit time.
FIA_DOC75_PROVISIONAL_GRID: dict[str, int] = {
    "VER": 1,
    "LEC": 2,
    "HAM": 3,
    "ANT": 4,
    "NOR": 5,
    "PIA": 6,
    "GAS": 7,
    "LAW": 8,
    "LIN": 9,
    "HUL": 10,
    "BEA": 11,
    "OCO": 12,
    "SAI": 13,
    "BOR": 14,
    "COL": 15,
    "ALO": 16,
    "ALB": 17,
    "STR": 18,
    "PER": 19,
    "BOT": 20,
    "RUS": 21,
    "HAD": 22,
}

FIA_DOC74_QUALI: dict[str, int] = {
    "VER": 1,
    "LEC": 2,
    "HAM": 3,
    "ANT": 4,
    "NOR": 5,
    "RUS": 6,
    "PIA": 7,
    "GAS": 8,
    "LAW": 9,
    "LIN": 10,
    "HUL": 11,
    "BEA": 12,
    "OCO": 13,
    "SAI": 14,
    "BOR": 15,
    "COL": 16,
    "ALO": 17,
    "ALB": 18,
    "STR": 19,
    "PER": 20,
    "BOT": 21,
}

OUTPUT_PATHS = {
    "grid_resolution": OUTPUT_DIR / "singapore_2026_sunday_grid_resolution.csv",
    "ridge_validation": OUTPUT_DIR / "singapore_2026_ridge_validation_comparison.csv",
    "post_model_validation": OUTPUT_DIR / "singapore_2026_post_model_validation_comparison.csv",
    "ridge_report": OUTPUT_DIR / "singapore_2026_ridge_validation_report.md",
    "audit_md": OUTPUT_DIR / "singapore_2026_pre_race_audit.md",
    "weather_context": OUTPUT_DIR / "singapore_2026_race_weather_context.json",
    "pred_grid_corrected": PREDICTIONS_DIR / "singapore_2026_final_prediction_experimental_grid_corrected.csv",
    "snap_grid_corrected": OUTPUT_DIR / "singapore_2026_final_snapshot_experimental_grid_corrected.md",
    "pred_ridge": PREDICTIONS_DIR / "singapore_2026_final_prediction_experimental_ridge.csv",
    "snap_ridge": OUTPUT_DIR / "singapore_2026_final_snapshot_experimental_ridge.md",
}

STAGE_E_SHA256 = "90808895c601a30895fd38117f2db4736afb389b00acafd183dfcf60431007ae"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _singapore_extended_split(df: pd.DataFrame) -> SplitSpec:
    val_rounds = sorted(
        int(r) for r in df.loc[(df["season"] == SINGAPORE_SEASON) & (df["round"] >= 14), "round"].unique()
    )
    if len(val_rounds) < 2:
        raise ValueError("Insufficient 2026 validation rounds in modeling matrix")
    val_rounds_used = val_rounds[-3:]
    first_val = val_rounds[0]
    val_mask = (df["season"] == SINGAPORE_SEASON) & (df["round"].isin(val_rounds_used))
    train_mask = ~val_mask & (
        df["season"].isin([2024, 2025])
        | ((df["season"] == SINGAPORE_SEASON) & (df["round"] < first_val))
    )
    return SplitSpec("singapore_extended_val", train_mask, val_mask)


def _inner_train_masks(train_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Inner chronological split on training rows only: hold out 2026 R11–R13 for Ridge alpha."""
    inner_val = (train_df["season"] == SINGAPORE_SEASON) & (train_df["round"].isin([11, 12, 13]))
    inner_train = ~inner_val
    return inner_train, inner_val


def build_ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def tune_ridge_alpha(train_df: pd.DataFrame, feature_cols: list[str]) -> tuple[float, pd.DataFrame]:
    inner_train, inner_val = _inner_train_masks(train_df)
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    rows = []
    best_alpha = 1.0
    best_mae = float("inf")
    for alpha in alphas:
        pipe = build_ridge_pipeline(alpha)
        tr = train_df.loc[inner_train]
        va = train_df.loc[inner_val]
        if va.empty or tr.empty:
            continue
        pipe.fit(tr[feature_cols], tr[TARGET])
        pred = pipe.predict(va[feature_cols])
        mae = mean_absolute_error(va[TARGET], pred)
        rows.append({"alpha": alpha, "inner_val_mae": mae, "inner_val_rows": len(va)})
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
    return best_alpha, pd.DataFrame(rows)


def spearman_rank_corr(y_true: pd.Series, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float("nan")
    r, _ = spearmanr(y_true, y_pred)
    return float(r) if r is not None else float("nan")


def run_post_model_validation(train_df: pd.DataFrame, ridge_alpha: float) -> pd.DataFrame:
    split = _singapore_extended_split(train_df)
    verify_no_overlap(train_df, split.train_mask, split.val_mask)
    val_df = train_df.loc[split.val_mask]
    rows = []

    for model_name, model in get_models().items():
        pipe = build_pipeline(model_name, model)
        result, fitted, _ = train_and_evaluate(
            train_df, split, POST_QUALIFYING_FEATURES, "POST", model_name, pipe
        )
        y_pred = fitted.predict(val_df[POST_QUALIFYING_FEATURES])
        sp = spearman_rank_corr(val_df[TARGET], y_pred)
        rows.append(
            {
                "split": split.name,
                "model": model_name,
                "mae": result.mae,
                "rmse": result.rmse,
                "spearman_rank_corr": sp,
                "winner_accuracy": result.winner_accuracy,
                "podium_hit_rate": result.podium_hit_rate,
                "train_rows": result.train_rows,
                "val_rows": result.val_rows,
            }
        )

    ridge_pipe = build_ridge_pipeline(ridge_alpha)
    result, fitted, _ = train_and_evaluate(
        train_df, split, POST_QUALIFYING_FEATURES, "POST", "Ridge", ridge_pipe
    )
    y_pred = fitted.predict(val_df[POST_QUALIFYING_FEATURES])
    sp = spearman_rank_corr(val_df[TARGET], y_pred)
    rows.append(
        {
            "split": split.name,
            "model": "Ridge",
            "mae": result.mae,
            "rmse": result.rmse,
            "spearman_rank_corr": sp,
            "winner_accuracy": result.winner_accuracy,
            "podium_hit_rate": result.podium_hit_rate,
            "train_rows": result.train_rows,
            "val_rows": result.val_rows,
            "ridge_alpha": ridge_alpha,
        }
    )

    rows.append(
        {
            "split": split.name,
            "model": "qualifying_position_baseline",
            "mae": baseline_mae(val_df, "qualifying_position", "quali"),
            "rmse": float("nan"),
            "spearman_rank_corr": float("nan"),
            "val_rows": len(val_df),
        }
    )
    rows.append(
        {
            "split": split.name,
            "model": "grid_position_baseline",
            "mae": baseline_mae(val_df, "grid_position", "grid"),
            "rmse": float("nan"),
            "spearman_rank_corr": float("nan"),
            "val_rows": len(val_df),
        }
    )
    return pd.DataFrame(rows)


def build_grid_resolution(gp_table: pd.DataFrame, retrieved_at: str) -> pd.DataFrame:
    rows = []
    drivers = sorted(set(gp_table["driver"]) | set(FIA_DOC75_PROVISIONAL_GRID))
    quali_map = gp_table.set_index("driver") if not gp_table.empty else pd.DataFrame()
    for driver in drivers:
        q_fia = FIA_DOC74_QUALI.get(driver)
        q_gp = None
        gap = None
        if driver in quali_map.index:
            q_gp = quali_map.loc[driver].get("gp_qualifying_position")
            gap = quali_map.loc[driver].get("gp_quali_gap_to_pole_s")
        quali_pos = int(q_gp) if pd.notna(q_gp) else q_fia
        grid = FIA_DOC75_PROVISIONAL_GRID.get(driver)
        pen_notes = ""
        has_pen = False
        grid_pen = None
        if driver == "RUS":
            pen_notes = (
                "FIA Doc 20: 40-place PU penalty; grid per Art. B2.5.4(b)(iv) — Doc 75 P21"
            )
            has_pen = True
            if quali_pos is not None and grid is not None:
                grid_pen = grid - quali_pos
        elif driver == "HAD":
            pen_notes = "FIA Doc 71: permitted to start; no quali time (Doc 74 not classified)"
            quali_pos = np.nan
        status = "provisional_official_fia_doc75"
        if grid is None:
            status = "unresolved"
        rows.append(
            {
                "driver": driver,
                "gp_qualifying_position": quali_pos,
                "gp_quali_gap_to_pole_s": gap,
                "verified_sunday_grid_position": grid,
                "gp_grid_penalty_positions": grid_pen,
                "gp_has_documented_grid_penalty": has_pen,
                "penalty_or_grid_notes": pen_notes,
                "grid_source_document": "FIA SGP Doc 75 Provisional Starting Grid (2026-10-11 03:00)",
                "grid_source_status": status,
                "quali_source_document": "FIA Doc 74 Final Qualifying Classification; FastF1 Q session",
                "retrieved_at_ist": retrieved_at,
            }
        )
    return pd.DataFrame(rows)


def apply_verified_grid_to_gp_table(gp_table: pd.DataFrame, grid_res: pd.DataFrame) -> pd.DataFrame:
    out = gp_table.copy()
    gmap = grid_res.set_index("driver")["verified_sunday_grid_position"]
    for idx, row in out.iterrows():
        d = row["driver"]
        if d not in gmap or pd.isna(gmap[d]):
            continue
        g = int(gmap[d])
        out.at[idx, "gp_grid_position"] = g
        out.at[idx, "gp_grid_source"] = "fia_doc75_provisional_starting_grid"
        q = row.get("gp_qualifying_position")
        if pd.notna(q):
            out.at[idx, "gp_grid_penalty_positions"] = int(g) - int(q)
        if d == "RUS":
            out.at[idx, "gp_has_documented_grid_penalty"] = True
            out.at[idx, "gp_quali_penalty_notes"] = grid_res.loc[
                grid_res.driver == "RUS", "penalty_or_grid_notes"
            ].iloc[0]
    return out


def _order_table(df: pd.DataFrame, rank_col: str) -> str:
    lines = []
    for _, r in df.sort_values(rank_col).iterrows():
        lines.append(f"P{int(r[rank_col])} {r['driver']}")
    return "\n".join(lines)


def write_experimental_snapshot(
    path: Path,
    pred_df: pd.DataFrame,
    title: str,
    extra_notes: list[str],
    *,
    rank_col: str | None = None,
    model_label: str = "RF",
) -> None:
    ts = now_ist().isoformat()
    lines = [f"# {title}", "", f"**Generated (IST):** {ts}", ""]
    lines.extend(extra_notes)
    lines.append("")
    if rank_col is None:
        rank_col = (
            "ridge_post_quali_rank"
            if "ridge_post_quali_rank" in pred_df.columns
            and "Ridge" in title
            and pred_df["ridge_post_quali_rank"].notna().any()
            else "rf_post_quali_rank"
        )
    if rank_col not in pred_df.columns:
        raise KeyError(f"Snapshot rank column missing: {rank_col}")
    lines.append(f"## Predicted finishing order ({model_label})")
    lines.append("")
    lines.append(f"Primary sort: `{rank_col}`.")
    lines.append("")
    for _, r in pred_df.sort_values(rank_col, kind="mergesort").iterrows():
        g = r.get("gp_grid_position", "n/a")
        q = r.get("gp_qualifying_position", "n/a")
        lines.append(f"- P{int(r[rank_col])} {r['driver']} — grid {g} | quali {q}")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit() -> int:
    retrieved = now_ist().isoformat()
    for p in OUTPUT_PATHS.values():
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {p}")

    frozen_e = PREDICTIONS_DIR / STAGE_FILES["final_csv"]
    if not frozen_e.exists():
        raise FileNotFoundError("Frozen Stage E prediction missing")
    e_hash = _sha256(frozen_e)
    if e_hash != STAGE_E_SHA256:
        logger.warning(
            "Stage E SHA256 %s differs from audit baseline %s — document in report",
            e_hash,
            STAGE_E_SHA256,
        )

    gp_table, gp_status, gp_weather, _, _ = build_gp_qualifying_table()
    grid_res = build_grid_resolution(gp_table, retrieved)
    grid_res.to_csv(OUTPUT_PATHS["grid_resolution"], index=False)

    gp_corrected = apply_verified_grid_to_gp_table(gp_table, grid_res)

    modeling_path = PROCESSED_DIR / MODELING_THROUGH_R16
    train_df = augment_post_dataset(pd.read_csv(modeling_path))
    ridge_alpha, alpha_log = tune_ridge_alpha(train_df.loc[_singapore_extended_split(train_df).train_mask], POST_QUALIFYING_FEATURES)
    alpha_log.to_csv(OUTPUT_PATHS["ridge_validation"], index=False)

    val_cmp = run_post_model_validation(train_df, ridge_alpha)
    val_cmp.to_csv(OUTPUT_PATHS["post_model_validation"], index=False)

    feature_cols = list(POST_QUALIFYING_FEATURES)
    models = train_final_models(train_df, feature_cols)
    sg_corr = build_singapore_post_feature_rows(gp_corrected)
    preds_corr = generate_predictions(sg_corr, models, feature_cols)
    stage_d = load_frozen_stage_d()
    exp_corr = stage_e_merge_final(stage_d, gp_corrected, preds_corr)
    exp_corr["prediction_stage"] = "experimental_grid_corrected_post_v1"
    exp_corr["prediction_timestamp_ist"] = retrieved
    exp_corr.to_csv(OUTPUT_PATHS["pred_grid_corrected"], index=False)

    ridge_pipe = build_ridge_pipeline(ridge_alpha)
    ridge_pipe.fit(train_df[feature_cols], train_df[TARGET])
    sg_corr["ridge_predicted_finish"] = ridge_pipe.predict(sg_corr[feature_cols])
    sg_corr["ridge_post_quali_rank"] = sg_corr["ridge_predicted_finish"].rank(method="min").astype(int)
    ridge_out = stage_e_merge_final(stage_d, gp_corrected, preds_corr)
    ridge_merge = sg_corr[["driver", "ridge_predicted_finish", "ridge_post_quali_rank"]]
    ridge_out = ridge_out.merge(ridge_merge, on="driver", how="left")
    ridge_out["prediction_stage"] = "experimental_ridge_post_v1"
    ridge_out["prediction_timestamp_ist"] = retrieved
    ridge_out.to_csv(OUTPUT_PATHS["pred_ridge"], index=False)

    frozen_df = pd.read_csv(frozen_e)
    stage_e_order = frozen_df.sort_values("rf_post_quali_rank")["driver"].tolist()
    corr_order = exp_corr.sort_values("rf_post_quali_rank")["driver"].tolist()
    ridge_order = ridge_out.sort_values("ridge_post_quali_rank")["driver"].tolist()

    weather = {
        "retrieved_at_ist": retrieved,
        "category": "race_start_forecast_context_not_ml_features",
        "sources": [
            {
                "name": "Formula 1 official weather article",
                "url": "https://www.formula1.com/en/latest/article/what-is-the-weather-forecast-for-the-2026-singapore-grand-prix.4DFAwBl2esxhHp1mCv3VSF",
                "issue_time_utc": "2026-10-07T09:54:35Z",
            },
            {
                "name": "f1weather.com (Open-Meteo/OpenWeatherMap)",
                "url": "https://f1weather.com/",
                "updated_utc": "2026-10-11T07:05:00Z",
                "race_window_notes": "~39% rain probability; air ~27.3C; humidity ~86%; wind NE ~2.2 km/h",
            },
        ],
        "fastf1_gp_quali_observed": gp_weather if gp_weather else {},
        "uncertainty": (
            "Third-party forecasts disagree on rain timing (afternoon showers vs ~40% at race start). "
            "Marina Bay is humid; convection can develop quickly. Not used in ML features."
        ),
    }
    OUTPUT_PATHS["weather_context"].write_text(json.dumps(weather, indent=2), encoding="utf-8")

    ridge_report = [
        "# Ridge validation report (Singapore extended val)",
        "",
        f"**Generated (IST):** {retrieved}",
        "",
        f"**Selected Ridge alpha (inner val 2026 R11–R13):** {ridge_alpha}",
        "",
        "## Alpha search (inner validation)",
        "",
        "```\n" + alpha_log.to_string(index=False) + "\n```",
        "",
        "## Model comparison (train < 2026 R14, val 2026 R14–R16)",
        "",
        "```\n" + val_cmp.to_string(index=False) + "\n```",
        "",
        "## Limitations",
        "- POST features only; no weekend session or weather features.",
        "- Ridge pipeline uses median imputer + StandardScaler (RF/GB/XGB use median imputer only).",
        "- Singapore R17 excluded from training and validation.",
        "- Provisional grid (Doc 75); no FIA Final Starting Grid PDF listed at audit time.",
    ]
    OUTPUT_PATHS["ridge_report"].write_text("\n".join(ridge_report), encoding="utf-8")

    rf_row = val_cmp[val_cmp["model"] == "RandomForestRegressor"].iloc[0]
    ridge_row = val_cmp[val_cmp["model"] == "Ridge"].iloc[0]
    quali_b = val_cmp[val_cmp["model"] == "qualifying_position_baseline"].iloc[0]["mae"]

    audit = [
        "# Singapore 2026 pre-race audit",
        "",
        f"**Generated (IST):** {retrieved}",
        "",
        "## Phase 0 — Stage E provenance",
        "",
        f"- `singapore_2026_final_prediction.csv` SHA256: `{e_hash}`",
        f"- Audit baseline SHA256 (2026-10-10 Stage E run): `{STAGE_E_SHA256}`",
        "- File is **untracked in git** (Singapore work never committed); no git recovery path.",
        "- **No experimental file** overwrote Stage E; no `*experimental*` files existed before this audit.",
        "- Conclusion: current Stage E is the **original local frozen run** unless SHA differs (then note above).",
        "",
        "## Phase 1 — Sunday grid",
        "",
        "- **Source:** FIA **Doc 75 — Provisional Starting Grid** (11 Oct 2026, 03:00).",
        "- **Final Starting Grid** PDF not listed on FIA event page at audit time (Doc 76 = Pre-Race Procedure).",
        f"- **George Russell:** quali **P6** (Doc 74); grid **P21** (Doc 75); FIA Doc 20 PU penalty (40 places, B2.5.4 allocation).",
        "- Stage E erroneously used **grid = quali** when FastF1 `GridPosition` was null.",
        f"- Grid resolution: `{OUTPUT_PATHS['grid_resolution'].name}`",
        "",
        "## Phase 2 — Weather",
        "",
        f"- Context JSON: `{OUTPUT_PATHS['weather_context'].name}` (forecast only; not ML input).",
        "",
        "## Phase 3 — Historical Singapore pole conversion",
        "",
        "| Year | Pole | Winner | Pole won? |",
        "|------|------|--------|-----------|",
        "| 2024 | NOR | NOR | Yes |",
        "| 2023 | SAI | SAI | Yes |",
        "| 2022 | PER | VER | No |",
        "| 2019 | LEC | VET | No |",
        "| 2018 | HAM | HAM | Yes |",
        "",
        "**3 of 5** included races (2020–21 cancelled). Small sample; not used to adjust ranks.",
        "",
        "## Phase 4 — Stage E model",
        "",
        "- **Primary order:** `rf_post_quali_rank` from **RandomForestRegressor** POST pipeline.",
        "- **Alternatives in CSV:** GB/XGB ranks and `model_agreement`; **no ensemble average** for headline order.",
        "- **Features:** exactly `POST_QUALIFYING_FEATURES` (11 columns).",
        "- Sectors/teammate gaps/championship/circuit/FP1/SQ/Sprint **not** in POST model.",
        "",
        "## Phase 5 — Ridge",
        "",
        f"- Selected alpha: **{ridge_alpha}**",
        f"- RF MAE: **{rf_row['mae']:.3f}**; Ridge MAE: **{ridge_row['mae']:.3f}**; Quali baseline: **{quali_b:.3f}**",
        f"- Details: `{OUTPUT_PATHS['ridge_report'].name}`",
        "",
        "**Primary model for frozen Stage E remains RF** (existing pipeline). Ridge is experimental only.",
        "",
        "## Phase 6 — Order comparison",
        "",
        f"**Frozen Stage E (RF):** {' → '.join(stage_e_order[:10])} …",
        f"**Grid-corrected RF:** {' → '.join(corr_order[:10])} …",
        f"**Ridge (corrected grid):** {' → '.join(ridge_order[:10])} …",
        "",
        "Position changes vs Stage E are driven mainly by **grid_position** feature updates (midfield +1 slot after RUS penalty); RF top order may be unchanged.",
        "",
        "## Outputs",
        "",
    ]
    for k, p in OUTPUT_PATHS.items():
        audit.append(f"- `{p}`")
    OUTPUT_PATHS["audit_md"].write_text("\n".join(audit), encoding="utf-8")

    write_experimental_snapshot(
        OUTPUT_PATHS["snap_grid_corrected"],
        exp_corr,
        "Singapore 2026 — experimental grid-corrected POST (RF/GB/XGB)",
        [
            "FIA Doc 75 provisional Sunday grid applied to `grid_position` / `gp_grid_position`.",
            "Qualifying positions and `gp_quali_gap_to_pole_s` unchanged.",
        ],
    )
    write_experimental_snapshot(
        OUTPUT_PATHS["snap_ridge"],
        ridge_out,
        "Singapore 2026 — experimental Ridge POST (grid-corrected)",
        [f"Ridge alpha={ridge_alpha} from inner chronological validation."],
        rank_col="ridge_post_quali_rank",
        model_label="Ridge Regression",
    )

    print("Pre-race audit complete.")
    print("Stage E SHA256:", e_hash, "(unchanged:", e_hash == STAGE_E_SHA256, ")")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    return run_audit()


if __name__ == "__main__":
    raise SystemExit(main())
