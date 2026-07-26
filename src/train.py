"""
V1 baseline ML experiment — PRE vs POST-qualifying model comparison.

Temporal splits only (no random train_test_split):
  Primary:   train 2024 + 2025 + 2026 R1–7  |  val 2026 R8–10
  Secondary: train 2024                      |  val 2025

Preprocessing (median imputation) is fit on training rows only via sklearn Pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.config import MODELING_DATASET_FILE, PROCESSED_DATA_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_FILE = "model_results.csv"

TARGET = "finishing_position"

IDENTIFIER_COLUMNS = [
    "season",
    "round",
    "race_name",
    "circuit",
    "driver",
    "constructor",
]

# Excluded from model inputs (identifiers + target + unused high-missing features)
EXCLUDED_FROM_FEATURES = IDENTIFIER_COLUMNS + [
    TARGET,
    "driver_previous_circuit_finish",
    "driver_circuit_avg_finish",
]

PRE_QUALIFYING_FEATURES = [
    "avg_finish_last_3",
    "avg_finish_last_5",
    "avg_quali_last_3",
    "points_last_5",
    "dnf_rate_last_5",
    "constructor_avg_finish_last_3",
    "constructor_points_last_5",
    "is_2026",
]

POST_QUALIFYING_EXTRA_FEATURES = [
    "grid_position",
    "qualifying_position",
    "gap_to_pole_s",
]

POST_QUALIFYING_FEATURES = PRE_QUALIFYING_FEATURES + POST_QUALIFYING_EXTRA_FEATURES

RANDOM_STATE = 42

PRIMARY_VAL_SEASON = 2026
PRIMARY_VAL_ROUNDS = (8, 9, 10)

SECONDARY_VAL_SEASON = 2025


@dataclass
class SplitSpec:
    name: str
    train_mask: pd.Series
    val_mask: pd.Series


@dataclass
class EvalResult:
    split: str
    configuration: str
    model: str
    mae: float
    rmse: float
    winner_accuracy: float
    podium_hit_rate: float
    train_rows: int
    val_rows: int


def load_modeling_dataset(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or (PROCESSED_DATA_DIR / MODELING_DATASET_FILE)
    df = pd.read_csv(csv_path)
    df["season"] = df["season"].astype(int)
    df["round"] = df["round"].astype(int)
    return df


def primary_train_mask(df: pd.DataFrame) -> pd.Series:
    """2024, 2025, and 2026 rounds 1–7."""
    return (
        df["season"].isin([2024, 2025])
        | ((df["season"] == PRIMARY_VAL_SEASON) & (df["round"] <= 7))
    )


def primary_val_mask(df: pd.DataFrame) -> pd.Series:
    """2026 rounds 8–10 (Austrian, British, Belgian)."""
    return (df["season"] == PRIMARY_VAL_SEASON) & (df["round"].isin(PRIMARY_VAL_ROUNDS))


def secondary_train_mask(df: pd.DataFrame) -> pd.Series:
    return df["season"] == 2024


def secondary_val_mask(df: pd.DataFrame) -> pd.Series:
    return df["season"] == SECONDARY_VAL_SEASON


def verify_no_overlap(df: pd.DataFrame, train_mask: pd.Series, val_mask: pd.Series) -> None:
    """Ensure no (season, round) appears in both train and validation."""
    train_events = set(zip(df.loc[train_mask, "season"], df.loc[train_mask, "round"]))
    val_events = set(zip(df.loc[val_mask, "season"], df.loc[val_mask, "round"]))
    overlap = train_events & val_events
    if overlap:
        raise ValueError(f"Train/validation event overlap detected: {overlap}")


def get_models() -> dict[str, object]:
    return {
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
        ),
        "RandomForestRegressor": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            n_jobs=-1,
        ),
        "XGBRegressor": XGBRegressor(
            random_state=RANDOM_STATE,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            objective="reg:squarederror",
            n_jobs=-1,
            verbosity=0,
        ),
    }


def build_pipeline(model_name: str, model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def _race_groups(val_df: pd.DataFrame) -> list[tuple[int, int]]:
    return [
        (int(s), int(r))
        for s, r in val_df[["season", "round"]].drop_duplicates().sort_values(["season", "round"]).itertuples(index=False, name=None)
    ]


def winner_accuracy(y_true: pd.Series, y_pred: pd.Series, groups: pd.DataFrame) -> float:
    """
    Per race: did the driver with the lowest predicted finish actually win?
    Returns mean across races.
    """
    hits = []
    for season, rnd in _race_groups(groups):
        mask = (groups["season"] == season) & (groups["round"] == rnd)
        yt = y_true[mask]
        yp = y_pred[mask]
        if yt.empty:
            continue
        pred_winner_idx = yp.idxmin()
        actual_winner_idx = yt.idxmin()
        hits.append(pred_winner_idx == actual_winner_idx)
    return float(np.mean(hits)) if hits else float("nan")


def podium_hit_rate(y_true: pd.Series, y_pred: pd.Series, groups: pd.DataFrame) -> float:
    """
    Per race: of the 3 drivers with lowest predicted finish, how many finished
    in the actual top 3? Returns mean hit count / 3 across races.
    """
    rates = []
    for season, rnd in _race_groups(groups):
        mask = (groups["season"] == season) & (groups["round"] == rnd)
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) < 3:
            continue
        pred_top3 = set(yp.nsmallest(3).index)
        actual_top3 = set(yt.nsmallest(3).index)
        rates.append(len(pred_top3 & actual_top3) / 3.0)
    return float(np.mean(rates)) if rates else float("nan")


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    val_meta: pd.DataFrame,
) -> tuple[float, float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    y_pred_series = pd.Series(y_pred, index=y_true.index)
    wacc = winner_accuracy(y_true, y_pred_series, val_meta)
    podium = podium_hit_rate(y_true, y_pred_series, val_meta)
    return mae, rmse, wacc, podium


def train_and_evaluate(
    df: pd.DataFrame,
    split: SplitSpec,
    feature_cols: list[str],
    configuration: str,
    model_name: str,
    pipeline: Pipeline,
) -> tuple[EvalResult, Pipeline, pd.DataFrame]:
    train_df = df.loc[split.train_mask]
    val_df = df.loc[split.val_mask]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    mae, rmse, wacc, podium = evaluate_predictions(y_val, y_pred, val_df)

    result = EvalResult(
        split=split.name,
        configuration=configuration,
        model=model_name,
        mae=mae,
        rmse=rmse,
        winner_accuracy=wacc,
        podium_hit_rate=podium,
        train_rows=len(train_df),
        val_rows=len(val_df),
    )
    val_out = val_df[IDENTIFIER_COLUMNS + [TARGET]].copy()
    val_out["predicted_finishing_position"] = y_pred
    return result, pipeline, val_out


def baseline_mae(
    val_df: pd.DataFrame,
    column: str,
    label: str,
) -> float:
    """MAE when predicting finishing position directly from a baseline column."""
    valid = val_df[[column, TARGET]].dropna()
    if valid.empty:
        return float("nan")
    return mean_absolute_error(valid[TARGET], valid[column])


def extract_feature_importance(pipeline: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "feature_importance"):
        imp = model.feature_importance()
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": feature_cols, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def print_results_table(results: list[EvalResult]) -> None:
    rows = [
        {
            "Configuration": r.configuration,
            "Model": r.model,
            "Split": r.split,
            "MAE": round(r.mae, 3),
            "RMSE": round(r.rmse, 3),
            "Winner Accuracy": round(r.winner_accuracy, 3),
            "Podium Hit Rate": round(r.podium_hit_rate, 3),
        }
        for r in results
    ]
    table = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)
    print(table.to_string(index=False))


def print_top5_sanity(predictions: pd.DataFrame, season: int, rnd: int, race_label: str) -> None:
    """Print predicted vs actual top 5 for one validation race."""
    race = predictions[
        (predictions["season"] == season) & (predictions["round"] == rnd)
    ].copy()

    pred_top5 = (
        race.nsmallest(5, "predicted_finishing_position")[["driver", "predicted_finishing_position"]]
        .reset_index(drop=True)
    )
    actual_top5 = (
        race.nsmallest(5, TARGET)[["driver", TARGET]]
        .reset_index(drop=True)
    )

    print(f"\n--- {race_label} ---")
    print("Predicted Top 5:")
    for i, row in pred_top5.iterrows():
        print(f"  {i+1}. {row['driver']} (pred {row['predicted_finishing_position']:.2f})")
    print("Actual Top 5:")
    for i, row in actual_top5.iterrows():
        print(f"  {i+1}. {row['driver']} (finish {int(row[TARGET])})")


def save_results(results: list[EvalResult], baselines: list[dict], path: Path | None = None) -> Path:
    out_path = (path or OUTPUT_DIR) / RESULTS_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "split": r.split,
            "configuration": r.configuration,
            "model": r.model,
            "mae": r.mae,
            "rmse": r.rmse,
            "winner_accuracy": r.winner_accuracy,
            "podium_hit_rate": r.podium_hit_rate,
            "train_rows": r.train_rows,
            "val_rows": r.val_rows,
        }
        for r in results
    ]
    rows.extend(baselines)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def run_experiment() -> dict:
    df = load_modeling_dataset()

    splits = [
        SplitSpec(
            name="primary_2026_r8_r10",
            train_mask=primary_train_mask(df),
            val_mask=primary_val_mask(df),
        ),
        SplitSpec(
            name="secondary_2024_train_2025_val",
            train_mask=secondary_train_mask(df),
            val_mask=secondary_val_mask(df),
        ),
    ]

    for split in splits:
        verify_no_overlap(df, split.train_mask, split.val_mask)
        train_events = df.loc[split.train_mask, ["season", "round", "race_name"]].drop_duplicates()
        val_events = df.loc[split.val_mask, ["season", "round", "race_name"]].drop_duplicates()
        logger.info(
            "%s — train: %d rows (%d events), val: %d rows (%d events)",
            split.name,
            split.train_mask.sum(),
            len(train_events),
            split.val_mask.sum(),
            len(val_events),
        )

    configurations = {
        "PRE-QUALIFYING": PRE_QUALIFYING_FEATURES,
        "POST-QUALIFYING": POST_QUALIFYING_FEATURES,
    }

    all_results: list[EvalResult] = []
    fitted_primary_post: dict[str, tuple[Pipeline, pd.DataFrame]] = {}

    for split in splits:
        for config_name, feature_cols in configurations.items():
            for model_name, model in get_models().items():
                pipeline = build_pipeline(model_name, model)
                result, fitted, val_preds = train_and_evaluate(
                    df, split, feature_cols, config_name, model_name, pipeline
                )
                all_results.append(result)
                if split.name == "primary_2026_r8_r10" and config_name == "POST-QUALIFYING":
                    fitted_primary_post[model_name] = (fitted, val_preds)

    # Baselines on primary POST-QUALIFYING validation only
    primary_val = df.loc[primary_val_mask(df)]
    baseline_rows = []
    grid_mae = baseline_mae(primary_val, "grid_position", "Grid baseline")
    quali_mae = baseline_mae(primary_val, "qualifying_position", "Qualifying baseline")
    for label, mae in [("Baseline: grid_position", grid_mae), ("Baseline: qualifying_position", quali_mae)]:
        baseline_rows.append(
            {
                "split": "primary_2026_r8_r10",
                "configuration": "POST-QUALIFYING",
                "model": label,
                "mae": mae,
                "rmse": np.nan,
                "winner_accuracy": np.nan,
                "podium_hit_rate": np.nan,
                "train_rows": np.nan,
                "val_rows": len(primary_val),
            }
        )

    save_results(all_results, baseline_rows)

    primary_results = [r for r in all_results if r.split == "primary_2026_r8_r10"]
    pre_results = [r for r in primary_results if r.configuration == "PRE-QUALIFYING"]
    post_results = [r for r in primary_results if r.configuration == "POST-QUALIFYING"]

    best_pre = min(pre_results, key=lambda r: r.mae)
    best_post = min(post_results, key=lambda r: r.mae)

    print_results_table(all_results)

    print("\n" + "=" * 90)
    print("PRIMARY HOLDOUT BASELINES (POST-QUALIFYING validation races)")
    print("=" * 90)
    print(f"  Baseline A — grid_position MAE:        {grid_mae:.3f}")
    print(f"  Baseline B — qualifying_position MAE:  {quali_mae:.3f}")

    print("\n" + "=" * 90)
    print("BEST MODELS (primary split, selected by MAE)")
    print("=" * 90)
    print(
        f"  BEST PRE-QUALIFYING:  {best_pre.model}  "
        f"MAE={best_pre.mae:.3f}  RMSE={best_pre.rmse:.3f}  "
        f"WinnerAcc={best_pre.winner_accuracy:.3f}  Podium={best_pre.podium_hit_rate:.3f}"
    )
    print(
        f"  BEST POST-QUALIFYING: {best_post.model}  "
        f"MAE={best_post.mae:.3f}  RMSE={best_post.rmse:.3f}  "
        f"WinnerAcc={best_post.winner_accuracy:.3f}  Podium={best_post.podium_hit_rate:.3f}"
    )

    best_post_pipeline, best_post_preds = fitted_primary_post[best_post.model]
    importance = extract_feature_importance(best_post_pipeline, POST_QUALIFYING_FEATURES)

    print("\n" + "=" * 90)
    print(f"FEATURE IMPORTANCE — best POST-QUALIFYING ({best_post.model})")
    print("=" * 90)
    if importance.empty:
        print("  (not available for this model)")
    else:
        print(importance.to_string(index=False))

    print("\n" + "=" * 90)
    print("SANITY CHECK — Predicted vs Actual Top 5 (best POST-QUALIFYING)")
    print("=" * 90)
    for rnd, label in [(8, "2026 Austrian GP"), (9, "2026 British GP"), (10, "2026 Belgian GP")]:
        race_preds = best_post_preds[
            (best_post_preds["season"] == 2026) & (best_post_preds["round"] == rnd)
        ]
        print_top5_sanity(race_preds, 2026, rnd, label)

    secondary_results = [r for r in all_results if r.split == "secondary_2024_train_2025_val"]
    print("\n" + "=" * 90)
    print("SECONDARY HISTORICAL CHECK (train 2024 → val 2025)")
    print("=" * 90)
    sec_table = pd.DataFrame(
        [
            {
                "Configuration": r.configuration,
                "Model": r.model,
                "MAE": round(r.mae, 3),
                "RMSE": round(r.rmse, 3),
                "Winner Accuracy": round(r.winner_accuracy, 3),
                "Podium Hit Rate": round(r.podium_hit_rate, 3),
            }
            for r in secondary_results
        ]
    )
    print(sec_table.to_string(index=False))

    return {
        "df": df,
        "results": all_results,
        "best_pre": best_pre,
        "best_post": best_post,
        "grid_mae": grid_mae,
        "quali_mae": quali_mae,
        "importance": importance,
        "best_post_preds": best_post_preds,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    print("=" * 90)
    print("V1 BASELINE ML EXPERIMENT")
    print("=" * 90)
    print(f"\nTarget: {TARGET}")
    print(f"PRE-QUALIFYING features ({len(PRE_QUALIFYING_FEATURES)}): {PRE_QUALIFYING_FEATURES}")
    print(f"POST-QUALIFYING features ({len(POST_QUALIFYING_FEATURES)}): {POST_QUALIFYING_FEATURES}")

    out = run_experiment()

    primary = [r for r in out["results"] if r.split == "primary_2026_r8_r10"]
    train_rows = primary[0].train_rows
    val_rows = primary[0].val_rows

    print("\n" + "=" * 90)
    print("FINAL REPORT")
    print("=" * 90)
    print(f"\n1. Training rows (primary):   {train_rows}")
    print(f"2. Validation rows (primary): {val_rows}")
    print(f"3. PRE features:  {PRE_QUALIFYING_FEATURES}")
    print(f"   POST extra:     {POST_QUALIFYING_EXTRA_FEATURES}")
    print(f"5. Grid baseline MAE:       {out['grid_mae']:.3f}")
    print(f"6. Qualifying baseline MAE:  {out['quali_mae']:.3f}")
    print(f"7. Best PRE:  {out['best_pre'].model} (MAE {out['best_pre'].mae:.3f})")
    print(f"8. Best POST: {out['best_post'].model} (MAE {out['best_post'].mae:.3f})")

    print("\n11. Overfitting / sanity notes:")
    sec_post = [r for r in out["results"] if r.split == "secondary_2024_train_2025_val" and r.configuration == "POST-QUALIFYING"]
    if sec_post:
        sec_best = min(sec_post, key=lambda r: r.mae)
        print(f"    Secondary (2024→2025) best POST MAE: {sec_best.mae:.3f} ({sec_best.model})")
        print(f"    Primary (→2026 R8-10) best POST MAE:  {out['best_post'].mae:.3f}")
        if out["best_post"].mae < sec_best.mae * 0.7:
            print("    ⚠ Primary MAE much lower than secondary — inspect for split-specific ease or leakage.")
        else:
            print("    Primary vs secondary MAE gap is plausible for different validation eras.")
    if out["best_post"].mae > out["grid_mae"]:
        print(f"    Best POST model MAE ({out['best_post'].mae:.3f}) exceeds grid baseline ({out['grid_mae']:.3f}) — grid is a strong heuristic.")
    else:
        print(f"    Best POST model beats grid baseline by {out['grid_mae'] - out['best_post'].mae:.3f} MAE.")

    print(f"\nResults saved to: {OUTPUT_DIR / RESULTS_FILE}")
    print("\nExperiment complete — no model binaries saved.")


if __name__ == "__main__":
    main()
