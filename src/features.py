"""
V1 feature engineering — simple modeling dataset from Race + Qualifying results.

Leakage rule (applied throughout):
    Every rolling / history feature uses ONLY information from races BEFORE the
    current row. We sort chronologically by (season, round), then apply shift(1)
    within the relevant group before any rolling or expanding window.

    Example: avg_finish_last_3 for VER at 2025 Hungarian GP uses his finishes
    from the three races immediately before Hungary 2025 — never Hungary itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    MODELING_DATASET_FILE,
    PROCESSED_DATA_DIR,
    QUALIFYING_OUTPUT_FILE,
    RACE_OUTPUT_FILE,
    RAW_DATA_DIR,
    REGULATION_ERA_START_SEASON,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constructor normalization — spelling variants only (same team, same era).
# Does NOT merge genuine rebrands (e.g. Kick Sauber → Audi for 2026).
# ---------------------------------------------------------------------------
CONSTRUCTOR_NORMALIZATION: dict[str, str] = {
    "Alpine F1 Team": "Alpine",
    "Cadillac F1 Team": "Cadillac",
    "Red Bull": "Red Bull Racing",
    "RB F1 Team": "Racing Bulls",
}

# ---------------------------------------------------------------------------
# DNF definition (from FastF1 classified_status values in our raw data)
# ---------------------------------------------------------------------------
# Observed values: Finished, Lapped, Retired, Did not start, Disqualified
#
# Finished / Lapped  → classified race finish (driver completed the GP distance
#                      or was running at the flag, possibly lapped).
# Retired            → mechanical / crash / etc. — DNF.
# Did not start      → entered but did not start the race — DNF.
# Disqualified       → result voided after the fact — treated as non-finish (DNF)
#                      for rolling DNF-rate purposes.
DNF_STATUSES: frozenset[str] = frozenset({"Retired", "Did not start", "Disqualified"})
NON_DNF_STATUSES: frozenset[str] = frozenset({"Finished", "Lapped"})

IDENTIFIER_COLUMNS = [
    "season",
    "round",
    "race_name",
    "circuit",
    "driver",
    "constructor",
]

DRIVER_FORM_COLUMNS = [
    "avg_finish_last_3",
    "avg_finish_last_5",
    "avg_quali_last_3",
    "points_last_5",
    "dnf_rate_last_5",
]

CONSTRUCTOR_FORM_COLUMNS = [
    "constructor_avg_finish_last_3",
    "constructor_points_last_5",
]

CIRCUIT_HISTORY_COLUMNS = [
    "driver_previous_circuit_finish",
    "driver_circuit_avg_finish",
]

SEASON_CONTEXT_COLUMNS = ["is_2026"]

QUALIFYING_FEATURE_COLUMNS = [
    "grid_position",
    "qualifying_position",
    "gap_to_pole_s",
]

TARGET_COLUMN = "finishing_position"

FEATURE_COLUMNS = (
    DRIVER_FORM_COLUMNS
    + CONSTRUCTOR_FORM_COLUMNS
    + CIRCUIT_HISTORY_COLUMNS
    + SEASON_CONTEXT_COLUMNS
    + QUALIFYING_FEATURE_COLUMNS
)

OUTPUT_COLUMNS = IDENTIFIER_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]


def load_raw_results(
    raw_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load race and qualifying CSVs from data/raw/."""
    base = raw_dir or RAW_DATA_DIR
    race = pd.read_csv(base / RACE_OUTPUT_FILE)
    quali = pd.read_csv(base / QUALIFYING_OUTPUT_FILE)
    race["season"] = race["season"].astype(int)
    quali["season"] = quali["season"].astype(int)
    race["round"] = race["round"].astype(int)
    quali["round"] = quali["round"].astype(int)
    return race, quali


def normalize_constructors(race: pd.DataFrame) -> pd.DataFrame:
    """Apply explicit constructor name normalization mapping."""
    out = race.copy()
    out["constructor"] = out["constructor"].replace(CONSTRUCTOR_NORMALIZATION)
    return out


def print_constructor_normalization(race: pd.DataFrame, normalized: pd.DataFrame) -> None:
    """Print constructor names before and after normalization."""
    print("\n" + "=" * 70)
    print("CONSTRUCTOR NORMALIZATION")
    print("=" * 70)
    print("\nMapping applied (variant → canonical):")
    for variant, canonical in sorted(CONSTRUCTOR_NORMALIZATION.items()):
        print(f"  {variant!r} → {canonical!r}")

    for season in sorted(race["season"].unique()):
        before = sorted(race[race["season"] == season]["constructor"].unique())
        after = sorted(normalized[normalized["season"] == season]["constructor"].unique())
        print(f"\nSeason {season} — BEFORE ({len(before)}): {before}")
        print(f"Season {season} — AFTER  ({len(after)}): {after}")

    r26 = normalized[normalized["season"] == REGULATION_ERA_START_SEASON]
    counts = r26.groupby("round")["constructor"].nunique()
    print("\n2026 constructors per race (after normalization):")
    for rnd, n in counts.items():
        status = "OK" if n == 11 else f"UNEXPECTED ({n})"
        name = r26.loc[r26["round"] == rnd, "race_name"].iloc[0]
        print(f"  R{int(rnd):>2} {name}: {n} constructors — {status}")


def is_dnf(classified_status: str | float) -> bool | float:
    """Return True if status counts as DNF; NaN if status is missing."""
    if pd.isna(classified_status):
        return pd.NA
    return classified_status in DNF_STATUSES


def merge_race_qualifying(race: pd.DataFrame, quali: pd.DataFrame) -> pd.DataFrame:
    """
    One modeling row per driver per race (race results are the spine).

    Qualifying columns are left-joined; missing quali for a race entrant stays NaN.
    """
    quali_cols = [
        "season",
        "round",
        "driver",
        "qualifying_position",
        "qualifying_best_lap_s",
        "gap_to_pole_s",
    ]
    quali_sub = quali[quali_cols].drop_duplicates(subset=["season", "round", "driver"])

    merged = race.merge(
        quali_sub,
        on=["season", "round", "driver"],
        how="left",
        validate="one_to_one",
    )
    return merged.sort_values(["season", "round", "driver"]).reset_index(drop=True)


def _chronological_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["season", "round", "driver"]).reset_index(drop=True)


def _shifted_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """shift(1) then rolling mean — excludes current row from the window."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def _shifted_rolling_sum(series: pd.Series, window: int) -> pd.Series:
    """shift(1) then rolling sum — excludes current row from the window."""
    return series.shift(1).rolling(window, min_periods=1).sum()


def add_driver_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Driver rolling form from prior races only (grouped by driver, chronological).

    shift(1) ensures the current race finish / quali / points / DNF flag is never
    included in the feature value for that same row.
    """
    out = _chronological_sort(df)
    out["is_dnf"] = out["classified_status"].map(is_dnf)

    g_finish = out.groupby("driver")["finishing_position"]
    out["avg_finish_last_3"] = g_finish.transform(lambda s: _shifted_rolling_mean(s, 3))
    out["avg_finish_last_5"] = g_finish.transform(lambda s: _shifted_rolling_mean(s, 5))

    g_quali = out.groupby("driver")["qualifying_position"]
    out["avg_quali_last_3"] = g_quali.transform(lambda s: _shifted_rolling_mean(s, 3))

    g_points = out.groupby("driver")["points"]
    out["points_last_5"] = g_points.transform(lambda s: _shifted_rolling_sum(s, 5))

    g_dnf = out.groupby("driver")["is_dnf"]
    out["dnf_rate_last_5"] = g_dnf.transform(lambda s: _shifted_rolling_mean(s.astype(float), 5))

    return out.drop(columns=["is_dnf"])


def add_constructor_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructor rolling form from prior races only.

    First aggregate to one row per (season, round, constructor), then shift(1)
    and roll, then merge back to driver rows. The current race weekend is never
    included in the constructor's rolling window for that weekend.
    """
    race_level = (
        df.groupby(["season", "round", "constructor"], as_index=False)
        .agg(
            constructor_race_avg_finish=("finishing_position", "mean"),
            constructor_race_points=("points", "sum"),
        )
        .sort_values(["season", "round", "constructor"])
    )

    g_avg = race_level.groupby("constructor")["constructor_race_avg_finish"]
    race_level["constructor_avg_finish_last_3"] = g_avg.transform(
        lambda s: _shifted_rolling_mean(s, 3)
    )

    g_pts = race_level.groupby("constructor")["constructor_race_points"]
    race_level["constructor_points_last_5"] = g_pts.transform(
        lambda s: _shifted_rolling_sum(s, 5)
    )

    merge_cols = [
        "season",
        "round",
        "constructor",
        "constructor_avg_finish_last_3",
        "constructor_points_last_5",
    ]
    return df.merge(race_level[merge_cols], on=["season", "round", "constructor"], how="left")


def add_circuit_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Circuit history from prior visits to the same circuit only.

    shift(1) within (driver, circuit) excludes the current race before any
    expanding mean is computed.
    """
    out = _chronological_sort(df)
    g = out.groupby(["driver", "circuit"])["finishing_position"]
    out["driver_previous_circuit_finish"] = g.shift(1)
    out["driver_circuit_avg_finish"] = g.transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    return out


def add_season_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_2026"] = out["season"] >= REGULATION_ERA_START_SEASON
    return out


def build_modeling_dataset(
    raw_dir: Path | None = None,
    *,
    race_frame: pd.DataFrame | None = None,
    quali_frame: pd.DataFrame | None = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Build the full V1 modeling dataset from raw Race + Qualifying CSVs.

    Optional ``race_frame`` / ``quali_frame`` allow callers (e.g. Singapore pipeline)
    to supply venue-normalized copies without writing raw CSVs.
    """
    if race_frame is None or quali_frame is None:
        race_raw, quali = load_raw_results(raw_dir)
    else:
        race_raw, quali = race_frame, quali_frame

    if not quiet:
        print_constructor_normalization(race_raw, normalize_constructors(race_raw))

    race = normalize_constructors(race_raw)
    merged = merge_race_qualifying(race, quali)

    merged = add_driver_form_features(merged)
    merged = add_constructor_form_features(merged)
    merged = add_circuit_history_features(merged)
    merged = add_season_context(merged)

    # grid_position comes from race results; qualifying features from quali merge
    dataset = merged[OUTPUT_COLUMNS].copy()
    return _chronological_sort(dataset)


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts for engineered feature columns."""
    rows = []
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        n_missing = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "missing": n_missing,
                "pct": round(100 * n_missing / len(df), 1) if len(df) else 0.0,
            }
        )
    report = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("MISSING VALUES BY FEATURE")
    print("=" * 70)
    print(report.to_string(index=False))
    return report


def _prior_driver_races(
    df: pd.DataFrame,
    driver: str,
    season: int,
    round_number: int,
    n: int,
) -> pd.DataFrame:
    """Return the n races immediately before (season, round) for a driver."""
    hist = df[
        (df["driver"] == driver)
        & ((df["season"] < season) | ((df["season"] == season) & (df["round"] < round_number)))
    ].sort_values(["season", "round"])
    return hist.tail(n)


def _prior_constructor_races(
    race: pd.DataFrame,
    constructor: str,
    season: int,
    round_number: int,
    n: int,
) -> pd.DataFrame:
    """Return constructor-level aggregates for the n races before this round."""
    race_level = (
        race.groupby(["season", "round", "constructor", "race_name"], as_index=False)
        .agg(constructor_race_avg_finish=("finishing_position", "mean"))
        .sort_values(["season", "round"])
    )
    hist = race_level[
        (race_level["constructor"] == constructor)
        & (
            (race_level["season"] < season)
            | ((race_level["season"] == season) & (race_level["round"] < round_number))
        )
    ]
    return hist.tail(n)


def validate_leakage(
    dataset: pd.DataFrame,
    race: pd.DataFrame,
    examples: list[tuple[int, int, str, str]] | None = None,
) -> None:
    """
    Manually verify rolling features exclude the current race.

    examples: list of (season, round, driver, label) tuples to inspect.
    """
    if examples is None:
        examples = [
            (2025, 14, "VER", "2025 Hungarian GP — VER"),
            (2026, 9, "NOR", "2026 British GP — NOR"),
            (2025, 24, "LEC", "2025 Abu Dhabi GP — LEC"),
        ]

    print("\n" + "=" * 70)
    print("LEAKAGE VALIDATION — manual spot checks")
    print("=" * 70)

    for season, round_number, driver, label in examples:
        row = dataset[
            (dataset["season"] == season)
            & (dataset["round"] == round_number)
            & (dataset["driver"] == driver)
        ]
        if row.empty:
            print(f"\n{label}: row not found — skipped")
            continue
        row = row.iloc[0]
        print(f"\n--- {label} ---")
        print(f"  Target finishing_position: {row[TARGET_COLUMN]}")
        print(f"  avg_finish_last_3:         {row['avg_finish_last_3']:.3f}")
        print(f"  avg_quali_last_3:          {row['avg_quali_last_3']:.3f}")
        print(f"  constructor_avg_finish_last_3: {row['constructor_avg_finish_last_3']:.3f}")

        prior_finish = _prior_driver_races(race, driver, season, round_number, 3)
        print(f"\n  Prior 3 races used for avg_finish_last_3:")
        if prior_finish.empty:
            print("    (none — early career / first races)")
        else:
            for _, r in prior_finish.iterrows():
                print(
                    f"    {int(r['season'])} R{int(r['round']):>2} {r['race_name']:<28} "
                    f"finish={int(r['finishing_position'])}"
                )
            expected = prior_finish["finishing_position"].mean()
            print(f"    Manual mean (must NOT include current race): {expected:.3f}")
            assert abs(row["avg_finish_last_3"] - expected) < 1e-9, "avg_finish_last_3 leakage!"

        prior_quali = _prior_driver_races(
            race.merge(
                dataset[["season", "round", "driver", "qualifying_position"]],
                on=["season", "round", "driver"],
                how="left",
            ),
            driver,
            season,
            round_number,
            3,
        )
        print(f"\n  Prior 3 races used for avg_quali_last_3:")
        if prior_quali.empty:
            print("    (none)")
        else:
            for _, r in prior_quali.iterrows():
                q = r.get("qualifying_position", pd.NA)
                print(
                    f"    {int(r['season'])} R{int(r['round']):>2} {r['race_name']:<28} "
                    f"quali={q}"
                )
            valid = prior_quali["qualifying_position"].dropna()
            if len(valid):
                expected_q = valid.mean()
                print(f"    Manual mean (non-null quali only): {expected_q:.3f}")
                if pd.notna(row["avg_quali_last_3"]):
                    assert abs(row["avg_quali_last_3"] - expected_q) < 1e-9, "avg_quali_last_3 leakage!"

        constructor = row["constructor"]
        prior_const = _prior_constructor_races(race, constructor, season, round_number, 3)
        print(f"\n  Prior 3 constructor races for constructor_avg_finish_last_3 ({constructor}):")
        if prior_const.empty:
            print("    (none — new constructor entry)")
        else:
            for _, r in prior_const.iterrows():
                print(
                    f"    {int(r['season'])} R{int(r['round']):>2} {r['race_name']:<28} "
                    f"team_avg_finish={r['constructor_race_avg_finish']:.3f}"
                )
            expected_c = prior_const["constructor_race_avg_finish"].mean()
            print(f"    Manual mean: {expected_c:.3f}")
            assert abs(row["constructor_avg_finish_last_3"] - expected_c) < 1e-9, (
                "constructor_avg_finish_last_3 leakage!"
            )

        # Confirm current race is NOT in prior lists
        current_in_finish = (
            (prior_finish["season"] == season) & (prior_finish["round"] == round_number)
        ).any()
        assert not current_in_finish, "Current race found in prior finish list — LEAKAGE"
        print("\n  ✓ Current race excluded from all rolling windows")


def save_modeling_dataset(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    filename: str | None = None,
) -> Path:
    out_dir = output_dir or PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (filename or MODELING_DATASET_FILE)
    df.to_csv(path, index=False)
    logger.info("Wrote modeling dataset: %s (%s rows)", path, len(df))
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    print("=" * 70)
    print("V1 MODELING DATASET BUILD")
    print("=" * 70)
    print(f"\nDNF statuses ({len(DNF_STATUSES)}): {sorted(DNF_STATUSES)}")
    print(f"Non-DNF statuses ({len(NON_DNF_STATUSES)}): {sorted(NON_DNF_STATUSES)}")

    race_raw, _ = load_raw_results()
    dataset = build_modeling_dataset()

    path = save_modeling_dataset(dataset)

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"\nFinal shape: {dataset.shape}")
    print(f"Columns ({len(dataset.columns)}): {list(dataset.columns)}")
    print(f"Saved to: {path}")

    print("\nRows by season:")
    for season, count in dataset.groupby("season").size().items():
        print(f"  {season}: {count}")

    report_missing_values(dataset)

    race_norm = normalize_constructors(race_raw)
    validate_leakage(dataset, race_norm)

    print("\n" + "=" * 70)
    print("BUILD COMPLETE — no model training performed")
    print("=" * 70)


if __name__ == "__main__":
    main()
