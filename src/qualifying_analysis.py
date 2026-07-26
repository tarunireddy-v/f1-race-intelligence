"""
Hungary 2026 qualifying extraction from FastF1.

Qualifying session only — no race data loaded.
"""

from __future__ import annotations

import logging
from typing import Any

import fastf1
import numpy as np
import pandas as pd

from src.data_loader import enable_cache
from src.practice_analysis import normalize_constructor

logger = logging.getLogger(__name__)

# Official post-penalty starting grid (project-supplied, confirmed vs FastF1 quali).
OFFICIAL_GRID_2026_HUNGARY: dict[str, int] = {
    "NOR": 1,
    "LEC": 2,
    "PIA": 3,
    "VER": 4,
    "HAM": 5,
    "RUS": 6,
    "ANT": 7,
    "HAD": 8,
    "LIN": 9,
    "HUL": 10,
    "LAW": 11,
    "GAS": 12,
    "COL": 13,
    "BOR": 14,
    "OCO": 15,
    "ALO": 16,
    "BEA": 17,
    "SAI": 18,
    "ALB": 19,
    "STR": 20,
    "BOT": 21,
    "PER": 22,
}

# Drivers with confirmed sporting grid penalties (not grid gains from others' penalties).
KNOWN_GRID_PENALTY_DRIVERS: frozenset[str] = frozenset({"HAM", "ANT"})


def _td_to_seconds(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    return float(value)


def _valid_laps(laps: pd.DataFrame) -> pd.DataFrame:
    if laps is None or laps.empty:
        return pd.DataFrame()
    out = laps.copy()
    if "Deleted" in out.columns:
        out = out[out["Deleted"].isna() | (out["Deleted"] == False)]  # noqa: E712
    out = out[out["LapTime"].notna()]
    return out


def _quali_stage(q1: Any, q2: Any, q3: Any) -> str | None:
    if _td_to_seconds(q3) is not None:
        return "Q3"
    if _td_to_seconds(q2) is not None:
        return "Q2"
    if _td_to_seconds(q1) is not None:
        return "Q1"
    return None


def _best_session_lap(q1: Any, q2: Any, q3: Any) -> float | None:
    times = [_td_to_seconds(q1), _td_to_seconds(q2), _td_to_seconds(q3)]
    valid = [t for t in times if t is not None]
    return min(valid) if valid else None


def _sector_seconds(lap_row: pd.Series, sector: int) -> float | None:
    col = f"Sector{sector}Time"
    if col not in lap_row.index:
        return None
    return _td_to_seconds(lap_row[col])


def load_hungary_qualifying_session(season: int = 2026, event: str = "Hungarian Grand Prix"):
    """Load qualifying from FastF1 cache — qualifying only."""
    enable_cache()
    session = fastf1.get_session(season, event, "Q")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    return session


def extract_qualifying_table(session) -> pd.DataFrame:
    """
    Build per-driver qualifying analysis for all official entrants.

    qualifying_position = official session classification (pre-penalty).
    grid_position = official post-penalty race grid (project-supplied).
    """
    results = session.results.copy()
    laps = _valid_laps(session.laps)

    rows: list[dict] = []
    for _, res in results.iterrows():
        driver = str(res["Abbreviation"])
        constructor = normalize_constructor(res["TeamName"])
        quali_pos = int(res["Position"])
        q1_s = _td_to_seconds(res.get("Q1"))
        q2_s = _td_to_seconds(res.get("Q2"))
        q3_s = _td_to_seconds(res.get("Q3"))
        best_lap_s = _best_session_lap(res.get("Q1"), res.get("Q2"), res.get("Q3"))
        if best_lap_s is None:
            best_lap_s = _td_to_seconds(res.get("Time"))

        driver_laps = laps[laps["Driver"] == driver] if not laps.empty else pd.DataFrame()
        s1 = s2 = s3 = None
        sector_source = "unavailable"
        if not driver_laps.empty:
            best_idx = driver_laps["LapTime"].idxmin()
            best_lap_row = driver_laps.loc[best_idx]
            s1 = _sector_seconds(best_lap_row, 1)
            s2 = _sector_seconds(best_lap_row, 2)
            s3 = _sector_seconds(best_lap_row, 3)
            if any(v is not None for v in (s1, s2, s3)):
                sector_source = "fastf1_laps"

        grid_pos = OFFICIAL_GRID_2026_HUNGARY.get(driver)
        if grid_pos is None:
            raise ValueError(f"Driver {driver} missing from official post-penalty grid")

        rows.append(
            {
                "driver": driver,
                "constructor": constructor,
                "qualifying_position": quali_pos,
                "grid_position": grid_pos,
                "qualifying_best_lap_s": best_lap_s,
                "q1_best_s": q1_s,
                "q2_best_s": q2_s,
                "q3_best_s": q3_s,
                "driver_quali_stage": _quali_stage(res.get("Q1"), res.get("Q2"), res.get("Q3")),
                "quali_s1_best_s": s1,
                "quali_s2_best_s": s2,
                "quali_s3_best_s": s3,
                "sector_data_source": sector_source,
            }
        )

    df = pd.DataFrame(rows)
    pole_lap = df["qualifying_best_lap_s"].min()
    df["quali_gap_to_pole_s"] = df["qualifying_best_lap_s"] - pole_lap

    df["grid_penalty_positions"] = df["grid_position"] - df["qualifying_position"]
    df["has_grid_penalty"] = df["driver"].isin(KNOWN_GRID_PENALTY_DRIVERS)

    # Teammate qualifying comparison
    tm_gaps: list[float | None] = []
    tm_stages: list[str | None] = []
    for _, row in df.iterrows():
        team = df[df["constructor"] == row["constructor"]]
        others = team[team["driver"] != row["driver"]]
        if len(others) != 1:
            tm_gaps.append(None)
            tm_stages.append(None)
            continue
        mate = others.iloc[0]
        if row["qualifying_best_lap_s"] is not None and mate["qualifying_best_lap_s"] is not None:
            tm_gaps.append(row["qualifying_best_lap_s"] - mate["qualifying_best_lap_s"])
        else:
            tm_gaps.append(None)
        tm_stages.append(mate["driver_quali_stage"])
    df["teammate_quali_gap_s"] = tm_gaps
    df["teammate_quali_stage"] = tm_stages

    # Sector gaps and ranks (where reliable)
    for sec, col in [(1, "quali_s1"), (2, "quali_s2"), (3, "quali_s3")]:
        best_col = f"{col}_best_s"
        gap_col = f"{col}_gap_to_best_s"
        rank_col = f"{col}_rank"
        valid = df[best_col].notna()
        if valid.any():
            best = df.loc[valid, best_col].min()
            df[gap_col] = np.where(valid, df[best_col] - best, np.nan)
            df[rank_col] = np.where(valid, df[best_col].rank(method="min"), np.nan)
        else:
            df[gap_col] = np.nan
            df[rank_col] = np.nan

    # Q progression (descriptive)
    df["q1_to_q3_improvement_s"] = np.where(
        df["q1_best_s"].notna() & df["q3_best_s"].notna(),
        df["q1_best_s"] - df["q3_best_s"],
        np.nan,
    )

    return df.sort_values("qualifying_position").reset_index(drop=True)


def team_front_row_lockout(quali_df: pd.DataFrame) -> bool:
    """True if P1 and P2 on actual starting grid share a constructor."""
    grid = quali_df.sort_values("grid_position")
    top2 = grid.head(2)
    if len(top2) < 2:
        return False
    return top2.iloc[0]["constructor"] == top2.iloc[1]["constructor"]


def print_quali_sector2_top10(quali_df: pd.DataFrame) -> None:
    valid = quali_df[quali_df["quali_s2_best_s"].notna()].copy()
    valid = valid.sort_values("quali_s2_best_s")
    print("\n" + "=" * 70)
    print("QUALIFYING SECTOR 2 TOP 10")
    print("=" * 70)
    if valid.empty:
        print("  Sector 2 data unavailable.")
        return
    for i, row in valid.head(10).iterrows():
        print(
            f"  {int(row['quali_s2_rank']):>2}. {row['driver']:<4} "
            f"{row['quali_s2_best_s']:.3f}s  gap {row['quali_s2_gap_to_best_s']:.3f}s  "
            f"(Q{row['driver_quali_stage'][-1] if row['driver_quali_stage'] else '?'})"
        )
    missing = quali_df[quali_df["quali_s2_best_s"].isna()]["driver"].tolist()
    if missing:
        print(f"\n  Missing/unreliable S2: {', '.join(missing)}")
