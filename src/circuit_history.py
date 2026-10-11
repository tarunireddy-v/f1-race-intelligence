"""
Circuit-specific historical features (analytical + optional merge to prediction tables).

Uses only races completed before the target (season, round). No leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import DNF_STATUSES, normalize_constructors
from src.singapore_config import (
    SINGAPORE_EVENT_NAME,
    SINGAPORE_HISTORY_START_SEASON,
    SINGAPORE_SEASON_WEIGHTS,
    STREET_COMPARABLE_EVENTS,
)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    v = values[valid]
    w = weights[valid]
    return float(np.average(v, weights=w))


def build_driver_circuit_history(
    race: pd.DataFrame,
    quali: pd.DataFrame,
    *,
    race_name: str,
    before_season: int,
    before_round: int,
    season_weights: dict[int, float] | None = None,
    min_season: int | None = None,
) -> pd.DataFrame:
    """
    Per-driver historical stats at one circuit before a target event.

    Excludes DNS from average finish (non-classified); counts DNF rate separately.
    """
    race = normalize_constructors(race)
    hist_r = race[
        (race["race_name"] == race_name)
        & (
            (race["season"] < before_season)
            | ((race["season"] == before_season) & (race["round"] < before_round))
        )
    ].copy()
    if min_season is not None:
        hist_r = hist_r[hist_r["season"] >= min_season]

    hist_q = quali[
        (quali["race_name"] == race_name)
        & (
            (quali["season"] < before_season)
            | ((quali["season"] == before_season) & (quali["round"] < before_round))
        )
    ].copy()
    if min_season is not None:
        hist_q = hist_q[hist_q["season"] >= min_season]

    drivers = sorted(set(hist_r["driver"].dropna()) | set(hist_q["driver"].dropna()))
    rows = []
    for driver in drivers:
        dr = hist_r[hist_r["driver"] == driver]
        dq = hist_q[hist_q["driver"] == driver]
        weights = dr["season"].map(season_weights or {}).fillna(1.0)

        classified = dr[~dr["classified_status"].isin({"Did not start"})]
        finishes = classified["finishing_position"]
        dnf_count = int(dr["classified_status"].isin(DNF_STATUSES).sum())
        n_starts = len(dr)
        podiums = int((finishes <= 3).sum()) if len(finishes) else 0

        rows.append(
            {
                "driver": driver,
                "circuit_starts": n_starts,
                "circuit_weighted_avg_finish": _weighted_mean(finishes, weights.loc[finishes.index])
                if len(finishes)
                else np.nan,
                "circuit_best_finish": finishes.min() if len(finishes) else np.nan,
                "circuit_podium_count": podiums,
                "circuit_podium_rate": podiums / n_starts if n_starts else np.nan,
                "circuit_dnf_count": dnf_count,
                "circuit_dnf_rate": dnf_count / n_starts if n_starts else np.nan,
                "circuit_avg_qualifying": dq["qualifying_position"].mean() if len(dq) else np.nan,
                "circuit_quali_observations": len(dq),
            }
        )
    return pd.DataFrame(rows)


def build_constructor_circuit_history(
    race: pd.DataFrame,
    *,
    race_name: str,
    before_season: int,
    before_round: int,
    min_season: int | None = None,
) -> pd.DataFrame:
    race = normalize_constructors(race)
    hist = race[
        (race["race_name"] == race_name)
        & (
            (race["season"] < before_season)
            | ((race["season"] == before_season) & (race["round"] < before_round))
        )
    ].copy()
    if min_season is not None:
        hist = hist[hist["season"] >= min_season]

    rows = []
    for constructor, grp in hist.groupby("constructor"):
        classified = grp[~grp["classified_status"].isin({"Did not start"})]
        rows.append(
            {
                "constructor": constructor,
                "constructor_circuit_starts": len(grp),
                "constructor_circuit_avg_finish": classified["finishing_position"].mean()
                if len(classified)
                else np.nan,
                "constructor_circuit_dnf_rate": grp["classified_status"].isin(DNF_STATUSES).mean()
                if len(grp)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_street_circuit_driver_form(
    race: pd.DataFrame,
    *,
    before_season: int,
    before_round: int,
    event_names: tuple[str, ...] = STREET_COMPARABLE_EVENTS,
) -> pd.DataFrame:
    """Comparable street-circuit finishes — separate from Singapore history."""
    race = normalize_constructors(race)
    hist = race[
        race["race_name"].isin(event_names)
        & (
            (race["season"] < before_season)
            | ((race["season"] == before_season) & (race["round"] < before_round))
        )
    ].copy()
    classified = hist[~hist["classified_status"].isin({"Did not start"})]

    rows = []
    for driver, grp in classified.groupby("driver"):
        rows.append(
            {
                "driver": driver,
                "street_circuit_starts": len(grp),
                "street_circuit_avg_finish": grp["finishing_position"].mean(),
                "street_circuit_obs_events": grp["race_name"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def singapore_driver_history(
    race: pd.DataFrame,
    quali: pd.DataFrame,
    before_season: int,
    before_round: int,
) -> pd.DataFrame:
    return build_driver_circuit_history(
        race,
        quali,
        race_name=SINGAPORE_EVENT_NAME,
        before_season=before_season,
        before_round=before_round,
        season_weights=SINGAPORE_SEASON_WEIGHTS,
        min_season=SINGAPORE_HISTORY_START_SEASON,
    )
