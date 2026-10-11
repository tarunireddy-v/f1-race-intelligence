"""FastF1 session loading utilities for historical data ingestion."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

from src.config import (
    CACHE_DIR,
    HISTORICAL_SESSION_TYPES,
    RACE_LOAD_KWARGS,
    TARGET_EVENT,
    TARGET_SEASON,
    TIMING_LOAD_KWARGS,
)

logger = logging.getLogger(__name__)


def enable_cache(cache_dir: str | None = None) -> str:
    """Enable FastF1 disk cache and return the cache path."""
    path = str(cache_dir or CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(path)
    logger.info("FastF1 cache enabled at %s", path)
    return path


def get_event_schedule(season: int) -> pd.DataFrame:
    """Return the FastF1 event schedule for a season."""
    try:
        schedule = fastf1.get_event_schedule(season)
    except RateLimitExceededError as exc:
        logger.error("Rate limit exceeded fetching schedule for %s — %s", season, exc)
        raise
    if schedule is None or schedule.empty:
        raise ValueError(f"No event schedule returned for season {season}")
    return schedule


def get_grand_prix_events(schedule: pd.DataFrame) -> pd.DataFrame:
    """Filter schedule to Grand Prix events (positive round numbers)."""
    gps = schedule[schedule["RoundNumber"] > 0].copy()
    gps = gps.sort_values("RoundNumber").reset_index(drop=True)
    return gps


def get_event_round(schedule: pd.DataFrame, event_name: str) -> int:
    """
    Return the round number for a named Grand Prix from schedule data.

    Raises ValueError if the event cannot be found.
    """
    gps = get_grand_prix_events(schedule)
    matches = gps[gps["EventName"].str.strip().str.lower() == event_name.strip().lower()]
    if matches.empty:
        # Fallback: substring match (e.g. partial name)
        matches = gps[gps["EventName"].str.contains(event_name, case=False, na=False)]
    if matches.empty:
        raise ValueError(f"Event '{event_name}' not found in schedule")
    if len(matches) > 1:
        logger.warning(
            "Multiple schedule rows matched '%s'; using first (round %s)",
            event_name,
            matches.iloc[0]["RoundNumber"],
        )
    return int(matches.iloc[0]["RoundNumber"])


def _normalize_event_date(event_date: Any) -> date:
    """Convert schedule EventDate values to plain dates."""
    if isinstance(event_date, datetime):
        return event_date.date()
    if isinstance(event_date, date):
        return event_date
    return pd.Timestamp(event_date).date()


def is_event_completed(event_date: Any, as_of: date | None = None) -> bool:
    """True if the Grand Prix race date is strictly before as_of."""
    reference = as_of or date.today()
    return _normalize_event_date(event_date) < reference


def get_completed_grand_prix(
    season: int,
    *,
    before_event: str | None = None,
    before_round: int | None = None,
    as_of: date | None = None,
) -> pd.DataFrame:
    """
    Return completed Grand Prix events for a season.

    Parameters
    ----------
    season :
        Championship year.
    before_event :
        If set, exclude this event and all later rounds (resolved via schedule).
    before_round :
        If set, include only rounds strictly less than this value.
        Takes precedence over ``before_event`` when both supplied explicitly,
        but ``before_event`` is typically resolved from config instead.
    as_of :
        Reference date for completion checks (default: today).
    """
    schedule = get_event_schedule(season)
    gps = get_grand_prix_events(schedule)

    if before_event is not None and before_round is None:
        before_round = get_event_round(schedule, before_event)

    completed = gps[gps["EventDate"].apply(lambda d: is_event_completed(d, as_of))]

    if before_round is not None:
        completed = completed[completed["RoundNumber"] < before_round]

    return completed.reset_index(drop=True)


def get_training_events(
    season: int,
    as_of: date | None = None,
    *,
    before_event: str | None = None,
) -> pd.DataFrame:
    """
    Return Grand Prix events to ingest for a training season.

    2024 / 2025: all completed races.
    2026: completed races strictly before ``before_event`` when provided;
    otherwise completed races strictly before ``TARGET_EVENT`` (Hungary V1 default).
    """
    if season == TARGET_SEASON:
        schedule = get_event_schedule(season)
        cutoff_event = before_event or TARGET_EVENT
        target_round = get_event_round(schedule, cutoff_event)
        return get_completed_grand_prix(
            season,
            before_round=target_round,
            as_of=as_of,
        )
    if season in (2024, 2025):
        return get_completed_grand_prix(season, as_of=as_of)
    raise ValueError(f"Season {season} is not configured for training ingestion")


def load_session(
    season: int,
    round_number: int,
    session_type: str,
    *,
    load_messages: bool = False,
):
    """
    Load a FastF1 session with lightweight settings.

    Returns the loaded session, or None if the session does not exist.
    """
    _WEEKEND_SESSION_TYPES = ("FP1", "FP2", "FP3", "SQ", "S")
    if session_type not in HISTORICAL_SESSION_TYPES and session_type not in _WEEKEND_SESSION_TYPES:
        raise ValueError(f"Unsupported session type: {session_type}")

    label = HISTORICAL_SESSION_TYPES.get(session_type, session_type)
    try:
        session = fastf1.get_session(season, round_number, session_type)
    except Exception as exc:
        logger.warning(
            "%s %s round %s: could not get %s session — %s",
            season,
            label,
            round_number,
            session_type,
            exc,
        )
        return None

    if session_type == "R":
        load_kwargs = dict(RACE_LOAD_KWARGS)
    else:
        load_kwargs = dict(TIMING_LOAD_KWARGS)
        if load_messages:
            load_kwargs["messages"] = True
    try:
        session.load(**load_kwargs)
    except ValueError as exc:
        # Sprint weekends may not have FP2, etc.
        logger.warning(
            "%s round %s: %s session unavailable — %s",
            season,
            round_number,
            session_type,
            exc,
        )
        return None
    except RateLimitExceededError as exc:
        logger.error(
            "%s round %s: rate limit hit loading %s session — %s",
            season,
            round_number,
            session_type,
            exc,
        )
        raise
    except Exception as exc:
        logger.error(
            "%s round %s: failed loading %s session — %s",
            season,
            round_number,
            session_type,
            exc,
        )
        return None

    return session


def _event_metadata(session, season: int) -> dict[str, Any]:
    """Extract shared event identifiers from a loaded session."""
    event = session.event
    event_date = _normalize_event_date(event["EventDate"])
    return {
        "season": season,
        "round": int(event["RoundNumber"]),
        "race_name": str(event["EventName"]),
        "circuit": str(event["Location"]),
        "race_date": event_date.isoformat(),
    }


def _valid_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Return laps with valid lap times, excluding deleted laps when flagged."""
    if laps is None or laps.empty:
        return pd.DataFrame()
    valid = laps.dropna(subset=["LapTime"]).copy()
    if "Deleted" in valid.columns:
        valid = valid[~valid["Deleted"].astype("boolean").fillna(False)]
    valid["LapTime (s)"] = valid["LapTime"].dt.total_seconds()
    return valid


def extract_race_results(session, season: int) -> pd.DataFrame:
    """Extract lightweight race results per driver."""
    meta = _event_metadata(session, season)
    results = session.results
    if results is None or results.empty:
        return pd.DataFrame()

    rows = []
    for _, row in results.iterrows():
        position = row.get("Position")
        rows.append(
            {
                **meta,
                "driver": row.get("Abbreviation"),
                "constructor": row.get("TeamName"),
                "finishing_position": position,
                "classified_status": row.get("Status"),
                "points": row.get("Points"),
                "grid_position": row.get("GridPosition"),
            }
        )
    return pd.DataFrame(rows)


def extract_qualifying_results(session, season: int) -> pd.DataFrame:
    """
    Extract qualifying results per driver.

    ``qualifying_position`` is the official FastF1 classification and must not
    be overwritten by fastest-lap ordering. ``qualifying_best_lap_s`` is a
    separate pace feature derived from valid (non-deleted) laps.
    """
    meta = _event_metadata(session, season)
    results = session.results
    if results is None or results.empty:
        return pd.DataFrame()

    best_laps: dict[str, float] = {}
    valid = _valid_laps(session.laps)
    if not valid.empty:
        best_laps = valid.groupby("Driver")["LapTime (s)"].min().to_dict()

    rows = []
    for _, row in results.iterrows():
        driver = row.get("Abbreviation")
        best_lap = best_laps.get(driver)
        if pd.isna(best_lap) and driver in best_laps:
            best_lap = best_laps[driver]
        rows.append(
            {
                **meta,
                "driver": driver,
                "qualifying_position": row.get("Position"),
                "qualifying_best_lap_s": best_lap,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    pole_time = df["qualifying_best_lap_s"].min()
    df["gap_to_pole_s"] = df["qualifying_best_lap_s"] - pole_time
    return df


def extract_practice_results(session, season: int, session_label: str) -> pd.DataFrame:
    """Extract FP1 or FP2 lightweight timing summary per driver."""
    meta = _event_metadata(session, season)
    valid = _valid_laps(session.laps)
    if valid.empty:
        return pd.DataFrame()

    summary = (
        valid.groupby("Driver")
        .agg(
            best_lap_s=("LapTime (s)", "min"),
            laps_completed=("LapTime (s)", "count"),
        )
        .reset_index()
        .rename(columns={"Driver": "driver"})
    )

    session_fastest = summary["best_lap_s"].min()
    summary["gap_to_fastest_s"] = summary["best_lap_s"] - session_fastest
    summary["practice_position"] = summary["best_lap_s"].rank(method="min").astype(int)

    summary = summary.sort_values("practice_position").reset_index(drop=True)

    for key, value in meta.items():
        summary[key] = value

    # Standardize column names to schema (best lap time in seconds)
    summary = summary.rename(columns={"best_lap_s": "best_lap_s"})
    summary["session"] = session_label

    column_order = [
        "season",
        "round",
        "race_name",
        "driver",
        "best_lap_s",
        "practice_position",
        "gap_to_fastest_s",
        "laps_completed",
        "circuit",
        "race_date",
        "session",
    ]
    return summary[column_order]


def collect_event_sessions(
    season: int,
    round_number: int,
    event_name: str,
    *,
    include_practice: bool = False,
) -> dict[str, pd.DataFrame | None]:
    """
    Load and extract sessions for one Grand Prix.

    V1 historical ingestion uses ``include_practice=False`` (Race + Qualifying
    only). Set ``include_practice=True`` for current-weekend FP collection.

    Returns dict with keys: race, qualifying, and optionally fp1, fp2.
    """
    logger.info("Collecting %s %s (round %s)", season, event_name, round_number)
    extracted: dict[str, pd.DataFrame | None] = {
        "race": None,
        "qualifying": None,
        "fp1": None,
        "fp2": None,
    }

    race_session = load_session(season, round_number, "R")
    if race_session is not None:
        extracted["race"] = extract_race_results(race_session, season)

    quali_session = load_session(season, round_number, "Q")
    if quali_session is not None:
        extracted["qualifying"] = extract_qualifying_results(quali_session, season)

    if include_practice:
        fp1_session = load_session(season, round_number, "FP1")
        if fp1_session is not None:
            extracted["fp1"] = extract_practice_results(fp1_session, season, "FP1")

        fp2_session = load_session(season, round_number, "FP2")
        if fp2_session is not None:
            extracted["fp2"] = extract_practice_results(fp2_session, season, "FP2")

    return extracted


def estimate_cache_coverage(season: int) -> dict:
    """
    Estimate Race/Qualifying cache availability without bulk downloading.

    Inspects FastF1 cache directories on disk for Race and Qualifying folders.
    """
    cache_year = CACHE_DIR / str(season)
    events = get_training_events(season)
    cached_race = cached_quali = 0
    per_event: list[dict] = []

    for _, row in events.iterrows():
        rnd = int(row["RoundNumber"])
        name = str(row["EventName"])
        event_dir = None
        has_r = has_q = False

        if cache_year.exists():
            for folder in cache_year.iterdir():
                if not folder.is_dir():
                    continue
                subs = [d.name for d in folder.iterdir() if d.is_dir()]
                if any("Race" in s for s in subs):
                    # Match by checking round via schedule is expensive; count folder
                    pass
            # Match event folders by date substring from EventDate
            event_date = pd.Timestamp(row["EventDate"]).strftime("%Y-%m-%d")
            matches = [d for d in cache_year.iterdir() if d.is_dir() and event_date in d.name]
            if matches:
                event_dir = matches[0]
                subs = [d.name for d in event_dir.iterdir() if d.is_dir()]
                has_r = any("Race" in s for s in subs)
                has_q = any("Qualifying" in s for s in subs)

        if has_r:
            cached_race += 1
        if has_q:
            cached_quali += 1
        per_event.append(
            {
                "round": rnd,
                "race_name": name,
                "race_cached": has_r,
                "qualifying_cached": has_q,
            }
        )

    total = len(events)
    return {
        "season": season,
        "total_events": total,
        "race_cached": cached_race,
        "qualifying_cached": cached_quali,
        "events": per_event,
    }
