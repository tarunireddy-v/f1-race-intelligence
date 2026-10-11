"""
Current-weekend session analysis (FP1, Sprint Qualifying, Sprint, GP Qualifying).

Reuses filtering ideas from practice_analysis; not fed into historical ML training.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import (
    _valid_laps,
    get_event_schedule,
    get_grand_prix_events,
    load_session,
)
from src.practice_analysis import (
    DISRUPTED_TRACK_STATUSES,
    SessionConditions,
    _empty_long_run,
    analyze_fp2_long_run,
    inspect_session_conditions,
    normalize_constructor,
)
from src.singapore_config import IST, SINGAPORE_ROUND, SINGAPORE_SEASON

logger = logging.getLogger(__name__)

SQ_NOT_SUNDAY_GRID_NOTE = (
    "Sprint Qualifying sets the Sprint starting order only — not Sunday Grand Prix "
    "qualifying or the race grid."
)

SPRINT_NOT_SUNDAY_GP_NOTE = (
    "Sprint race classification only — not Sunday Grand Prix qualifying, grid, or race result."
)

GP_QUAL_NOT_SQ_OR_SPRINT_NOTE = (
    "Grand Prix qualifying sets the Sunday race grid — not Sprint Qualifying (SQ) or the Sprint (S)."
)

DAYTIME_FP1_NIGHT_RACE_NOTE = (
    "FP1 is a daytime session at Singapore; lap times are not directly comparable "
    "to night-time Sprint Qualifying, GP qualifying, or the race."
)


@dataclass
class SessionRetrievalStatus:
    season: int
    round: int
    session_type: str
    retrieved_at_ist: str
    success: bool
    driver_count: int
    message: str
    data_source: str = "fastf1"


def _ist_now_str() -> str:
    return datetime.now(IST).isoformat()


def resolve_sprint_qualifying_session_code(
    season: int,
    round_number: int,
    event_name: str | None = None,
) -> tuple[str, str]:
    """
    Resolve FastF1 session code from schedule metadata (Session1–Session5 labels).

    Returns (fastf1_code, schedule_label).
    """
    schedule = get_event_schedule(season)
    gps = get_grand_prix_events(schedule)
    row = gps[gps["RoundNumber"] == round_number]
    if row.empty and event_name:
        row = gps[gps["EventName"].str.contains(event_name, case=False, na=False)]
    if row.empty:
        raise ValueError(f"No schedule row for season={season} round={round_number}")
    ev = row.iloc[0]
    for i in range(1, 6):
        label = ev.get(f"Session{i}")
        if label and "sprint" in str(label).lower() and "qualif" in str(label).lower():
            return "SQ", str(label)
    raise ValueError("Sprint Qualifying not found on event schedule")


def resolve_sprint_race_session_code(
    season: int,
    round_number: int,
    event_name: str | None = None,
) -> tuple[str, str]:
    """
    Resolve FastF1 Sprint *race* session code from schedule (not Sprint Qualifying).

    Returns (fastf1_code, schedule_label). Code is typically ``S``.
    """
    schedule = get_event_schedule(season)
    gps = get_grand_prix_events(schedule)
    row = gps[gps["RoundNumber"] == round_number]
    if row.empty and event_name:
        row = gps[gps["EventName"].str.contains(event_name, case=False, na=False)]
    if row.empty:
        raise ValueError(f"No schedule row for season={season} round={round_number}")
    ev = row.iloc[0]
    for i in range(1, 6):
        label = ev.get(f"Session{i}")
        if not label:
            continue
        low = str(label).lower()
        if "sprint" in low and "qualif" not in low:
            return "S", str(label)
    raise ValueError("Sprint race session not found on event schedule")


def resolve_gp_qualifying_session_code(
    season: int,
    round_number: int,
    event_name: str | None = None,
) -> tuple[str, str]:
    """
    Resolve FastF1 Grand Prix qualifying code from schedule (not SQ).

    Returns (fastf1_code, schedule_label). Code is typically ``Q``.
    """
    schedule = get_event_schedule(season)
    gps = get_grand_prix_events(schedule)
    row = gps[gps["RoundNumber"] == round_number]
    if row.empty and event_name:
        row = gps[gps["EventName"].str.contains(event_name, case=False, na=False)]
    if row.empty:
        raise ValueError(f"No schedule row for season={season} round={round_number}")
    ev = row.iloc[0]
    for i in range(1, 6):
        label = ev.get(f"Session{i}")
        if not label:
            continue
        low = str(label).lower()
        if "qualif" in low and "sprint" not in low:
            return "Q", str(label)
    raise ValueError("Grand Prix qualifying session not found on event schedule")


def try_load_session(
    season: int,
    round_number: int,
    session_type: str,
    *,
    load_messages: bool = False,
):
    """Load session; return (session, status)."""
    session = load_session(
        season,
        round_number,
        session_type,
        load_messages=load_messages,
    )
    if session is None:
        return None, SessionRetrievalStatus(
            season=season,
            round=round_number,
            session_type=session_type,
            retrieved_at_ist=_ist_now_str(),
            success=False,
            driver_count=0,
            message="load_session returned None",
        )
    results = session.results
    n = len(results) if results is not None else 0
    if n == 0:
        return session, SessionRetrievalStatus(
            season=season,
            round=round_number,
            session_type=session_type,
            retrieved_at_ist=_ist_now_str(),
            success=False,
            driver_count=0,
            message="No results in FastF1 (session may not be published yet)",
        )
    return session, SessionRetrievalStatus(
        season=season,
        round=round_number,
        session_type=session_type,
        retrieved_at_ist=_ist_now_str(),
        success=True,
        driver_count=n,
        message="OK",
    )


def _best_lap_table(session, session_label: str) -> pd.DataFrame:
    valid = _valid_laps(session.laps)
    if valid.empty:
        return pd.DataFrame()

    # Exclude disrupted track-status laps when column present
    if "TrackStatus" in valid.columns:
        valid = valid[~valid["TrackStatus"].astype(str).isin(DISRUPTED_TRACK_STATUSES)]

    best = (
        valid.groupby("Driver")
        .agg(
            best_lap_s=("LapTime (s)", "min"),
            laps_valid=("LapTime (s)", "count"),
        )
        .reset_index()
        .rename(columns={"Driver": "driver"})
    )
    pole = best["best_lap_s"].min()
    best["gap_to_fastest_s"] = best["best_lap_s"] - pole
    best[f"{session_label.lower()}_rank"] = best["best_lap_s"].rank(method="min").astype(int)

    # Sectors on best lap per driver
    for sec in (1, 2, 3):
        col = f"Sector{sec}Time"
        if col not in valid.columns:
            continue
        sec_best = []
        for drv in best["driver"]:
            dl = valid[valid["Driver"] == drv]
            if dl.empty:
                sec_best.append(np.nan)
                continue
            idx = dl["LapTime (s)"].idxmin()
            t = dl.loc[idx, col]
            sec_best.append(t.total_seconds() if hasattr(t, "total_seconds") else np.nan)
        best[f"{session_label.lower()}_s{sec}_best_s"] = sec_best

    return best


def session_weather_summary(session) -> dict[str, Any]:
    """Aggregate session weather if FastF1 weather data was loaded."""
    out: dict[str, Any] = {"weather_available": False}
    try:
        w = session.weather_data
    except Exception:
        return out
    if w is None or w.empty:
        return out
    out["weather_available"] = True
    out["label"] = "OBSERVED_SESSION"
    for col, key in [
        ("AirTemp", "air_temp_avg_c"),
        ("TrackTemp", "track_temp_avg_c"),
        ("Humidity", "humidity_avg_pct"),
        ("WindSpeed", "wind_speed_avg"),
    ]:
        if col in w.columns:
            out[key] = float(w[col].mean())
    if "Rainfall" in w.columns:
        out["rainfall_occurred"] = bool(w["Rainfall"].astype(bool).any())
    return out


def build_fp1_analysis_table(
    season: int = SINGAPORE_SEASON,
    round_number: int = SINGAPORE_ROUND,
) -> tuple[pd.DataFrame, SessionRetrievalStatus, dict[str, Any], SessionConditions | None]:
    """
    FP1 timing + long-run + weather for one event.

    Returns empty table if session unavailable.
    """
    session, status = try_load_session(season, round_number, "FP1")
    if not status.success or session is None:
        return pd.DataFrame(), status, {}, None

    # Weather requires weather=True on load — reload with weather if needed
    try:
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    except Exception as exc:
        logger.warning("FP1 weather reload failed: %s", exc)

    results = session.results
    valid = _valid_laps(session.laps)
    drivers = []
    for _, row in results.iterrows():
        drivers.append(
            {
                "driver": str(row["Abbreviation"]),
                "constructor": normalize_constructor(row.get("TeamName")),
            }
        )
    base = pd.DataFrame(drivers).drop_duplicates(subset=["driver"])

    best = _best_lap_table(session, "FP1")
    if best.empty:
        merged = base
    else:
        merged = base.merge(best, on="driver", how="left")

    lr_map = analyze_fp2_long_run(session)
    lr_rows = []
    for drv in merged["driver"]:
        lr = lr_map.get(drv, _empty_long_run())
        lr_rows.append(
            {
                "driver": drv,
                "fp1_long_run_avg_s": lr.long_run_avg_lap_s,
                "fp1_long_run_laps": lr.long_run_laps,
                "fp1_long_run_reliability": lr.reliability,
            }
        )
    merged = merged.merge(pd.DataFrame(lr_rows), on="driver", how="left")

    conditions = inspect_session_conditions(session, "FP1")
    weather = session_weather_summary(session)
    weather["daytime_session_note"] = DAYTIME_FP1_NIGHT_RACE_NOTE
    weather["session"] = "FP1"

    return merged, status, weather, conditions


def _td_to_seconds(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    return float(value)


def _best_session_lap(q1, q2, q3) -> float | None:
    times = [_td_to_seconds(q1), _td_to_seconds(q2), _td_to_seconds(q3)]
    valid = [t for t in times if t is not None]
    return min(valid) if valid else None


def _sq_stage(q1, q2, q3) -> str | None:
    if _td_to_seconds(q3) is not None:
        return "SQ3"
    if _td_to_seconds(q2) is not None:
        return "SQ2"
    if _td_to_seconds(q1) is not None:
        return "SQ1"
    return None


def _gp_quali_stage(q1, q2, q3) -> str | None:
    if _td_to_seconds(q3) is not None:
        return "Q3"
    if _td_to_seconds(q2) is not None:
        return "Q2"
    if _td_to_seconds(q1) is not None:
        return "Q1"
    return None


def build_sprint_qualifying_table(
    season: int = SINGAPORE_SEASON,
    round_number: int = SINGAPORE_ROUND,
    event_name: str = "Singapore Grand Prix",
) -> tuple[pd.DataFrame, SessionRetrievalStatus, dict[str, Any], str]:
    """
    Sprint Qualifying classification and pace (NOT Grand Prix qualifying).

    Requires ``messages=True`` so FastF1 can apply deleted-lap logic for SQ results.
    """
    sq_code, schedule_label = resolve_sprint_qualifying_session_code(
        season, round_number, event_name
    )
    session, status = try_load_session(
        season,
        round_number,
        sq_code,
        load_messages=True,
    )
    status.session_type = f"{sq_code} ({schedule_label})"

    if session is None:
        return pd.DataFrame(), status, {}, sq_code

    results = session.results
    n_drv = len(results) if results is not None else 0
    if results is None or results.empty or results["Position"].isna().all():
        status = SessionRetrievalStatus(
            season=season,
            round=round_number,
            session_type=status.session_type,
            retrieved_at_ist=_ist_now_str(),
            success=False,
            driver_count=len(results) if results is not None else 0,
            message="SQ results unavailable (load with messages or timing not published)",
        )
        return pd.DataFrame(), status, {}, sq_code

    status = SessionRetrievalStatus(
        season=season,
        round=round_number,
        session_type=status.session_type,
        retrieved_at_ist=_ist_now_str(),
        success=True,
        driver_count=n_drv,
        message="OK",
    )

    laps = _valid_laps(session.laps)
    if "TrackStatus" in laps.columns:
        laps = laps[~laps["TrackStatus"].astype(str).isin(DISRUPTED_TRACK_STATUSES)]

    rows: list[dict] = []
    for _, res in results.iterrows():
        driver = str(res["Abbreviation"])
        pos = res.get("Position")
        q1_s = _td_to_seconds(res.get("Q1"))
        q2_s = _td_to_seconds(res.get("Q2"))
        q3_s = _td_to_seconds(res.get("Q3"))
        best_lap_s = _best_session_lap(res.get("Q1"), res.get("Q2"), res.get("Q3"))
        if best_lap_s is None and not laps.empty:
            dl = laps[laps["Driver"] == driver]
            if not dl.empty:
                best_lap_s = float(dl["LapTime (s)"].min())

        s1 = s2 = s3 = None
        if not laps.empty:
            dl = laps[laps["Driver"] == driver]
            if not dl.empty:
                best_idx = dl["LapTime (s)"].idxmin()
                br = dl.loc[best_idx]
                for sec, attr in [(1, "s1"), (2, "s2"), (3, "s3")]:
                    col = f"Sector{sec}Time"
                    if col in br.index:
                        val = _td_to_seconds(br[col])
                        if sec == 1:
                            s1 = val
                        elif sec == 2:
                            s2 = val
                        else:
                            s3 = val

        rows.append(
            {
                "driver": driver,
                "constructor": normalize_constructor(res.get("TeamName")),
                "sq_classification_position": int(pos) if pd.notna(pos) else np.nan,
                "sq_best_lap_s": best_lap_s,
                "sq_q1_best_s": q1_s,
                "sq_q2_best_s": q2_s,
                "sq_q3_best_s": q3_s,
                "driver_sq_stage": _sq_stage(res.get("Q1"), res.get("Q2"), res.get("Q3")),
                "sq_s1_best_s": s1,
                "sq_s2_best_s": s2,
                "sq_s3_best_s": s3,
                "sq_grid_position_fastf1": res.get("GridPosition"),
                "sq_not_sunday_grid_note": SQ_NOT_SUNDAY_GRID_NOTE,
            }
        )

    df = pd.DataFrame(rows)
    if df["sq_best_lap_s"].notna().any():
        pole = df["sq_best_lap_s"].min()
        df["sq_gap_to_fastest_s"] = df["sq_best_lap_s"] - pole
        df["sq_pace_rank"] = df["sq_best_lap_s"].rank(method="min")

    # Official classification rank (may differ from pace if deleted laps / rulings)
    if df["sq_classification_position"].notna().any():
        df["sq_official_rank"] = df["sq_classification_position"]
    else:
        df["sq_official_rank"] = df.get("sq_pace_rank")

    # Teammate gap on best SQ lap
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
        if pd.notna(row["sq_best_lap_s"]) and pd.notna(mate["sq_best_lap_s"]):
            tm_gaps.append(row["sq_best_lap_s"] - mate["sq_best_lap_s"])
        else:
            tm_gaps.append(None)
        tm_stages.append(mate.get("driver_sq_stage"))
    df["sq_teammate_gap_s"] = tm_gaps
    df["teammate_sq_stage"] = tm_stages

    for sec, col in [(1, "sq_s1"), (2, "sq_s2"), (3, "sq_s3")]:
        best_col = f"{col}_best_s"
        if best_col not in df.columns:
            continue
        valid = df[best_col].notna()
        if valid.any():
            best = df.loc[valid, best_col].min()
            df[f"{col}_gap_to_best_s"] = np.where(valid, df[best_col] - best, np.nan)
            df[f"{col}_rank"] = np.where(valid, df[best_col].rank(method="min"), np.nan)

    weather = session_weather_summary(session)
    weather["session"] = schedule_label
    weather["fastf1_session_code"] = sq_code
    weather["sq_night_session_note"] = (
        "Sprint Qualifying is a night session; not directly comparable to daytime FP1."
    )

    conditions = inspect_session_conditions(session, schedule_label)
    if conditions.notes:
        weather["session_conditions_notes"] = "; ".join(conditions.notes)

    return df.sort_values("sq_official_rank").reset_index(drop=True), status, weather, sq_code


def _session_metadata_matches(
    session,
    season: int,
    round_number: int,
    event_name: str | None,
) -> tuple[bool, str]:
    event = session.event
    meta_round = int(event["RoundNumber"])
    meta_name = str(event["EventName"])
    if meta_round != round_number or int(event.get("Year", season)) != season:
        return False, f"metadata mismatch season={season} round={round_number} vs {meta_name} R{meta_round}"
    if event_name and event_name.lower() not in meta_name.lower():
        return False, f"event name mismatch expected '{event_name}' got '{meta_name}'"
    return True, meta_name


def _sprint_results_complete(results: pd.DataFrame) -> tuple[bool, str]:
    if results is None or results.empty:
        return False, "empty results table"
    if results["Position"].notna().sum() < 5:
        return False, "too few classified finishing positions (session may be in progress)"
    return True, "OK"


def _time_column_to_label(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value):
        return None
    return str(value)


def _gap_to_winner_seconds(results: pd.DataFrame, driver: str) -> float | None:
    """Parse gap to race winner when FastF1 provides a timedelta offset."""
    if results is None or results.empty:
        return None
    winner = results[results["Position"] == 1]
    if winner.empty:
        return None
    win_time = winner.iloc[0].get("Time")
    row = results[results["Abbreviation"] == driver]
    if row.empty:
        return None
    t = row.iloc[0].get("Time")
    if t is None or pd.isna(t):
        return None
    if hasattr(t, "total_seconds"):
        if hasattr(win_time, "total_seconds") and int(results.loc[row.index[0], "Position"]) == 1:
            return 0.0
        return float(t.total_seconds())
    return None


def _dnf_from_status(status) -> bool:
    if status is None or (isinstance(status, float) and np.isnan(status)):
        return False
    if pd.isna(status):
        return False
    s = str(status).strip()
    if not s:
        return False
    if s == "Finished":
        return False
    if s.startswith("+") and "Lap" in s:
        return False
    lowered = s.lower()
    if lowered in ("did not start", "dns", "disqualified", "dsq"):
        return True
    if any(k in lowered for k in ("retired", "accident", "collision", "engine", "gearbox", "hydraulic")):
        return True
    return False


def _sq_sprint_grid_positions(
    season: int,
    round_number: int,
) -> dict[str, int]:
    """Sprint starting grid from SQ official classification (FastF1 SQ session)."""
    session, status = try_load_session(season, round_number, "SQ", load_messages=True)
    if not status.success or session is None or session.results is None:
        return {}
    out: dict[str, int] = {}
    for _, res in session.results.iterrows():
        pos = res.get("Position")
        if pd.notna(pos):
            out[str(res["Abbreviation"])] = int(pos)
    return out


def _race_control_penalty_notes(session, driver: str, driver_number: str | int | None) -> str | None:
    try:
        rc = session.race_control_messages
    except Exception:
        return None
    if rc is None or rc.empty:
        return None
    notes: list[str] = []
    num = str(driver_number) if driver_number is not None and not pd.isna(driver_number) else ""
    for _, msg in rc.iterrows():
        category = str(msg.get("Category", "")).lower()
        text = str(msg.get("Message", ""))
        if "penalty" not in category and "penalty" not in text.lower():
            continue
        if driver in text or (num and num in text):
            notes.append(text.strip())
    if not notes:
        return None
    return "; ".join(dict.fromkeys(notes[:5]))


def build_sprint_race_table(
    season: int = SINGAPORE_SEASON,
    round_number: int = SINGAPORE_ROUND,
    event_name: str = "Singapore Grand Prix",
) -> tuple[pd.DataFrame, SessionRetrievalStatus, dict[str, Any], str, dict[str, Any]]:
    """
    Sprint race results (session ``S``) — not Sprint Qualifying or Grand Prix qualifying.

    Uses official FastF1 classification; penalties noted only from race-control messages.
    """
    sprint_code, schedule_label = resolve_sprint_race_session_code(
        season, round_number, event_name
    )
    session, status = try_load_session(
        season,
        round_number,
        sprint_code,
        load_messages=True,
    )
    status.session_type = f"{sprint_code} ({schedule_label})"
    verification: dict[str, Any] = {
        "fastf1_session_code": sprint_code,
        "schedule_session_label": schedule_label,
        "session_complete": False,
    }

    if session is None:
        return pd.DataFrame(), status, {}, sprint_code, verification

    ok_meta, meta_msg = _session_metadata_matches(session, season, round_number, event_name)
    verification["verified_event_name"] = meta_msg if ok_meta else None
    verification["verified_round"] = int(session.event["RoundNumber"])
    if not ok_meta:
        status = SessionRetrievalStatus(
            season=season,
            round=round_number,
            session_type=status.session_type,
            retrieved_at_ist=_ist_now_str(),
            success=False,
            driver_count=0,
            message=f"Event metadata check failed: {meta_msg}",
        )
        return pd.DataFrame(), status, {}, sprint_code, verification

    results = session.results
    n_drv = len(results) if results is not None else 0
    complete, complete_msg = _sprint_results_complete(results)
    verification["session_complete"] = complete
    if not complete:
        status = SessionRetrievalStatus(
            season=season,
            round=round_number,
            session_type=status.session_type,
            retrieved_at_ist=_ist_now_str(),
            success=False,
            driver_count=n_drv,
            message=f"Sprint results incomplete: {complete_msg}",
        )
        return pd.DataFrame(), status, {}, sprint_code, verification

    status = SessionRetrievalStatus(
        season=season,
        round=round_number,
        session_type=status.session_type,
        retrieved_at_ist=_ist_now_str(),
        success=True,
        driver_count=n_drv,
        message="OK",
    )

    sq_grid = _sq_sprint_grid_positions(season, round_number)

    laps = _valid_laps(session.laps)
    rows: list[dict] = []
    for _, res in results.iterrows():
        driver = str(res["Abbreviation"])
        finish = res.get("Position")
        if pd.isna(finish) and pd.notna(res.get("ClassifiedPosition")):
            finish = res.get("ClassifiedPosition")
        grid = res.get("GridPosition")
        grid_source = "fastf1_sprint_session"
        if pd.isna(grid) and driver in sq_grid:
            grid = sq_grid[driver]
            grid_source = "sq_official_classification"
        status_txt = res.get("Status")
        if status_txt is None or (isinstance(status_txt, str) and not str(status_txt).strip()):
            status_txt = "Finished" if pd.notna(finish) else None
        gained = None
        if pd.notna(finish) and pd.notna(grid):
            gained = int(grid) - int(finish)

        best_lap_s = None
        if not laps.empty:
            dl = laps[laps["Driver"] == driver]
            if not dl.empty:
                best_lap_s = float(dl["LapTime (s)"].min())

        rows.append(
            {
                "driver": driver,
                "constructor": normalize_constructor(res.get("TeamName")),
                "sprint_finish_position": int(finish) if pd.notna(finish) else np.nan,
                "sprint_official_rank": int(finish) if pd.notna(finish) else np.nan,
                "sprint_classified_status": status_txt,
                "sprint_points": res.get("Points"),
                "sprint_grid_position": int(grid) if pd.notna(grid) else np.nan,
                "sprint_grid_source": grid_source if pd.notna(grid) else None,
                "sprint_positions_gained_vs_grid": gained,
                "sprint_time_label": _time_column_to_label(res.get("Time")),
                "sprint_gap_to_winner_s": _gap_to_winner_seconds(results, driver),
                "sprint_best_lap_s": best_lap_s,
                "sprint_laps_completed": res.get("Laps"),
                "sprint_dnf": _dnf_from_status(status_txt),
                "sprint_race_penalty_notes": _race_control_penalty_notes(
                    session,
                    driver,
                    res.get("DriverNumber"),
                ),
                "sprint_is_not_sunday_gp_note": SPRINT_NOT_SUNDAY_GP_NOTE,
            }
        )

    df = pd.DataFrame(rows)
    if df["sprint_best_lap_s"].notna().any():
        fastest = df["sprint_best_lap_s"].min()
        df["sprint_gap_to_fastest_lap_s"] = df["sprint_best_lap_s"] - fastest

    weather = session_weather_summary(session)
    try:
        session.load(laps=True, telemetry=False, weather=True, messages=True)
        weather = session_weather_summary(session)
    except Exception as exc:
        logger.warning("Sprint weather reload failed: %s", exc)
    weather["session"] = schedule_label
    weather["fastf1_session_code"] = sprint_code
    weather["sprint_not_sunday_gp_note"] = SPRINT_NOT_SUNDAY_GP_NOTE

    conditions = inspect_session_conditions(session, schedule_label)
    if conditions.notes:
        weather["session_conditions_notes"] = "; ".join(conditions.notes)

    return (
        df.sort_values("sprint_official_rank", na_position="last").reset_index(drop=True),
        status,
        weather,
        sprint_code,
        verification,
    )


def _gp_qual_results_complete(results: pd.DataFrame) -> tuple[bool, str]:
    if results is None or results.empty:
        return False, "empty results table"
    if results["Position"].notna().sum() < 5:
        return False, "too few classified qualifying positions (session may be in progress)"
    return True, "OK"


def build_gp_qualifying_table(
    season: int = SINGAPORE_SEASON,
    round_number: int = SINGAPORE_ROUND,
    event_name: str = "Singapore Grand Prix",
) -> tuple[pd.DataFrame, SessionRetrievalStatus, dict[str, Any], str, dict[str, Any]]:
    """
    Grand Prix qualifying (session ``Q``) — not Sprint Qualifying or Sprint.

    ``gp_qualifying_position`` is official session classification (pre-grid penalties).
    ``gp_grid_position`` uses FastF1 ``GridPosition`` when published, else qualifying position.
    """
    q_code, schedule_label = resolve_gp_qualifying_session_code(season, round_number, event_name)
    session, status = try_load_session(
        season,
        round_number,
        q_code,
        load_messages=True,
    )
    status.session_type = f"{q_code} ({schedule_label})"
    verification: dict[str, Any] = {
        "fastf1_session_code": q_code,
        "schedule_session_label": schedule_label,
        "session_complete": False,
        "distinct_from_sq": True,
        "distinct_from_sprint": True,
    }

    if session is None:
        return pd.DataFrame(), status, {}, q_code, verification

    ok_meta, meta_msg = _session_metadata_matches(session, season, round_number, event_name)
    verification["verified_event_name"] = meta_msg if ok_meta else None
    verification["verified_round"] = int(session.event["RoundNumber"])
    if not ok_meta:
        status = SessionRetrievalStatus(
            season=season,
            round=round_number,
            session_type=status.session_type,
            retrieved_at_ist=_ist_now_str(),
            success=False,
            driver_count=0,
            message=f"Event metadata check failed: {meta_msg}",
        )
        return pd.DataFrame(), status, {}, q_code, verification

    results = session.results
    n_drv = len(results) if results is not None else 0
    complete, complete_msg = _gp_qual_results_complete(results)
    verification["session_complete"] = complete
    if not complete:
        status = SessionRetrievalStatus(
            season=season,
            round=round_number,
            session_type=status.session_type,
            retrieved_at_ist=_ist_now_str(),
            success=False,
            driver_count=n_drv,
            message=f"GP qualifying results incomplete: {complete_msg}",
        )
        return pd.DataFrame(), status, {}, q_code, verification

    status = SessionRetrievalStatus(
        season=season,
        round=round_number,
        session_type=status.session_type,
        retrieved_at_ist=_ist_now_str(),
        success=True,
        driver_count=n_drv,
        message="OK",
    )

    laps = _valid_laps(session.laps)
    if "TrackStatus" in laps.columns:
        laps = laps[~laps["TrackStatus"].astype(str).isin(DISRUPTED_TRACK_STATUSES)]

    rows: list[dict] = []
    for _, res in results.iterrows():
        driver = str(res["Abbreviation"])
        quali_pos = res.get("Position")
        q1_s = _td_to_seconds(res.get("Q1"))
        q2_s = _td_to_seconds(res.get("Q2"))
        q3_s = _td_to_seconds(res.get("Q3"))
        best_lap_s = _best_session_lap(res.get("Q1"), res.get("Q2"), res.get("Q3"))
        if best_lap_s is None and not laps.empty:
            dl = laps[laps["Driver"] == driver]
            if not dl.empty:
                best_lap_s = float(dl["LapTime (s)"].min())

        s1 = s2 = s3 = None
        if not laps.empty:
            dl = laps[laps["Driver"] == driver]
            if not dl.empty:
                best_idx = dl["LapTime (s)"].idxmin()
                br = dl.loc[best_idx]
                for sec, attr in [(1, "s1"), (2, "s2"), (3, "s3")]:
                    col = f"Sector{sec}Time"
                    if col in br.index:
                        val = _td_to_seconds(br[col])
                        if sec == 1:
                            s1 = val
                        elif sec == 2:
                            s2 = val
                        else:
                            s3 = val

        grid = res.get("GridPosition")
        grid_source = "fastf1_gp_qual_session"
        if pd.isna(grid) and pd.notna(quali_pos):
            grid = quali_pos
            grid_source = "qualifying_position_when_grid_missing"
        penalty_notes = _race_control_penalty_notes(session, driver, res.get("DriverNumber"))
        grid_pen = None
        has_doc_penalty = False
        if pd.notna(quali_pos) and pd.notna(grid):
            grid_pen = int(grid) - int(quali_pos)
            has_doc_penalty = bool(penalty_notes) and grid_pen > 0

        rows.append(
            {
                "driver": driver,
                "constructor": normalize_constructor(res.get("TeamName")),
                "gp_qualifying_position": int(quali_pos) if pd.notna(quali_pos) else np.nan,
                "gp_qualifying_best_lap_s": best_lap_s,
                "gp_q1_best_s": q1_s,
                "gp_q2_best_s": q2_s,
                "gp_q3_best_s": q3_s,
                "driver_gp_quali_stage": _gp_quali_stage(res.get("Q1"), res.get("Q2"), res.get("Q3")),
                "gp_quali_s1_best_s": s1,
                "gp_quali_s2_best_s": s2,
                "gp_quali_s3_best_s": s3,
                "gp_grid_position": int(grid) if pd.notna(grid) else np.nan,
                "gp_grid_source": grid_source if pd.notna(grid) else None,
                "gp_grid_penalty_positions": grid_pen,
                "gp_has_documented_grid_penalty": has_doc_penalty,
                "gp_quali_penalty_notes": penalty_notes,
                "gp_qual_not_sq_or_sprint_note": GP_QUAL_NOT_SQ_OR_SPRINT_NOTE,
            }
        )

    df = pd.DataFrame(rows)
    if df["gp_qualifying_best_lap_s"].notna().any():
        pole = df["gp_qualifying_best_lap_s"].min()
        df["gp_quali_gap_to_pole_s"] = df["gp_qualifying_best_lap_s"] - pole

    if df["gp_qualifying_position"].notna().any():
        df["gp_quali_official_rank"] = df["gp_qualifying_position"]
    else:
        df["gp_quali_official_rank"] = df["gp_qualifying_best_lap_s"].rank(method="min")

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
        if pd.notna(row["gp_qualifying_best_lap_s"]) and pd.notna(mate["gp_qualifying_best_lap_s"]):
            tm_gaps.append(row["gp_qualifying_best_lap_s"] - mate["gp_qualifying_best_lap_s"])
        else:
            tm_gaps.append(None)
        tm_stages.append(mate.get("driver_gp_quali_stage"))
    df["gp_teammate_quali_gap_s"] = tm_gaps
    df["teammate_gp_quali_stage"] = tm_stages

    for sec, col in [(1, "gp_quali_s1"), (2, "gp_quali_s2"), (3, "gp_quali_s3")]:
        best_col = f"{col}_best_s"
        if best_col not in df.columns:
            continue
        valid = df[best_col].notna()
        if valid.any():
            best = df.loc[valid, best_col].min()
            df[f"{col}_gap_to_best_s"] = np.where(valid, df[best_col] - best, np.nan)
            df[f"{col}_rank"] = np.where(valid, df[best_col].rank(method="min"), np.nan)

    weather = session_weather_summary(session)
    try:
        session.load(laps=True, telemetry=False, weather=True, messages=True)
        weather = session_weather_summary(session)
    except Exception as exc:
        logger.warning("GP quali weather reload failed: %s", exc)
    weather["session"] = schedule_label
    weather["fastf1_session_code"] = q_code
    weather["gp_qual_not_sq_note"] = GP_QUAL_NOT_SQ_OR_SPRINT_NOTE

    conditions = inspect_session_conditions(session, schedule_label)
    if conditions.notes:
        weather["session_conditions_notes"] = "; ".join(conditions.notes)

    return (
        df.sort_values("gp_quali_official_rank").reset_index(drop=True),
        status,
        weather,
        q_code,
        verification,
    )


def save_retrieval_status(path, status: SessionRetrievalStatus, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**asdict(status), **extra}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
