"""
Current-weekend FP1/FP2 practice analysis (V1 analytical layer).

NOT used as ML training features. Separate signal alongside the baseline model.

Hungary 2026 entry point:
    python -m src.practice_analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, TARGET_EVENT, TARGET_SEASON
from src.data_loader import (
    _valid_laps,
    enable_cache,
    get_event_round,
    get_event_schedule,
    load_session,
)
from src.features import CONSTRUCTOR_NORMALIZATION

logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRACTICE_OUTPUT_FILE = "hungary_2026_practice_analysis.csv"

MIN_LONG_RUN_LAPS = 4
# Within a stint, drop laps more than this many seconds SLOWER than the stint
# median. Catches red-flag crawl laps and obvious interruptions without
# removing valid push laps that happen to be faster than the median.
STINT_SLOW_LAP_THRESHOLD_S = 3.0

# Track-status codes (FastF1) that indicate disrupted running.
# 2=Yellow, 5=Red, 6=Safety Car, 7=VSC — laps set under these are excluded.
DISRUPTED_TRACK_STATUSES = frozenset({"2", "5", "6", "7"})

RELIABILITY_HIGH_MIN_LAPS = 8
RELIABILITY_HIGH_MAX_STD = 1.0
RELIABILITY_MEDIUM_MIN_LAPS = 5
RELIABILITY_MEDIUM_MAX_STD = 1.5


@dataclass
class LongRunResult:
    long_run_avg_lap_s: float | None
    long_run_median_lap_s: float | None
    long_run_laps: int | None
    long_run_std_s: float | None
    reliability: str  # HIGH | MEDIUM | LOW | UNAVAILABLE — descriptive only, not for ML


@dataclass
class SessionConditions:
    session: str
    red_flags: int
    yellow_flags: int
    aborted: bool
    notes: list[str]


def normalize_constructor(name: str | None) -> str | None:
    if name is None or pd.isna(name):
        return name
    return CONSTRUCTOR_NORMALIZATION.get(str(name), str(name))


def _official_entry_list(*sessions) -> list[str]:
    """
    Canonical driver codes from session results (prefer latest session).

    Uses lap ``Driver`` codes matched by ``DriverNumber`` when available, because
    FastF1 ``results.Abbreviation`` can disagree with timing data for the same car.
    """
    for session in sessions:  # caller passes newest session first (FP3, FP2, FP1)
        if session.results is None or session.results.empty:
            continue
        res = session.results
        valid = _valid_laps(session.laps)
        lap_by_number: dict[str, str] = {}
        if not valid.empty and "DriverNumber" in valid.columns:
            for num, group in valid.groupby("DriverNumber"):
                lap_by_number[str(int(num))] = str(group["Driver"].iloc[0])

        drivers: list[str] = []
        for _, row in res.iterrows():
            num = str(int(row["DriverNumber"]))
            if num in lap_by_number:
                drivers.append(lap_by_number[num])
            elif pd.notna(row.get("Abbreviation")):
                drivers.append(str(row["Abbreviation"]))
        if drivers:
            return sorted(set(drivers))
    return []


def _driver_constructor_map(session) -> dict[str, str]:
    """Map driver code → constructor from session results (via driver number)."""
    mapping: dict[str, str] = {}
    if session.results is None or session.results.empty:
        return mapping

    valid = _valid_laps(session.laps)
    lap_by_number: dict[str, str] = {}
    if not valid.empty and "DriverNumber" in valid.columns:
        for num, group in valid.groupby("DriverNumber"):
            lap_by_number[str(int(num))] = str(group["Driver"].iloc[0])

    for _, row in session.results.iterrows():
        num = str(int(row["DriverNumber"]))
        driver = lap_by_number.get(num, str(row.get("Abbreviation")))
        team = row.get("TeamName")
        if pd.notna(driver) and pd.notna(team):
            mapping[str(driver)] = normalize_constructor(str(team))
    return mapping


def _exclude_pit_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiable pit-in and pit-out laps."""
    out = laps.copy()
    if "PitOutTime" in out.columns:
        out = out[out["PitOutTime"].isna()]
    if "PitInTime" in out.columns:
        out = out[out["PitInTime"].isna()]
    return out


def _exclude_disrupted_laps(laps: pd.DataFrame, session) -> pd.DataFrame:
    """
    Exclude laps completed under red flag, yellow, SC, or VSC track status.

    Uses merge-asof on session time so only clearly affected laps are removed.
    """
    if laps.empty or not hasattr(session, "track_status") or session.track_status is None:
        return laps

    ts = session.track_status.copy()
    ts["Time"] = pd.to_timedelta(ts["Time"])
    ts = ts.sort_values("Time")
    ts["Status"] = ts["Status"].astype(str)

    out = laps.copy()
    out["_lap_time_td"] = pd.to_timedelta(out["Time"])
    merged = pd.merge_asof(
        out.sort_values("_lap_time_td"),
        ts[["Time", "Status"]].rename(columns={"Time": "_status_time", "Status": "_track_status"}),
        left_on="_lap_time_td",
        right_on="_status_time",
        direction="backward",
    )
    keep = ~merged["_track_status"].isin(DISRUPTED_TRACK_STATUSES)
    filtered = merged.loc[keep].drop(columns=["_lap_time_td", "_status_time", "_track_status"])
    return filtered.sort_values(["Driver", "LapNumber"])


def _exclude_slow_outliers(stint_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Drop laps > STINT_SLOW_LAP_THRESHOLD_S slower than the stint median.

    Only removes clearly slow/disrupted laps — unusually fast laps are kept.
    """
    if stint_laps.empty:
        return stint_laps
    median = stint_laps["LapTime (s)"].median()
    cutoff = median + STINT_SLOW_LAP_THRESHOLD_S
    return stint_laps[stint_laps["LapTime (s)"] <= cutoff]


def _filter_representative_laps(stint_laps: pd.DataFrame) -> pd.DataFrame:
    """Pit- and slow-outlier-filtered laps within a stint."""
    filtered = _exclude_pit_laps(stint_laps)
    if filtered.empty:
        return pd.DataFrame()
    filtered = _exclude_slow_outliers(filtered)
    return filtered.sort_values("LapNumber")


def _longest_consecutive_block(laps: pd.DataFrame) -> pd.DataFrame | None:
    """Return the longest block of consecutive lap numbers."""
    if laps.empty:
        return None

    best_block: list[int] = []
    current_block: list[int] = []
    prev_lap: int | None = None

    for idx, row in laps.iterrows():
        lap_no = int(row["LapNumber"])
        if prev_lap is not None and lap_no != prev_lap + 1:
            if len(current_block) > len(best_block):
                best_block = current_block
            current_block = [idx]
        else:
            current_block.append(idx)
        prev_lap = lap_no

    if len(current_block) > len(best_block):
        best_block = current_block

    if len(best_block) < MIN_LONG_RUN_LAPS:
        return None
    return laps.loc[best_block]


def _longest_stint_run(stint_laps: pd.DataFrame) -> pd.DataFrame | None:
    """
    From one stint, return the longest contiguous lap-number sequence of
    representative laps (after pit/outlier filtering), if >= MIN_LONG_RUN_LAPS.
    """
    reps = _filter_representative_laps(stint_laps)
    if reps.empty:
        return None
    return _longest_consecutive_block(reps)


def _reliability_label(lap_count: int, std_s: float) -> str:
    """
    Descriptive confidence label for a long-run sample (not for ML).

    HIGH:   ≥8 laps and σ≤1.0s — stable, long representative run
    MEDIUM: ≥5 laps and σ≤1.5s — usable but shorter or noisier
    LOW:    ≥4 laps but fails HIGH/MEDIUM thresholds
    UNAVAILABLE: fewer than 4 representative laps after all filtering
    """
    if lap_count < MIN_LONG_RUN_LAPS:
        return "UNAVAILABLE"
    if lap_count >= RELIABILITY_HIGH_MIN_LAPS and std_s <= RELIABILITY_HIGH_MAX_STD:
        return "HIGH"
    if lap_count >= RELIABILITY_MEDIUM_MIN_LAPS and std_s <= RELIABILITY_MEDIUM_MAX_STD:
        return "MEDIUM"
    return "LOW"


def _empty_long_run() -> LongRunResult:
    return LongRunResult(None, None, None, None, "UNAVAILABLE")


def analyze_fp2_long_run(session) -> dict[str, LongRunResult]:
    """
    Approximate FP2 race-pace signal from longest valid long run per driver.

    Labelled as practice signal only — fuel loads and run plans vary.
    Laps under flags and laps >3s slower than stint median are excluded.
    """
    valid = _valid_laps(session.laps)
    valid = _exclude_disrupted_laps(valid, session)
    if valid.empty:
        return {}

    results: dict[str, LongRunResult] = {}
    for driver, driver_laps in valid.groupby("Driver"):
        best_run: pd.DataFrame | None = None
        for _, stint_laps in driver_laps.groupby("Stint"):
            run = _longest_stint_run(stint_laps)
            if run is None:
                continue
            if best_run is None or len(run) > len(best_run):
                best_run = run

        if best_run is None or len(best_run) < MIN_LONG_RUN_LAPS:
            results[str(driver)] = _empty_long_run()
            continue

        times = best_run["LapTime (s)"]
        lap_count = int(len(times))
        std_s = float(times.std(ddof=0)) if lap_count > 1 else 0.0
        reliability = _reliability_label(lap_count, std_s)

        if reliability == "UNAVAILABLE":
            results[str(driver)] = _empty_long_run()
            continue

        results[str(driver)] = LongRunResult(
            long_run_avg_lap_s=float(times.mean()),
            long_run_median_lap_s=float(times.median()),
            long_run_laps=lap_count,
            long_run_std_s=std_s,
            reliability=reliability,
        )
    return results


def extract_session_pace(session, session_label: str) -> pd.DataFrame:
    """
    Single-lap pace from valid laps — rank by best lap, not FastF1 classification.
    """
    valid = _valid_laps(session.laps)
    constructors = _driver_constructor_map(session)

    if valid.empty:
        return pd.DataFrame()

    summary = (
        valid.groupby("Driver")
        .agg(
            best_lap_s=("LapTime (s)", "min"),
            valid_laps=("LapTime (s)", "count"),
        )
        .reset_index()
        .rename(columns={"Driver": "driver"})
    )

    summary["constructor"] = summary["driver"].map(constructors)
    session_fastest = summary["best_lap_s"].min()
    summary["gap_s"] = summary["best_lap_s"] - session_fastest
    summary["rank"] = summary["best_lap_s"].rank(method="min").astype(int)
    summary["session"] = session_label
    return summary.sort_values("rank")


def inspect_session_conditions(session, session_label: str) -> SessionConditions:
    """Summarise flags and interruptions visible in FastF1 session data."""
    notes: list[str] = []
    red_flags = yellow_flags = 0
    aborted = False

    if hasattr(session, "track_status") and session.track_status is not None:
        ts = session.track_status
        red_flags = int((ts["Status"] == "5").sum())
        yellow_flags = int((ts["Status"] == "2").sum())
        if red_flags:
            notes.append(f"{red_flags} red-flag period(s) on track")
        if yellow_flags:
            notes.append(f"{yellow_flags} yellow-flag period(s) on track")

    if hasattr(session, "session_status") and session.session_status is not None:
        aborted = session.session_status["Status"].astype(str).str.contains("Aborted", case=False).any()
        if aborted:
            notes.append("session aborted / restarted")

    return SessionConditions(session_label, red_flags, yellow_flags, aborted, notes)


def _sector_seconds(laps: pd.DataFrame, sector_col: str) -> pd.Series:
    """Convert sector timedelta column to seconds."""
    return laps[sector_col].dt.total_seconds()


def extract_fp3_sectors(session) -> pd.DataFrame:
    """
    Best valid FP3 sector times per driver (analytical, not FIA classification).
    """
    valid = _valid_laps(session.laps)
    if valid.empty:
        return pd.DataFrame()

    sector_cols = ["Sector1Time", "Sector2Time", "Sector3Time"]
    rows = []
    for driver, group in valid.groupby("Driver"):
        row: dict = {"driver": str(driver)}
        for i, col in enumerate(sector_cols, start=1):
            secs = group.loc[group[col].notna(), col].dt.total_seconds()
            row[f"fp3_s{i}_best_s"] = float(secs.min()) if not secs.empty else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    for i in (1, 2, 3):
        col = f"fp3_s{i}_best_s"
        if col not in df.columns:
            continue
        best = df[col].min()
        df[f"fp3_s{i}_gap_to_best_s"] = df[col] - best
        df[f"fp3_s{i}_rank"] = df[col].rank(method="min")
    return df


def extract_fp3_weather(session) -> dict:
    """
    Summarise actual FP3 weather from FastF1 (requires weather=True on load).
    """
    summary = {
        "air_temp_avg_c": np.nan,
        "air_temp_min_c": np.nan,
        "air_temp_max_c": np.nan,
        "track_temp_avg_c": np.nan,
        "track_temp_min_c": np.nan,
        "track_temp_max_c": np.nan,
        "humidity_avg_pct": np.nan,
        "wind_speed_avg_ms": np.nan,
        "rainfall_occurred": False,
        "available": False,
    }
    try:
        wd = session.weather_data
    except Exception:
        return summary

    if wd is None or wd.empty:
        return summary

    summary["available"] = True
    if "AirTemp" in wd.columns:
        summary["air_temp_avg_c"] = float(wd["AirTemp"].mean())
        summary["air_temp_min_c"] = float(wd["AirTemp"].min())
        summary["air_temp_max_c"] = float(wd["AirTemp"].max())
    if "TrackTemp" in wd.columns:
        summary["track_temp_avg_c"] = float(wd["TrackTemp"].mean())
        summary["track_temp_min_c"] = float(wd["TrackTemp"].min())
        summary["track_temp_max_c"] = float(wd["TrackTemp"].max())
    if "Humidity" in wd.columns:
        summary["humidity_avg_pct"] = float(wd["Humidity"].mean())
    if "WindSpeed" in wd.columns:
        summary["wind_speed_avg_ms"] = float(wd["WindSpeed"].mean())
    if "Rainfall" in wd.columns:
        summary["rainfall_occurred"] = bool((wd["Rainfall"] > 0).any())
    return summary


def load_fp3_session(season: int, round_number: int):
    """Load FP3 with lap and weather data."""
    enable_cache()
    import fastf1

    session = fastf1.get_session(season, round_number, "FP3")
    session.load(laps=True, weather=True, telemetry=False, messages=False)
    return session


def compute_teammate_gaps(table: pd.DataFrame, lap_col: str = "fp3_best_lap_s") -> pd.Series:
    """
    Teammate gap from best valid lap (negative = faster than teammate).
    """
    gaps: dict[str, float] = {}
    for _, group in table.groupby("constructor"):
        valid = group.dropna(subset=[lap_col])
        if len(valid) != 2:
            for driver in group["driver"]:
                gaps[driver] = np.nan
            continue
        d1, d2 = valid.iloc[0], valid.iloc[1]
        t1, t2 = d1[lap_col], d2[lap_col]
        gaps[d1["driver"]] = t1 - t2
        gaps[d2["driver"]] = t2 - t1
    return table["driver"].map(gaps)


def add_practice_flags(table: pd.DataFrame) -> pd.DataFrame:
    """Simple descriptive practice flags — not numeric weights."""
    out = table.copy()

    def top6(col: str) -> set[str]:
        if col not in out.columns:
            return set()
        valid = out.dropna(subset=[col])
        return set(valid.nsmallest(6, col)["driver"])

    fp1_t6 = top6("fp1_rank")
    fp2_t6 = top6("fp2_rank")
    fp3_t6 = top6("fp3_rank")

    out["consistently_fast"] = out["driver"].apply(
        lambda d: sum(d in s for s in (fp1_t6, fp2_t6, fp3_t6)) >= 2
    )

    out["improving"] = False
    for _, row in out.iterrows():
        if pd.notna(row.get("fp1_rank")) and pd.notna(row.get("fp3_rank")):
            out.loc[out["driver"] == row["driver"], "improving"] = (
                row["fp3_rank"] <= row["fp1_rank"] - 3
            )

    lr_ok = out["fp2_long_run_reliability"].isin(["HIGH", "MEDIUM"])
    lr_top6 = set(out.loc[lr_ok].nsmallest(6, "fp2_long_run_rank")["driver"])
    out["fp2_long_run_strong"] = out["driver"].isin(lr_top6)

    s2_top6 = top6("fp3_s2_rank")
    out["technical_sector_strong"] = out["driver"].isin(s2_top6)

    out["teammate_advantage"] = out["fp3_teammate_gap_s"] <= -0.20
    return out


def detect_reserve_appearances(
    fp1_session,
    fp2_session,
    fp3_session,
    official_drivers: list[str],
) -> list[str]:
    """Drivers with lap data in practice but not on the official entry list."""
    official = set(official_drivers)
    reserve: set[str] = set()
    for session in (fp1_session, fp2_session, fp3_session):
        if session is None:
            continue
        valid = _valid_laps(session.laps)
        if valid.empty:
            continue
        reserve |= set(valid["Driver"].astype(str)) - official
    return sorted(reserve)


def build_hungary_practice_table(
    season: int = TARGET_SEASON,
    event_name: str = TARGET_EVENT,
) -> tuple[pd.DataFrame, dict[str, SessionConditions], pd.DataFrame, dict, list[str]]:
    """
    Load FP1 + FP2 + FP3 and build the combined current-weekend practice table.
    """
    enable_cache()
    schedule = get_event_schedule(season)
    round_number = get_event_round(schedule, event_name)

    fp1_session = load_session(season, round_number, "FP1")
    fp2_session = load_session(season, round_number, "FP2")
    fp3_session = load_fp3_session(season, round_number)

    if fp1_session is None or fp2_session is None or fp3_session is None:
        raise RuntimeError("Could not load FP1, FP2, and/or FP3 for Hungary 2026")

    conditions = {
        "FP1": inspect_session_conditions(fp1_session, "FP1"),
        "FP2": inspect_session_conditions(fp2_session, "FP2"),
        "FP3": inspect_session_conditions(fp3_session, "FP3"),
    }
    fp3_weather = extract_fp3_weather(fp3_session)

    fp1 = extract_session_pace(fp1_session, "FP1")
    fp2 = extract_session_pace(fp2_session, "FP2")
    fp3 = extract_session_pace(fp3_session, "FP3")
    fp3_sectors = extract_fp3_sectors(fp3_session)
    long_runs = analyze_fp2_long_run(fp2_session)

    all_drivers = _official_entry_list(fp3_session, fp2_session, fp1_session)
    if not all_drivers:
        all_drivers = sorted(set(fp1["driver"]) | set(fp2["driver"]) | set(fp3["driver"]))

    reserve_drivers = detect_reserve_appearances(
        fp1_session, fp2_session, fp3_session, all_drivers
    )

    constructors = {
        **_driver_constructor_map(fp1_session),
        **_driver_constructor_map(fp2_session),
        **_driver_constructor_map(fp3_session),
    }

    rows = []
    for driver in sorted(all_drivers):
        fp1_row = fp1[fp1["driver"] == driver]
        fp2_row = fp2[fp2["driver"] == driver]
        fp3_row = fp3[fp3["driver"] == driver]
        sec_row = fp3_sectors[fp3_sectors["driver"] == driver]
        lr = long_runs.get(driver, _empty_long_run())

        row: dict = {
            "driver": driver,
            "constructor": constructors.get(driver),
            "fp1_best_lap_s": fp1_row["best_lap_s"].iloc[0] if not fp1_row.empty else np.nan,
            "fp1_rank": int(fp1_row["rank"].iloc[0]) if not fp1_row.empty else np.nan,
            "fp1_gap_s": fp1_row["gap_s"].iloc[0] if not fp1_row.empty else np.nan,
            "fp2_best_lap_s": fp2_row["best_lap_s"].iloc[0] if not fp2_row.empty else np.nan,
            "fp2_rank": int(fp2_row["rank"].iloc[0]) if not fp2_row.empty else np.nan,
            "fp2_gap_s": fp2_row["gap_s"].iloc[0] if not fp2_row.empty else np.nan,
            "fp2_long_run_avg_s": lr.long_run_avg_lap_s,
            "fp2_long_run_median_lap_s": lr.long_run_median_lap_s,
            "fp2_long_run_laps": lr.long_run_laps,
            "fp2_long_run_std_s": lr.long_run_std_s,
            "fp2_long_run_reliability": lr.reliability,
            "fp3_best_lap_s": fp3_row["best_lap_s"].iloc[0] if not fp3_row.empty else np.nan,
            "fp3_rank": int(fp3_row["rank"].iloc[0]) if not fp3_row.empty else np.nan,
            "fp3_gap_s": fp3_row["gap_s"].iloc[0] if not fp3_row.empty else np.nan,
        }
        for i in (1, 2, 3):
            for suffix in ("best_s", "gap_to_best_s", "rank"):
                col = f"fp3_s{i}_{suffix}"
                row[col] = (
                    sec_row[col].iloc[0]
                    if not sec_row.empty and col in sec_row.columns
                    else np.nan
                )
        rows.append(row)

    table = pd.DataFrame(rows)
    table["fp3_teammate_gap_s"] = compute_teammate_gaps(table, "fp3_best_lap_s")

    valid_lr = table["fp2_long_run_avg_s"].dropna()
    if not valid_lr.empty:
        fastest_lr = valid_lr.min()
        table["fp2_long_run_gap_s"] = table["fp2_long_run_avg_s"] - fastest_lr
        table["fp2_long_run_rank"] = table["fp2_long_run_avg_s"].rank(method="min").where(
            table["fp2_long_run_avg_s"].notna()
        )
    else:
        table["fp2_long_run_gap_s"] = np.nan
        table["fp2_long_run_rank"] = np.nan

    table = add_practice_flags(table)

    output_cols = [
        "driver",
        "constructor",
        "fp1_best_lap_s",
        "fp1_rank",
        "fp1_gap_s",
        "fp2_best_lap_s",
        "fp2_rank",
        "fp2_gap_s",
        "fp2_long_run_avg_s",
        "fp2_long_run_median_lap_s",
        "fp2_long_run_gap_s",
        "fp2_long_run_rank",
        "fp2_long_run_laps",
        "fp2_long_run_std_s",
        "fp2_long_run_reliability",
        "fp3_best_lap_s",
        "fp3_rank",
        "fp3_gap_s",
        "fp3_teammate_gap_s",
        "fp3_s1_best_s",
        "fp3_s1_gap_to_best_s",
        "fp3_s1_rank",
        "fp3_s2_best_s",
        "fp3_s2_gap_to_best_s",
        "fp3_s2_rank",
        "fp3_s3_best_s",
        "fp3_s3_gap_to_best_s",
        "fp3_s3_rank",
        "consistently_fast",
        "improving",
        "fp2_long_run_strong",
        "technical_sector_strong",
        "teammate_advantage",
    ]
    table = table[output_cols].sort_values(["fp3_rank", "driver"], na_position="last").reset_index(
        drop=True
    )

    team_summary = build_constructor_fp2_summary(table)
    return table, conditions, team_summary, fp3_weather, reserve_drivers


def build_constructor_fp2_summary(table: pd.DataFrame) -> pd.DataFrame:
    """
    Constructor-level FP2 pace summary.

    Team long-run average uses only drivers with HIGH or MEDIUM reliability.
    """
    usable_reliability = frozenset({"HIGH", "MEDIUM"})
    rows = []
    for constructor, group in table.groupby("constructor"):
        usable = group[group["fp2_long_run_reliability"].isin(usable_reliability)]
        contributors = usable["driver"].tolist()
        rows.append(
            {
                "constructor": constructor,
                "fp2_best_team_lap_s": group["fp2_best_lap_s"].min(),
                "fp2_best_team_lap_rank": group["fp2_rank"].min(),
                "fp2_avg_long_run_s": usable["fp2_long_run_avg_s"].mean()
                if not usable.empty
                else np.nan,
                "fp2_long_run_contributors": ", ".join(contributors) if contributors else "",
                "fp2_long_run_usable_samples": len(contributors),
                "drivers_with_fp2_lap": int(group["fp2_best_lap_s"].notna().sum()),
            }
        )
    out = pd.DataFrame(rows).sort_values("fp2_best_team_lap_s")
    out["fp2_team_long_run_rank"] = out["fp2_avg_long_run_s"].rank(method="min")
    return out


def print_rankings(table: pd.DataFrame, team_summary: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("FP1 TOP 10 — single-lap pace (valid laps, analytical ranking)")
    print("=" * 70)
    fp1 = table.dropna(subset=["fp1_best_lap_s"]).nsmallest(10, "fp1_rank")
    for _, r in fp1.iterrows():
        print(
            f"  {int(r['fp1_rank']):>2}. {r['driver']:<4} {r['constructor']:<18} "
            f"{r['fp1_best_lap_s']:.3f}s  (+{r['fp1_gap_s']:.3f}s)"
        )

    print("\n" + "=" * 70)
    print("FP2 TOP 10 — single-lap pace")
    print("=" * 70)
    fp2 = table.dropna(subset=["fp2_best_lap_s"]).nsmallest(10, "fp2_rank")
    for _, r in fp2.iterrows():
        print(
            f"  {int(r['fp2_rank']):>2}. {r['driver']:<4} {r['constructor']:<18} "
            f"{r['fp2_best_lap_s']:.3f}s  (+{r['fp2_gap_s']:.3f}s)"
        )

    print("\n" + "=" * 70)
    print("FP2 TOP 10 — approximate long-run pace (*practice signal, not race pace*)")
    print("=" * 70)
    lr = table.dropna(subset=["fp2_long_run_avg_s"]).nsmallest(10, "fp2_long_run_rank")
    for _, r in lr.iterrows():
        print(
            f"  {int(r['fp2_long_run_rank']):>2}. {r['driver']:<4} {r['constructor']:<18} "
            f"avg {r['fp2_long_run_avg_s']:.3f}s  med {r['fp2_long_run_median_lap_s']:.3f}s  "
            f"(+{r['fp2_long_run_gap_s']:.3f}s)  "
            f"[{int(r['fp2_long_run_laps'])} laps, σ={r['fp2_long_run_std_s']:.3f}, "
            f"{r['fp2_long_run_reliability']}]"
        )

    print("\n" + "=" * 70)
    print("FP2 LONG-RUN RELIABILITY SUMMARY")
    print("=" * 70)
    for label in ("HIGH", "MEDIUM", "LOW", "UNAVAILABLE"):
        drivers = table.loc[table["fp2_long_run_reliability"] == label, "driver"].tolist()
        print(f"  {label:<12} ({len(drivers):>2}): {', '.join(drivers) if drivers else '—'}")

    print("\n" + "=" * 70)
    print("CONSTRUCTOR FP2 PACE SUMMARY (long-run: HIGH/MEDIUM drivers only)")
    print("=" * 70)
    for _, r in team_summary.iterrows():
        if pd.notna(r["fp2_avg_long_run_s"]):
            lr = f"{r['fp2_avg_long_run_s']:.3f}s"
            contrib = r["fp2_long_run_contributors"]
        else:
            lr = "unavailable"
            contrib = "—"
        print(
            f"  {r['constructor']:<18} best lap {r['fp2_best_team_lap_s']:.3f}s  "
            f"avg long-run {lr}  (from: {contrib})"
        )


def interpret_practice(
    table: pd.DataFrame,
    team_summary: pd.DataFrame,
    conditions: dict[str, SessionConditions],
) -> None:
    print("\n" + "=" * 70)
    print("CAUTIOUS INTERPRETATION (not a race prediction)")
    print("=" * 70)

    print("\nSession conditions / red flags:")
    for label, cond in conditions.items():
        flag_str = ", ".join(cond.notes) if cond.notes else "no major interruptions flagged"
        print(f"  {label}: {flag_str}")

    # Consistently quick FP1 + FP2 single lap (both top 10)
    fp1_top10 = set(table.nsmallest(10, "fp1_rank")["driver"])
    fp2_top10 = set(table.nsmallest(10, "fp2_rank")["driver"])
    both = sorted(fp1_top10 & fp2_top10)
    print(f"\nConsistently quick (FP1 & FP2 single-lap top 10): {', '.join(both) or 'none'}")

    # Strong FP2 long-run (top 10)
    lr_top10 = table.nsmallest(10, "fp2_long_run_rank")
    print(
        "\nStrong FP2 approximate long-run pace (top 10): "
        + ", ".join(lr_top10["driver"].tolist())
    )

    # Team-wide pace — constructors with usable long-run from both drivers
    strong_teams = []
    for _, row in team_summary.iterrows():
        if row["fp2_long_run_usable_samples"] >= 2:
            strong_teams.append(row["constructor"])
    print(
        f"\nConstructors with HIGH/MEDIUM long-run from both drivers: "
        f"{', '.join(strong_teams) if strong_teams else 'none'}"
    )

    # Single-lap vs long-run divergence (>5 places)
    divergent = []
    for _, r in table.dropna(subset=["fp2_rank", "fp2_long_run_rank"]).iterrows():
        diff = abs(r["fp2_rank"] - r["fp2_long_run_rank"])
        if diff >= 5:
            divergent.append(
                f"{r['driver']} (FP2 lap P{int(r['fp2_rank'])}, long-run P{int(r['fp2_long_run_rank'])})"
            )
    print(
        "\nDrivers with large single-lap vs long-run rank gap (≥5 places): "
        + ("; ".join(divergent) if divergent else "none")
    )

    missing_lr = table[table["fp2_long_run_reliability"] == "UNAVAILABLE"]["driver"].tolist()
    low_lr = table[table["fp2_long_run_reliability"] == "LOW"]["driver"].tolist()
    print(f"\nUNAVAILABLE FP2 long-run: {', '.join(missing_lr) or 'none'}")
    print(f"LOW reliability long-run: {', '.join(low_lr) or 'none'}")

    no_fp2 = table[table["fp2_best_lap_s"].isna()]["driver"].tolist()
    if no_fp2:
        print(f"No FP2 lap data at all: {', '.join(no_fp2)}")

    print(
        "\nNote: Long-run gaps are approximate practice signals only — fuel loads, "
        "engine modes, and run plans differ across teams. Do not treat as true race pace."
    )


def save_practice_table(table: pd.DataFrame, path: Path | None = None) -> Path:
    out = path or (OUTPUT_DIR / PRACTICE_OUTPUT_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return out


def print_long_run_changes(
    old_table: pd.DataFrame | None,
    new_table: pd.DataFrame,
    new_team_summary: pd.DataFrame,
) -> None:
    """Report drivers/constructors whose long-run values changed materially."""
    if old_table is None or old_table.empty:
        return

    print("\n" + "=" * 70)
    print("LONG-RUN CORRECTIONS vs PREVIOUS VERSION")
    print("=" * 70)

    merged = new_table.merge(
        old_table[["driver", "fp2_long_run_avg_s", "fp2_long_run_std_s", "fp2_long_run_laps"]],
        on="driver",
        how="left",
        suffixes=("_new", "_old"),
    )

    for _, r in merged.iterrows():
        old_avg = r.get("fp2_long_run_avg_s_old")
        new_avg = r.get("fp2_long_run_avg_s_new")
        old_std = r.get("fp2_long_run_std_s_old")
        new_std = r.get("fp2_long_run_std_s_new")
        if pd.isna(old_avg) and pd.isna(new_avg):
            continue
        avg_delta = (new_avg - old_avg) if pd.notna(old_avg) and pd.notna(new_avg) else np.nan
        material = (
            (pd.notna(old_std) and old_std > 3.0)
            or (pd.notna(avg_delta) and abs(avg_delta) > 2.0)
            or (pd.notna(old_avg) and pd.isna(new_avg))
            or (pd.isna(old_avg) and pd.notna(new_avg))
        )
        if material:
            print(
                f"  {r['driver']}: avg {old_avg} → {new_avg}, "
                f"σ {old_std} → {new_std}, "
                f"laps {r.get('fp2_long_run_laps_old')} → {r.get('fp2_long_run_laps_new')}, "
                f"reliability → {r['fp2_long_run_reliability']}"
            )

    old_team = build_constructor_fp2_summary(old_table) if "fp2_long_run_reliability" in old_table.columns else None
    if old_team is None:
        # Previous CSV had no reliability — approximate old constructor avg (all drivers)
        old_rows = []
        for constructor, group in old_table.groupby("constructor"):
            lr = group["fp2_long_run_avg_s"].dropna()
            old_rows.append(
                {
                    "constructor": constructor,
                    "fp2_avg_long_run_s": lr.mean() if not lr.empty else np.nan,
                }
            )
        old_team = pd.DataFrame(old_rows)

    print("\n  Constructor long-run averages:")
    for _, row in new_team_summary.iterrows():
        old_row = old_team[old_team["constructor"] == row["constructor"]]
        old_val = old_row["fp2_avg_long_run_s"].iloc[0] if not old_row.empty else np.nan
        new_val = row["fp2_avg_long_run_s"]
        if pd.notna(old_val) or pd.notna(new_val):
            if pd.isna(old_val) or pd.isna(new_val) or abs(old_val - new_val) > 0.5:
                print(
                    f"    {row['constructor']}: {old_val} → {new_val} "
                    f"(contributors: {row['fp2_long_run_contributors'] or 'none'})"
                )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    print("=" * 70)
    print("HUNGARY 2026 — FP1/FP2 PRACTICE ANALYSIS (V1 analytical layer)")
    print("=" * 70)
    print("\nThis is NOT fed into the ML model. Historical V1 training excluded FP1/FP2.")

    out_path = OUTPUT_DIR / PRACTICE_OUTPUT_FILE
    old_table = pd.read_csv(out_path) if out_path.exists() else None

    table, conditions, team_summary, fp3_weather, reserve_drivers = build_hungary_practice_table()
    save_path = save_practice_table(table)

    print(f"\n2026 Hungarian GP — {len(table)} drivers in table")
    if reserve_drivers:
        print(f"Reserve appearances (excluded): {', '.join(reserve_drivers)}")
    print(f"Saved: {save_path}")

    print_rankings(table, team_summary)
    print_long_run_changes(old_table, table, team_summary)
    interpret_practice(table, team_summary, conditions)

    print("\n" + "=" * 70)
    print("Analysis complete — no ML retrain, no winner declared.")
    print("=" * 70)


if __name__ == "__main__":
    main()
