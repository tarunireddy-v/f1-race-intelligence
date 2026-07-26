"""
Collect historical FastF1 session data for model training.

V1 historical ingestion collects Race + Qualifying only.

Usage:
    # Estimate cache coverage (no bulk download):
    python -m src.collect_historical_data --estimate

    # Lightweight test (2025 Hungarian GP, Race + Qualifying):
    python -m src.collect_historical_data --test

    # Full historical ingestion (2024–2026 pre-Hungary, Race + Qualifying):
    python -m src.collect_historical_data --full
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import (
    QUALIFYING_OUTPUT_FILE,
    RACE_OUTPUT_FILE,
    RAW_DATA_DIR,
    TARGET_SEASON,
    TEST_EVENT,
    TEST_SEASON,
    TRAINING_SEASONS,
)
from src.data_loader import (
    collect_event_sessions,
    enable_cache,
    estimate_cache_coverage,
    get_event_round,
    get_event_schedule,
    get_training_events,
)

try:
    from fastf1.exceptions import RateLimitExceededError
except ImportError:
    RateLimitExceededError = Exception  # noqa: A001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

HISTORICAL_SESSION_KEYS = ("race", "qualifying")


def _entrant_counts_for_event(
    data: dict[str, pd.DataFrame],
    season: int,
    round_number: int,
) -> dict[str, int]:
    """Return per-session entrant counts for one Grand Prix."""
    counts: dict[str, int] = {}
    for key in HISTORICAL_SESSION_KEYS:
        df = data.get(key, pd.DataFrame())
        if df.empty:
            counts[key] = 0
            continue
        event_df = df[(df["season"] == season) & (df["round"] == round_number)]
        counts[key] = len(event_df)
    return counts


def _check_qualifying_classification_mismatch(quali_df: pd.DataFrame) -> list[str]:
    """
    Warn when official qualifying position order differs from fastest-lap order.

    Does not modify either value — both are retained as separate features.
    """
    warnings: list[str] = []
    if quali_df.empty:
        return warnings

    group_cols = ["season", "round"]
    for (_, rnd), group in quali_df.groupby(group_cols):
        valid = group.dropna(subset=["qualifying_position", "qualifying_best_lap_s"])
        if valid.empty:
            continue

        official_pole = valid.loc[valid["qualifying_position"].idxmin(), "driver"]
        pace_pole = valid.loc[valid["qualifying_best_lap_s"].idxmin(), "driver"]

        season = int(group["season"].iloc[0])
        race_name = group["race_name"].iloc[0]

        if official_pole != pace_pole:
            warnings.append(
                f"qualifying: {season} round {rnd} ({race_name}) — "
                f"official pole {official_pole} vs fastest valid lap {pace_pole} "
                f"(deleted laps / session rulings may apply)"
            )

        official_order = valid.sort_values("qualifying_position")["driver"].tolist()
        pace_order = valid.sort_values("qualifying_best_lap_s")["driver"].tolist()
        if official_order != pace_order:
            mismatched = sum(
                1 for a, b in zip(official_order, pace_order, strict=False) if a != b
            )
            if mismatched and official_pole == pace_pole:
                warnings.append(
                    f"qualifying: {season} round {rnd} ({race_name}) — "
                    f"position order differs from best-lap order "
                    f"({mismatched} rank mismatch(es))"
                )

    return warnings


def _ensure_raw_dir() -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DATA_DIR


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [df for df in frames if df is not None and not df.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)


def collect_season(season: int) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Collect all training events for one season."""
    try:
        events = get_training_events(season)
    except RateLimitExceededError as exc:
        logger.error("Skipping season %s — rate limit fetching schedule: %s", season, exc)
        empty = {key: pd.DataFrame() for key in HISTORICAL_SESSION_KEYS}
        return empty, [
            {
                "season": season,
                "round": None,
                "race_name": f"SEASON {season}",
                "missing_sessions": ["ALL (rate limit on schedule fetch)"],
            }
        ]

    frames: dict[str, list[pd.DataFrame]] = {key: [] for key in HISTORICAL_SESSION_KEYS}
    missing_log: list[dict] = []

    for _, event in events.iterrows():
        round_number = int(event["RoundNumber"])
        event_name = str(event["EventName"])
        extracted = collect_event_sessions(season, round_number, event_name)

        missing_sessions = [
            key for key in HISTORICAL_SESSION_KEYS
            if extracted[key] is None or extracted[key].empty
        ]
        if missing_sessions:
            missing_log.append(
                {
                    "season": season,
                    "round": round_number,
                    "race_name": event_name,
                    "missing_sessions": missing_sessions,
                }
            )

        for key in HISTORICAL_SESSION_KEYS:
            if extracted[key] is not None and not extracted[key].empty:
                frames[key].append(extracted[key])

    combined = {key: _concat_or_empty(frames[key]) for key in HISTORICAL_SESSION_KEYS}
    return combined, missing_log


def collect_test_event(season: int, event_name: str) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Collect a single Grand Prix for pipeline verification."""
    schedule = get_event_schedule(season)
    round_number = get_event_round(schedule, event_name)
    extracted = collect_event_sessions(season, round_number, event_name)

    missing_sessions = [
        key for key in HISTORICAL_SESSION_KEYS
        if extracted[key] is None or extracted[key].empty
    ]
    missing_log = []
    if missing_sessions:
        missing_log.append(
            {
                "season": season,
                "round": round_number,
                "race_name": event_name,
                "missing_sessions": missing_sessions,
            }
        )

    combined = {
        key: extracted[key] if extracted[key] is not None else pd.DataFrame()
        for key in HISTORICAL_SESSION_KEYS
    }
    return combined, missing_log


def save_raw_data(data: dict[str, pd.DataFrame]) -> dict[str, Path]:
    """Write V1 historical Race/Qualifying CSVs under data/raw/."""
    _ensure_raw_dir()
    output_map = {
        "race": RACE_OUTPUT_FILE,
        "qualifying": QUALIFYING_OUTPUT_FILE,
    }
    written: dict[str, Path] = {}
    for key, filename in output_map.items():
        path = RAW_DATA_DIR / filename
        df = data.get(key, pd.DataFrame())
        df.to_csv(path, index=False)
        written[key] = path
        logger.info("Wrote %s rows to %s", len(df), path)
    return written


def _duplicate_keys(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df[df.duplicated(subset=key_cols, keep=False)].sort_values(key_cols)


def validate_data(
    data: dict[str, pd.DataFrame],
    missing_log: list[dict],
    *,
    seasons: list[int],
) -> dict:
    """Run validation checks and return a summary dict."""
    summary: dict = {"seasons": {}, "issues": [], "event_entrants": []}

    for season in seasons:
        season_summary = {
            "races_collected": 0,
            "race_rows": 0,
            "qualifying_rows": 0,
            "missing_sessions": [],
            "events": [],
        }

        race_df = data.get("race", pd.DataFrame())
        if not race_df.empty and "season" in race_df.columns:
            season_races = race_df[race_df["season"] == season]
            season_summary["races_collected"] = season_races["round"].nunique()
            season_summary["race_rows"] = len(season_races)

            for round_number, event_df in season_races.groupby("round"):
                rnd = int(round_number)
                entrant_counts = _entrant_counts_for_event(data, season, rnd)
                classified = event_df["finishing_position"].notna().sum()
                season_summary["events"].append(
                    {
                        "round": rnd,
                        "race_name": event_df["race_name"].iloc[0],
                        "race_entrants": entrant_counts["race"],
                        "classified_finishers": int(classified),
                        "qualifying_entrants": entrant_counts["qualifying"],
                    }
                )
                summary["event_entrants"].append(
                    {"season": season, **season_summary["events"][-1]}
                )

                # Report session count differences (informational, not errors)
                counts = [v for k, v in entrant_counts.items() if v > 0]
                if counts and max(counts) != min(counts):
                    summary["issues"].append(
                        f"entrant counts differ — {season} round {rnd} "
                        f"{event_df['race_name'].iloc[0]}: {entrant_counts}"
                    )

        df = data.get("qualifying", pd.DataFrame())
        if not df.empty and "season" in df.columns:
            season_summary["qualifying_rows"] = len(df[df["season"] == season])

        season_summary["missing_sessions"] = [
            entry for entry in missing_log if entry["season"] == season
        ]
        summary["seasons"][season] = season_summary

    # Cross-cutting checks
    key_cols = ["season", "round", "driver"]
    for name in HISTORICAL_SESSION_KEYS:
        df = data.get(name, pd.DataFrame())
        if df.empty:
            summary["issues"].append(f"{name}: empty dataset")
            continue

        dupes = _duplicate_keys(df, key_cols)
        if not dupes.empty:
            summary["issues"].append(
                f"{name}: {len(dupes)} duplicate rows on {key_cols}"
            )

        if "driver" in df.columns and df["driver"].isna().any():
            n_missing = int(df["driver"].isna().sum())
            summary["issues"].append(f"{name}: {n_missing} missing driver codes")

    race_df = data.get("race", pd.DataFrame())
    if not race_df.empty and race_df["finishing_position"].isna().any():
        n = int(race_df["finishing_position"].isna().sum())
        summary["issues"].append(
            f"race: {n} missing finishing_position values (may include DNFs/NC — not auto-deleted)"
        )

    quali_df = data.get("qualifying", pd.DataFrame())
    for warning in _check_qualifying_classification_mismatch(quali_df):
        summary["issues"].append(warning)

    if missing_log:
        for entry in missing_log:
            summary["issues"].append(
                f"missing sessions — {entry['season']} round {entry['round']} "
                f"{entry['race_name']}: {entry['missing_sessions']}"
            )

    return summary


def print_validation_summary(summary: dict) -> None:
    """Print human-readable validation output."""
    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)

    for season, stats in summary["seasons"].items():
        print(f"\nSeason {season}:")
        print(f"  Races collected:     {stats['races_collected']}")
        print(f"  Race result rows:      {stats['race_rows']}")
        print(f"  Qualifying rows:       {stats['qualifying_rows']}")
        if stats.get("events"):
            print("  Entrants per event:")
            for ev in stats["events"]:
                print(
                    f"    R{ev['round']:>2} {ev['race_name']:<28} "
                    f"race={ev['race_entrants']} (classified={ev['classified_finishers']}) "
                    f"quali={ev['qualifying_entrants']}"
                )
        if stats["missing_sessions"]:
            print(f"  Missing sessions:      {len(stats['missing_sessions'])} event(s)")
        else:
            print("  Missing sessions:      none")

    if summary["issues"]:
        print("\nAnomalies / warnings (not auto-deleted):")
        for issue in summary["issues"]:
            print(f"  - {issue}")
    else:
        print("\nNo anomalies detected.")


def print_test_preview(data: dict[str, pd.DataFrame]) -> None:
    """Print first rows and shapes for test-mode inspection."""
    labels = {
        "race": "Race",
        "qualifying": "Qualifying",
    }
    print("\n" + "=" * 70)
    print("TEST PREVIEW — first 5 rows per session")
    print("=" * 70)
    for key, label in labels.items():
        df = data.get(key, pd.DataFrame())
        print(f"\n{label} shape: {df.shape}")
        if df.empty:
            print("  (empty)")
        else:
            print(df.head(5).to_string(index=False))


def _cache_size_mb(cache_dir: Path) -> float:
    if not cache_dir.exists():
        return 0.0
    total = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def probe_rate_limit() -> tuple[bool, str]:
    """
    Make a minimal API call to detect active FastF1 rate limiting.

    Returns (is_rate_limited, message).
    """
    try:
        get_event_schedule(TARGET_SEASON)
        return False, "FastF1 API responsive (schedule fetch succeeded)"
    except RateLimitExceededError as exc:
        return True, f"Rate limited: {exc}"
    except Exception as exc:
        return False, f"Probe inconclusive (not rate limit): {exc}"


def print_coverage_estimate() -> None:
    """Print cache coverage for V1 Race + Qualifying historical data."""
    print("\n" + "=" * 70)
    print("V1 CACHE COVERAGE ESTIMATE (Race + Qualifying only)")
    print("=" * 70)

    raw_race = RAW_DATA_DIR / RACE_OUTPUT_FILE
    raw_quali = RAW_DATA_DIR / QUALIFYING_OUTPUT_FILE
    if raw_race.exists():
        df = pd.read_csv(raw_race)
        print(f"\nExisting {RACE_OUTPUT_FILE}: {df.shape[0]} rows")
        if "season" in df.columns:
            for s in sorted(df["season"].unique()):
                sub = df[df["season"] == s]
                print(f"  season {int(s)}: {sub['round'].nunique()} rounds, {len(sub)} rows")

    for season in TRAINING_SEASONS:
        cov = estimate_cache_coverage(season)
        print(
            f"\nSeason {season}: {cov['race_cached']}/{cov['total_events']} Race cached, "
            f"{cov['qualifying_cached']}/{cov['total_events']} Qualifying cached"
        )
        missing_r = [e for e in cov["events"] if not e["race_cached"]]
        missing_q = [e for e in cov["events"] if not e["qualifying_cached"]]
        if missing_r:
            print(f"  Missing Race cache: {len(missing_r)} event(s)")
        if missing_q:
            print(f"  Missing Qualifying cache: {len(missing_q)} event(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect historical FastF1 data (V1: Race + Qualifying)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate cache/raw coverage without downloading",
    )
    mode.add_argument(
        "--test",
        action="store_true",
        help=f"Test Race+Qualifying on {TEST_SEASON} {TEST_EVENT}",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Collect Race+Qualifying for all training seasons",
    )
    args = parser.parse_args(argv)

    from src.config import CACHE_DIR

    cache_path = enable_cache()
    missing_log: list[dict] = []
    rate_limited = False

    if args.estimate:
        limited, msg = probe_rate_limit()
        print(f"Rate limit status: {msg}")
        print_coverage_estimate()
        cache_mb = _cache_size_mb(Path(cache_path))
        print(f"\nFastF1 cache size: {cache_mb:.2f} MB ({cache_path})")
        return 1 if limited else 0

    limited, msg = probe_rate_limit()
    if limited:
        print(f"STOPPING — FastF1 rate limit active: {msg}")
        print("Wait before running --full or --test. Use --estimate to inspect cache.")
        return 1
    logger.info("Rate limit probe: %s", msg)

    if args.test:
        logger.info("Running test ingestion: %s %s (Race + Qualifying)", TEST_SEASON, TEST_EVENT)
        data, missing_log = collect_test_event(TEST_SEASON, TEST_EVENT)
        seasons = [TEST_SEASON]
    else:
        logger.info("Running V1 full ingestion (Race + Qualifying) for %s", TRAINING_SEASONS)
        all_frames: dict[str, list[pd.DataFrame]] = {key: [] for key in HISTORICAL_SESSION_KEYS}
        missing_log = []
        seasons = TRAINING_SEASONS
        for season in TRAINING_SEASONS:
            try:
                season_data, season_missing = collect_season(season)
            except RateLimitExceededError as exc:
                logger.error("Rate limit stopped season %s — not retrying: %s", season, exc)
                rate_limited = True
                missing_log.append(
                    {
                        "season": season,
                        "round": None,
                        "race_name": f"SEASON {season}",
                        "missing_sessions": ["ALL (rate limit)"],
                    }
                )
                break  # stop remaining seasons — do not hammer API
            missing_log.extend(season_missing)
            for key in HISTORICAL_SESSION_KEYS:
                if not season_data[key].empty:
                    all_frames[key].append(season_data[key])
            data = {key: _concat_or_empty(all_frames[key]) for key in HISTORICAL_SESSION_KEYS}
            save_raw_data(data)
            logger.info("Checkpoint save completed after season %s", season)
        data = {key: _concat_or_empty(all_frames[key]) for key in HISTORICAL_SESSION_KEYS}

    save_raw_data(data)
    summary = validate_data(data, missing_log, seasons=seasons)
    print_validation_summary(summary)
    print_test_preview(data)

    cache_mb = _cache_size_mb(Path(cache_path))
    print(f"\nFastF1 cache size after run: {cache_mb:.2f} MB ({cache_path})")

    if args.full and rate_limited:
        print("\n⚠ Ingestion stopped due to rate limit — checkpoint saved. Re-run later.")
        return 1

    if args.test:
        required = all(not data[key].empty for key in HISTORICAL_SESSION_KEYS)
        if required and not any(entry.get("missing_sessions") for entry in missing_log):
            print("\n✓ 2025 Hungarian GP test PASSED (Race + Qualifying).")
            return 0
        print("\n✗ 2025 Hungarian GP test FAILED — review logs above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
