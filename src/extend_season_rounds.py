"""
Append missing completed Grand Prix rounds to raw race/qualifying CSVs.

Does not remove or alter existing rows. Used to extend 2026 through R16 before Singapore.

Usage:
    python -m src.extend_season_rounds --season 2026 --before-event "Singapore Grand Prix"
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

import pandas as pd

from src.config import QUALIFYING_OUTPUT_FILE, RACE_OUTPUT_FILE, RAW_DATA_DIR
from src.data_loader import (
    collect_event_sessions,
    enable_cache,
    get_completed_grand_prix,
    get_event_schedule,
)
from src.singapore_config import SINGAPORE_FORM_CUTOFF_EVENT, SINGAPORE_SEASON

logger = logging.getLogger(__name__)


def _existing_rounds(df: pd.DataFrame, season: int) -> set[int]:
    if df.empty:
        return set()
    sub = df[df["season"] == season]
    return set(int(r) for r in sub["round"].unique())


def extend_raw_season(
    season: int,
    before_event: str,
    as_of: date | None = None,
) -> dict[str, int]:
    """Collect Race+Q for completed rounds not yet in raw CSVs."""
    enable_cache()
    schedule = get_event_schedule(season)
    to_collect = get_completed_grand_prix(
        season,
        before_event=before_event,
        as_of=as_of or date.today(),
    )

    race_path = RAW_DATA_DIR / RACE_OUTPUT_FILE
    quali_path = RAW_DATA_DIR / QUALIFYING_OUTPUT_FILE
    race_existing = pd.read_csv(race_path) if race_path.exists() else pd.DataFrame()
    quali_existing = pd.read_csv(quali_path) if quali_path.exists() else pd.DataFrame()
    have = _existing_rounds(race_existing, season)

    new_race: list[pd.DataFrame] = []
    new_quali: list[pd.DataFrame] = []
    added_rounds: list[int] = []

    for _, ev in to_collect.iterrows():
        rnd = int(ev["RoundNumber"])
        if rnd in have:
            continue
        name = str(ev["EventName"])
        logger.info("Collecting missing %s round %s (%s)", season, rnd, name)
        extracted = collect_event_sessions(season, rnd, name)
        if extracted["race"] is not None and not extracted["race"].empty:
            new_race.append(extracted["race"])
        if extracted["qualifying"] is not None and not extracted["qualifying"].empty:
            new_quali.append(extracted["qualifying"])
        added_rounds.append(rnd)

    stats = {"rounds_added": len(added_rounds), "race_rows_added": 0, "quali_rows_added": 0}

    if new_race:
        race_df = pd.concat([race_existing, *new_race], ignore_index=True)
        race_df = race_df.drop_duplicates(subset=["season", "round", "driver"], keep="first")
        race_df = race_df.sort_values(["season", "round", "driver"]).reset_index(drop=True)
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        race_df.to_csv(race_path, index=False)
        stats["race_rows_added"] = len(race_df) - len(race_existing)

    if new_quali:
        quali_df = pd.concat([quali_existing, *new_quali], ignore_index=True)
        quali_df = quali_df.drop_duplicates(subset=["season", "round", "driver"], keep="first")
        quali_df = quali_df.sort_values(["season", "round", "driver"]).reset_index(drop=True)
        quali_df.to_csv(quali_path, index=False)
        stats["quali_rows_added"] = len(quali_df) - len(quali_existing)

    stats["added_round_numbers"] = added_rounds
    return stats


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    parser = argparse.ArgumentParser(description="Extend raw CSVs with missing completed rounds")
    parser.add_argument("--season", type=int, default=SINGAPORE_SEASON)
    parser.add_argument(
        "--before-event",
        type=str,
        default=SINGAPORE_FORM_CUTOFF_EVENT,
        help="Collect completed GPs strictly before this event",
    )
    parser.add_argument("--as-of", type=str, default=None, help="YYYY-MM-DD reference date")
    args = parser.parse_args(argv)
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    stats = extend_raw_season(args.season, args.before_event, as_of=as_of)
    print("Extension complete:", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
