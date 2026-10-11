"""
Canonical venue metadata for events where FastF1 ``Location`` ≠ actual circuit.

Official ``race_name`` / ``EventName`` is never altered. Only ``circuit`` and
optional ``country`` are normalized for feature engineering and documentation.
"""

from __future__ import annotations

import pandas as pd

# (season, round) -> canonical fields to apply on top of source rows.
CANONICAL_VENUES: dict[tuple[int, int], dict[str, str]] = {
    (2026, 16): {
        # Official event: "Bahrain Grand Prix" (Malaysia host) — venue is Sepang.
        "circuit": "Sepang International Circuit",
        "country": "Malaysia",
    },
}

SOURCE_LOCATION_NOTE: dict[tuple[int, int], str] = {
    (2026, 16): (
        "FastF1 schedule lists Location='Kuala Lumpur', Country='Bahrain' for the "
        "2026 R16 Bahrain Grand Prix in Malaysia; canonical circuit is Sepang."
    ),
}


def apply_canonical_venues(
    race: pd.DataFrame,
    quali: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return copies with canonical circuit/country where configured."""
    race_out = race.copy()
    quali_out = quali.copy() if quali is not None else None

    for (season, round_num), fields in CANONICAL_VENUES.items():
        r_mask = (race_out["season"] == season) & (race_out["round"] == round_num)
        for col, value in fields.items():
            if col not in race_out.columns:
                race_out[col] = pd.NA
            race_out.loc[r_mask, col] = value
        if quali_out is not None:
            q_mask = (quali_out["season"] == season) & (quali_out["round"] == round_num)
            for col in ("circuit", "country"):
                if col in fields:
                    if col not in quali_out.columns:
                        quali_out[col] = pd.NA
                    quali_out.loc[q_mask, col] = fields[col]

    return race_out, quali_out
