# Singapore 2026 — Circuit and weekend context

## Event

| Field | Value |
|-------|-------|
| Event | Singapore Grand Prix |
| Season / Round | 2026 R17 |
| Format | Sprint qualifying weekend |
| Circuit | Marina Bay |

## Night race and climate (interpretive context)

- Permanent night race; track and air temperatures typically fall from daytime practice toward qualifying and the race.
- High humidity and limited overtaking historically — track position matters.
- 2023 Marina Bay layout revision: older Singapore results are down-weighted in historical features.

## Session schedule (supplied project schedule — verify with F1)

See `src/singapore_config.py` `SINGAPORE_WEEKEND_SCHEDULE_IST`.

## Data notes

- 2026 season form uses completed races **before** Singapore (2026 R1–R16).
- **Round 16** official event name: **Bahrain Grand Prix** (Gulf Air Bahrain GP **in Malaysia**).
- **Actual venue:** Sepang International Circuit, Malaysia — not Bahrain International Circuit.
- FastF1 stores `Location=Kuala Lumpur` for this event; the Singapore pipeline maps R16 to canonical `circuit=Sepang International Circuit`, `country=Malaysia` while preserving `race_name`.

Weather forecasts and observed session weather are stored separately when available.
