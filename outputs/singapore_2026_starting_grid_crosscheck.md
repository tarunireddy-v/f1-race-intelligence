# Singapore 2026 — Sunday starting grid cross-check

**Generated (IST):** 2026-10-11 (publication prep)

## Purpose

Record how the published F1 starting-grid order (user-supplied graphic transcription) compares to project grid evidence. **Does not replace** `outputs/singapore_2026_sunday_grid_resolution.csv` (preserved unchanged).

## Evidence tiers

| Tier | Source | Status | Notes |
|------|--------|--------|-------|
| Qualifying classification | FastF1 `Q` + FIA Doc 74 | Project evidence | `gp_qualifying_position` / `gp_quali_gap_to_pole_s` in grid-resolution CSV |
| Sunday grid (project canonical) | FIA **Doc 75 — Provisional Starting Grid** (11 Oct 2026 03:00) | **Provisional** (not “final”) | Encoded in `verified_sunday_grid_position`; PDF: [FIA decision document](https://www.fia.com/system/files/decision-document/2026_singapore_grand_prix_-_provisional_starting_grid.pdf) |
| F1 published graphic | User transcription (22 positions, full names) | **Cross-check reference** | Not independently verified as an FIA PDF; matched project grid 22/22 on audit |
| Formula 1 results page | [formula1.com starting grid](https://www.formula1.com/en/results/2026/races/1296/singapore/starting-grid) | Secondary | Listed as public F1 results URL; page body not re-scraped in this audit |
| FIA final starting grid | — | **Not located** | No separate FIA “Final Starting Grid” document was verified beyond Doc 75 provisional |

## User graphic transcription vs `verified_sunday_grid_position`

Comparison date: publication prep audit. **Mismatches: none** (22 drivers, positions 1–22 unique).

| Grid P | Driver (graphic) | Code | Project `verified_sunday_grid_position` |
|-------:|------------------|------|----------------------------------------|
| 1 | Max Verstappen | VER | 1 |
| 2 | Charles Leclerc | LEC | 2 |
| 3 | Lewis Hamilton | HAM | 3 |
| 4 | Andrea Kimi Antonelli | ANT | 4 |
| 5 | Lando Norris | NOR | 5 |
| 6 | Oscar Piastri | PIA | 6 |
| 7 | Pierre Gasly | GAS | 7 |
| 8 | Liam Lawson | LAW | 8 |
| 9 | Arvid Lindblad | LIN | 9 |
| 10 | Nico Hulkenberg | HUL | 10 |
| 11 | Oliver Bearman | BEA | 11 |
| 12 | Esteban Ocon | OCO | 12 |
| 13 | Carlos Sainz | SAI | 13 |
| 14 | Gabriel Bortoleto | BOR | 14 |
| 15 | Franco Colapinto | COL | 15 |
| 16 | Fernando Alonso | ALO | 16 |
| 17 | Alex Albon | ALB | 17 |
| 18 | Lance Stroll | STR | 18 |
| 19 | Sergio Perez | PER | 19 |
| 20 | Valtteri Bottas | BOT | 20 |
| 21 | George Russell | RUS | 21 |
| 22 | Isack Hadjar | HAD | 22 |

**Penalty rows (grid ≠ quali, quali unchanged):**

- **RUS:** quali P6, grid **P21** (FIA Doc 20 — 40-place PU penalty; grid per sporting regulations, not quali−40 arithmetic).
- **HAD:** unclassified in qualifying (Doc 74); permitted to start (Doc 71); grid **P22**.

## Experimental grid-corrected RF — model inputs (no rerun)

Path: `src/singapore_pre_race_audit.py` → `apply_verified_grid_to_gp_table()` → `build_singapore_post_feature_rows(gp_corrected)`.

POST feature columns fed to RF/Ridge at inference:

- `grid_position` ← `gp_grid_position` (Doc 75 / resolution CSV for all 22 drivers)
- `qualifying_position` ← `gp_qualifying_position` (qualifying classification)
- `gap_to_pole_s` ← `gp_quali_gap_to_pole_s` (qualifying pace)

Audit: `gp_grid_position` in `singapore_2026_final_prediction_experimental_grid_corrected.csv` equals `verified_sunday_grid_position` for every driver. **No rerun required.**

## Frozen Stage E limitation (unchanged)

`outputs/predictions/singapore_2026_final_prediction.csv` (SHA256 `90808895…`) used FastF1 quali-as-grid fallback (`gp_grid_position = gp_qualifying_position`) when session grid was missing. That is **not** the verified Sunday grid. Headline **predicted** order remains `rf_post_quali_rank` and must not be confused with the starting grid.
