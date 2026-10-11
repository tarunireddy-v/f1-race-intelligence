# PRE-RACE PREDICTION — GENERATED BEFORE THE 2026 SINGAPORE GRAND PRIX

**F1 Race Intelligence — Singapore Grand Prix (2026 R17)**

**This file must not be modified based on race results. Post-race evaluation (Stage F) is not included.**

**Report assembled (IST):** 2026-10-11 13:42 UTC+05:30
**Frozen Stage E RF POST generated (IST):** 2026-10-10T22:06:05.979380+05:30
**Experimental grid-corrected RF generated (IST):** 2026-10-11T13:15:45.893321+05:30
**Experimental Ridge POST generated (IST):** 2026-10-11T13:15:45.893321+05:30

**Training data cutoff (Stage E):** 2026 R1–16 race+qualifying; historical Singapore 2019+; + Singapore 2026 Sprint Qualifying (not GP quali / not Sunday grid); + Singapore 2026 Sprint race (not GP quali / not Sunday race); + Singapore 2026 GP qualifying & Sunday grid (POST-ML race prediction)

**Sessions used for evidence (not all are ML inputs):** FP1, Sprint Qualifying (SQ), Sprint (S), Grand Prix qualifying (Q). **No Singapore Grand Prix race result was used.**

---

## Official Grand Prix qualifying classification

Source: FastF1 `Q` session + FIA Doc 74 (see `outputs/singapore_2026_gp_qual_session_evidence.csv`).

- P1 VER (Red Bull Racing) — 91.373s (+0.000s)
- P2 LEC (Ferrari) — 91.427s (+0.054s)
- P3 HAM (Ferrari) — 91.496s (+0.123s)
- P4 ANT (Mercedes) — 91.562s (+0.189s)
- P5 NOR (McLaren) — 91.712s (+0.339s)
- P6 RUS (Mercedes) — 91.747s (+0.374s)
- P7 PIA (McLaren) — 91.860s (+0.487s)
- P8 GAS (Alpine) — 92.225s (+0.852s)
- P9 LAW (Racing Bulls) — 92.319s (+0.946s)
- P10 LIN (Racing Bulls) — 92.498s (+1.125s)
- P11 HUL (Audi) — 92.784s (+1.411s)
- P12 BEA (Haas F1 Team) — 92.908s (+1.535s)
- P13 OCO (Haas F1 Team) — 93.323s (+1.950s)
- P14 SAI (Williams) — 93.526s (+2.153s)
- P15 BOR (Audi) — 93.658s (+2.285s)
- P16 COL (Alpine) — 93.261s (+1.888s)
- P17 ALO (Aston Martin) — 94.223s (+2.850s)
- P18 ALB (Williams) — 94.289s (+2.916s)
- P19 STR (Aston Martin) — 94.946s (+3.573s)
- P20 PER (Cadillac) — 95.123s (+3.750s)
- P21 BOT (Cadillac) — 95.984s (+4.611s)
- P22 HAD (Red Bull Racing) — no classified time

## Sunday starting grid (provisional — not predicted finishing order)

**Canonical project grid:** `outputs/singapore_2026_sunday_grid_resolution.csv` column `verified_sunday_grid_position`.

**Primary regulatory source:** FIA Singapore GP **Document 75 — Provisional Starting Grid** (11 Oct 2026 03:00 UTC). PDF: [FIA decision document](https://www.fia.com/system/files/decision-document/2026_singapore_grand_prix_-_provisional_starting_grid.pdf). This is **provisional**, not an independently verified FIA **final** starting-grid document.

**Cross-check:** A full 22-position order transcribed from the published F1 starting-grid graphic (user reference) matches every `verified_sunday_grid_position` in the resolution CSV — see `outputs/singapore_2026_starting_grid_crosscheck.md`. Public F1 results URL (secondary, not re-scraped here): [formula1.com starting grid](https://www.formula1.com/en/results/2026/races/1296/singapore/starting-grid).

**Verified back-of-grid rows:** **George Russell P21** (quali P6; FIA Doc 20 PU penalty) and **Isack Hadjar P22** (unclassified in qualifying; permitted to start per Doc 71).

Qualifying classification, qualifying gap to pole, and Sunday grid position are **separate fields**. Grid penalties change starting position only — not qualifying result or `gap_to_pole_s`.

- Grid P1 VER — quali P1
- Grid P2 LEC — quali P2
- Grid P3 HAM — quali P3
- Grid P4 ANT — quali P4
- Grid P5 NOR — quali P5
- Grid P6 PIA — quali P7
- Grid P7 GAS — quali P8
- Grid P8 LAW — quali P9
- Grid P9 LIN — quali P10
- Grid P10 HUL — quali P11
- Grid P11 BEA — quali P12
- Grid P12 OCO — quali P13
- Grid P13 SAI — quali P14
- Grid P14 BOR — quali P15
- Grid P15 COL — quali P16
- Grid P16 ALO — quali P17
- Grid P17 ALB — quali P18
- Grid P18 STR — quali P19
- Grid P19 PER — quali P20
- Grid P20 BOT — quali P21
- Grid P21 RUS — quali P6 — FIA Doc 20: 40-place PU penalty; grid per Art. B2.5.4(b)(iv) — Doc 75 P21
- Grid P22 HAD — quali unclassified (no classified Q time)

**Frozen Stage E grid-input limitation (checksum unchanged):** At Stage E generation, FastF1 returned `GridPosition` NaN for all drivers; the frozen pipeline set `gp_grid_position = gp_qualifying_position` for POST feature construction. That fallback is **not** the verified Sunday grid above. The frozen CSV headline **race forecast** is `rf_post_quali_rank` (predicted *finishing* positions) and must not be read as the starting grid.

**Experimental grid-corrected RF / Ridge:** Re-inference used Doc 75 grid via `apply_verified_grid_to_gp_table()` → `build_singapore_post_feature_rows()` (`grid_position` = verified Sunday grid; `qualifying_position` and `gap_to_pole_s` from qualifying only). No model rerun in this publication pass — existing experimental CSVs already match the resolution grid.

**Ridge publication snapshot:** Use `outputs/singapore_2026_final_snapshot_experimental_ridge_companion.md` (or Ridge `*_sorted.csv`). Retain `singapore_2026_final_snapshot_experimental_ridge.md` for provenance only — it incorrectly labels RF order.

---

## Weekend evidence (context — not POST-ML inputs)

### FP1 (practice)
Merged in Stage B as `fp1_rank` and sector times. **Not** in `POST_QUALIFYING_FEATURES`. See `outputs/singapore_2026_post_fp1_snapshot.md`.

### Sprint Qualifying (SQ)
Official SQ classification in Stage C (`sq_official_rank`). Sets Sprint grid only. **Not** in POST-ML. Top 3: VER, RUS, LEC.

### Sprint (S)
Official sprint finish in Stage D (`sprint_official_rank`). **Not** in POST-ML. Winner: VER; podium VER / HAM / LEC.

### Historical PRE-ML (`rf_pre_weekend_rank`)
Retrospective Random Forest trained on 2026 R1–16 **without** Singapore session data; frozen unchanged Stages A–E. Headline for Stages A–D CSVs when sorting presentation copies.

---

## Weather context (not ML features)

- Retrieved (IST): 2026-10-11T13:15:45.893321+05:30
- Category: race_start_forecast_context_not_ml_features
- **Formula 1 official weather article** — https://www.formula1.com/en/latest/article/what-is-the-weather-forecast-for-the-2026-singapore-grand-prix.4DFAwBl2esxhHp1mCv3VSF
- **f1weather.com (Open-Meteo/OpenWeatherMap)** — https://f1weather.com/
- **Observed GP quali (FastF1):** air ~29.8°C, track ~39.4°C, humidity ~75.1%, rainfall in session: False
- **Uncertainty:** Third-party forecasts disagree on rain timing (afternoon showers vs ~40% at race start). Marina Bay is humid; convection can develop quickly. Not used in ML features.

---

## Complete predicted finishing orders (P1–P22)

Sort keys are model rank columns — **not** physical CSV row order.

### 1. Frozen Stage E — Random Forest POST (`rf_post_quali_rank`)

- P1 LEC (Ferrari) — quali P2 | Sunday grid P2 (provisional Doc 75)
- P2 HAM (Ferrari) — quali P3 | Sunday grid P3 (provisional Doc 75)
- P3 VER (Red Bull Racing) — quali P1 | Sunday grid P1 (provisional Doc 75)
- P4 NOR (McLaren) — quali P5 | Sunday grid P5 (provisional Doc 75)
- P5 ANT (Mercedes) — quali P4 | Sunday grid P4 (provisional Doc 75)
- P6 PIA (McLaren) — quali P7 | Sunday grid P6 (provisional Doc 75)
- P7 RUS (Mercedes) — quali P6 | Sunday grid P21 (provisional Doc 75)
- P8 GAS (Alpine) — quali P8 | Sunday grid P7 (provisional Doc 75)
- P9 LIN (Racing Bulls) — quali P10 | Sunday grid P9 (provisional Doc 75)
- P10 HAD (Red Bull Racing) — quali P22 | Sunday grid P22 (provisional Doc 75)
- P11 LAW (Racing Bulls) — quali P9 | Sunday grid P8 (provisional Doc 75)
- P12 BEA (Haas F1 Team) — quali P12 | Sunday grid P11 (provisional Doc 75)
- P13 HUL (Audi) — quali P11 | Sunday grid P10 (provisional Doc 75)
- P14 OCO (Haas F1 Team) — quali P13 | Sunday grid P12 (provisional Doc 75)
- P15 SAI (Williams) — quali P14 | Sunday grid P13 (provisional Doc 75)
- P16 BOR (Audi) — quali P15 | Sunday grid P14 (provisional Doc 75)
- P17 COL (Alpine) — quali P16 | Sunday grid P15 (provisional Doc 75)
- P18 ALO (Aston Martin) — quali P17 | Sunday grid P16 (provisional Doc 75)
- P19 ALB (Williams) — quali P18 | Sunday grid P17 (provisional Doc 75)
- P20 PER (Cadillac) — quali P20 | Sunday grid P19 (provisional Doc 75)
- P21 STR (Aston Martin) — quali P19 | Sunday grid P18 (provisional Doc 75)
- P22 BOT (Cadillac) — quali P21 | Sunday grid P20 (provisional Doc 75)

### 2. Experimental — grid-corrected RF POST (`rf_post_quali_rank`)

- P1 LEC (Ferrari) — quali P2 | Sunday grid P2 (provisional Doc 75)
- P2 HAM (Ferrari) — quali P3 | Sunday grid P3 (provisional Doc 75)
- P3 VER (Red Bull Racing) — quali P1 | Sunday grid P1 (provisional Doc 75)
- P4 NOR (McLaren) — quali P5 | Sunday grid P5 (provisional Doc 75)
- P5 ANT (Mercedes) — quali P4 | Sunday grid P4 (provisional Doc 75)
- P6 PIA (McLaren) — quali P7 | Sunday grid P6 (provisional Doc 75)
- P7 RUS (Mercedes) — quali P6 | Sunday grid P21 (provisional Doc 75)
- P8 GAS (Alpine) — quali P8 | Sunday grid P7 (provisional Doc 75)
- P9 LIN (Racing Bulls) — quali P10 | Sunday grid P9 (provisional Doc 75)
- P10 HAD (Red Bull Racing) — quali P22 | Sunday grid P22 (provisional Doc 75)
- P11 LAW (Racing Bulls) — quali P9 | Sunday grid P8 (provisional Doc 75)
- P12 HUL (Audi) — quali P11 | Sunday grid P10 (provisional Doc 75)
- P13 BEA (Haas F1 Team) — quali P12 | Sunday grid P11 (provisional Doc 75)
- P14 OCO (Haas F1 Team) — quali P13 | Sunday grid P12 (provisional Doc 75)
- P15 SAI (Williams) — quali P14 | Sunday grid P13 (provisional Doc 75)
- P16 COL (Alpine) — quali P16 | Sunday grid P15 (provisional Doc 75)
- P17 BOR (Audi) — quali P15 | Sunday grid P14 (provisional Doc 75)
- P18 ALO (Aston Martin) — quali P17 | Sunday grid P16 (provisional Doc 75)
- P19 ALB (Williams) — quali P18 | Sunday grid P17 (provisional Doc 75)
- P20 PER (Cadillac) — quali P20 | Sunday grid P19 (provisional Doc 75)
- P21 BOT (Cadillac) — quali P21 | Sunday grid P20 (provisional Doc 75)
- P22 STR (Aston Martin) — quali P19 | Sunday grid P18 (provisional Doc 75)

### 3. Experimental — Ridge Regression POST (`ridge_post_quali_rank`)

- P1 VER (Red Bull Racing) — quali P1 | Sunday grid P1 (provisional Doc 75)
- P2 ANT (Mercedes) — quali P4 | Sunday grid P4 (provisional Doc 75)
- P3 LEC (Ferrari) — quali P2 | Sunday grid P2 (provisional Doc 75)
- P4 HAM (Ferrari) — quali P3 | Sunday grid P3 (provisional Doc 75)
- P5 NOR (McLaren) — quali P5 | Sunday grid P5 (provisional Doc 75)
- P6 PIA (McLaren) — quali P7 | Sunday grid P6 (provisional Doc 75)
- P7 RUS (Mercedes) — quali P6 | Sunday grid P21 (provisional Doc 75)
- P8 LAW (Racing Bulls) — quali P9 | Sunday grid P8 (provisional Doc 75)
- P9 GAS (Alpine) — quali P8 | Sunday grid P7 (provisional Doc 75)
- P10 LIN (Racing Bulls) — quali P10 | Sunday grid P9 (provisional Doc 75)
- P11 HUL (Audi) — quali P11 | Sunday grid P10 (provisional Doc 75)
- P12 BEA (Haas F1 Team) — quali P12 | Sunday grid P11 (provisional Doc 75)
- P13 OCO (Haas F1 Team) — quali P13 | Sunday grid P12 (provisional Doc 75)
- P14 SAI (Williams) — quali P14 | Sunday grid P13 (provisional Doc 75)
- P15 COL (Alpine) — quali P16 | Sunday grid P15 (provisional Doc 75)
- P16 BOR (Audi) — quali P15 | Sunday grid P14 (provisional Doc 75)
- P17 HAD (Red Bull Racing) — quali P22 | Sunday grid P22 (provisional Doc 75)
- P18 ALO (Aston Martin) — quali P17 | Sunday grid P16 (provisional Doc 75)
- P19 ALB (Williams) — quali P18 | Sunday grid P17 (provisional Doc 75)
- P20 STR (Aston Martin) — quali P19 | Sunday grid P18 (provisional Doc 75)
- P21 PER (Cadillac) — quali P20 | Sunday grid P19 (provisional Doc 75)
- P22 BOT (Cadillac) — quali P21 | Sunday grid P20 (provisional Doc 75)

### 4. Gradient Boosting POST (`gb_post_quali_rank`)

- P1 LEC (Ferrari) — quali P2 | Sunday grid P2 (provisional Doc 75)
- P2 HAM (Ferrari) — quali P3 | Sunday grid P3 (provisional Doc 75)
- P3 VER (Red Bull Racing) — quali P1 | Sunday grid P1 (provisional Doc 75)
- P4 RUS (Mercedes) — quali P6 | Sunday grid P21 (provisional Doc 75)
- P5 NOR (McLaren) — quali P5 | Sunday grid P5 (provisional Doc 75)
- P6 ANT (Mercedes) — quali P4 | Sunday grid P4 (provisional Doc 75)
- P7 GAS (Alpine) — quali P8 | Sunday grid P7 (provisional Doc 75)
- P8 LIN (Racing Bulls) — quali P10 | Sunday grid P9 (provisional Doc 75)
- P9 LAW (Racing Bulls) — quali P9 | Sunday grid P8 (provisional Doc 75)
- P10 PIA (McLaren) — quali P7 | Sunday grid P6 (provisional Doc 75)
- P11 OCO (Haas F1 Team) — quali P13 | Sunday grid P12 (provisional Doc 75)
- P12 BEA (Haas F1 Team) — quali P12 | Sunday grid P11 (provisional Doc 75)
- P13 HUL (Audi) — quali P11 | Sunday grid P10 (provisional Doc 75)
- P14 BOR (Audi) — quali P15 | Sunday grid P14 (provisional Doc 75)
- P15 SAI (Williams) — quali P14 | Sunday grid P13 (provisional Doc 75)
- P16 COL (Alpine) — quali P16 | Sunday grid P15 (provisional Doc 75)
- P17 HAD (Red Bull Racing) — quali P22 | Sunday grid P22 (provisional Doc 75)
- P18 ALO (Aston Martin) — quali P17 | Sunday grid P16 (provisional Doc 75)
- P19 ALB (Williams) — quali P18 | Sunday grid P17 (provisional Doc 75)
- P20 STR (Aston Martin) — quali P19 | Sunday grid P18 (provisional Doc 75)
- P21 PER (Cadillac) — quali P20 | Sunday grid P19 (provisional Doc 75)
- P22 BOT (Cadillac) — quali P21 | Sunday grid P20 (provisional Doc 75)

### 5. XGBoost POST (`xgb_post_quali_rank`)

- P1 LEC (Ferrari) — quali P2 | Sunday grid P2 (provisional Doc 75)
- P2 HAM (Ferrari) — quali P3 | Sunday grid P3 (provisional Doc 75)
- P3 VER (Red Bull Racing) — quali P1 | Sunday grid P1 (provisional Doc 75)
- P4 RUS (Mercedes) — quali P6 | Sunday grid P21 (provisional Doc 75)
- P5 ANT (Mercedes) — quali P4 | Sunday grid P4 (provisional Doc 75)
- P6 NOR (McLaren) — quali P5 | Sunday grid P5 (provisional Doc 75)
- P7 GAS (Alpine) — quali P8 | Sunday grid P7 (provisional Doc 75)
- P8 LIN (Racing Bulls) — quali P10 | Sunday grid P9 (provisional Doc 75)
- P9 LAW (Racing Bulls) — quali P9 | Sunday grid P8 (provisional Doc 75)
- P10 HAD (Red Bull Racing) — quali P22 | Sunday grid P22 (provisional Doc 75)
- P11 HUL (Audi) — quali P11 | Sunday grid P10 (provisional Doc 75)
- P12 OCO (Haas F1 Team) — quali P13 | Sunday grid P12 (provisional Doc 75)
- P13 BEA (Haas F1 Team) — quali P12 | Sunday grid P11 (provisional Doc 75)
- P14 PIA (McLaren) — quali P7 | Sunday grid P6 (provisional Doc 75)
- P15 BOR (Audi) — quali P15 | Sunday grid P14 (provisional Doc 75)
- P16 SAI (Williams) — quali P14 | Sunday grid P13 (provisional Doc 75)
- P17 COL (Alpine) — quali P16 | Sunday grid P15 (provisional Doc 75)
- P18 ALB (Williams) — quali P18 | Sunday grid P17 (provisional Doc 75)
- P19 ALO (Aston Martin) — quali P17 | Sunday grid P16 (provisional Doc 75)
- P20 PER (Cadillac) — quali P20 | Sunday grid P19 (provisional Doc 75)
- P21 STR (Aston Martin) — quali P19 | Sunday grid P18 (provisional Doc 75)
- P22 BOT (Cadillac) — quali P21 | Sunday grid P20 (provisional Doc 75)

---

## Predicted winner and podium

- **Stage E RF:** winner **LEC** — podium LEC / HAM / VER
- **Grid-corrected RF:** winner **LEC** — podium LEC / HAM / VER
- **Ridge (experimental):** winner **VER** — podium VER / ANT / LEC
- **Gradient Boosting:** winner **LEC** — podium LEC / HAM / VER
- **XGBoost:** winner **LEC** — podium LEC / HAM / VER

---

## Historical model validation (2026 R14–R16 holdout)

Split: `singapore_extended_val` — train rows before 2026 R14 (**1244**), validation **66** driver-race rows (R14–R16). Singapore R17 excluded. Ridge α=**100** (inner val R11–R13).

| Model | MAE | RMSE | Spearman rank corr |
|-------|-----|------|-------------------|
| GradientBoostingRegressor | 3.727 | 4.863 | 0.630 |
| RandomForestRegressor | 3.632 | 4.643 | 0.676 |
| XGBRegressor | 3.853 | 5.088 | 0.605 |
| Ridge | 3.607 | 4.731 | 0.665 |
| qualifying_position_baseline | 3.625 | — | — |
| grid_position_baseline | 3.667 | — | — |

**How to read:** lower MAE/RMSE is better; higher Spearman is better. Ridge’s slightly lower MAE does not automatically dominate RF (RF has higher rank correlation on this split). Small validation sample; grid correction is a data-quality fix, not proof of improved Singapore accuracy without post-race evaluation.

---

## POST-qualifying ML features (verified from code)

**`POST_QUALIFYING_FEATURES` (11):** `avg_finish_last_3`, `avg_finish_last_5`, `avg_quali_last_3`, `points_last_5`, `dnf_rate_last_5`, `constructor_avg_finish_last_3`, `constructor_points_last_5`, `is_2026`, `grid_position`, `qualifying_position`, `gap_to_pole_s`.

**Excluded from headline POST forest (confirmed):** FP1/SQ/Sprint session pace, weather, circuit history columns present in CSV export, championship position, teammate quali gap, sector times.

**Missing-data / grid caveat:** Frozen Stage E inference used `gp_*` fields at predict time while legacy export columns `grid_position` / `qualifying_position` are often NaN. Experimental runs re-inferred with Doc 75 `gp_grid_position`.

---

## Why orders differ

1. **CSV row order vs headline rank:** Stage E CSV rows follow the Stage D skeleton sorted by `rf_pre_weekend_rank` (ANT, HAM, LEC first). Headline POST order sorts by `rf_post_quali_rank` (LEC, HAM, VER…).
2. **RF vs Ridge:** Different estimators on the same 11 POST features; Ridge companion snapshot required because `singapore_2026_final_snapshot_experimental_ridge.md` incorrectly lists RF order.
3. **Frozen vs grid-corrected RF:** Top five identical (LEC, HAM, VER, NOR, ANT). Midfield changes include HUL/BEA swap, COL/BOR swap, BOT/STR swap — consistent with Doc 75 grid shifts while RUS remains RF P7.

---

## Artifact index

| Role | Path |
|------|------|
| Frozen Stage E prediction | `outputs/predictions/singapore_2026_final_prediction.csv` |
| Stage E sorted presentation | `outputs/predictions/singapore_2026_final_prediction_sorted.csv` |
| Grid-corrected experimental | `outputs/predictions/singapore_2026_final_prediction_experimental_grid_corrected.csv` |
| Ridge experimental | `outputs/predictions/singapore_2026_final_prediction_experimental_ridge.csv` |
| Sorted copies (all stages) | `outputs/predictions/*_sorted.csv` |
| Stage snapshots A–E | `outputs/singapore_2026_*_snapshot.md` (frozen originals) |
| Ridge companion (CSV-aligned) | `outputs/singapore_2026_final_snapshot_experimental_ridge_companion.md` |
| Grid resolution | `outputs/singapore_2026_sunday_grid_resolution.csv` |
| Grid cross-check (F1 graphic vs project) | `outputs/singapore_2026_starting_grid_crosscheck.md` |
| Validation | `outputs/singapore_2026_post_model_validation_comparison.csv`, `outputs/singapore_2026_ridge_validation_report.md` |
| Consistency audits | `outputs/singapore_2026_prediction_snapshot_consistency_audit.md`, `outputs/singapore_2026_csv_standardisation_audit.md` |

**Reproduce frozen Stage E (does not regenerate experiments):**
```bash
PYTHONPATH=. .venv/bin/python -m src.predict_singapore --stage final
```

**Pre-race audit experiments** (refuses to overwrite existing outputs): `PYTHONPATH=. .venv/bin/python -m src.singapore_pre_race_audit`
