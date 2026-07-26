# Dataset Schema — F1 Race Intelligence

## Overview

**Core modeling unit:** one row = one driver entering one Grand Prix.

**Historical training window (initial):**

| Period | Scope |
|--------|-------|
| 2024 | Full season |
| 2025 | Full season |
| 2026 | Completed races **before** the Hungarian GP (round resolved from FastF1 schedule) |

**Grid size:** not fixed. 2024–2025 seasons typically field **~20 drivers**; 2026 expands to **22 drivers / 11 constructors** under new regulations. Entrant counts may also differ between practice, qualifying, and race within the same weekend (e.g. reserve drivers). The pipeline must never assume exactly 20 rows per event.

**Temporal integrity rule:** every feature for a historical race must use only information that would have been available **before that race started** (or before qualifying / post-qualifying cutoff, depending on model mode). Rolling and circuit-history features must be computed with an explicit **shift** so the current race is never included in its own feature vector.

**Primary V1 target:** `actual_finishing_position` (integer).

**Not in V1:** winner probability, total race time regression, telemetry-derived features, **bulk historical FP1/FP2 as model training features**.

### V1 data ingestion scope

| Data | V1 historical bulk collection | V1 model features |
|------|------------------------------|-------------------|
| Race results | Yes (2024, 2025, 2026 pre-Hungary) | Targets + form/history derivation |
| Qualifying results | Yes | Post-qualifying model only |
| FP1 / FP2 (historical) | **No** — postponed to V2 | **No** |
| FP1 / FP2 (2026 Hungarian GP) | Separate later task | **Not ML features in V1** — analytical signal alongside prediction |

**Important:** Hungary 2026 FP1/FP2 cannot be inserted directly as training features in V1 because historical practice data is not part of the training matrix. For the 2026 Hungarian GP, practice pace will be presented as a **separate current-weekend analytical signal** alongside the ML model output (qualitative/quantitative comparison, not a joined feature column in V1).

---

## 1. Identifiers

These columns uniquely define a row and support joins, filtering, and grouping. They are **not** automatically model features — using `race_name`, `circuit`, or raw `driver` / `constructor` strings directly in a tree model may cause overfitting or require encoding; `season`, `round`, and encoded IDs are safer when needed.

| Column | Type | Description |
|--------|------|-------------|
| `season` | int | Championship year (e.g. 2024) |
| `round` | int | Round number on the calendar (1–24) |
| `race_name` | str | Official event name (e.g. `Australian Grand Prix`) |
| `circuit` | str | Circuit identifier — prefer FastF1 `Location` or a normalized circuit key |
| `driver` | str | FastF1 3-letter driver code (e.g. `VER`) |
| `constructor` | str | Team name at time of entry (handle mid-season renames via mapping table) |
| `race_date` | date | Race session date (UTC or local — pick one convention and document it) |

**Composite primary key:** `(season, round, driver)`.

**Notes:**

- `constructor` must reflect the team the driver raced for **that weekend** (not career-default).
- Sprint weekends share the same `season`, `round`, `race_name`, `circuit`, and `race_date` as the GP; session-level features (FP1/FP2/quali) distinguish timing.

---

## 2. Regulation / Season Context

Fields that distinguish performance across regulation eras and account for grid expansion in 2026. **No arbitrary numeric performance boosts** (contrast with baseline `RegChangeBoost` hand-tuning). These are structural indicators for later feature engineering and modeling.

| Column | Type | Definition |
|--------|------|------------|
| `regulation_era` | category | `pre_2026` for seasons 2024–2025; `2026_regulation` for 2026+ |
| `is_2026_regulation` | bool | True when `season >= 2026` |
| `current_season_races_completed` | int | Number of completed GPs in this season **before** the current round |
| `constructor_is_new` | bool | True if constructor has **no** rows in training history from prior regulation era (see below) |
| `driver_constructor_races` | int | Count of prior GP entries for this `(driver, constructor)` pairing in the **current season**, shifted (excludes current race) |
| `driver_constructor_avg_finish` | float | Mean finishing position for this `(driver, constructor)` pairing in the current season, shifted |

**Era mapping:**

| Seasons | `regulation_era` | Notes |
|---------|------------------|-------|
| 2024–2025 | `pre_2026` | Previous car / aero regulations; DRS-based overtaking |
| 2026+ | `2026_regulation` | New technical regulations; Active Aero and **Overtake Mode** replace DRS as primary overtaking aid |

**Leakage risk:** **HIGH** for `driver_constructor_*` if the current race is included. **LOW** for `regulation_era` / `is_2026_regulation` (deterministic from season).

**Pre-qualifying availability:** Yes (all derivable before the race).

**Post-qualifying availability:** Yes.

**Implementation status:** designed in schema; **not yet computed** in ingestion layer.

### Driver / constructor changes

Driver form and constructor form remain **separate feature groups**. When a driver changes teams between seasons (or mid-season):

- **Driver form** (`driver_avg_finish_prev_*`, etc.) follows the **driver** across constructors using shifted prior-race history.
- **Constructor form** follows the **current constructor** only.
- **`driver_constructor_*`** features measure experience with the **current** constructor only.

**Critical rule:** do **not** carry statistics from a driver's **previous** constructor into `driver_constructor_races` or `driver_constructor_avg_finish`.

For a new driver/constructor pairing (e.g. driver moved teams for 2026):

```text
driver_constructor_races = 0
driver_constructor_avg_finish = NaN
```

Historical **driver** form from prior seasons/constructors may still exist in driver form columns — the model can combine driver pedigree + current team context explicitly.

### New constructors (2026)

Constructors debuting in 2026 (e.g. rebrands / new entries) have **no** 2024–2025 historical rows. Do **not** invent backfilled statistics.

| Feature type | Handling for new constructor |
|--------------|----------------------------|
| `constructor_is_new` | `True` |
| Constructor form (rolling points, avg finish) | `NaN` until sufficient **current-season** observations exist |
| Constructor circuit history | `NaN` until prior editions under that constructor identity exist |
| Driver form for their drivers | May still use driver history from previous teams |

Maintain a **constructor identity mapping table** (rebrand continuity is a separate, documented decision — e.g. whether Audi inherits Sauber history).

### Regulation-aware recency (future modeling — not implemented)

The 2026 regulation change may make 2024–2025 performance less predictive of 2026 outcomes. During model development, evaluate (but do **not** pre-select) approaches such as:

- **Sample weighting by recency** — higher weight on recent seasons/races
- **Regulation-era indicator** — `regulation_era` or `is_2026_regulation` as a model feature or split factor
- **Training-window comparison** — train on all seasons vs 2026-only vs post-reg-change subset; compare validation metrics
- **2026-only validation** — hold out 2026 races for out-of-sample evaluation while training on 2024–2025

No weighting formula is chosen yet. Document findings when modeling begins.

---

## 3. Driver Form Features

All rolling windows use **prior races only**. Implementation pattern:

```text
For driver D at (season S, round R):
  include results from races where (season < S) OR (season = S AND round < R)
  NEVER include (season = S, round = R)
```

DNFs and non-classified finishes should be handled consistently (document choice in pipeline — recommended: include DNF as finishing position = max starters + 1 or use a separate `dnf_rate` rather than imputing positions in rolling averages without flagging).

| Column | Type | Definition |
|--------|------|------------|
| `driver_avg_finish_prev_3` | float | Mean classified finishing position over previous 3 GP entries |
| `driver_avg_finish_prev_5` | float | Mean classified finishing position over previous 5 GP entries |
| `driver_avg_quali_pos_prev_3` | float | Mean qualifying position over previous 3 GPs |
| `driver_points_prev_5` | int | Sum of championship points scored in previous 5 GPs |
| `driver_avg_positions_gained` | float | Mean `(qualifying_position − finishing_position)` over previous N races (recommend N=5; store N in pipeline config) |
| `driver_dnf_rate_prev_n` | float | DNFs / entries over previous N races (recommend N=10 for stability) |
| `driver_podium_rate_prev_n` | float | Podium finishes / entries over previous N races |
| `driver_championship_position_pre_race` | int | Driver standings position **before** this GP (points from all prior rounds in season + carry-over if applicable) |
| `driver_championship_points_pre_race` | int | Cumulative championship points **before** this GP |

**Leakage risk:** **HIGH** if the current race result, current quali, or post-race points update is included in rolling windows or championship columns. Must shift by one race.

**Pre-qualifying availability:** Yes (all are known before FP1).

**Post-qualifying availability:** Yes (unchanged).

---

## 4. Constructor Form Features

Computed by aggregating **all drivers** who scored a classified result for the constructor in prior races, or by primary team entry if multiple cars — pipeline must pick one rule and apply consistently.

| Column | Type | Definition |
|--------|------|------------|
| `constructor_avg_finish_prev_3` | float | Mean finishing position of constructor's cars over previous 3 GPs |
| `constructor_points_prev_5` | int | Constructor points scored in previous 5 GPs |
| `constructor_dnf_rate_prev_n` | float | Share of constructor entries that DNF'd over previous N GPs |
| `constructor_championship_position_pre_race` | int | Constructor standings position before this GP |
| `constructor_championship_points_pre_race` | int | Constructor cumulative points before this GP |

**Leakage risk:** **HIGH** if current-race finishes or points are included. Same shift rule as driver form.

**Pre-qualifying availability:** Yes.

**Post-qualifying availability:** Yes.

---

## 5. Circuit History Features

Only **previous editions** of the same circuit (or normalized circuit key) may be used. Same shift: exclude current `(season, round)`.

| Column | Type | Definition |
|--------|------|------------|
| `driver_circuit_avg_finish` | float | Driver's mean finishing position at this circuit in all prior seasons/rounds |
| `driver_circuit_avg_quali_pos` | float | Driver's mean qualifying position at this circuit (prior editions only) |
| `driver_circuit_best_finish` | int | Driver's best classified finish at this circuit |
| `constructor_circuit_avg_finish` | float | Constructor's mean car finishing position at this circuit (prior editions) |

**Minimum sample handling:** if no prior visits (rookie / new circuit), set to `NaN` and impute or use global prior — do not use current-race data.

**Leakage risk:** **HIGH** if current weekend quali or race is included in "historical" circuit stats.

**Pre-qualifying availability:** Yes.

**Post-qualifying availability:** Yes.

---

## 6. Circuit Metadata

Static or slow-changing circuit descriptors. **Do not invent numeric scores in the pipeline until an objective sourcing/scoring methodology is defined.**

| Column | Type | Definition | Status |
|--------|------|------------|--------|
| `circuit_length_km` | float | Track length in km | Objective — source from FastF1 event metadata / official circuit data |
| `circuit_corner_count` | int | Number of corners | Objective — source from official circuit data |
| `circuit_track_type` | category | e.g. `permanent`, `street`, `hybrid` | Objective — curated mapping table |
| `circuit_overtaking_difficulty` | float / ordinal | How hard it is to overtake | **Requires objective methodology** — e.g. overtakes per race from FastF1 historical data, not hand scores |
| `circuit_sc_rate` | float | Historical safety-car / VSC incidence rate at circuit | Derive from FastF1 race control messages (prior seasons) |
| `circuit_pit_lane_time_loss_s` | float | Estimated time loss for a pit stop vs staying out | Derive from FastF1 pit-stop duration data (prior races at circuit) |
| `circuit_tyre_degradation` | float / ordinal | Typical degradation severity | **Requires objective methodology** — e.g. stint length vs lap-time drop from historical stints |
| `circuit_downforce_requirement` | category | e.g. `high`, `medium`, `low` | **Requires objective methodology** — e.g. derived from wing level distribution in FP/race telemetry or team consensus mapping with documented rules |
| `circuit_active_aero_suitability` | float / ordinal | How well 2026 Active Aero rules reward this circuit profile | **Requires objective methodology** — regulation-specific; define before use |
| `circuit_overtake_mode_effectiveness` | float / ordinal | Expected benefit of **Overtake Mode** at this circuit under 2026 regulations | **Requires objective methodology** — regulation-specific; define before use |

**2026 aerodynamic context:** the 2026 regulations replace the previous DRS-centric overtaking system with **Active Aero** and **Overtake Mode**. Do not assign circuit scores until an objective methodology is documented. Do not use legacy DRS assumptions for 2026 races.

**Leakage risk:** **LOW** for static physical attributes. **MEDIUM** for `circuit_sc_rate` and `circuit_pit_lane_time_loss_s` if current-race incidents are included — use only prior editions.

**Pre-qualifying availability:** Yes (static / historical).

**Post-qualifying availability:** Yes.

**V1 recommendation:** ship `circuit_length_km`, `circuit_corner_count`, `circuit_track_type`, and data-derived `circuit_sc_rate` / `circuit_pit_lane_time_loss_s` first; hold subjective ordinals until methodology is documented in a separate `docs/circuit_scoring.md`.

---

## 7. Practice Features (FP1 & FP2)

> **V1 status:** Historical FP1/FP2 columns below are **postponed to V2** for model training. V1 bulk ingestion collects **Race + Qualifying only**. Current-weekend FP (2026 Hungarian GP) is collected separately and used as an **analytical overlay**, not as a V1 model input.

Practice data comes from FastF1 session loads (`"FP1"`, `"FP2"`). All times in seconds. Positions computed within session among classified laps.

| Column | Type | Definition |
|--------|------|------------|
| `fp1_best_lap_s` | float | Fastest valid lap time in FP1 |
| `fp1_position` | int | Rank by FP1 best lap (1 = fastest) |
| `fp1_gap_to_fastest_s` | float | `fp1_best_lap_s − session_fastest_fp1` |
| `fp1_laps_completed` | int | Count of completed laps in FP1 |
| `fp2_best_lap_s` | float | Fastest valid lap time in FP2 |
| `fp2_position` | int | Rank by FP2 best lap |
| `fp2_gap_to_fastest_s` | float | Gap to FP2 session fastest |
| `fp2_laps_completed` | int | Count of completed laps in FP2 |
| `fp2_long_run_avg_pace_s` | float | Mean lap time on long-run stints (see below) |
| `fp2_long_run_pace_rank` | int | Rank by `fp2_long_run_avg_pace_s` (1 = fastest) |
| `fp2_lap_time_std_s` | float | Std dev of valid lap times in FP2 (consistency proxy) |
| `fp1_to_fp2_improvement_s` | float | `fp1_best_lap_s − fp2_best_lap_s` (positive = improved) |

### Long-run pace — ideal calculation

1. Load FP2 laps from FastF1; exclude in/out laps, pit laps, and laps flagged deleted/trapped.
2. Identify **stints** as consecutive laps on the same compound without pitting.
3. Keep stints with ≥ **5 consecutive flying laps** (configurable threshold).
4. Optionally exclude first 1–2 laps of each stint ( tyre scrub / push lap ).
5. Compute mean lap time per qualifying stint; take the **best stint average** per driver as `fp2_long_run_avg_pace_s`.
6. Rank drivers by that best-stint average.

**Uncertainty and distortions:**

| Factor | Effect | Mitigation |
|--------|--------|------------|
| Tyre compound | Soft vs hard not comparable raw | Compute long-run per compound; prefer like-for-like stints or normalize using compound offset learned from prior sessions |
| Fuel load | Unknown fuel correction | Compare relative ranks within session rather than absolute seconds cross-weekend; optional fuel-corrected model later |
| Red flags | Truncated stints | Drop stints below lap threshold; flag `fp2_session_disrupted` |
| Rain / mixed conditions | Lap times not comparable | Use `TrackStatus` from FastF1; exclude wet laps or entire session if predominantly wet |
| Disrupted sessions | Few flying laps | Fall back to `NaN` for long-run fields; rely on single-lap FP2 metrics |

### Missing FP1 / FP2 handling

| Scenario | Handling |
|----------|----------|
| **Sprint weekend** (no FP2) | `fp2_*` = `NaN`; optionally derive long-run proxy from FP1 only or from sprint quali/race the day before — document if used; do not mix without a `session_type` flag |
| **FP1 cancelled** | `fp1_*` = `NaN`; model uses FP2-only practice signal |
| **Both disrupted** | All practice columns `NaN`; rely on form / circuit / weather features |
| **Rookie / no timed lap** | `NaN` with `fp1_laps_completed = 0` |

Add boolean flags (recommended): `fp1_available`, `fp2_available`, `is_sprint_weekend`.

**Leakage risk:** **LOW** if only FP1/FP2 from **current weekend** are used and quali/race are excluded. **MEDIUM** if FP3 or quali laps accidentally included in practice aggregates.

**Pre-qualifying availability:** Yes — FP1 and FP2 occur before qualifying.

**Post-qualifying availability:** Yes.

---

## 8. Weather Features

Separate **forecast** (known pre-race) from **actual** (observed during sessions) to avoid lookahead.

### Forecast features (pre-race decision time)

Available after forecast is issued — for historical backtests, use forecast valid at a defined cutoff (e.g. 24h before FP1 or official FIA forecast if archived).

| Column | Type | Definition |
|--------|------|------------|
| `forecast_air_temp_c` | float | Forecast air temperature at race start window |
| `forecast_track_temp_c` | float | Forecast track temperature (if available; else derive from air temp model) |
| `forecast_humidity_pct` | float | Forecast relative humidity |
| `forecast_rainfall` | bool / float | Rain expected (bool) or probability (0–1) |
| `forecast_wind_speed_kmh` | float | Forecast wind speed |

### Actual session weather (for analysis & optional post-hoc features)

**Do not use actual race-day weather in pre-qualifying historical features** unless simulating a forecast that would have existed at prediction time.

| Column | Type | Definition |
|--------|------|------------|
| `actual_fp1_air_temp_c` | float | Observed air temp during FP1 |
| `actual_fp2_air_temp_c` | float | Observed during FP2 |
| `actual_race_air_temp_c` | float | Observed during race |
| `actual_race_track_temp_c` | float | Observed during race |
| `actual_race_rainfall` | bool | Rain during race |

**Leakage risk:** **HIGH** for `actual_race_*` in pre-qualifying or post-qualifying models if prediction is made before race start. **LOW** for forecast fields if cutoff is enforced. **MEDIUM** for `actual_fp2_*` in pre-qualifying model if prediction cutoff is before FP2 — use only sessions available at cutoff.

**Pre-qualifying model:** `forecast_*` + optionally `actual_fp1_*` if predicting after FP1.

**Post-qualifying model:** `forecast_*` + `actual_fp1_*` + `actual_fp2_*` (not race actuals).

**Sources:** FastF1 `Session.weather_data` for actuals; external forecast API (e.g. OpenWeatherMap) or archived forecast for `forecast_*`.

---

## 9. Qualifying Features

**POST-QUALIFYING MODEL ONLY.** Must be excluded from pre-qualifying feature set.

| Column | Type | Definition |
|--------|------|------------|
| `qualifying_position` | int | **Official** qualifying classification (1 = pole) — from FastF1 results |
| `grid_position` | int | Starting grid position (may differ from quali due to penalties) |
| `qualifying_best_lap_s` | float | Best **valid** qualifying lap time (seconds) — from non-deleted laps |
| `gap_to_pole_s` | float | `qualifying_best_lap_s − pole_time` (pace-based gap) |
| `gap_to_teammate_s` | float | Driver best lap minus teammate best lap (negative = faster than teammate) |

### Official classification vs pace lap

`qualifying_position` and `qualifying_best_lap_s` are **separate features**:

- **Do not** overwrite official `qualifying_position` based on fastest lap ordering.
- Deleted laps, track limits, and session rulings can cause the official pole sitter to differ from the fastest stored lap (observed in 2025 Hungary: LEC official P1 vs NOR fastest valid lap).

The ingestion pipeline emits a **validation warning** when position order and best-lap order disagree. Neither value is auto-corrected.

**Leakage risk:** **CRITICAL** if used in pre-qualifying model. Quali occurs after FP2; including these features in a "pre-quali" pipeline invalidates backtests.

**Pre-qualifying availability:** No.

**Post-qualifying availability:** Yes.

---

## 10. Targets

| Column | Type | Definition |
|--------|------|------------|
| `actual_finishing_position` | int | Classified race finishing position (1 = winner) |
| `classified_status` | category | e.g. `Finished`, `+N Laps`, `DNF`, `DSQ`, `NC` |
| `dnf` | bool | True if driver did not classify as finisher |
| `points_scored` | int | Championship points awarded for this race |

### Primary target: `actual_finishing_position`

**Complications:**

1. **DNFs:** Driver may have no meaningful "position" vs competitors on pace — options: (a) assign last-place + 1 among classified finishers, (b) model DNF as separate binary task, (c) use rank-based loss that handles ties. Document choice in modeling spec.
2. **Not classified (NC) / DSQ:** Exclude from training or treat as missing target; do not impute arbitrary positions.
3. **Regulation changes (2026):** Same target definition but field spread and DNF rate may shift — monitor calibration by season.
4. **Shared last positions:** Multiple DNFs may share position; ranking metrics (Spearman) may be more stable than raw MAE on position integers.
5. **Sprint weekends:** Target remains **Sunday GP finishing position**, not sprint result (unless building a separate sprint model).

**Leakage risk:** **CRITICAL** — targets must never appear in features.

**Winner probability:** Not in V1 schema.

---

## 11. Two Prediction Modes

### Mode A — Pre-Qualifying Model (V1)

**Prediction cutoff:** before qualifying (no grid information).

**V1 feature groups included:**

| Group | Included in V1 |
|-------|----------------|
| Identifiers | For indexing only — not all as model inputs |
| Driver form | Yes |
| Constructor form | Yes |
| Regulation / season context | Yes |
| Circuit history | Yes |
| Circuit metadata | Yes (objective fields first) |
| Weather (forecast only at cutoff) | Yes |
| Practice (FP1, FP2) | **No — V2** |
| Qualifying / grid | **No** |

**V1 note:** Without historical practice in training, pre-qualifying V1 relies on form, constructor, circuit, and weather signals. Hungary 2026 FP1/FP2 are compared **alongside** the model output, not fed into it.

**Use case:** Early-weekend predictions before quali; supplemented by practice analysis for Hungary 2026.

---

### Mode B — Post-Qualifying Model

**Prediction cutoff:** after qualifying (grid known).

**Feature groups included:**

| Group | Included |
|-------|----------|
| Everything in Mode A | Yes |
| Qualifying features | Yes |
| Grid position | Yes |

**Use case:** Race-day predictions with full grid information.

---

Store `model_mode` ∈ `{pre_quali, post_quali}` in dataset metadata or maintain two exported feature matrices from the same base table.

---

## 12. Data Sources — Full Column Registry

Legend:

- **Pre-Q:** available for pre-qualifying model at defined cutoff
- **Post-Q:** available for post-qualifying model
- **Leakage:** None / Low / Medium / High / Critical

| Column Name | Feature Group | Data Type | Source | Pre-Q? | Post-Q? | Leakage Risk | Notes |
|-------------|---------------|-----------|--------|--------|---------|--------------|-------|
| `season` | Identifiers | int | FastF1 `EventSchedule` | Yes | Yes | None | Not a default model feature |
| `round` | Identifiers | int | FastF1 | Yes | Yes | None | |
| `race_name` | Identifiers | str | FastF1 | Yes | Yes | None | |
| `circuit` | Identifiers | str | FastF1 `Location` | Yes | Yes | None | Normalize to circuit key |
| `driver` | Identifiers | str | FastF1 | Yes | Yes | None | 3-letter code |
| `constructor` | Identifiers | str | FastF1 `Session.results` | Yes | Yes | None | Weekend-specific |
| `race_date` | Identifiers | date | FastF1 event schedule | Yes | Yes | None | |
| `regulation_era` | Regulation / season | category | Derived from season | Yes | Yes | None | `pre_2026` or `2026_regulation` |
| `is_2026_regulation` | Regulation / season | bool | Derived from season | Yes | Yes | None | |
| `current_season_races_completed` | Regulation / season | int | Derived from schedule | Yes | Yes | High | Shift required |
| `constructor_is_new` | Regulation / season | bool | Derived vs constructor history | Yes | Yes | Low | True when no prior-era rows |
| `driver_constructor_races` | Regulation / season | int | Derived (current pairing) | Yes | Yes | High | Shift; exclude prior constructor |
| `driver_constructor_avg_finish` | Regulation / season | float | Derived (current pairing) | Yes | Yes | High | Shift; exclude prior constructor |
| `driver_avg_finish_prev_3` | Driver form | float | Derived from FastF1 race results | Yes | Yes | High | Shift required |
| `driver_avg_finish_prev_5` | Driver form | float | Derived | Yes | Yes | High | Shift required |
| `driver_avg_quali_pos_prev_3` | Driver form | float | Derived from FastF1 quali | Yes | Yes | High | Shift required |
| `driver_points_prev_5` | Driver form | int | Derived from FastF1 points | Yes | Yes | High | Shift required |
| `driver_avg_positions_gained` | Driver form | float | Derived (quali + race) | Yes | Yes | High | Shift required |
| `driver_dnf_rate_prev_n` | Driver form | float | Derived | Yes | Yes | High | Shift required |
| `driver_podium_rate_prev_n` | Driver form | float | Derived | Yes | Yes | High | Shift required |
| `driver_championship_position_pre_race` | Driver form | int | Derived / FastF1 standings | Yes | Yes | High | Points before this round only |
| `driver_championship_points_pre_race` | Driver form | int | Derived / FastF1 | Yes | Yes | High | |
| `constructor_avg_finish_prev_3` | Constructor form | float | Derived | Yes | Yes | High | Shift required |
| `constructor_points_prev_5` | Constructor form | int | Derived | Yes | Yes | High | |
| `constructor_dnf_rate_prev_n` | Constructor form | float | Derived | Yes | Yes | High | |
| `constructor_championship_position_pre_race` | Constructor form | int | Derived | Yes | Yes | High | |
| `constructor_championship_points_pre_race` | Constructor form | int | Derived | Yes | Yes | High | |
| `driver_circuit_avg_finish` | Circuit history | float | Derived from prior FastF1 races | Yes | Yes | High | Prior editions only |
| `driver_circuit_avg_quali_pos` | Circuit history | float | Derived | Yes | Yes | High | |
| `driver_circuit_best_finish` | Circuit history | int | Derived | Yes | Yes | High | |
| `constructor_circuit_avg_finish` | Circuit history | float | Derived | Yes | Yes | High | |
| `circuit_length_km` | Circuit metadata | float | FastF1 / official | Yes | Yes | Low | |
| `circuit_corner_count` | Circuit metadata | int | Official / curated CSV | Yes | Yes | Low | |
| `circuit_track_type` | Circuit metadata | category | Curated mapping | Yes | Yes | Low | |
| `circuit_overtaking_difficulty` | Circuit metadata | float | **TBD methodology** | Yes | Yes | Medium | Derive from overtake counts |
| `circuit_sc_rate` | Circuit metadata | float | FastF1 race control | Yes | Yes | Medium | Prior races only |
| `circuit_pit_lane_time_loss_s` | Circuit metadata | float | FastF1 pit data | Yes | Yes | Medium | Prior races only |
| `circuit_tyre_degradation` | Circuit metadata | float | **TBD methodology** | Yes | Yes | Medium | |
| `circuit_downforce_requirement` | Circuit metadata | category | **TBD methodology** | Yes | Yes | Low | |
| `circuit_active_aero_suitability` | Circuit metadata | float | **TBD methodology** | Yes | Yes | Low | 2026 Active Aero |
| `circuit_overtake_mode_effectiveness` | Circuit metadata | float | **TBD methodology** | Yes | Yes | Low | 2026 Overtake Mode |
| `fp1_best_lap_s` | Practice | float | FastF1 FP1 laps | Yes | Yes | Low | |
| `fp1_position` | Practice | int | Derived | Yes | Yes | Low | |
| `fp1_gap_to_fastest_s` | Practice | float | Derived | Yes | Yes | Low | |
| `fp1_laps_completed` | Practice | int | FastF1 FP1 laps | Yes | Yes | Low | |
| `fp2_best_lap_s` | Practice | float | FastF1 FP2 laps | Yes* | Yes | Low | *Sprint: missing |
| `fp2_position` | Practice | int | Derived | Yes* | Yes | Low | |
| `fp2_gap_to_fastest_s` | Practice | float | Derived | Yes* | Yes | Low | |
| `fp2_laps_completed` | Practice | int | FastF1 FP2 laps | Yes* | Yes | Low | |
| `fp2_long_run_avg_pace_s` | Practice | float | Derived from FP2 stints | Yes* | Yes | Medium | Compound/rain aware |
| `fp2_long_run_pace_rank` | Practice | int | Derived | Yes* | Yes | Low | |
| `fp2_lap_time_std_s` | Practice | float | Derived | Yes* | Yes | Low | |
| `fp1_to_fp2_improvement_s` | Practice | float | Derived | Yes* | Yes | Low | |
| `fp1_available` | Practice | bool | Derived | Yes | Yes | None | |
| `fp2_available` | Practice | bool | Derived | Yes | Yes | None | |
| `is_sprint_weekend` | Practice | bool | FastF1 schedule | Yes | Yes | None | |
| `forecast_air_temp_c` | Weather | float | Forecast API / archive | Yes | Yes | Medium | Enforce forecast cutoff |
| `forecast_track_temp_c` | Weather | float | Forecast API | Yes | Yes | Medium | |
| `forecast_humidity_pct` | Weather | float | Forecast API | Yes | Yes | Medium | |
| `forecast_rainfall` | Weather | bool/float | Forecast API | Yes | Yes | Medium | |
| `forecast_wind_speed_kmh` | Weather | float | Forecast API | Yes | Yes | Medium | |
| `actual_fp1_air_temp_c` | Weather | float | FastF1 weather | Yes** | Yes | Medium | **Only if cutoff after FP1 |
| `actual_fp2_air_temp_c` | Weather | float | FastF1 weather | Yes*** | Yes | Medium | ***Only if cutoff after FP2 |
| `actual_race_air_temp_c` | Weather | float | FastF1 weather | No | No**** | Critical | ****Analysis only |
| `actual_race_track_temp_c` | Weather | float | FastF1 weather | No | No | Critical | |
| `actual_race_rainfall` | Weather | bool | FastF1 weather | No | No | Critical | |
| `qualifying_position` | Qualifying | int | FastF1 quali results | No | Yes | Critical | Official classification; post-Q only |
| `grid_position` | Qualifying | int | FastF1 starting grid | No | Yes | Critical | Post-Q only |
| `qualifying_best_lap_s` | Qualifying | float | FastF1 quali laps (valid) | No | Yes | Critical | Separate pace feature |
| `gap_to_pole_s` | Qualifying | float | Derived | No | Yes | Critical | |
| `gap_to_teammate_s` | Qualifying | float | Derived | No | Yes | Critical | |
| `actual_finishing_position` | Target | int | FastF1 race results | No | No | Critical | Never a feature |
| `classified_status` | Target | category | FastF1 | No | No | Critical | |
| `dnf` | Target | bool | FastF1 | No | No | Critical | |
| `points_scored` | Target | int | FastF1 | No | No | Critical | |

---

## 13. Final Modeling Table — Example Row

Fictional placeholders only (Driver `XXX` at `Example Circuit`, Round 99, 2099):

| Column | Example Value |
|--------|---------------|
| `season` | 2099 |
| `round` | 99 |
| `race_name` | Example Grand Prix |
| `circuit` | example_circuit |
| `driver` | XXX |
| `constructor` | Example Team |
| `race_date` | 2099-06-15 |
| `regulation_era` | pre_2026 |
| `is_2026_regulation` | False |
| `current_season_races_completed` | 5 |
| `constructor_is_new` | False |
| `driver_constructor_races` | 6 |
| `driver_constructor_avg_finish` | 4.50 |
| `driver_avg_finish_prev_3` | 5.33 |
| `driver_avg_finish_prev_5` | 6.20 |
| `driver_avg_quali_pos_prev_3` | 4.00 |
| `driver_points_prev_5` | 47 |
| `driver_avg_positions_gained` | 1.23 |
| `driver_dnf_rate_prev_n` | 0.10 |
| `driver_podium_rate_prev_n` | 0.25 |
| `driver_championship_position_pre_race` | 3 |
| `driver_championship_points_pre_race` | 112 |
| `constructor_avg_finish_prev_3` | 4.50 |
| `constructor_points_prev_5` | 89 |
| `constructor_dnf_rate_prev_n` | 0.05 |
| `constructor_championship_position_pre_race` | 2 |
| `constructor_championship_points_pre_race` | 198 |
| `driver_circuit_avg_finish` | 7.00 |
| `driver_circuit_avg_quali_pos` | 6.50 |
| `driver_circuit_best_finish` | 2 |
| `constructor_circuit_avg_finish` | 5.00 |
| `circuit_length_km` | 5.00 |
| `circuit_corner_count` | 15 |
| `circuit_track_type` | permanent |
| `circuit_overtaking_difficulty` | NaN |
| `circuit_sc_rate` | 0.45 |
| `circuit_pit_lane_time_loss_s` | 22.5 |
| `circuit_tyre_degradation` | NaN |
| `circuit_downforce_requirement` | NaN |
| `circuit_active_aero_suitability` | NaN |
| `circuit_overtake_mode_effectiveness` | NaN |
| `fp1_best_lap_s` | 89.123 |
| `fp1_position` | 5 |
| `fp1_gap_to_fastest_s` | 0.456 |
| `fp1_laps_completed` | 23 |
| `fp2_best_lap_s` | 88.901 |
| `fp2_position` | 4 |
| `fp2_gap_to_fastest_s` | 0.312 |
| `fp2_laps_completed` | 28 |
| `fp2_long_run_avg_pace_s` | 90.500 |
| `fp2_long_run_pace_rank` | 3 |
| `fp2_lap_time_std_s` | 0.87 |
| `fp1_to_fp2_improvement_s` | 0.222 |
| `fp1_available` | True |
| `fp2_available` | True |
| `is_sprint_weekend` | False |
| `forecast_air_temp_c` | 24.0 |
| `forecast_track_temp_c` | 38.0 |
| `forecast_humidity_pct` | 55.0 |
| `forecast_rainfall` | False |
| `forecast_wind_speed_kmh` | 12.0 |
| `actual_fp1_air_temp_c` | 23.5 |
| `actual_fp2_air_temp_c` | 25.1 |
| `qualifying_position` | 4 |
| `grid_position` | 4 |
| `qualifying_best_lap_s` | 87.654 |
| `gap_to_pole_s` | 0.321 |
| `gap_to_teammate_s` | -0.098 |
| `actual_finishing_position` | 3 |
| `classified_status` | Finished |
| `dnf` | False |
| `points_scored` | 15 |

---

## 14. Dataset Build Order

Recommended pipeline implementation sequence:

1. **Reference tables**
   - Circuit key normalization (`circuit` ↔ FastF1 `Location`)
   - Driver / constructor ID mapping (handle renames, mid-season changes, 2026 grid expansion)
   - Constructor identity / `constructor_is_new` mapping (2026 entrants vs 2024–2025 history)
   - Sprint weekend calendar flags per `(season, round)`

2. **Raw session ingestion (FastF1) — V1**
   - **Race + Qualifying only** for 2024, 2025, 2026 pre-Hungary → `data/raw/race_results.csv`, `qualifying_results.csv`
   - Do **not** bulk-download historical FP1/FP2 (V2)
   - Hungary 2026 FP1/FP2: separate current-weekend script later
   - Dynamic entrant-count validation; qualifying official-vs-pace mismatch warnings
   - Per-season checkpoint saves; stop cleanly on rate limit (no retry loops)

3. **Results spine**
   - Build one row per `(season, round, driver)` with identifiers and targets only
   - Report entrant counts per event; do not flag fewer classified finishers as corruption

4. **Regulation / season context**
   - `regulation_era`, `is_2026_regulation`, `current_season_races_completed`
   - `constructor_is_new`, `driver_constructor_races`, `driver_constructor_avg_finish`

5. **Target columns**
   - `actual_finishing_position`, `classified_status`, `dnf`, `points_scored`

6. **Championship snapshot features**
   - Pre-race driver and constructor points / positions (shifted)

7. **Rolling form features**
   - Driver and constructor rolling windows with strict temporal shift
   - Unit tests: no row's target round appears in its own history

8. **Circuit history features**
   - Per driver and constructor at normalized circuit key

9. **Practice features**
   - FP1/FP2 single-lap metrics, then long-run stint logic
   - Sprint-weekend missingness flags

10. **Weather features**
    - Actual session weather from FastF1
    - Forecast weather from external source with documented cutoff

11. **Qualifying features**
    - Post-qualifying column set only; keep in separate feature view or column list

12. **Circuit metadata join**
    - Objective fields first (`length`, `corners`, `track_type`, data-derived SC/pit loss)
    - Placeholder NaN for subjective fields pending methodology doc

13. **Quality checks**
    - Leakage audit script (feature max date < race date)
    - Missingness report by season and sprint vs non-sprint
    - Duplicate key check on `(season, round, driver)`

14. **Export modeling datasets**
    - `data/processed/modeling_base.parquet` — full table
    - `data/processed/features_pre_quali.parquet` — Mode A column list
    - `data/processed/features_post_quali.parquet` — Mode B column list

15. **Documentation**
    - Update this schema when columns are added
    - Add `docs/circuit_scoring.md` before filling subjective circuit columns

---

*Schema version: V1 — baseline design. No race data downloaded; no models trained.*
