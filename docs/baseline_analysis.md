# Baseline Analysis: Mariana Antaya F1 Prediction Repositories

Reference repositories (read-only, not executed):

- `reference/mariana_2025/` — [2025_f1_predictions](https://github.com/mar-antaya/2025_f1_predictions)
- `reference/mariana_2026/` — [2026_f1_predictions](https://github.com/mar-antaya/2026_f1_predictions)

This document summarizes how the baseline works, how it evolved across 2025 and 2026, and what we should reuse, improve, or replace in the F1 Race Intelligence project.

---

## Repository Structure

### 2025 (`mariana_2025`)

Flat repository with one Python script per race weekend (plus a few early-season variants):

| File | Role |
|------|------|
| `README.md` | Project overview, dependencies, usage |
| `prediction1.py` … `prediction8.py` | Race-by-race prediction scripts (Rounds 1–8) |
| `prediction24.py` | Late-season script (Abu Dhabi, Round 24) |
| `prediction2.py`, `prediction2_nochange.py`, `prediction2_olddrivers.py` | Variants for Round 2 (China) showing iterative experimentation |

**Evolution pattern:** Each file is a self-contained copy-paste script for one upcoming Grand Prix. There is no shared `src/` module. As the 2025 season progresses, scripts accumulate features:

1. **Rounds 1–2:** Qualifying time only → qualifying + 2024 sector times from FastF1
2. **Round 3:** Weather (OpenWeatherMap), wet-performance factors, sector times
3. **Round 4:** Season driver points added
4. **Round 5:** Constructor standings, average 2025 race pace (manual), qualifying time squared, `LastYearWinner` flag
5. **Round 6+:** Clean-air race pace (manual dict), imputation, constructor score; Monaco adds circuit-specific average position change
6. **Round 24:** Switches from `GradientBoostingRegressor` to `XGBRegressor` with monotone constraints

The README claims scripts through `prediction24`, but the cloned repo only contains scripts for rounds 1–8 and 24 (11 prediction files total).

### 2026 (`mariana_2026`)

Smaller, more deliberate structure:

| File | Role |
|------|------|
| `README.md` | Setup, script descriptions, OpenWeatherMap config |
| `prediction1.py` | 2026 Australian GP — XGBoost on manually entered quali + team/regulation features |
| `prediction2.py` | 2026 Chinese GP — XGBoost ranking model blending Australia results + sprint pace |
| `racepace.py` | Non-ML sector-time pace analysis for China |
| `requirements.txt`, `.env.example`, `.gitignore` | Environment scaffolding |

**Evolution pattern:** 2026 decouples from FastF1 entirely in the prediction scripts. Data is manually curated. Two distinct modeling approaches appear by Round 2: (1) quali-to-race time regression with synthetic targets, and (2) composite race-score ranking with leave-one-out validation.

---

## Data Sources

### FastF1 (2025 only)

| Source | Season | Session | Variables | Dynamic / Manual |
|--------|--------|---------|-----------|----------------|
| Same-circuit prior-year session | 2024 | `"R"` (Race) in most scripts; `"Q"` (Qualifying) in `prediction7.py` | `Driver`, `LapTime`, `Sector1Time`, `Sector2Time`, `Sector3Time` | **Dynamic** via `fastf1.get_session(...).load()` |
| FastF1 cache | — | — | Cached API responses | **Dynamic** (`fastf1.Cache.enable_cache("f1_cache")`) |

**Not pulled from FastF1 despite README claims:** telemetry, car data, weather, practice sessions, results tables, grid positions, or multi-race history.

### Manually entered (both repos)

| Source | Season | Used for | Variables | Dynamic / Manual |
|--------|--------|----------|-----------|----------------|
| Current-year qualifying | 2025 / 2026 | Prediction features | `QualifyingTime (s)`, sector splits (2026 China), `GridPosition` | **Manual** — hardcoded per script |
| Driver name mapping | 2025 early rounds | Join quali names → 3-letter codes | Full name → `NOR`, `PIA`, etc. | **Manual** dict |
| Driver → team mapping | 2025 R5+, all 2026 | Constructor features | Team name | **Manual** dict (updated per script) |
| Constructor championship points | 2025 R5+, 2026 | `TeamPerformanceScore` / `TeamScore` | Points normalized by max | **Manual** — updated each race |
| Wet performance factors | 2025 R3–R4, referenced elsewhere | Rain adjustment multiplier | Per-driver float | **Manual** static dict |
| Season driver points | 2025 R4 | `SeasonPoints` | Points | **Manual** |
| Average 2025 race pace | 2025 R5 | `Average2025Performance` | Approximate mean lap time | **Manual** estimates |
| Clean-air race pace | 2025 R6–R8, R24 | `CleanAirRacePace (s)` | Seconds per lap | **Manual** — fixed dict attributed to external `racepace.py` (not in 2025 repo) |
| Monaco avg position change | 2025 R8 | `AveragePositionChange` | Quali→race delta | **Manual** historical estimate |
| 2025 full-season team points | 2026 R1 | `TeamPerformanceScore` base | Championship points | **Manual** |
| 2026 regulation change boost | 2026 R1 | `RegChangeBoost` | Per-team multiplier | **Manual** subjective factors |
| 2026 Australia grid/finish | 2026 R2 | `AusGrid`, `AusFinishPos` | Positions | **Manual** |
| China sprint gaps | 2026 R2 | `SprintGapPerLap_s` | Gap to leader / 15 laps | **Manual** |
| Sprint penalties | 2026 R2 | Adjusted sprint gaps | Time penalties | **Manual** |

### External APIs

| Source | Season | Variables | Dynamic / Manual |
|--------|--------|-----------|----------------|
| OpenWeatherMap forecast | 2025 R3+, 2026 R2 | `RainProbability` (`pop`), `Temperature` | **Dynamic** when API key works; **Manual fallback** (defaults like 20°C, 0–15% rain) |
| OpenWeatherMap (hardcoded defaults) | 2026 R1 | `RainProbability`, `Temperature` | **Manual** constants in script |

### 2026 sector-level quali (China)

Manually entered per driver: `S1`, `S2`, `S3`, `ChinaQuali_s`, `ChinaGrid` — used to derive `UltimateLap_s` (best S1+S2+S3) and gap-to-pole.

---

## Data Collection

### How FastF1 sessions are loaded (2025)

Standard pattern in every 2025 script:

```python
fastf1.Cache.enable_cache("f1_cache")
session_2024 = fastf1.get_session(2024, <round_or_name>, <"R"|"Q">)
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]]
```

- **Round matching:** Each script loads the **2024 edition of the same circuit** (e.g., Bahrain 2025 prediction uses 2024 Bahrain Race).
- **Exception:** `prediction7.py` loads 2024 Emilia Romagna **Qualifying** (`"Q"`) instead of Race — the target becomes qualifying lap times, not race pace.
- **Exception:** `prediction1.py` loads 2024 Australian GP (round 3) but prints "Chinese GP" in output (labeling inconsistency).

### Qualifying information collected

- **2025:** Total qualifying lap time in seconds; early scripts use full driver names, later scripts use 3-letter codes.
- **2026 R1:** Qualifying time + grid position (1–22).
- **2026 R2:** Sector times S1/S2/S3, total quali time, grid position; derived ultimate lap and gap from pole.

All current-year qualifying is **typed manually** into each script after the session — not fetched from FastF1.

### Race information collected

- **2025 training target:** Mean `LapTime` per driver from the loaded 2024 session (race laps aggregated by driver).
- **2025 prediction output:** Predicted mean race lap time per 2025 driver; drivers ranked ascending (fastest predicted lap = predicted winner).
- **2026 R1:** No real race data — synthetic target `QualifyingTime × 1.055`.
- **2026 R2:** Manual Australia finish positions and China sprint gaps used to build training target `RaceScore`, not lap time.

### Lap-level data usage

**Yes, partially:**

- 2025 uses individual race laps from FastF1, then aggregates to per-driver mean lap time and mean sector times.
- `prediction1.py` merges qualifying rows with **every lap row** (not aggregated first), inflating training rows.
- 2026 does not use FastF1 lap data in ML scripts; `racepace.py` uses manually entered sector times only.

### Telemetry usage

**No.** Despite the 2025 README mentioning telemetry, no script calls `get_car_data()`, `pick_driver()`, speed traces, throttle/brake, or DRS data. All timing data is lap/sector level only.

---

## Data Cleaning

| Operation | Where | Details |
|-----------|-------|---------|
| Drop NA lap times | 2025 all | `dropna()` on `LapTime` and/or sector columns |
| Timedelta → seconds | 2025 all | `.dt.total_seconds()` on `LapTime`, `Sector1/2/3Time` |
| Driver name normalization | 2025 R1–R2 | Manual dict: full names → 3-letter codes for FastF1 merge |
| Missing sector times | 2025 R2–R4 | `.fillna(0)` |
| Missing feature imputation | 2025 R6+ | `SimpleImputer(strategy="median")` on feature matrix |
| Driver filtering | 2025 R8, R24 | Keep only drivers present in 2024 FastF1 session: `valid_drivers = merged_data["Driver"].isin(laps_2024["Driver"].unique())` |
| Merge quali + historical | 2025 all | `merge` on driver code; left joins for sector times |
| Qualifying time transform | 2025 R5–R6 | Squared quali time when rain probability < 75% |
| Wet quali adjustment | 2025 R5+ | If `rain_probability >= 0.75`, multiply quali by `WetPerformanceFactor` (factor often undefined in later scripts — latent bug) |
| Constructor score normalization | 2025 R5+, 2026 | `team_points / max(team_points)` |
| Grid penalty feature | 2026 R1 | `(GridPosition - 1) × 0.15` |
| Gap from pole | 2026 R1–R2 | Quali time minus pole time |
| Ultimate lap | 2026 R2, racepace | `S1 + S2 + S3` |
| Sprint gap per lap | 2026 R2 | `(sprint_gap - penalty) / 15` |
| Race score normalization | 2026 R2 | `(AusFinishPos - 1) / 21` blended 50/50 with normalized sprint gap |
| Feature scaling | 2026 R1 | Qualifying columns multiplied by 3.0 before training |
| Sorting for output | All | Sort by predicted time or predicted score ascending |

---

## Features

Features supplied to ML models (final feature sets by era):

| Feature | Source | Meaning | 2025 / 2026 | Keep / Improve / Replace |
|---------|--------|---------|-------------|--------------------------|
| `QualifyingTime (s)` | Manual quali | Current GP qualifying lap time | Both | **Improve** — automate from FastF1; centralize |
| `QualifyingTime` (transformed) | Derived | Quali time or quali² or wet-adjusted quali | 2025 R5+ | **Replace** — ad hoc transforms; use principled weather model |
| `GapFromPole (s)` | Derived | Delta to pole sitter | 2026 R1 | **Keep** — strong, interpretable |
| `GridPosition` / grid penalty | Manual / derived | Starting slot; linear penalty proxy | 2026 R1 | **Improve** — use actual grid from FastF1 |
| `Sector1/2/3Time (s)` | FastF1 2024 race mean | Historical circuit sector pace | 2025 R2–R4 | **Improve** — pull from current-year FP/quali, not prior-year race |
| `TotalSectorTime (s)` | Derived | Sum of mean sector times | 2025 R5+ | **Improve** — same as above |
| `UltimateLap_s` | Manual S1+S2+S3 | Theoretical best lap from sectors | 2026 R2, racepace | **Keep** — useful pace proxy; automate ingestion |
| `ChinaGapFromPole_s` | Derived | China quali gap to pole | 2026 R2 | **Keep** |
| `ChinaGrid` / `AusGrid` | Manual | Qualifying positions at two circuits | 2026 R2 | **Improve** — generalize to rolling recent form |
| `WetPerformanceFactor` | Manual dict | Driver wet-weather multiplier | 2025 R3–R4 | **Replace** — estimate from historical wet sessions |
| `RainProbability` | OpenWeatherMap / default | Forecast rain probability | Both | **Improve** — structured weather features + uncertainty |
| `Temperature` | OpenWeatherMap / default | Forecast air temperature | Both | **Improve** — same |
| `SeasonPoints` | Manual | Driver championship points mid-season | 2025 R4 | **Improve** — automate; watch leakage timing |
| `TeamPerformanceScore` | Manual constructor points / max | Normalized team strength | 2025 R5+, 2026 | **Improve** — automate with point-in-time standings |
| `TeamScore` | Manual 2026 standings | Early-season constructor form | 2026 R2 | **Improve** |
| `AdjustedTeamScore` | Derived | Team score × regulation boost | 2026 R1 | **Replace** — subjective; model team effects from data |
| `RegChangeBoost` | Manual dict | Subjective 2026 regulation impact | 2026 R1 | **Replace** — too hand-tuned for new project |
| `Average2025Performance` | Manual estimates | Rough mean lap pace in current season | 2025 R5 | **Replace** — compute from actual 2025 results programmatically |
| `CleanAirRacePace (s)` | Manual dict (external) | Estimated clean-air race lap time | 2025 R6–R8, R24 | **Improve** — derive from FP long runs / prior races |
| `AveragePositionChange` | Manual circuit history | Avg quali→finish delta at Monaco | 2025 R8 only | **Improve** — generalize circuit overtaking difficulty |
| `LastYearWinner` | Derived binary | 1 if driver won prior year at circuit | 2025 R5–R6 (often unused in X) | **Replace** — weak, sparse signal |
| `AusFinishPos` / sprint gaps | Manual 2026 results | Recent race and sprint performance | 2026 R2 target/features | **Improve** for features; **Replace** as training target component |

---

## Prediction Target

### What the model predicts

| Repo / Script | Target variable | Interpretation | Ranking method |
|---------------|-----------------|----------------|----------------|
| 2025 (all ML scripts) | Mean **race lap time** (seconds) from 2024 FastF1 session | Predicted mean lap time for each 2025 driver | Ascending predicted time → finish order |
| 2026 `prediction1.py` | `TargetLapTime = QualifyingTime × 1.055` | **Synthetic** race lap time derived from quali | Ascending predicted time |
| 2026 `prediction2.py` | `RaceScore` = 0.5 × norm(Aus finish) + 0.5 × norm(sprint gap/lap) | Composite **relative performance score** (lower = better) | Ascending predicted score |

**The baseline does NOT directly predict:**

- Finishing position (classification)
- Win/podium probability
- Total race time or points

**Indirectly:** Finishing order is inferred by sorting predicted lap times or scores. Podium is `head(3)` after sorting.

**Not predicted:** pit stops, DNFs, safety cars, penalties (except manual sprint penalty adjustment in 2026 data prep).

---

## Model

### Algorithm

| Period | Algorithm |
|--------|-----------|
| 2025 R1–R8 | `sklearn.ensemble.GradientBoostingRegressor` |
| 2025 R24 | `xgboost.XGBRegressor` |
| 2026 | `xgboost.XGBRegressor` exclusively |

### Hyperparameters (representative)

| Script | n_estimators | learning_rate | max_depth | Other |
|--------|--------------|---------------|-----------|-------|
| 2025 R1 | 100 | 0.1 | default | `random_state=39` |
| 2025 R3–R4 | 200 | 0.1 | default | `random_state=38` |
| 2025 R5–R6 | 300 | 0.05 | 5 | `random_state=38–39` |
| 2025 R7–R8 | 100 | 0.05–0.7 | 3 | varies by race |
| 2025 R24 | 300 | 0.9 | 3 | `monotone_constraints='(1,0,0,-1,-1)'`, `random_state=39` |
| 2026 R1 | 400 | 0.05 | 4 | `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=0.5`, `max_delta_step=1`, monotone constraints `(1,1,-1,1,1,0)` |
| 2026 R2 | 200 | 0.05 | 3 | `reg_lambda=1.5`, `subsample=0.8`, `colsample_bytree=0.8` |

### Training data

- **2025:** Single 2024 session (~15–20 drivers, expanded to many rows if lap-level merge). Features mix 2025 quali (current race) with 2024 historical pace.
- **2026 R1:** All 22 drivers from Australian GP quali; target is synthetic from their own quali times (circular).
- **2026 R2:** All 22 China quali drivers; target built from Australia race + China sprint outcomes being predicted.

### Testing strategy

| Script | Split | Issues |
|--------|-------|--------|
| 2025 most | `train_test_split`, 10–30% test, random | ~12–20 samples; random split on single-race data |
| 2026 R1 | `train_test_split`, 15% test, `random_state=42` | Train and test from same race; target derived from quali |
| 2026 R2 | **Leave-One-Out CV** (22 folds) | Better for tiny N, but target includes future race information |

**Random states used:** 34, 37, 38, 39, 42 (vary by script, not consistent).

**No temporal cross-validation** across races or seasons.

---

## Evaluation

### Metrics reported

Only **Mean Absolute Error (MAE)** is reported in every ML script. Some scripts also plot feature importances (matplotlib bar charts). No RMSE, R², rank correlation (Spearman/Kendall), podium accuracy, or betting-calibration metrics.

### How MAE is calculated

```python
from sklearn.metrics import mean_absolute_error
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
```

**MAE** = average of `|y_true - y_pred|` across the test fold, in **seconds** (for lap-time targets) or **unitless score** (2026 R2).

**What the reported value means:**

- **2025:** Average absolute error in predicted **mean race lap time** vs. held-out 2024 drivers from the same session. A reported MAE of ~3 seconds means the model's lap-time predictions are typically within 3 seconds of the historical mean lap time. This measures fit to **last year's same-circuit data**, not actual 2025 prediction accuracy.
- **2026 R1:** MAE on synthetic targets (`quali × 1.055`) from a random 15% split of the same 22 drivers — largely measures noise around a deterministic transform, not predictive quality.
- **2026 R2:** Leave-one-out MAE on `RaceScore` — average absolute error in the composite score across LOO folds.

**Important:** MAE is **not** computed against actual 2025/2026 race results in these scripts.

---

## Prediction Pipeline

### 2025 pipeline (typical mid/late season)

```
2024 same-circuit FastF1 session (Race)
        ↓
Extract laps → convert to seconds → aggregate mean lap/sector times per driver
        ↓
Manually enter 2025 qualifying times (+ optional manual pace/team/weather dicts)
        ↓
Merge quali with 2024 aggregates; impute/fill missing values
        ↓
Train GBR/XGB on (features → mean 2024 lap time)
        ↓
Predict mean lap time for each 2025 driver using current quali + manual features
        ↓
Sort ascending by predicted lap time
        ↓
Output predicted winner / podium + MAE on random holdout
```

### 2026 pipeline — Australia (`prediction1.py`)

```
Manual 2026 quali times + grid + team points + regulation boosts + weather defaults
        ↓
Engineer features (gap from pole, grid penalty, adjusted team score)
        ↓
Create synthetic target = quali × 1.055
        ↓
Impute → scale quali features × 3 → random train/test split
        ↓
Train XGBRegressor with monotone constraints
        ↓
Predict race lap time for all 22 drivers
        ↓
Sort by predicted time → print podium + MAE
```

### 2026 pipeline — China (`prediction2.py`)

```
Manual China sector quali + manual Australia grid/finish + manual sprint gaps
        ↓
Derive UltimateLap, gaps, sprint gap/lap, team score, weather (API or default)
        ↓
Build RaceScore target from Australia finish + sprint performance
        ↓
Train XGBRegressor with Leave-One-Out CV
        ↓
Predict RaceScore for all drivers → sort ascending
        ↓
Print full finishing order + LOO MAE + feature importances
```

### 2026 pipeline — `racepace.py` (non-ML)

```
Manual sector times → UltimateLap → delta to fastest → estimated race pace from sprint pole reference + tire delta → sort
```

---

## Changes Between 2025 and 2026

### Architectural shifts

| Area | 2025 | 2026 |
|------|------|------|
| FastF1 usage | Required for training data | **Removed** from ML scripts |
| Model family | GBR → XGB (late season) | XGB only |
| Code organization | One monolithic script per race | Fewer scripts + separate pace analysis |
| Validation | Random `train_test_split` | LOO CV introduced (China) |
| Constraints | Monotone constraints in R24 | Monotone constraints in Australia |
| Environment | Undocumented deps in README | `requirements.txt`, `.env.example` |

### Feature evolution

| Feature | 2025 | 2026 |
|---------|------|------|
| Practice pace | ❌ Not used | ❌ Not used |
| Race pace | ✅ Clean-air dict (manual, static all season) | ✅ Sprint gaps + `racepace.py` sector analysis |
| Weather | ✅ OpenWeatherMap from R3 | ✅ API in R2; hardcoded defaults in R1 |
| Constructor info | ✅ From R5 (manual points) | ✅ 2025 full-season + 2026 early standings |
| Circuit info | ✅ Monaco-only position change | ❌ No circuit-specific features |
| Recent driver form | ✅ `Average2025Performance` (manual, R5 only) | ✅ Australia finish + sprint in China model |
| Additional seasons | ❌ Only 2024 same race | ❌ Still no multi-season FastF1 panel |
| Regulation handling | ❌ | ✅ Manual `RegChangeBoost` (2026 R1) |
| Sector features | From 2024 FastF1 race | From manual 2026 quali sectors |

### Model parameter changes

- Learning rates generally lower in 2026 (0.05 vs up to 0.9 in R24).
- 2026 adds explicit regularization (`reg_lambda`, `max_delta_step`) and monotone constraints tied to domain intuition.
- 2026 R1 trains on **synthetic targets** instead of historical FastF1 lap times — major methodology change.

### Target variable change

- 2025: historical mean race lap time (seconds).
- 2026 R1: synthetic quali-derived lap time (no real race labels).
- 2026 R2: composite ranking score blending prior race + sprint — closer to learning-to-rank but still regression on a hand-built score.

---

## Potential Problems

### Data leakage

| Issue | Severity | Details |
|-------|----------|---------|
| 2026 R2 target includes sprint/race outcomes | **High** | `RaceScore` is built from Australia finishes and China sprint gaps — the same event being predicted. Model learns to reproduce known outcomes. |
| 2026 R1 synthetic target from quali | **High** | Target is a deterministic function of a feature (`Quali × 1.055`). Model mostly learns the transform, not race dynamics. |
| 2026 R2 features include sprint performance | **Medium** | Sprint gaps available before race but highly correlated with target construction. |
| Same-race train/predict (2026 R1) | **High** | Model fit and inference on identical 22-driver matrix. |

### Training on future information

- Constructor points and `Average2025Performance` in mid-2025 scripts are updated manually but not always clearly point-in-time.
- 2026 R1 uses **full 2025** championship standings to predict 2026 Round 1 — acceptable as prior season, but `RegChangeBoost` is subjective hindsight.

### Incorrect validation for temporal data

- Random `train_test_split` on 12–20 drivers from a **single historical race** does not simulate predicting a future race.
- No walk-forward or grouped-by-race CV.
- LOO in 2026 R2 is better for sample size but still optimizes reproducing a leaky target.

### Very small datasets

- Training N ≈ 12–22 drivers per script; sometimes hundreds of lap rows but only ~15 independent drivers.
- High variance; feature importances unstable.

### Manual data entry

- All current-year quali, team points, sprint gaps, regulation boosts, and clean-air pace are hand-entered — transcription error risk (e.g., `prediction1.py` mislabels GP; `prediction5.py` comment still says "Bahrain" for Jeddah; `prediction2.py` X/y row misalignment bug).

### Inconsistent driver names

- Early 2025: full names; later: 3-letter codes.
- 2026 new drivers (Antonelli, Hadjar, etc.) have no 2025 FastF1 mapping in 2025 scripts.
- `prediction7.py` references `WetPerformanceFactor` in rain branch but never defines it.

### Missing values

- RUS quali `None` in Monaco script; handled by imputer.
- New 2025 drivers lack 2024 sector times → filled with 0 or median.

### Features unavailable before race day

- Most features are quali-based (available post-quali) — OK for pre-race prediction timing.
- `CleanAirRacePace` dict appears mid-season but uses static values — not race-specific.
- Weather forecasts are legitimate pre-race inputs.

### Target leakage

- Squared quali time (`QualifyingTime ** 2`) applied to features that correlate directly with lap-time target.
- 2026 synthetic targets guarantee low training error without generalization.

### 2026 regulation assumptions

- Manual `RegChangeBoost` encodes opinions (Mercedes 1.15, Red Bull 0.95) that may not hold under new PU/chassis rules.
- Team rebrands (Audi, Cadillac, Racing Bulls) handled via manual mappings only.

### Implementation bugs (2025)

- `prediction2.py`: `X` (quali drivers) and `y` (2024 lap-time aggregate) are not aligned by driver before `train_test_split`.
- `prediction7.py`: trains on qualifying session lap times as target instead of race pace.
- `prediction1.py`: wrong GP name in print output; uses Australian 2024 data for what README maps to Round 1.

---

## Reusable Components

### KEEP

- **Qualifying time and gap-to-pole** as core pre-race features — strong domain signal.
- **Per-circuit historical baseline** idea (2025's same-race-last-year FastF1 load) — sound starting point if implemented systematically.
- **Sector-level ultimate lap** concept (`racepace.py`, 2026 R2) — interpretable pace proxy.
- **Constructor strength** as a feature — normalize championship points at a point in time.
- **Weather forecast integration** — OpenWeatherMap (or similar) with graceful fallback defaults.
- **Monotone constraints** in XGBoost — sensible when faster quali should not predict slower race time.
- **Ranking by predicted pace** — convert regression output to finishing order for communication.
- **One script per race weekend** workflow for exploratory baselines (adapt into notebooks/pipelines later).

### IMPROVE

- **FastF1 ingestion** — centralize session loading, driver mapping, and lap aggregation in reusable modules instead of copy-paste.
- **Clean-air / race pace** — compute from FP long runs and prior races dynamically instead of static dicts.
- **Cross-validation** — replace random splits with leave-one-driver-out or walk-forward across races/seasons.
- **Feature engineering** — sector times from current weekend FP/quali; rolling recent form across last N races.
- **Evaluation** — add Spearman rank correlation vs actual results, podium/top-10 hit rate, and true out-of-sample backtesting.
- **Team/driver mappings** — single maintained mapping table updated once per season.
- **2026 R2 LOO approach** — keep the validation idea but apply to non-leaky targets.

### REPLACE

- **Manual hardcoding of entire DataFrames per script** — replace with data files or automated FastF1 pulls.
- **Synthetic targets (quali × 1.055)** — replace with real historical race lap times or finishing-position targets.
- **Composite RaceScore target built from the race being predicted** — replace with prior-race features only; predict position or time separately.
- **Subjective `RegChangeBoost` multipliers** — replace with data-driven team performance trends.
- **Random train/test split on single-race micro datasets** — replace with proper temporal backtesting.
- **MAE on training distribution as sole metric** — replace with race-outcome validation against actual results.
- **Static wet-performance dict** — replace with empirical wet-session analysis or drop until supported.
- **Copy-paste script proliferation** — replace with parameterized pipeline (`src/` modules + config per race).

---

*Analysis completed without executing prediction scripts or downloading FastF1 data.*
