# Hungary 2026 — Pre-Qualifying Context

Objective circuit and weekend context for interpreting the V1 pre-qualifying assessment.
This document does **not** contain arbitrary numeric model features.

## Circuit Characteristics

| Attribute | Value | Notes |
|-----------|-------|-------|
| Circuit | Hungaroring | |
| Location | Budapest, Hungary | FastF1: Budapest |
| Length | 4.381 km | Official circuit length |
| Corners | 14 | |
| Character | High-downforce, technical, rhythm-heavy | |
| Overtaking | Relatively limited | Track position historically important |
| 2026 Straight Mode | 4 Straight Mode zones | 2026 regulation context |
| 2026 Overtake Mode | Detection approaching Turn 14, deployment onto main straight | Subject to official 2026 configuration |

## Tyres (weekend allocation)

Pirelli compounds **C3 (Hard) / C4 (Medium) / C5 (Soft)** — confirm against official Pirelli/FIA source before race.

## Pit loss estimate

Approximately **20.56 seconds** total pit-lane time loss vs staying out — useful race strategy context; confirm against official or timing source.

## Historical disruption rates (Hungaroring)

| Event | Approximate rate | Notes |
|-------|------------------|-------|
| Safety Car | ~13% | Historical incidence — context only |
| VSC | ~25% | Historical incidence — context only |

## Same-circuit constants vs driver-varying signals

Circuit length, corner count, pit-loss estimate, and historical SC/VSC rates describe the **race environment** shared by all 22 drivers. They cannot distinguish between drivers within one Hungary prediction.

Driver-varying interpretation in V1 relies on separate evidence layers (ML form, FP pace, sectors, teammate gaps).

## Track-Specific Signals Used (V1)

Rather than arbitrary downforce or Active Aero scores, V1 uses objective driver-varying signals:

1. **FP3 Sector 2 relative pace** — proxy for performance through Hungary's technical medium-speed section (high-downforce/chassis signal).
2. **FP3 teammate gap** — separates driver performance from general car performance.
3. **FP2 long-run pace** — approximate race-run evidence with reliability filtering (HIGH/MEDIUM only for team summaries).
4. **FP1 / FP2 / FP3 progression** — current-weekend performance trend (descriptive flags, not ML inputs).
5. **Qualifying position and gap to pole** — will be added to the POST-qualifying model after qualifying.
6. **Team front-row lockout** — will be calculated after qualifying as contextual information.

### Possible V2 historical feature (not implemented)

`high_downforce_form` — rolling form at comparable high-downforce circuits (Monaco, Singapore, Hungary). Not implemented in V1 because the comparable-race sample is small and would increase complexity.

## Actual FP3 Weather (FastF1)

| Measure | Value |
|---------|-------|
| Air temperature (avg) | 24.9 °C |
| Air temperature (min / max) | 23.3 / 26.0 °C |
| Track temperature (avg) | 52.1 °C |
| Track temperature (min / max) | 48.8 / 54.5 °C |
| Humidity (avg) | 28.6 % |
| Wind speed (avg) | 1.59 m/s |
| Rainfall during FP3 | No |

**Race weather FORECAST** (Formula 1 official, 26 July 2026 race day — not FP3 actuals):

| Field | Value |
|-------|-------|
| Label | **FORECAST** |
| Expected air temp | 17–32 °C (max ~32 °C at race time) |
| Rain probability | ~40 % (F1 official); bulk precipitation expected after session |
| Conditions | Warm, slightly humid; mainly dry race expected |
| Wind | Light–moderate (~10 km/h), gusts 35–40 km/h |
| Track temp note | Potentially ~50 °C given warm ambient |

Source: [F1 Hungarian GP weather forecast](https://www.formula1.com/en/latest/article/what-is-the-weather-forecast-for-the-2026-hungarian-grand-prix.3yj8zISPt1alwyYCD8ecYL)

Do **not** conflate this forecast with FP3 actual conditions above. Forecast weather is **not** used as an ML input in V1.

### Weather interpretation

FP3 conditions were dry with moderate air and track temperatures. FP3 session notes: 1 red-flag period(s) on track, 3 yellow-flag period(s) on track, session aborted / restarted. These observations are contextual only and are **not** used as ML model inputs.

