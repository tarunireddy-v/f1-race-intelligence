# PRE-RACE PREDICTION — GENERATED BEFORE THE 2026 HUNGARIAN GRAND PRIX

**This file must not be modified based on race results. Post-race evaluation will be stored separately.**

**Generation timestamp:** 2026-07-26 04:10 UTC
**Data cutoff:** Historical training through 2026 Belgian GP (R10); Hungary qualifying loaded; **no Hungary race data**.

## Official Qualifying Top 10 (pre-penalty)

- P1 NOR (McLaren) — 77.207s (+0.000s)
- P2 HAM (Ferrari) — 77.219s (+0.012s)
- P3 LEC (Ferrari) — 77.445s (+0.238s)
- P4 ANT (Mercedes) — 77.479s (+0.272s)
- P5 PIA (McLaren) — 77.684s (+0.477s)
- P6 VER (Red Bull Racing) — 77.725s (+0.518s)
- P7 RUS (Mercedes) — 77.760s (+0.553s)
- P8 HAD (Red Bull Racing) — 77.856s (+0.649s)
- P9 LIN (Racing Bulls) — 78.281s (+1.074s)
- P10 HUL (Audi) — 78.639s (+1.432s)

## Actual Starting Grid (post-penalty)

- P1 NOR
- P2 LEC
- P3 PIA
- P4 VER
- P5 HAM [penalty +3]
- P6 RUS
- P7 ANT [penalty +3]
- P8 HAD
- P9 LIN
- P10 HUL
- P11 LAW
- P12 GAS
- P13 COL
- P14 BOR
- P15 OCO
- P16 ALO
- P17 BEA
- P18 SAI
- P19 ALB
- P20 STR
- P21 BOT
- P22 PER

## RF Top 10

- P1 NOR
- P2 HAM
- P3 VER
- P4 LEC
- P5 ANT
- P6 PIA
- P7 HAD
- P8 RUS
- P9 LIN
- P10 GAS

## GB Top 10

- P1 NOR
- P2 HAM
- P3 VER
- P4 LEC
- P5 HAD
- P6 PIA
- P7 ANT
- P8 LIN
- P9 HUL
- P10 RUS

## XGB Top 10

- P1 NOR
- P2 HAM
- P3 LEC
- P4 VER
- P5 HAD
- P6 PIA
- P7 ANT
- P8 HUL
- P9 GAS
- P10 LIN

## Analytical Top 10 (evidence-based ordering)

P1 NOR, P2 LEC, P3 PIA, P4 VER, P5 HAM, P6 RUS, P7 ANT, P8 HAD, P9 LIN, P10 HUL

**Predicted winner:** NOR
**Predicted podium:** NOR / LEC / PIA

## Key Assumptions

- POST feature set: avg_finish_last_3, avg_finish_last_5, avg_quali_last_3, points_last_5, dnf_rate_last_5, constructor_avg_finish_last_3, constructor_points_last_5, is_2026, grid_position, qualifying_position, gap_to_pole_s, teammate_quali_gap_s
- No practice or weather features in ML
- No arbitrary combined score or ensemble
- HAM quali P2 → grid P5; ANT quali P4 → grid P7 (confirmed)
- Full forecast: `hungary_2026_final_forecast.md`