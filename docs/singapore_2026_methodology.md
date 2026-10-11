# Singapore 2026 — Methodology

## Stages

| Stage | Information cutoff | ML inputs |
|-------|-------------------|-----------|
| A Pre-weekend (retrospective) | 2026 R1–R16 + historical Singapore | PRE features only |
| B Post-FP1 | + FP1 timing (analytical) | Practice not in ML; `rf_pre_weekend_rank` unchanged from A |
| C Post-Sprint Qualifying | + SQ (planned) | TBD |
| D Post-Sprint | + Sprint (planned) | TBD |
| E Final pre-race | + GP qualifying + grid | POST features |

## Leakage rules

- Rolling features use `shift(1)` before windows (same as Hungary V1).
- Singapore 2026 race and future sessions never enter earlier stages.

## Baselines

Qualifying-position and grid-position baselines are reported alongside ML when qualifying/grid exist.
