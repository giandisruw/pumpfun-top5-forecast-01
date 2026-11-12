# pumpfun-top5-forecast-01 — entry

Forecast the top 5 new pump.fun coins by market cap, 24 hours out, for
make.fun competition `pumpfun-top5-forecast-01`.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=pumpfun-top5-forecast-01-kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads `$MAKEFUN_INPUT_DIR/history.ndjson` + `task.json` (defaults `/input`)
and writes `$MAKEFUN_OUTPUT_DIR/forecast.json` (default `/output`).
Python 3.13, standard library only, no network, no installs.
Deterministic: two runs on the same history produce byte-identical output
(fixed RNG seed). Runtime ~2–3 s on the practice input, far under the
300 s limit.

## How it works

Each coin's log10 market cap (in SOL) is tracked across the marks it
appears in: last level, recent slope (least squares over the last few
marks), venue and age.

* **Median forecast.** The recent slope is extrapolated, but growth fades
  with coin age and is capped at ~2.5 h of horizon; the result is pulled
  toward a venue-specific long-run level (pool 2.3, migrating 2.1, curve
  1.7) with a ~12 h mean-reversion time. The slope shrink is
  horizon-dependent (~0.25 at short horizons, ~0.55 at the final's
  ~10 h): at a short horizon the last level pins the answer and the
  slope is mostly noise; at a long horizon the slope is the growth
  signal. Curve coins take a haircut (pool pricing after graduation is
  usually below the curve's virtual price); a curve coin close to
  completing (≥ 55 SOL in quote reserves) is treated as a likely
  graduate landing near 2.3–2.8.
* **Uncertainty.** Sigma grows with horizon (random walk + model error),
  but the random-walk term saturates at ~4 h of effective horizon: a
  coin's log-cap peaks and decays, it does not wander without bound.
  Without the saturation the Monte-Carlo tail invents 10^6 SOL caps at
  long horizons. Each simulated cap is capped at mu + 2 sigma, so no
  mediocre coin can claim a top place with a 500x moonshot the lognormal
  tail would otherwise invent.
* **Places.** A seeded Monte-Carlo (15,000 sims) over the top 150
  candidates by predicted median gives place frequencies, with newborn
  coins drawn INSIDE each sim: a Poisson count from the entry rate
  measured from the history (newborn coins entering the top 5 over the
  recent marks), caps near the cohort's place-3 level. A newcomer must
  beat the cap a named coin actually draws, not its reverted mean. A sim
  where a newcomer holds a place leaves that place's mass unnamed, so
  each place's unnamed share is whatever the newcomer model leaves —
  there is no fixed unnamed share any more (v1 used 0.50).
* **Chart fillers.** Each place is filled to the 100-row cap with the
  coins most likely to be on the answer chart but near-certain not to
  hold the place, at p = 1e-6. The scorer shares a place's unnamed mass
  over the chart's coins NOT named at that place, so naming them shrinks
  the denominator and leaves the unnamed share to the newcomers who may
  take the place. Near-graduation curve coins are excluded from fillers.
* **Sizes.** Per place, the Monte-Carlo order-statistic histogram is
  mixed with a trend normal centred on the cohort level: the place-k cap
  at the cut, pulled up toward the place-1 cap (the top 5 compress
  together as a cohort matures) and drifted by the recent place-1 drift,
  trusted fully. The trend normal's sd and the MC mix weight are both
  horizon-adaptive: tight at short horizons (sd ~0.15, MC weight 0.20)
  so the same code is tight on the 1–2 h practice, wide at long ones
  (sd ~0.50, MC weight 0.40) so it stays honest on the 7–13 h final.
  12% of each place's size mass is left unnamed.
* **top5** is ranked by total Monte-Carlo top-5 probability (after the
  newcomers), tie-broken by predicted median.

All horizon times, bin edges, places and chart size are read from
`task.json`, never hardcoded. Only mints that appear in the history are
named; mints are validated as base58, 32–44 chars.

## Local backtest

`backtest.py` is a **dev-only** script (do NOT upload it to the GitHub
repo). It truncates the practice history to its first 3 / 4 / 5 marks,
runs the entry on each truncated input, and scores the forecast against
the 20:50 read as a single answer read, using the same scoring rules as
`RULES.txt` (order + size log scores, floored at ln(0.0001), unnamed
shares shared over the chart's unnamed coins / bins).

```sh
python3 backtest.py
```

Results on this practice input (mean over the 3 splits):

| split | v2 (this entry) | v1 | house baseline |
|---|---|---|---|
| first 3 marks → mark 6 | -39.38 (order -22.89, size -16.49) | -47.00 | -52.72 |
| first 4 marks → mark 6 | -29.17 (order -16.34, size -12.84) | -34.21 | -37.31 |
| first 5 marks → mark 6 | -25.79 (order -13.18, size -12.61) | -29.16 | -29.87 |
| **mean** | **-31.45** | **-36.79** | **-39.96** |

v2 beats v1 on every split (mean +5.34) and beats the house baseline on
every split (mean +8.51). For reference, the even spread scores -45.06
and the baseline scores -40.41 on the full 7-read practice grading; v1
scored -37.84 there (order -19.85, size -17.99). The v2 backtest mean
order is -17.47 and mean size -13.98 (v1: -20.21 / -16.58).

What changed from v1 to v2, and what each change was worth on the
backtest mean (measured one edit at a time):

* Horizon-adaptive size sigma (0.45 fixed → 0.15 + 0.11·√h) and MC mix
  weight (0.40 fixed → 0.20 short / 0.40 long): -36.79 → -35.46.
* Newcomer displacement (first as a Poisson convolution on predicted
  medians, then as newborn coins drawn inside the Monte-Carlo, so a
  newcomer must beat a named coin's drawn cap, not its reverted mean):
  -35.46 → -33.15 (convolution) → -31.45 (inside the MC). The v1
  failure this fixes: the MC was overconfident about the modal order
  (Claude 0.34 at place 1, actual place 5) and gave displaced coins ~0
  at the place they landed on; Claude's place-5 p is now 0.080.
* Chart fillers (100 rows/place, p = 1e-6 for chart-likely,
  near-certain-non-holder coins): part of the -33.71 → -33.15 step.
* Horizon-dependent slope shrink (0.50 fixed → 0.25 short / 0.55 long)
  and mean-reversion tau 8 h → 12 h: -33.15 → -32.91.
* Cohort drift trusted fully (shrink 0.50 → 1.00): -32.91 → -31.97.
* Per-coin sigma saturating at ~4 h of effective horizon: no change on
  the backtests (they are all short-horizon), but it fixes the
  long-horizon stress check — without it the Monte-Carlo tail put 35.8%
  of the size mass in bin 81 (cap ≥ 10^6 SOL) at a 26 h horizon and
  3.0% at 10 h; with it, 0.0057 at 26 h and 0.00026 at 10 h.

Things tried and rejected on the backtests (kept at v1 values): slope
window 3 instead of 4 (no effect); slope shrink 0.65 (worse) and 0.0–
0.25 (better on backtests but kills the growth signal the final's 10 h
horizon needs, so the horizon-dependent shrink above was used instead);
per-coin sigma base 0.15 instead of 0.30 (worse — the 0.30 encodes real
model error); MC cap 2.5σ instead of 2σ (worse); N_SIMS 30000 instead
of 15000 (no change); N_CANDIDATES 250 instead of 150 (no change);
growth cap 3.5 h instead of 2.5 h (never binds at short horizons,
kept at 2.5); NEW_CAP_SD 0.35 instead of 0.25 (no change); rate window
3 marks instead of 12 (0.07 better on backtests, but a 20 min window
can catch 0 entrants on the final's 11 h history and switch the
newcomer model off entirely, so 12 marks was kept).

Long-horizon stress check (final-like task.json: t0 02:00Z, history
13:00Z, reads 20:00Z–02:00Z, run on the practice history; plus a true
~10 h horizon variant): runtime ~3 s, deterministic (two runs
byte-identical), exactly 3 keys, ≤ 1 MB, per-place sums ≤ 1, top5 five
distinct mints from the history. At 10 h the size mass is sensibly
wide (top bin p 0.03–0.17, 78 bins named, bin-81 mass 0.00026, no p
near 1); at 26 h bin-81 mass is 0.0057. One known tension, not a defect:
on the practice history (explosive phase, drift +0.9/h) paired with a
10 h horizon, the model leaves place 1 mostly unnamed (0.93) because
the drifted newcomer level sits far above the reverted incumbents; on
the real final the 11 h history will be mature, the measured drift
small, and the balance much more even — the backtests cannot
distinguish this, so no hack was applied to chase the practice number.

## Files for the repo

* `main.py` — the entry (only this is graded)
* `README.md` — this file
* `LICENSE` — MIT, copied from the kit's `starter/`

Do NOT upload `backtest.py`, `kit.tar.gz`, or the `pumpfun-top5-forecast-01-kit/`
directory.
