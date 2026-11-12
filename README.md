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
  there is no fixed unnamed share any more (v1 used 0.50). **v3:** the
  cap on the expected newcomer count, `N_NEW_MAX`, raised 3.0 → 6.0
  (see "v3" below).
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

## v3 — what changed and why (2026-11-12)

**The trigger.** v2 was submitted and ACCEPTED, but scored badly on the
real practice grading: **-41.4 (order -18.30, size -23.07), rank
399/459** — below the house baseline (-40.41) and far behind the leader
@chaboss-work (-31.8). v1 had scored -37.84 (order -19.85, size
-17.99). So v2 improved the order (+1.55) but **the size score
collapsed by -5.08**, and that is the whole story.

**Root cause (diagnosed with numbers).** v2 over-trusts the explosive
cohort drift and over-tightens the size sigma. Place-1 cap trajectory in
the practice history: 3.036 → 3.115 → 3.258 → 3.468 → 3.561 (increments
+0.079, +0.144, +0.210, +0.092 per 10 min); median drift g = +0.863/h,
trusted fully (shrink 1.0, up from v1's 0.5). At the mid read (0.667 h
horizon) the drift term pushes the size centre **+0.465 log10** above
the cut while the size sigma is only **0.24**. But the actual growth
decelerates (the last increment is already down to +0.092) and the
explosive phase does not sustain, so actual caps land ~1–2 sigma below
the drifted centre and get tiny probabilities. This matches the
literature: only ~0.63% of pump.fun tokens graduate, most die within a
day, and fast liquidity accumulation (not sustained drift) is the
strongest graduation predictor (arXiv 2602.14860); for log score,
calibration matters more than sharpness — an overconfident narrow
forecast is punished heavily when wrong (Gneiting et al. 2007).

**Why the single-read backtest missed it.** The backtest scores a
forecast against ONE read (mark 6, 10–30 min after the cut), where the
drift is still roughly valid and the answer sits near the cut. The real
practice scores ONE forecast against SEVEN reads (21:00–22:00, 10–70
min after the cut), where the drift has time to decelerate. The
single-read backtest systematically rewards the exact overconfidence
the 7-read practice punishes. A new **multi-read backtest**
(`backtest_multi.py`, dev-only, do NOT upload) scores each forecast
against ALL future marks after the cut (6 reads total) to mimic the
7-read grading; it shows the degradation pattern (split K=3: read 1
-34.83, read 2 -37.26, read 3 -41.25 — later reads are worse).

**The v3 change.** Only one lever improved BOTH backtests: the
newcomer cap `N_NEW_MAX` 3.0 → **6.0**. The entry rate is
hundreds/hour, so lam = 0.20·rate·h is large and the cap binds hard;
v2's 3.0 underestimated how many fresh coins displace the named
leaders, especially on the later reads where composition churns.
Multi-read backtest: 3.0 → -32.70, 4.0 → -32.38, 5.0 → -32.18,
6.0 → **-32.16** (best), 8.0 → -32.25 (order degrades past the
optimum). The real practice horizon (mid 0.667 h, reads to 1.17 h) is
longer than the backtest's max 0.5 h, so the cap binds even harder
there — 6.0 is the safer side of the optimum for the real grading
window. Single-read backtest also improves: -31.45 → **-31.02**.

**Rejected by the backtests (kept at v2 values).** Drift shrink
1.0 → 0.5 and size-sigma widening (SIZE_SD_A 0.15 → 0.30) both made the
multi-read backtest WORSE (-33.48 and -33.68 vs -32.70), even though
v1's wider size scored better on the real practice (-17.99 vs -23.07).
The local backtests only cover the explosive phase and cannot
reproduce the real practice's drift deceleration, so they are a biased
proxy for size parameters — but the newcomer cap is the one lever that
improves both the backtest and, by the v1-vs-v2 leaderboard evidence,
the real practice. The leader's approach ("day-calibrated crash
hazard") would need a fundamentally different size model (a crash/decay
mixture instead of a drifted normal); that is out of scope for this
iteration given the deadline.

**Honest assessment.** v3 beats v2 on both backtests (+0.54
multi-read, +0.43 single-read) and should recover much of the size
collapse, but it is **not expected to reach #1 (-31.8)** on the real
practice: the backtest total (-32.16) is already near the leader's
real-practice score, yet v2's backtest (-32.70) translated to a real
-41.4 — the backtests are optimistic proxies and the size parameters
cannot be validated locally. v3 is a solid improvement worth
submitting; #1 would need the crash-hazard size model.

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

| split | v3 (this entry) | v2 | v1 | house baseline |
|---|---|---|---|---|
| first 3 marks → mark 6 | -36.62 (order -21.35, size -15.27) | -39.38 (order -22.89, size -16.49) | -47.00 | -52.72 |
| first 4 marks → mark 6 | -30.41 (order -17.07, size -13.34) | -29.17 (order -16.34, size -12.84) | -34.21 | -37.31 |
| first 5 marks → mark 6 | -26.02 (order -13.33, size -12.69) | -25.79 (order -13.18, size -12.61) | -29.16 | -29.87 |
| **mean** | **-31.02** | **-31.45** | **-36.79** | **-39.96** |

v3 beats v2 on the single-read mean (+0.43), driven by a large win on
split K=3 (+2.76); splits K=4 (-1.24) and K=5 (-0.23) are slightly
worse — at the shortest horizons the newcomer cap barely binds, so the
higher cap costs a little there. v3 beats the house baseline on every
split (mean +8.94). For reference, the even spread scores -45.06 and
the baseline scores -40.41 on the full 7-read practice grading; v1
scored -37.84 there (order -19.85, size -17.99), v2 scored -41.4
(order -18.30, size -23.07). The v3 backtest mean order is -17.25 and
mean size -13.77 (v2: -17.47 / -13.98).

### Multi-read backtest (the more honest proxy)

`backtest_multi.py` (dev-only, do NOT upload) scores each forecast
against ALL future marks after the cut (6 reads total), mimicking the
real practice's 7-read grading instead of a single read. v2: **-32.70**
(order -18.36, size -14.35). v3: **-32.16** (order -18.06, size
-14.11) — **+0.54**. Per-read, v3 improves the longer-horizon reads
most (split K=3 read 3: -41.25 → -38.45, +2.80; read 2: -37.26 →
-34.77, +2.49; read 1: -34.83 → -32.70, +2.13) — exactly where v2's
size collapsed on the real practice — while the shortest-horizon reads
cost a little (split K=4 read 1: -27.92 → -30.62, -2.70; K=5 read 1:
-25.79 → -26.02, -0.23), where the named coins are more stable and
the higher newcomer cap over-dilutes them. The net is positive because
the long-horizon wins are larger, and the real practice (reads to
1.17 h) is longer-horizon than any backtest read, so the tradeoff
favours v3 there. Caveat: the multi-read backtest only reaches 30 min
past the cut (the real practice reaches 70 min), so it is still an
optimistic proxy — but it is the best local proxy available, and it is
the one that caught v2's failure mode.

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

Do NOT upload `backtest.py`, `backtest_multi.py`, `kit.tar.gz`, or the
`pumpfun-top5-forecast-01-kit/` directory.
