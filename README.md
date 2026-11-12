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
marks, shrunk 50%), venue and age.

* **Median forecast.** The recent slope is extrapolated, but growth fades
  with coin age and is capped at ~2.5 h of horizon; the result is pulled
  toward a venue-specific long-run level (pool 2.3, migrating 2.1, curve
  1.7) with an ~8 h mean-reversion time. Curve coins take a haircut
  (pool pricing after graduation is usually below the curve's virtual
  price); a curve coin close to completing (≥ 55 SOL in quote reserves)
  is treated as a likely graduate landing near 2.3–2.8.
* **Uncertainty.** Sigma grows with horizon (random walk + model error),
  so a 7–13 h forecast is honestly wide. Each simulated cap is capped at
  mu + 2 sigma, so no mediocre coin can claim a top place with a
  500x moonshot the lognormal tail would otherwise invent.
* **Places.** A seeded Monte-Carlo (15,000 sims) over the top 150
  candidates by predicted median gives place frequencies. Half of every
  place (0.50) is left unnamed for coins born after the history cut — in
  the practice, 3 of the top 5 at the last answer read could not be named
  from the history.
* **Sizes.** Per place, the Monte-Carlo order-statistic histogram
  (weight 0.4) is mixed with a trend normal (weight 0.6, sd 0.45)
  centred on the cohort level: the place-k cap at the cut, pulled up
  toward the place-1 cap (the top 5 compress together as a cohort
  matures) and drifted by the recent place-1 drift, saturating over
  ~1.5 h. This is what covers coins born after the cut, which the
  Monte-Carlo cannot see. 12% of each place's size mass is left unnamed.
* **top5** is the five coins with the highest predicted medians, in order.

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

| split | this entry | house baseline |
|---|---|---|
| first 3 marks → mark 6 | -47.00 | -52.72 |
| first 4 marks → mark 6 | -34.21 | -37.31 |
| first 5 marks → mark 6 | -29.16 | -29.87 |
| **mean** | **-36.79** | **-39.96** |

The entry beats the house baseline on every split (mean +3.17). For
reference, the even spread scores -45.06 and the baseline scores
-40.41 on the full 7-read practice grading; this entry's full-practice
forecast also names MIHAO VI (0.09) and Google (0.04) at their actual
places 2 and 3 on the last answer read, where 3 of the 5 coins were
not nameable from the history.

## Files for the repo

* `main.py` — the entry (only this is graded)
* `README.md` — this file
* `LICENSE` — MIT, copied from the kit's `starter/`

Do NOT upload `backtest.py`, `kit.tar.gz`, or the `pumpfun-top5-forecast-01-kit/`
directory.
