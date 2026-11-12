#!/usr/bin/env python3
"""Forecast the top 5 new pump.fun coins by market cap, 24 hours out.

Reads $MAKEFUN_INPUT_DIR/history.ndjson + task.json (defaults /input) and
writes $MAKEFUN_OUTPUT_DIR/forecast.json (default /output). Python 3.13,
standard library only, no network, deterministic (fixed RNG seed).

Model (original design, not the house baseline):
  * Each coin's log10 market cap (in SOL) is tracked across the marks it
    appears in: last level, recent slope (least squares over the last few
    marks, shrunk 50%), venue and age.
  * The median log-cap at the mid answer-read horizon is predicted as a
    damped extrapolation of the recent slope (growth fades with coin age
    and is capped at ~2.5 h of horizon) pulled toward a venue-specific
    long-run level with an ~8 h mean-reversion time: coins keep growing
    for a while after graduation, then fade. Curve coins take a haircut
    (pool pricing after graduation is usually below the curve's virtual
    price); a curve coin close to graduation (>= 55 SOL in quote
    reserves) is treated as a likely graduate landing near 2.3-2.8.
  * Uncertainty grows with horizon (random-walk + model error), so the
    forecast is honest about how little a 7-13 h prediction can pin down.
  * Place probabilities come from a seeded Monte-Carlo over the top
    candidates' lognormal caps; half of every place (0.50) is left
    unnamed for coins born after the history cut (in the practice, 3 of
    the top 5 at the last read were not nameable from the history).
  * Size distributions per place mix the Monte-Carlo order-statistic
    histogram (weight 0.4) with a trend normal (weight 0.6, sd 0.45)
    centred on the cohort level — the place-k cap at the cut, pulled up
    toward the place-1 cap and drifted by the recent place-1 drift —
    which is what covers coins born after the cut, and leave ~12%
    unnamed, so a missed bin costs far less than the -9.21 floor.
"""

import bisect
import json
import math
import os
import random
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------- constants

SEED = 20261112
N_SIMS = 15000
N_CANDIDATES = 150
UNNAMED_SHARE = 0.50          # share of each place left for unborn/unseen coins
SIZE_NAMED_SUM = 0.88         # share of each place's size mass named
SIZE_MIX_MC = 0.4             # weight of the Monte-Carlo histogram in sizes
SIZE_MIX_SD = 0.45            # sd of the trend normal mixed into sizes
DRIFT_TAU_H = 1.5             # cohort-level drift saturates over ~this long
GROWTH_CAP_H = 2.5            # growth only extrapolates this far (hours)
GROWTH_AGE_TAU_H = 4.0        # growth fades with coin age
REVERT_TAU_H = 8.0            # mean-reversion time (hours)
LONG_RUN = {"pool": 2.3, "migrating": 2.1, "curve": 1.7}
CURVE_HAIRCUT = 0.20          # pool pricing after graduation vs curve price
CURVE_GRAD_QUOTE_SOL = 55.0  # quote reserves suggesting graduation is near
CURVE_GRAD_LANDING = 2.35    # typical log-cap a fresh graduate lands near
MIN_NAMED_P = 0.000001        # rows below this (after flooring) are dropped
MICRO = 1_000_000

B58 = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")


# ---------------------------------------------------------------- helpers

def parse_iso(s):
    """ISO-8601 UTC string -> epoch seconds (float)."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def logcap_sol(lamports):
    """log10(market cap in SOL) from a lamports integer/string."""
    try:
        v = int(lamports)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return math.log10(v) - 9.0


def valid_mint(mint):
    return (
        isinstance(mint, str)
        and 32 <= len(mint) <= 44
        and all(c in B58 for c in mint)
    )


def floor6(p):
    return math.floor(p * MICRO) / MICRO


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bin_of(value, edges):
    """Size bin 0..81 for a log10 cap in SOL (bisect over the 81 edges)."""
    return bisect.bisect_right(edges, value)


# ---------------------------------------------------------------- input

def load_input(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    marks = []
    path = os.path.join(input_dir, "history.ndjson")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                if m.get("ok") is True and isinstance(m.get("rows"), list):
                    marks.append(m)
    marks.sort(key=lambda m: parse_iso(m.get("at")) or 0.0)
    return task, marks


def build_series(marks, t0):
    """mint -> list of (t_minutes_since_t0, logcap, venue, rank, quote_sol)."""
    series = {}
    for m in marks:
        t = parse_iso(m.get("at"))
        if t is None or t0 is None:
            continue
        tmin = (t - t0) / 60.0
        for r in m.get("rows") or []:
            mint = r.get("mint")
            if not valid_mint(mint):
                continue
            y = logcap_sol(r.get("market_cap_lamports"))
            if y is None:
                continue
            q = r.get("quote_reserves_lamports")
            try:
                qsol = int(q) / 1e9 if q not in (None, "") else None
            except (TypeError, ValueError):
                qsol = None
            series.setdefault(mint, []).append(
                (tmin, y, r.get("venue") or "curve", r.get("rank") or 9999, qsol,
                 parse_iso(r.get("created_at")))
            )
    for mint in series:
        series[mint].sort(key=lambda p: p[0])
    return series


# ---------------------------------------------------------------- model

def fit_slope(points):
    """Least-squares slope (log10 per hour) over the last <=4 observations."""
    pts = points[-4:]
    if len(pts) < 2:
        return 0.0
    n = len(pts)
    xs = [p[0] / 60.0 for p in pts]  # hours since t0
    ys = [p[1] for p in pts]
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return slope


def predict_coin(obs, cut_epoch, horizon_h):
    """Return (mu, sigma) of the coin's log10 cap at the mid answer read."""
    last = obs[-1]
    y0 = last[1]
    venue = last[2]
    created = last[5]
    age_h = max(0.0, (cut_epoch - created) / 3600.0) if created else 24.0

    slope = fit_slope(obs)
    slope = max(-1.0, min(1.5, slope)) * 0.5  # shrink: slopes do not persist
    growth = slope * min(horizon_h, GROWTH_CAP_H) * math.exp(-age_h / GROWTH_AGE_TAU_H)

    level = LONG_RUN.get(venue, 1.7)
    mu_raw = y0 + growth
    mu = level + (mu_raw - level) * math.exp(-horizon_h / REVERT_TAU_H)

    sigma2 = 0.30 ** 2 + (0.13 * horizon_h) ** 2
    if venue == "curve":
        mu -= CURVE_HAIRCUT
        # A curve coin close to completing is likely to graduate and land
        # near a fresh pool cap; give it that floor with wide uncertainty.
        qsol = last[4]
        if qsol is not None and qsol >= CURVE_GRAD_QUOTE_SOL:
            mu = max(mu, CURVE_GRAD_LANDING)
        sigma2 += 0.25 ** 2
    elif venue == "migrating":
        sigma2 += 0.15 ** 2
    if len(obs) == 1:
        sigma2 += 0.12 ** 2
    return mu, math.sqrt(sigma2)


# ---------------------------------------------------------------- forecast

def build_forecast(task, marks):
    places = int(task.get("places") or 5)
    edges = [float(e) for e in task.get("size", {}).get("edges", [])]
    if len(edges) != 81:
        edges = [2.0 + 0.05 * i for i in range(81)]
    n_bins = len(edges) + 1

    t0 = parse_iso(task.get("t0"))
    reads = [parse_iso(r) for r in task.get("reads") or []]
    reads = [r for r in reads if r is not None]
    cut_epoch = None
    for m in marks:
        t = parse_iso(m.get("at"))
        if t is not None and (cut_epoch is None or t > cut_epoch):
            cut_epoch = t
    if cut_epoch is None:
        cut_epoch = t0 or 0.0
    if reads:
        mid_read = sorted(reads)[len(reads) // 2]
        horizon_h = max(0.0, (mid_read - cut_epoch) / 3600.0)
    else:
        horizon_h = 1.0

    series = build_series(marks, t0)
    preds = {}
    for mint, obs in series.items():
        preds[mint] = predict_coin(obs, cut_epoch, horizon_h)

    # Rank candidates by predicted median; the Monte-Carlo decides places.
    ranked = sorted(preds.items(), key=lambda kv: (-kv[1][0], kv[0]))
    cands = ranked[:N_CANDIDATES]
    c_mints = [c[0] for c in cands]
    c_mus = [c[1][0] for c in cands]
    c_sigs = [c[1][1] for c in cands]
    n = len(cands)

    # --- seeded Monte-Carlo over the candidates' log-caps ----------------
    # Each coin's simulated cap is capped at mu + 2 sigma: a lognormal tail
    # would otherwise let any of ~150 mediocre coins draw +2.7 sigma and
    # claim the top places with a 500x moonshot no real coin makes, which
    # inflates the top places' sizes into absurd bins (a 10^6 SOL cap).
    # The cap barely touches ranking among the plausible leaders.
    place_counts = [[0] * n for _ in range(places)]
    place_bin_counts = [[0] * n_bins for _ in range(places)]
    place_sum = [0.0] * places
    place_sumsq = [0.0] * places
    if n:
        rng = random.Random(SEED)
        gauss = rng.gauss
        caps = [mu + 2.0 * sg for mu, sg in zip(c_mus, c_sigs)]
        for _ in range(N_SIMS):
            vals = [min(gauss(mu, sg), cap)
                    for mu, sg, cap in zip(c_mus, c_sigs, caps)]
            order = sorted(range(n), key=vals.__getitem__, reverse=True)[:places]
            for k, idx in enumerate(order):
                place_counts[k][idx] += 1
                v = vals[idx]
                place_sum[k] += v
                place_sumsq[k] += v * v
                place_bin_counts[k][bin_of(v, edges)] += 1

    # --- place probabilities ----------------------------------------------
    probabilities = []
    for k in range(places):
        rows = []
        for idx in range(n):
            freq = place_counts[k][idx] / N_SIMS
            if freq <= 0:
                continue
            p = floor6(freq * (1.0 - UNNAMED_SHARE))
            if p >= MIN_NAMED_P:
                rows.append((c_mints[idx], k + 1, p))
        rows.sort(key=lambda r: (-r[2], r[0]))
        probabilities.extend(
            {"mint": mint, "place": place, "p": p}
            for mint, place, p in rows[:100]
        )

    # --- sizes -------------------------------------------------------------
    # Two components. (a) The Monte-Carlo order-statistic histogram of the
    # surviving history coins. (b) A trend normal centred on the cohort
    # level: the place-k cap at the cut, pulled up toward the place-1 cap
    # (the top 5 compress together as a cohort matures: at the last
    # practice mark places 1-5 sat within 0.28 log of each other) and
    # drifted up by the recent place-1 drift, saturating over ~1.5 h.
    # Component (b) is what covers coins born after the cut, which the
    # Monte-Carlo cannot see and which take most top-5 places at long
    # horizons; without it the sizes sit far too low.
    last_caps = []
    if marks:
        last_rows = sorted(marks[-1].get("rows") or [], key=lambda r: r.get("rank") or 9999)
        last_caps = [logcap_sol(r.get("market_cap_lamports")) for r in last_rows]
        last_caps = [c for c in last_caps if c is not None]
    # Recent drift of the place-1 cap (log10 per hour), shrunk and capped.
    p1_series = []
    for m in marks:
        rows = sorted(m.get("rows") or [], key=lambda r: r.get("rank") or 9999)
        if rows:
            c = logcap_sol(rows[0].get("market_cap_lamports"))
            if c is not None:
                p1_series.append(c)
    if len(p1_series) >= 2:
        incs = sorted(p1_series[i + 1] - p1_series[i]
                      for i in range(max(0, len(p1_series) - 4), len(p1_series) - 1))
        cadence = float(task.get("cadence_minutes") or 10)
        drift_per_hour = incs[len(incs) // 2] * (60.0 / cadence)
        g = max(0.0, min(0.6, drift_per_hour * 0.5))
    else:
        g = 0.3
    h_drift = DRIFT_TAU_H * (1.0 - math.exp(-horizon_h / DRIFT_TAU_H))
    sizes = []
    for k in range(places):
        if n == 0 or N_SIMS == 0:
            continue
        mean = place_sum[k] / N_SIMS
        var = max(0.0, place_sumsq[k] / N_SIMS - mean * mean)
        sd_mc = math.sqrt(var)
        if last_caps and len(last_caps) >= k + 1:
            cap_k = last_caps[k]
            cap_1 = last_caps[0]
            center = max(cap_k, cap_1 - 0.10 * k) + g * h_drift
            center = min(center, cap_1 + 0.8)
            center = max(center, cap_k)
            sd = SIZE_MIX_SD
        else:
            center = mean
            sd = max(SIZE_MIX_SD, sd_mc)
        probs = []
        total = 0.0
        for b in range(n_bins):
            mc = place_bin_counts[k][b] / N_SIMS
            lo = -math.inf if b == 0 else edges[b - 1]
            hi = math.inf if b == n_bins - 1 else edges[b]
            broad = norm_cdf((hi - center) / sd) - norm_cdf((lo - center) / sd)
            p = (SIZE_MIX_MC * mc + (1.0 - SIZE_MIX_MC) * broad) * SIZE_NAMED_SUM
            p = floor6(p)
            if p >= MIN_NAMED_P:
                probs.append((b, p))
                total += p
        # Rounding down can only shrink the sum; guard anyway.
        if total > 1.0:
            scale = 1.0 / total
            probs = [(b, floor6(p * scale)) for b, p in probs if floor6(p * scale) > 0]
        sizes.extend({"place": k + 1, "bin": b, "p": p} for b, p in probs)

    # --- top5: highest predicted medians ----------------------------------
    top5 = [mint for mint, _ in ranked[:places]]
    pad = 32
    while len(top5) < places:
        top5.append("1" * pad)
        pad += 1

    return {"probabilities": probabilities, "top5": top5, "sizes": sizes}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, marks = load_input(input_dir)
    forecast = build_forecast(task, marks)
    out_name = task.get("output") or "forecast.json"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast))
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
