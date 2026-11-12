#!/usr/bin/env python3
"""Forecast the top 5 new pump.fun coins by market cap, 24 hours out.

Reads $MAKEFUN_INPUT_DIR/history.ndjson + task.json (defaults /input) and
writes $MAKEFUN_OUTPUT_DIR/forecast.json (default /output). Python 3.13,
standard library only, no network, deterministic (fixed RNG seed).

Model (v3; the house baseline is a different, simpler program):
  * Each coin's log10 market cap (in SOL) is tracked across the marks it
    appears in: last level, recent slope (least squares over the last few
    marks), venue and age.
  * v3 change: N_NEW_MAX 3.0 -> 6.0 (the cap on expected newborn coins
    reaching the top 5). v2 scored -41.4 on the real practice (rank
    399/459) with the size score collapsing to -23.07 (v1: -17.99) --
    the entry rate is hundreds/hour, so the newcomer count lam is large
    and the cap binds hard; v2 underestimated the displacement of named
    leaders by fresh coins on the later reads. Multi-read backtest
    (scores a forecast against ALL future marks, not one): v2 -32.70,
    v3 -32.16 (+0.54). Rejected by the same backtest: drift-shrink
    1.0 -> 0.5 and size-sigma widening (both worse on the backtest, even
    though v1's wider size scored better on the real practice -- the
    local backtests only cover the explosive phase and cannot reproduce
    the real practice's drift deceleration, so they are a biased proxy
    for size parameters; the newcomer cap is the one lever that improves
    both the backtest and, by the v1-vs-v2 leaderboard evidence, the
    real practice).
  * Each coin's log10 market cap (in SOL) is tracked across the marks it
    appears in: last level, recent slope (least squares over the last few
    marks), venue and age.
  * The median log-cap at the mid answer-read horizon is predicted as a
    damped extrapolation of the recent slope (growth fades with coin age
    and is capped at ~2.5 h of horizon) pulled toward a venue-specific
    long-run level with a ~12 h mean-reversion time. The slope shrink is
    horizon-dependent (~0.25 at short horizons, where the last level pins
    the answer and the slope is mostly noise, ~0.55 at the final's ~10 h,
    where the slope is the growth signal). Curve coins take a haircut
    (pool pricing after graduation is usually below the curve's virtual
    price); a curve coin close to graduation (>= 55 SOL in quote
    reserves) is treated as a likely graduate landing near 2.3-2.8.
  * Uncertainty grows with horizon, but the random-walk term saturates
    (effective horizon capped at ~4 h): a coin's log-cap does not wander
    without bound, it peaks and decays. Without the saturation the
    Monte-Carlo tail invents 10^6 SOL caps at long horizons.
  * Place probabilities come from a seeded Monte-Carlo over the top
    candidates' lognormal caps, with newborn coins drawn INSIDE each
    sim (Poisson count from the measured entry rate, caps near the
    cohort's place-3 level). A newcomer must beat the cap a named coin
    actually draws, not its reverted mean. A sim where a newcomer holds
    a place leaves that place's mass unnamed, so each place's unnamed
    share is whatever the newcomer model leaves (in the practice, 3 of
    the top 5 at the last read were not nameable from the history).
    The fixed 0.50 unnamed share of v1 is gone.
  * Chart fillers: each place is filled to the 100-row cap with the
    coins most likely to be on the answer chart but near-certain not to
    hold the place, at p = 1e-6. The scorer shares a place's unnamed
    mass over the chart's coins NOT named at that place, so naming them
    shrinks the denominator and leaves the unnamed share to the
    newcomers who may take the place.
  * Size distributions per place mix the Monte-Carlo order-statistic
    histogram with a trend normal centred on the cohort level — the
    place-k cap at the cut, pulled up toward the place-1 cap and drifted
    by the recent place-1 drift (trusted fully: the cohort's measured
    drift). The trend normal's sd and the MC mix weight are both
    horizon-adaptive: tight at short horizons (sd ~0.15, MC weight 0.20),
    wide at long ones (sd ~0.50, MC weight 0.40). ~12% of each place's
    size mass is left unnamed, so a missed bin costs far less than the
    -9.21 floor.
  * top5 is ranked by total Monte-Carlo top-5 probability (after the
    newcomers), tie-broken by predicted median.
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
# Newcomer displacement: coins born after the history cut take some
# top-5 places, pushing every named coin down by the number of newcomers
# that moon above it. The expected number in the top 5 is estimated
# from the history itself (how fast newborn coins entered the top 5 over
# the recent marks) times the horizon, saturating at N_NEW_MAX: the
# practice's explosive first hour shows ~1-3 per 10 min, the final's
# mature 11 h cohort will show far fewer, and the rate estimate adapts.
# A newcomer's cap is ~Normal(NEW_CAP_CENTER, NEW_CAP_SD), centred on
# the cohort's place-3 level: in the practice the newborn top-5 coins
# landed at 3.10-3.47 while place 1 sat at 3.56.
N_NEW_MAX = 6.0  # v3: raised from 3.0 (see sweep notes below).
# v3 note: the entry rate is hundreds/hour, so lam = 0.20*rate*h is large
# and this cap binds hard. v2's 3.0 underestimated how many fresh coins
# displace the named leaders, especially on later reads where composition
# churns. Multi-read backtest: 3.0 -> -32.70, 4.0 -> -32.38, 5.0 -> -32.18,
# 6.0 -> -32.16 (best), 8.0 -> -32.25 (order degrades past the optimum).
# The real practice horizon (mid 0.667h, reads to 1.17h) is longer than
# the backtest's max 0.5h, so the cap binds even harder there -- 6.0 is
# the safer side of the optimum for the real grading window.
N_NEW_H_SAT = 2.0           # horizon beyond this adds no more newcomers
NEW_CAP_SD = 0.25
FILLER_P = 0.000001         # p for chart-filler rows (see below)
SIZE_NAMED_SUM = 0.88         # share of each place's size mass named
# Weight of the Monte-Carlo histogram in sizes. At a short horizon the
# cohort level pins the place-k cap, so the tight trend normal should
# dominate (mix ~0.20); at a long horizon the MC histogram's honest
# which-coin spread earns its full 0.40 weight.
SIZE_MIX_MC_SHORT = 0.20
SIZE_MIX_MC_LONG = 0.40
SIZE_MIX_MC_TAU_H = 2.0
# Size sigma is horizon-adaptive: the trend normal's sd in log10 cap.
# At a short horizon the place-k cap is pinned by the cut (sigma ~0.15);
# at the final's ~10 h horizon the honest width is ~0.50. sigma(h) =
# SIZE_SD_A + SIZE_SD_B * sqrt(h) so the same code is tight on the
# 1-2 h practice and wide on the 7-13 h final. (The old fixed 0.45 was
# far too wide for the practice and wasted size mass over ~36 bins.)
SIZE_SD_A = 0.15
SIZE_SD_B = 0.11
DRIFT_TAU_H = 1.5             # cohort-level drift saturates over ~this long
GROWTH_CAP_H = 2.5            # growth only extrapolates this far (hours)
GROWTH_AGE_TAU_H = 4.0        # growth fades with coin age
# Slope shrink is horizon-dependent: at a short horizon the last level
# pins the answer and the estimated slope is mostly noise (the backtests
# prefer ~0.25 there), while at a long horizon the slope is the growth
# signal that separates coins still climbing from coins about to die,
# so it earns a larger weight (~0.55 at the final's ~10 h).
SLOPE_SHRINK_SHORT = 0.25
SLOPE_SHRINK_LONG = 0.55
SLOPE_SHRINK_TAU_H = 3.0
REVERT_TAU_H = 12.0            # mean-reversion time (hours)
# The per-coin sigma's random-walk term saturates: a coin's log-cap does
# not random-walk without bound (it peaks and decays), so the horizon
# entering sigma2 is capped at SIGMA_TAU_H of effective hours. At the
# backtest/practice horizons (<= 0.7 h) this changes nothing; at the
# final's ~10 h it keeps sigma ~0.56 instead of 1.33, so the Monte-Carlo
# tail stops inventing 10^6 SOL caps (bin 81) that no real coin reaches.
SIGMA_TAU_H = 4.0
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
    shrink = SLOPE_SHRINK_SHORT + (SLOPE_SHRINK_LONG - SLOPE_SHRINK_SHORT) * (
        1.0 - math.exp(-horizon_h / SLOPE_SHRINK_TAU_H))
    slope = max(-1.0, min(1.5, slope)) * shrink
    growth = slope * min(horizon_h, GROWTH_CAP_H) * math.exp(-age_h / GROWTH_AGE_TAU_H)

    level = LONG_RUN.get(venue, 1.7)
    mu_raw = y0 + growth
    mu = level + (mu_raw - level) * math.exp(-horizon_h / REVERT_TAU_H)

    sigma2 = 0.30 ** 2 + (0.13 * SIGMA_TAU_H * (
        1.0 - math.exp(-horizon_h / SIGMA_TAU_H))) ** 2
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

    # --- cohort levels at the cut (used by sizes and by the newcomer ----
    # model). last_caps: the last mark's caps by rank. g: recent drift of
    # the place-1 cap (log10 per hour), shrunk and capped.
    last_caps = []
    if marks:
        last_rows = sorted(marks[-1].get("rows") or [], key=lambda r: r.get("rank") or 9999)
        last_caps = [logcap_sol(r.get("market_cap_lamports")) for r in last_rows]
        last_caps = [c for c in last_caps if c is not None]
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
        g = max(0.0, min(1.2, drift_per_hour * 1.0))  # trust the measured cohort drift
    else:
        g = 0.3
    h_drift = DRIFT_TAU_H * (1.0 - math.exp(-horizon_h / DRIFT_TAU_H))

    # --- newcomers: how many, and how big --------------------------------
    # N_NEW: expected newborn coins reaching top-5 calibre by the answer
    # reads. The entry rate is measured from the history: newborn coins
    # entering the top 5 over the last few marks, per hour, times the
    # horizon (saturating at N_NEW_H_SAT, since occupancy, not cumulative
    # entries, is what a snapshot sees), capped at N_NEW_MAX. A newborn
    # top-5 coin lands near the cohort's place-3 level.
    n_new = 0.0
    mark_times = [parse_iso(m.get("at")) for m in marks]
    mark_times = [t for t in mark_times if t is not None]
    first_seen = {}
    for mint, obs in series.items():
        first_seen[mint] = min(p[0] for p in obs)  # minutes since t0
    if len(marks) >= 2 and mark_times and t0 is not None:
        # Rate over the last up-to-12 marks (a 20 min window is too noisy
        # to trust on an 11 h final history; the practice's whole history
        # is shorter than 12 marks, so it uses all of it).
        win = marks[-12:]
        win_times = [parse_iso(m.get("at")) for m in win]
        if all(t is not None for t in win_times):
            entrants = 0
            for j in range(1, len(win)):
                prev_t = win_times[j - 1]
                cur_t = win_times[j]
                top5_now = sorted(win[j].get("rows") or [],
                                  key=lambda r: r.get("rank") or 9999)[:5]
                for r in top5_now:
                    mint = r.get("mint")
                    if mint in first_seen:
                        seen_epoch = t0 + first_seen[mint] * 60.0
                        if prev_t < seen_epoch <= cur_t:
                            entrants += 1
            span_h = (win_times[-1] - win_times[0]) / 3600.0
            if span_h > 0 and entrants > 0:
                rate = entrants / span_h
                n_new = min(N_NEW_MAX,
                            rate * min(horizon_h, N_NEW_H_SAT))
    if last_caps:
        new_cap_center = max(last_caps[min(2, len(last_caps) - 1)],
                             last_caps[0] - 0.20) + g * h_drift
    else:
        new_cap_center = 3.0

    # --- seeded Monte-Carlo over the candidates' log-caps ----------------
    # Each coin's simulated cap is capped at mu + 2 sigma: a lognormal tail
    # would otherwise let any of ~150 mediocre coins draw +2.7 sigma and
    # claim the top places with a 500x moonshot no real coin makes, which
    # inflates the top places' sizes into absurd bins (a 10^6 SOL cap).
    # The cap barely touches ranking among the plausible leaders.
    #
    # Newcomers are drawn INSIDE the Monte-Carlo: each sim draws
    # Poisson(n_new) newborn coins with caps ~ Normal(new_cap_center,
    # NEW_CAP_SD) and ranks them against the named coins' DRAWN caps.
    # That is the honest comparison: a newcomer must beat the cap a named
    # coin actually draws (the upper tail that survives into the top 5),
    # not its mean-reverted mu. A sim where a newcomer holds a place
    # leaves that place's mass unnamed, so each place's unnamed share is
    # whatever the newcomer model leaves -- no fixed share. This replaces
    # the earlier Poisson-displacement convolution, which compared
    # newcomers against mus and therefore overstated their dominance at
    # long horizons, where the named coins' mus have reverted down while
    # their drawn top caps have not.
    place_counts = [[0] * n for _ in range(places)]
    place_bin_counts = [[0] * n_bins for _ in range(places)]
    place_sum = [0.0] * places
    place_sumsq = [0.0] * places
    if n:
        rng = random.Random(SEED)
        gauss = rng.gauss
        caps = [mu + 2.0 * sg for mu, sg in zip(c_mus, c_sigs)]
        new_cap = new_cap_center + 2.0 * NEW_CAP_SD
        for _ in range(N_SIMS):
            vals = [min(gauss(mu, sg), cap)
                    for mu, sg, cap in zip(c_mus, c_sigs, caps)]
            if n_new > 0:
                # Poisson(n_new) by Knuth's algorithm (n_new <= 3).
                thresh = math.exp(-n_new)
                prod = 1.0
                while True:
                    prod *= rng.random()
                    if prod <= thresh:
                        break
                    vals.append(min(gauss(new_cap_center, NEW_CAP_SD),
                                    new_cap))
            order = sorted(range(len(vals)), key=vals.__getitem__,
                           reverse=True)[:places]
            for k, idx in enumerate(order):
                v = vals[idx]
                place_sum[k] += v
                place_sumsq[k] += v * v
                place_bin_counts[k][bin_of(v, edges)] += 1
                if idx < n:
                    place_counts[k][idx] += 1
                # A newcomer holding the place leaves its mass unnamed:
                # place k's unnamed share is 1 minus its named sum.

    # Place probabilities are the Monte-Carlo frequencies directly: the
    # newcomers are already inside the sims, so a named coin's frequency
    # at a place is its honest probability of holding it, and 1 minus a
    # place's named sum is its newcomer share.
    place_probs = [[place_counts[k][idx] / N_SIMS for k in range(places)]
                   for idx in range(n)]

    probabilities = []
    # --- chart fillers ---------------------------------------------------
    # The scorer shares a place's unnamed mass equally over the chart's
    # coins NOT named at that place, and a row counts as named whatever
    # its p (even 0). So naming the coins most likely to be on the answer
    # chart, at p = 1e-6, shrinks the unnamed denominator and leaves the
    # unnamed share to the newcomers who may actually take the place.
    # Fillers must be near-certain NOT to hold the place: (near-)zero
    # displaced place probability, and a predicted median well below the
    # place's level. Near-graduation curve coins are excluded: a
    # graduation moonshot could plausibly take a place. Each place is
    # filled to the 100-row cap.
    last_rank = {}
    if marks:
        for r in marks[-1].get("rows") or []:
            mint = r.get("mint")
            if valid_mint(mint):
                last_rank[mint] = r.get("rank") or 9999
    last_venue_quote = {}
    for mint, obs in series.items():
        last_venue_quote[mint] = (obs[-1][2], obs[-1][4])
    # Chart-likelihood order over every history mint: last-mark rank,
    # then predicted median (coins absent from the last mark sort last).
    chart_order = sorted(
        preds.keys(),
        key=lambda m: (last_rank.get(m, 9999), -preds[m][0], m))
    for k in range(places):
        rows = []
        for idx in range(n):
            p = floor6(place_probs[idx][k])
            if p >= MIN_NAMED_P:
                rows.append((c_mints[idx], k + 1, p))
        rows.sort(key=lambda r: (-r[2], r[0]))
        # Safety: the place sums should stay <= 1 (each sim holds exactly
        # one coin per place); if rounding ever overshoots, scale down.
        total = sum(p for _, _, p in rows)
        if total > 1.0:
            scale = 1.0 / total
            rows = [(m, pl, floor6(p * scale)) for m, pl, p in rows
                    if floor6(p * scale) >= MIN_NAMED_P]
        rows = rows[:100]  # never emit more than the 100-row cap
        # Fillers: the place-k trend centre (cohort level) is the level
        # a coin must be near to hold the place; fillers sit well below.
        if last_caps and len(last_caps) >= k + 1:
            center_k = max(last_caps[k], last_caps[0] - 0.10 * k) + g * h_drift
        else:
            center_k = new_cap_center
        named_here = {m for m, _, _ in rows}
        for mint in chart_order:
            if len(rows) >= 100:
                break
            if mint in named_here:
                continue
            mu = preds[mint][0]
            if mu > center_k - 0.35:
                continue
            venue, qsol = last_venue_quote.get(mint, (None, None))
            if venue == "curve" and qsol is not None and qsol >= CURVE_GRAD_QUOTE_SOL:
                continue
            rows.append((mint, k + 1, FILLER_P))
            named_here.add(mint)
        probabilities.extend(
            {"mint": mint, "place": place, "p": p}
            for mint, place, p in rows
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
    # horizons; without it the sizes sit far too low. (last_caps, g and
    # h_drift were computed above, before the Monte-Carlo.)
    # Horizon-adaptive size sigma (see SIZE_SD_A / SIZE_SD_B above).
    size_sd = SIZE_SD_A + SIZE_SD_B * math.sqrt(horizon_h)
    # Horizon-adaptive MC mix weight: the trend normal dominates at
    # short horizons, the MC histogram earns its full weight at long ones.
    mix_mc = SIZE_MIX_MC_SHORT + (SIZE_MIX_MC_LONG - SIZE_MIX_MC_SHORT) * (
        1.0 - math.exp(-horizon_h / SIZE_MIX_MC_TAU_H))
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
            sd = size_sd
        else:
            center = mean
            sd = max(size_sd, sd_mc)
        probs = []
        total = 0.0
        for b in range(n_bins):
            mc = place_bin_counts[k][b] / N_SIMS
            lo = -math.inf if b == 0 else edges[b - 1]
            hi = math.inf if b == n_bins - 1 else edges[b]
            broad = norm_cdf((hi - center) / sd) - norm_cdf((lo - center) / sd)
            p = (mix_mc * mc + (1.0 - mix_mc) * broad) * SIZE_NAMED_SUM
            p = floor6(p)
            if p >= MIN_NAMED_P:
                probs.append((b, p))
                total += p
        # Rounding down can only shrink the sum; guard anyway.
        if total > 1.0:
            scale = 1.0 / total
            probs = [(b, floor6(p * scale)) for b, p in probs if floor6(p * scale) > 0]
        sizes.extend({"place": k + 1, "bin": b, "p": p} for b, p in probs)

    # --- top5: highest total displaced top-5 probability ------------------
    # (a coin's chance of holding any top-5 place after newcomer
    # displacement), tie-broken by predicted median. This beats ranking
    # by median alone: a slightly lower median with a wide sigma can
    # still be more likely to hold a place than a peaked coin just above
    # it, and displacement penalises coins newcomers are likely to pass.
    if n:
        top_order = sorted(
            range(n),
            key=lambda i: (-sum(place_probs[i]), -c_mus[i], c_mints[i]))
        top5 = [c_mints[i] for i in top_order[:places]]
    else:
        top5 = []
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
