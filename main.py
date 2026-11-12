#!/usr/bin/env python3
"""Forecast the top 5 new pump.fun coins by market cap, 24 hours out.

Reads $MAKEFUN_INPUT_DIR/history.ndjson + task.json (defaults /input) and
writes $MAKEFUN_OUTPUT_DIR/forecast.json (default /output). Python 3.13,
standard library only, no network, deterministic (fixed RNG seed).

Model (v5 -- the per-coin Monte-Carlo size engine from rank1-trace.md):
  * The cohort-level trend normal for sizes is GONE. Each candidate
    coin is simulated on its OWN trajectory to each of the 7 answer
    reads: an Ornstein-Uhlenbeck log-cap process with a per-coin
    momentum drift (own shrunk slope blended 50/50 with the cohort
    drift, active only in the first 2.5 h, faded by coin age), cohort
    vol 0.30/h (per-coin residual vols are noise-dominated on a
    <=6-mark history and are deliberately NOT used), mean reversion
    toward a venue-specific level (tau 12 h), and Merton crash jumps
    (lambda 0.30/h, drop ~N(0.55, 0.15)). At each simulated read the
    top-5 caps are taken, so the place-k size distribution is the
    distribution of the simulated place-k cap and the order
    probabilities come from the SAME paths -- order and size are
    JOINT, which is the structure the rank-1 trace says wins
    (leader centre error d ~0-0.16 dex vs our v1 d ~0.44).
  * Newcomers (coins born after the cut) are drawn INSIDE each sim at
    each read (Poisson from the measured entry rate, caps near the
    cohort's place-3 level), as in v2-v4.
  * Sizes = the MC order-statistic histogram per place, smoothed
    with a broad normal (weight 0.25, sd >= 0.30) so unhit bins keep
    some mass; 12% of each place's size mass stays unnamed.
  * Kept from v4: candidate ranking by predict_coin median (used
    only to select the 80 simulated candidates and for fillers /
    top5 tie-breaks), chart fillers at p=1e-6, top5 by total MC
    top-5 probability, curve haircut / near-graduation level.
  * Deliberately NOT in v5 (see v5-notes.md): no per-coin vol
    estimation, no per-read separate forecasts (the entry format is
    one forecast scored at all 7 reads), no sweep of the 50/50
    slope blend or crash lambda -- those are the backtest worker's
    job. N_SIMS is 3000 (not 15000): 7 reads x 3000 sims x ~80 coins
    x up to ~13 hourly steps is the pure-Python budget for a
    ~1-2 minute run.
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
N_SIMS = 3000
N_CANDIDATES = 80
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
# --- v5 per-coin trajectory constants ----------------------------------
# V5_SIGMA_H: cohort vol per hour. Measured: 10-min log-cap change sd
# 0.121 -> 0.121*sqrt(6) = 0.296/h. Per-coin residual vols are NOT used
# (median ~0.003 on this history -- coins print flat between marks).
V5_SIGMA_H = 0.30
# V5 crash jumps (Merton): hazard 0.30/h, drop ~N(0.55, 0.15). v4's
# top-20 hazard was 0.6/h on the young explosive cohort; 0.30/h sits
# between that and the all-coins >0.3-drop rate (~15%/h implied).
V5_CRASH_LAMBDA = 0.30
V5_CRASH_DROP = 0.55
V5_CRASH_DROP_SD = 0.15
# Weight of the broad-normal smoothing in sizes (rest = MC histogram).
SIZE_SMOOTH_W = 0.25
# Cap on the cohort drift's contribution to a coin's momentum (log10/h).
V5_G_CAP = 0.30
# Long-run level newborn top-5 coins revert toward (log10 cap in SOL).
V5_NEW_LEVEL = 2.6
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
    new_rate = 0.0
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
                new_rate = rate
                n_new = min(N_NEW_MAX,
                            rate * min(horizon_h, N_NEW_H_SAT))
    if last_caps:
        new_cap_center = max(last_caps[min(2, len(last_caps) - 1)],
                             last_caps[0] - 0.20) + g * h_drift
    else:
        new_cap_center = 3.0

    # --- v5: per-coin Monte-Carlo trajectories ---------------------------
    # The size model is now the simulation itself, not a cohort-level
    # trend normal. Each candidate coin gets its OWN trajectory to the
    # answer reads: log-cap y(t) follows an Ornstein-Uhlenbeck process
    # with a per-coin momentum drift, per-coin vol, and Merton crash
    # jumps. At each simulated read the top-5 caps are taken, so the
    # place-k size distribution is the distribution of the simulated
    # place-k cap and the order probabilities come from the SAME paths
    # (joint order+size -- a coin that moons becomes #1 AND pushes the
    # #1 cap up, which the split v1-v4 model could not express).
    #
    # Per step (dt hours, capped at 1 h per step):
    #   y += mom * dt_mom            (own momentum, first 2.5 h only)
    #   y += theta * (level - y) * dt (mean reversion, tau = 12 h)
    #   y += sigma * sqrt(dt) * Z     (diffusion)
    #   y -= Jump  with prob 1-exp(-lambda*dt)  (crash, ~N(0.55, 0.15))
    # mom = 0.75 * shrunk_own_slope * age_fade + 0.25 * cohort_drift g.
    # The own slope comes from the last <=4 marks (fit_slope); it is
    # shrunk (0.25 short horizon -> 0.55 long) because 10-min slopes
    # are mostly noise. The cohort drift enters only at weight 0.25
    # and capped at 0.30/h: the raw place-1 drift (0.9/h in this
    # explosive practice) is driven by a few hot coins, not the median
    # coin, and the practice evidence is that the cohort DECELERATES
    # (v1's centres, drifted up by the full g, landed 0.44 dex above
    # the realised caps). Giving every coin the full g as a rising
    # tide was the first version's bug: it put the place-1 size mean
    # at 4.13. The 75/25 blend and the 0.30 cap are priors, not
    # measurements; the backtest worker should sweep them.
    # sigma: cohort vol 0.30/h (measured: 10-min change sd 0.121 ->
    # 0.121*sqrt(6) = 0.296/h). Per-coin residual vols are NOT used:
    # on this history their median is ~0.003 (coins print flat between
    # marks), so they are noise-dominated and would understate vol.
    # Single-observation coins get extra vol (sqrt-add 0.10), curve
    # coins extra vol (sqrt-add 0.15, graduation uncertainty).
    # Crash hazard lambda = 0.30/h with drop ~N(0.55, 0.15): v4's
    # top-20 hazard was 0.6/h, but that was the young explosive
    # cohort; 0.30/h is the all-coins >0.3-drop rate implied by the
    # history (2.8% per 10 min -> ~15%/h for a >0.3 fall, and the
    # crash drop is larger than 0.3, so the hazard for a full 0.55
    # drop is lower -- 0.30/h sits between the two measurements).
    # With OU reversion the crashes create the left tail and the
    # dispersion without collapsing every coin: a crashed coin is
    # pulled back UP toward its level, a mooning coin pulled DOWN.
    #
    # Newcomers are drawn INSIDE each sim at each read (Poisson(n_new)
    # with the v4 rate model, caps ~ Normal(new_cap_center, 0.25)),
    # exactly as in v2-v4, so a newcomer must beat the cap a named
    # coin actually draws at that read.
    sim_y0 = []
    sim_level = []
    sim_mom = []
    sim_sig = []
    shrink = SLOPE_SHRINK_SHORT + (SLOPE_SHRINK_LONG - SLOPE_SHRINK_SHORT) * (
        1.0 - math.exp(-horizon_h / SLOPE_SHRINK_TAU_H))
    for mint in c_mints:
        obs = series[mint]
        last = obs[-1]
        y0 = last[1]
        venue = last[2]
        created = last[5]
        age_h = max(0.0, (cut_epoch - created) / 3600.0) if created else 24.0
        slope = fit_slope(obs)
        slope = max(-1.0, min(1.5, slope)) * shrink
        slope *= math.exp(-age_h / GROWTH_AGE_TAU_H)
        mom = 0.75 * slope + 0.25 * min(g, V5_G_CAP)
        mom = max(-0.5, min(1.5, mom))
        level = LONG_RUN.get(venue, 1.7)
        if venue == "curve":
            y0 = y0 - CURVE_HAIRCUT
            qsol = last[4]
            if qsol is not None and qsol >= CURVE_GRAD_QUOTE_SOL:
                level = 2.1  # likely graduate landing level, not 1.7
        sig2 = V5_SIGMA_H ** 2
        if len(obs) == 1:
            sig2 += 0.10 ** 2
        if venue == "curve":
            sig2 += 0.15 ** 2
        sim_y0.append(y0)
        sim_level.append(level)
        sim_mom.append(mom)
        sim_sig.append(math.sqrt(sig2))

    # Time grid: step from the cut to each read in <=1 h steps; record
    # the top 5 at every read. Reads at/before the cut are recorded
    # from the starting caps (no steps). The forecast is ONE set of
    # probabilities scored at all 7 reads, so counts accumulate over
    # (sim, read) pairs: total_draws = N_SIMS * n_reads.
    read_epochs = sorted(reads) if reads else [cut_epoch + 3600.0]
    steps = []  # (dt_hours, read_index_or_None)
    t_cursor = cut_epoch
    for j, r_epoch in enumerate(read_epochs):
        if r_epoch <= t_cursor:
            # Read at/before the cut: record from the starting caps.
            steps.append((0.0, j))
            continue
        while r_epoch - t_cursor > 3600.0 + 1e-9:
            steps.append((1.0, None))
            t_cursor += 3600.0
        dt = max(0.0, (r_epoch - t_cursor) / 3600.0)
        steps.append((dt, j))
        t_cursor = r_epoch
    n_reads = len(read_epochs)
    total_draws = N_SIMS * n_reads

    # Per-read newcomer parameters (v4 rate model, per-read horizon).
    # Centre: newborn top-5 coins land near the cohort's place-3 level
    # AT THE CUT (measured in the practice: newborns landed 3.10-3.47
    # while place 1 sat at 3.56) and revert toward the cohort level
    # V5_NEW_LEVEL with horizon -- they are NOT drifted up by the
    # explosive-phase place-1 drift g. The old centre (place-3 +
    # g*h_drift, = 4.04 here) was the same error as v1's size centre:
    # with n_new saturating at the 6.0 cap, ~6 newcomers per sim at
    # ~N(4.04, 0.25) swamped every named coin and pushed the place-1
    # size mean to 4.10. The rate itself (18/h here, capped at 6.0)
    # is inherited from v2-v4 untouched; only the centre is fixed.
    new_per_read = []
    if last_caps:
        new_base = max(last_caps[min(2, len(last_caps) - 1)],
                       last_caps[0] - 0.20)
    else:
        new_base = 3.0
    for r_epoch in read_epochs:
        h_j = max(0.0, (r_epoch - cut_epoch) / 3600.0)
        n_new_j = min(N_NEW_MAX,
                      new_rate * min(h_j, N_NEW_H_SAT)) if new_rate > 0 else 0.0
        center_j = V5_NEW_LEVEL + (new_base - V5_NEW_LEVEL) * math.exp(
            -h_j / REVERT_TAU_H)
        new_per_read.append((n_new_j, center_j))

    place_counts = [[0] * n for _ in range(places)]
    place_bin_counts = [[0] * n_bins for _ in range(places)]
    place_sum = [0.0] * places
    place_sumsq = [0.0] * places
    if n:
        rng = random.Random(SEED)
        gauss = rng.gauss
        rand = rng.random
        lam = V5_CRASH_LAMBDA
        theta = 1.0 / REVERT_TAU_H
        for _ in range(N_SIMS):
            y = list(sim_y0)
            elapsed = 0.0
            for dt, j in steps:
                if dt > 0:
                    # Momentum only in the first GROWTH_CAP_H hours.
                    dt_mom = max(0.0, min(dt, GROWTH_CAP_H - elapsed))
                    for i in range(n):
                        dy = sim_mom[i] * dt_mom
                        dy += theta * (sim_level[i] - y[i]) * dt
                        dy += sim_sig[i] * math.sqrt(dt) * gauss(0.0, 1.0)
                        if rand() < 1.0 - math.exp(-lam * dt):
                            drop = gauss(V5_CRASH_DROP, V5_CRASH_DROP_SD)
                            dy -= drop if drop > 0.05 else 0.05
                        y[i] += dy
                    elapsed += dt
                if j is not None:
                    n_new_j, center_j = new_per_read[j]
                    vals = list(y)
                    if n_new_j > 0:
                        # Poisson(n_new_j) by Knuth's algorithm.
                        thresh = math.exp(-n_new_j)
                        prod = 1.0
                        while True:
                            prod *= rand()
                            if prod <= thresh:
                                break
                            vals.append(gauss(center_j, NEW_CAP_SD))
                    order = sorted(range(len(vals)), key=vals.__getitem__,
                                   reverse=True)[:places]
                    for k, idx in enumerate(order):
                        v = vals[idx]
                        place_sum[k] += v
                        place_sumsq[k] += v * v
                        place_bin_counts[k][bin_of(v, edges)] += 1
                        if idx < n:
                            place_counts[k][idx] += 1
                        # A newcomer holding the place leaves its mass
                        # unnamed: place k's unnamed share is 1 minus
                        # its named sum.

    # Place probabilities are the Monte-Carlo frequencies directly,
    # averaged over sims AND reads: the newcomers are already inside
    # the sims, so a named coin's frequency at a place is its honest
    # probability of holding it, and 1 minus a place's named sum is
    # its newcomer share.
    place_probs = [[place_counts[k][idx] / total_draws for k in range(places)]
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

    # --- sizes (v5): the Monte-Carlo order-statistic histogram -----------
    # The size distribution per place IS the simulated distribution of
    # the place-k cap across (sim, read) paths -- no cohort-level trend
    # normal any more. The histogram is smoothed with a broad normal
    # (weight SIZE_SMOOTH_W) centred on the MC mean with sd at least
    # 0.30, so a bin the simulation never hit still gets some mass
    # instead of leaving the truth to the unnamed share alone, and so
    # neighbouring bins share mass when the MC histogram is spiky.
    # ~12% of each place's size mass is left unnamed, so a missed bin
    # costs far less than the -9.21 floor.
    sizes = []
    for k in range(places):
        if n == 0 or total_draws == 0:
            continue
        mean = place_sum[k] / total_draws
        var = max(0.0, place_sumsq[k] / total_draws - mean * mean)
        sd_mc = math.sqrt(var)
        sd_broad = max(0.30, sd_mc)
        probs = []
        total = 0.0
        for b in range(n_bins):
            mc = place_bin_counts[k][b] / total_draws
            lo = -math.inf if b == 0 else edges[b - 1]
            hi = math.inf if b == n_bins - 1 else edges[b]
            broad = (norm_cdf((hi - mean) / sd_broad)
                     - norm_cdf((lo - mean) / sd_broad))
            p = ((1.0 - SIZE_SMOOTH_W) * mc + SIZE_SMOOTH_W * broad) \
                * SIZE_NAMED_SUM
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
