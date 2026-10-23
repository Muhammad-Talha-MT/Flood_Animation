#!/usr/bin/env python3
"""Event hydrology for the animation (build/hydro.npz).

- Observed discharge: real NWIS instantaneous values at the gauge (15 min, UTC).
- Rain: NOAA MRMS hourly radar+gauge QPE (basin mean) if fetched, else an illustrative hyetograph.
- Channel animation: rain -> runoff (wetting-up coefficient) -> hillslope linear reservoir ->
  translation along NHDPlus flow distance. c, K and the coefficient are fitted so the modelled
  outlet hydrograph matches the observed one; per-reach series are normalised for display.
- Forecast: illustrative rolling forecast derived from the observed series (no HydroORBIT CSV).
"""
import json, pathlib
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import lfilter

SITE = "03451500"
HERE = pathlib.Path(__file__).parent
D, B = HERE / "data", HERE / "build"
DT = 0.25                                   # hours

# ------------------------------------------------------------------ observed
q = pd.read_csv(D / f"gauge_{SITE}_iv.csv", index_col=0, parse_dates=True).iloc[:, 0].astype(float)
q.index = pd.to_datetime(q.index, utc=True)
q = q[~q.index.duplicated()].sort_index()

rain_csv = D / f"mrms_prcp_{SITE}.csv"
RAIN_REAL = rain_csv.exists()
if RAIN_REAL:
    r = pd.read_csv(rain_csv, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    r.index = pd.to_datetime(r.index, utc=True)
    # indexed at the start of each accumulation hour
    wet = r[r > 0.5]
    rain_start = wet.index.min()
else:
    rain_start = q.index[np.argmax(q.diff().rolling(8).sum().values > 5)] - pd.Timedelta(hours=12)

tpk_obs = q.idxmax()
T0 = max((min(rain_start, tpk_obs - pd.Timedelta(hours=60)) - pd.Timedelta(hours=30)).floor("6h"),
         q.index.min().ceil("6h"))
print(f"rain starts {rain_start}")
T1 = (tpk_obs + pd.Timedelta(hours=84)).ceil("6h")
H_END = (T1 - T0).total_seconds() / 3600
T = np.arange(0, H_END + DT / 2, DT)
times = T0 + pd.to_timedelta(T, unit="h")

qe = q[(q.index >= T0 - pd.Timedelta(hours=3)) & (q.index <= T1 + pd.Timedelta(hours=3))]
gaps = qe.index.to_series().diff().dt.total_seconds().div(3600)
print(f"window {T0} -> {T1} ({H_END:.0f} h); obs records {len(qe)}, longest gap {gaps.max():.2f} h")
assert gaps.max() <= 2.0, "observed discharge has a gap > 2 h"
hrs = lambda idx: ((idx - T0).total_seconds() / 3600).values   # (index units vary: us vs ns)
QOBS = np.interp(T, hrs(qe.index), qe.values)
ipk = int(QOBS.argmax())
print(f"observed peak {QOBS[ipk]:.0f} m3/s at {times[ipk]} (+{T[ipk]:.1f} h)")

# ------------------------------------------------------------------ rain (mm/h, basin mean)
if RAIN_REAL:
    rh = r.reindex(pd.date_range(T0.floor("h"), T1, freq="h", tz="UTC")).fillna(0.0)
    RAIN = np.interp(T, (rh.index - T0).total_seconds() / 3600 + 0.5, rh.values, left=0, right=0)
    RAIN_HOURLY_T = ((rh.index - T0).total_seconds() / 3600).values
    RAIN_HOURLY = rh.values
else:
    # illustrative: two bursts that end before the main rising limb
    tp = T[ipk]
    RAIN = 6 * np.exp(-0.5 * ((T - (tp - 40)) / 5) ** 2) + 14 * np.exp(-0.5 * ((T - (tp - 12)) / 4) ** 2)
    RAIN_HOURLY_T = np.arange(0, H_END, 1.0)
    RAIN_HOURLY = np.interp(RAIN_HOURLY_T + 0.5, T, RAIN)
print(f"rain ({'MRMS' if RAIN_REAL else 'illustrative'}): total {RAIN.sum() * DT:.0f} mm, "
      f"max {RAIN.max():.1f} mm/h")

# ------------------------------------------------------------------ routing
cr = np.load(B / "catch_rivers.npz")
area, dn, dist = cr["area"].astype(float), cr["dn"], cr["dist"].astype(float)
n = area.size
# local contributing area = total area minus that of direct upstream reaches
loc = area.copy()
for i in range(n):
    if dn[i] >= 0:
        loc[dn[i]] -= area[i]
loc = np.clip(loc, 0.05, None)
loc *= area[int(cr["out_i"])] / loc.sum()             # divergences double-count area
print(f"reaches {n}; summed local area {loc.sum():.0f} km2; outlet area {area[int(cr['out_i'])]:.0f} km2")


def runoff(S, kf, frac, ks=40.0):
    """SCS-CN excess from cumulative rain (S = retention, mm), split into fast/slow reservoirs."""
    P = np.cumsum(RAIN) * DT
    Ia = 0.2 * S
    Qc = np.where(P > Ia, (P - Ia) ** 2 / (P + 0.8 * S), 0.0)
    ex = np.diff(Qc, prepend=0.0) / DT                               # mm/h
    af, as_ = DT / kf, DT / ks
    return (frac * lfilter([af], [1, -(1 - af)], ex)
            + (1 - frac) * lfilter([as_], [1, -(1 - as_)], ex))


def shift(x, lag_h):
    return np.interp(T - lag_h, T, x, left=x[0])


def kernel(c):
    # translation: each reach's local runoff reaches the outlet after dist/c hours
    k = np.bincount(np.round(dist / c / DT).astype(int), weights=loc) / 3.6   # mm/h*km2 -> m3/s
    return k


def outlet(R, c):
    return np.convolve(R, kernel(c))[:R.size]


base = np.minimum.accumulate(QOBS[::-1])[::-1]                     # rising-limb baseflow envelope
base = np.minimum(base, np.interp(T, [0, H_END], [QOBS[0], QOBS[-1]]))
best = None
for c in (2.0, 3.0, 4.0, 5.5, 7.0, 9.0, 12.0):    # km/h
    for S in (60, 100, 150, 220, 320, 450):        # mm
        for kf in (1.5, 3.0, 5.0, 8.0, 12.0):      # h
            for frac in (0.5, 0.7, 0.85, 1.0):
                o = outlet(runoff(S, kf, frac), c)
                if o.max() <= 0:
                    continue
                gain = np.dot(o, QOBS - base) / np.dot(o, o)  # absorbs areal/rain bias
                err = np.sqrt(np.mean((gain * o + base - QOBS) ** 2))
                if best is None or err < best[0]:
                    best = (err, c, S, kf, frac, gain)
err, C, S, KF, FRAC, GAIN = best
R = GAIN * runoff(S, KF, FRAC)
qmod = outlet(R, C) + base
lag_fix = T[ipk] - T[int(qmod.argmax())]
print(f"fit: c={C} km/h, S={S} mm, Kfast={KF} h, fast fraction {FRAC}, gain {GAIN:.2f}, rmse {err:.0f} m3/s, "
      f"model peak {qmod.max():.0f} m3/s, timing offset {lag_fix:+.2f} h (removed)")
R = shift(R, lag_fix)
qmod = outlet(R, C) + base
KH = KF

# per-reach discharge: Q_r(t) = sum over upstream reaches u of loc_u * R(t - (d_u - d_r)/c)
order = np.argsort(-dist)                                          # upstream first
Q = np.zeros((n, T.size))
for i in order:
    Q[i] += loc[i] / 3.6 * R
    # pass downstream with the translation lag of this reach's length
    if dn[i] >= 0:
        Q[dn[i]] += shift(Q[i], (dist[i] - dist[dn[i]]) / C)
QB = Q + base[None] * (area / area.max())[:, None]
# nudge every reach by the outlet's obs/model ratio, applied earlier by the reach's travel time,
# so the outlet reproduces the observed hydrograph exactly and upstream reaches lead it
ratio = np.clip(QOBS / np.maximum(QB[int(cr["out_i"])], 1e-3), 0.2, 5.0)
for i in range(n):
    QB[i] *= shift(ratio, -dist[i] / C)
QN = (QB - QB.min(1, keepdims=True)) / (np.ptp(QB, 1, keepdims=True) + 1e-9)
io = int(cr["out_i"])
print(f"outlet reach peak at +{T[QN[io].argmax()]:.1f} h (observed +{T[ipk]:.1f} h)")

# ------------------------------------------------------------------ illustrative rolling forecast
HORIZON = 48.0
ISSUES = np.arange(0, H_END + 1e-6, 1.0)                           # hourly issues


def forecast(t0):
    tt = np.arange(t0, min(t0 + HORIZON, H_END) + 1e-6, DT)
    lead = tt - t0
    w = 1 - np.exp(-lead / 9.0)
    tp = T[ipk]
    stage = np.clip((t0 - (tp - 60)) / 50.0, 0, 1)                  # skill improves as the storm is observed
    late = 3.0 * (1 - stage) ** 1.5 + 0.6                           # hours late
    mag = -0.18 * (1 - stage) ** 1.2 + 0.04                         # relative magnitude error
    qf = np.interp(tt - late * w, T, QOBS) * (1 + mag * w)
    q0 = np.interp(t0, T, QOBS)
    qf = qf + (q0 - qf[0]) * np.exp(-lead / 3.0)
    spread = (0.08 + 0.32 * (1 - stage)) * w
    return tt, qf, qf * (1 - spread), qf * (1 + 1.15 * spread)


lead_trace = np.array([np.interp(t, *forecast(max(t - 6.0, 0))[:2]) for t in T])
ipl = int(lead_trace.argmax())
peak_err = (lead_trace[ipl] - QOBS[ipk]) / QOBS[ipk] * 100
timing_err = T[ipl] - T[ipk]
print(f"illustrative 6-h lead: peak error {peak_err:+.1f} %, timing error {timing_err:+.2f} h")

# ------------------------------------------------------------------ video clock (PCHIP)
# storm scenes run 12 s -> 33 s; slow through the rising limb and the peak, fast in recession
tp = T[ipk]
rise = T[np.argmax(QOBS > QOBS[0] + 0.15 * (QOBS[ipk] - QOBS[0]))]
rs = (rain_start - T0).total_seconds() / 3600
knots_t = [12.0, 13.5, 17.0, 20.0, 24.0, 26.5, 33.0]
knots_h = [0.0, tp - 77, tp - 27, tp - 11, tp, tp + 11, H_END]
assert np.all(np.diff(knots_h) > 0), knots_h
print("clock knots", list(zip(knots_t, np.round(knots_h, 1))))

np.savez_compressed(
    B / "hydro.npz", T=T, t0_ns=np.int64(T0.value), RAIN=RAIN, RAIN_HT=RAIN_HOURLY_T, RAIN_H=RAIN_HOURLY,
    QOBS=QOBS, QMOD=qmod, QN=QN.astype(np.float16), RUNOFF=R, LEAD=lead_trace, H_END=H_END,
    ipk=ipk, ipl=ipl, peak_err=peak_err, timing_err=timing_err, clock_t=knots_t, clock_h=knots_h,
    rain_real=RAIN_REAL, C=C, KH=KH,
    FC_T=np.stack([np.pad(forecast(t)[0], (0, int(HORIZON / DT) + 1 - forecast(t)[0].size), constant_values=np.nan) for t in ISSUES]),
    FC_Q=np.stack([np.pad(forecast(t)[1], (0, int(HORIZON / DT) + 1 - forecast(t)[1].size), constant_values=np.nan) for t in ISSUES]),
    FC_LO=np.stack([np.pad(forecast(t)[2], (0, int(HORIZON / DT) + 1 - forecast(t)[2].size), constant_values=np.nan) for t in ISSUES]),
    FC_HI=np.stack([np.pad(forecast(t)[3], (0, int(HORIZON / DT) + 1 - forecast(t)[3].size), constant_values=np.nan) for t in ISSUES]),
    ISSUES=ISSUES)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(T, QOBS, "k", label="observed (NWIS)")
ax.plot(T, qmod, "C1", label="routed model (animation)")
ax.plot(T, lead_trace, "c", label="illustrative 6-h lead")
for t0 in (tp - 40, tp - 20, tp - 8):
    tt, qf, lo, hi = forecast(t0)
    ax.fill_between(tt, lo, hi, alpha=0.12); ax.plot(tt, qf, lw=1)
ax2 = ax.twinx(); ax2.bar(T, RAIN, width=DT, color="b", alpha=0.3); ax2.set_ylim(RAIN.max() * 4, 0)
for i in np.argsort(-area)[[50, 400]]:
    ax.plot(T, QN[i] * QOBS.max() * 0.5, ":", lw=1)
ax.legend(); plt.savefig(B / "hydro_preview.png", dpi=70)
