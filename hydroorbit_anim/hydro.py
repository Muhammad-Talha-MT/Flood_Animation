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
TAG = "2021mar"
ISSUE = pd.Timestamp("2021-03-25 22:00", tz="UTC")     # forecast issue time t0 (before the peak)
PRE_H, HORIZON = 30.0, 48.0                            # hours of history shown before t0; forecast horizon
HERE = pathlib.Path(__file__).parent
D, B = HERE / "data", HERE / "build"
DT = 0.25                                              # hours

q = pd.read_csv(D / f"gauge_{SITE}_{TAG}_iv.csv", index_col=0, parse_dates=True).iloc[:, 0].astype(float)
q.index = pd.to_datetime(q.index, utc=True)
q = q[~q.index.duplicated()].sort_index()
r = pd.read_csv(D / f"mrms_prcp_{SITE}_{TAG}.csv", index_col=0, parse_dates=True).iloc[:, 0].astype(float)
r.index = pd.to_datetime(r.index, utc=True)
RAIN_REAL = True

T0 = ISSUE - pd.Timedelta(hours=PRE_H)                 # clock origin: hours since T0
H_ISSUE = PRE_H
H_END = PRE_H + HORIZON
T = np.arange(0, H_END + DT / 2, DT)
hrs = lambda idx: ((idx - T0).total_seconds() / 3600).values
qe = q[(q.index >= T0 - pd.Timedelta(hours=3)) & (q.index <= T0 + pd.Timedelta(hours=H_END + 3))]
gaps = qe.index.to_series().diff().dt.total_seconds().div(3600)
print(f"window {T0} -> {T0 + pd.Timedelta(hours=H_END)} ({H_END:.0f} h); issue at +{H_ISSUE:.0f} h; "
      f"obs records {len(qe)}, longest gap {gaps.max():.2f} h")
assert gaps.max() <= 2.0, "observed discharge has a gap > 2 h"
QOBS = np.interp(T, hrs(qe.index), qe.values)
ipk = int(QOBS.argmax())
print(f"observed peak {QOBS[ipk]:.0f} m3/s at {T0 + pd.Timedelta(hours=T[ipk])} (+{T[ipk] - H_ISSUE:.1f} h after issue)")
rh = r.reindex(pd.date_range(T0.floor("h"), T0 + pd.Timedelta(hours=H_END), freq="h", tz="UTC")).fillna(0.0)
RAIN_HOURLY_T = hrs(rh.index)
RAIN_HOURLY = rh.values
RAIN = np.interp(T, RAIN_HOURLY_T + 0.5, RAIN_HOURLY, left=0, right=0)
rain_start = rh[rh > 0.5].index.min()
print(f"rain (MRMS): total {RAIN.sum() * DT:.0f} mm, max {RAIN.max():.1f} mm/h, starts {rain_start}")

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
for c in (3.0, 5.5, 9.0, 12.0, 16.0, 22.0, 30.0):    # km/h
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

# ------------------------------------------------------------------ synthetic forecast (one issue, never revised)
# Built once at t0 from the observed series with FIXED errors (peak 9 % low and 1.5 h late, slightly slow recession).
# Only the observations up to t0 are "known"; everything after t0 is the synthetic prediction.
tt = T[T >= H_ISSUE - 1e-9]
lead = tt - H_ISSUE
w = 1 - np.exp(-lead / 4.0)
qf = np.interp(tt - 1.5 * w, T, QOBS) * (1 - 0.09 * w) * (1 + 0.06 * (lead / HORIZON) ** 1.5 * w)
q0 = np.interp(H_ISSUE, T, QOBS)
qf = qf + (q0 - qf[0]) * np.exp(-lead / 2.0)
FC_T, FC_Q = tt, qf
ipf = int(qf.argmax())
print(f"synthetic forecast peak {qf[ipf]:.0f} m3/s at +{lead[ipf]:.1f} h after issue (illustrative; no skill claimed)")

pd.DataFrame({
    "time_utc": (T0 + pd.to_timedelta(T, unit="h")).strftime("%Y-%m-%d %H:%M"),
    "hours_after_issue": np.round(T - H_ISSUE, 2),
    "observed_streamflow_m3s": np.round(QOBS, 1),
    "synthetic_forecast_m3s": np.round(np.interp(T, FC_T, FC_Q, left=np.nan, right=np.nan), 1),
    "mrms_rain_mm_h": np.round(RAIN, 2),
}).to_csv(HERE / "out" / "hydrograph_data.csv", index=False)

# ------------------------------------------------------------------ video clock (PCHIP)
# 0-11 s static context; 11-18 s storm builds up to the issue time; 18-27 s clock holds at t0 (forecast issued);
# 27-34 s the event plays out (slow through the peak, faster in the recession); 34-40 s held at the end.
tp = T[ipk]
knots_t = [11.0, 14.0, 18.0, 27.0, 29.0, 30.6, 32.0, 34.0, 40.0]
knots_h = [rain_start_h := float((rain_start - T0).total_seconds() / 3600) - 3, H_ISSUE - 5.0, H_ISSUE, H_ISSUE,
           H_ISSUE + 5.0, tp + 2.0, tp + 14.0, H_END, H_END]
assert np.all(np.diff(knots_t) > 0) and np.all(np.diff(knots_h) >= 0), knots_h
print("clock knots", [(a, round(b, 1)) for a, b in zip(knots_t, knots_h)])

np.savez_compressed(
    B / "hydro.npz", T=T, t0_ns=np.int64(T0.as_unit("ns").value), RAIN=RAIN, RAIN_HT=RAIN_HOURLY_T, RAIN_H=RAIN_HOURLY,
    QOBS=QOBS, QMOD=qmod, QN=QN.astype(np.float16), RUNOFF=R, H_END=H_END, H_ISSUE=H_ISSUE, ipk=ipk,
    FC_T=FC_T, FC_Q=FC_Q, clock_t=knots_t, clock_h=knots_h, rain_real=RAIN_REAL, C=C, KH=KH)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(T - H_ISSUE, QOBS, "k", label="observed (NWIS)")
ax.plot(T - H_ISSUE, qmod, "C1", label="routed model (map animation)")
ax.plot(FC_T - H_ISSUE, FC_Q, "c", label="synthetic forecast")
ax.axvline(0, color="r"); ax.legend()
ax2 = ax.twinx(); ax2.bar(T - H_ISSUE, RAIN, width=DT, color="b", alpha=0.3); ax2.set_ylim(RAIN.max() * 4, 0)
plt.savefig(B / "hydro_preview.png", dpi=70)
