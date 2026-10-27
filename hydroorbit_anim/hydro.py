#!/usr/bin/env python3
"""Event hydrology for the animation (build/hydro.npz).

- Observed inflow: sum of real NWIS instantaneous values (15 min, UTC) at the three gauges just
  upstream of Douglas Lake (French Broad at Hot Springs, Pigeon at Newport, Nolichucky near Lowland;
  ~86 % of the drainage).
- Rain: NOAA MRMS hourly radar+gauge QPE (mean over the drainage above the focus dam).
- Channel animation: rain -> runoff (wetting-up coefficient) -> hillslope linear reservoir ->
  translation along NHDPlus flow distance, over the whole network above the dam. c, K and the
  coefficient are fitted so the modelled hydrograph at the dam reach matches the observed inflow;
  per-reach series are normalised for display.
- Forecasts: HydroORBIT ML (median + 5-95 % range) and TVA, from data/vinh/forecast.csv when it
  exists (columns time_utc, ml_q50, ml_q05, ml_q95, tva; m3/s). Until then, placeholder curves
  derived from the observed series (FC_PLACEHOLDER = True, flagged on screen).
- Flood thresholds for the combined inflow, from its hourly annual maxima WY2012-2025 (common record of
  the three gauges): median flood = median annual peak; moderate flood = 5-year flood (20 % annual
  chance, lognormal fit without Helene 2024).
"""
import json, pathlib
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import lfilter

CATCH = "douglas"
TAG = "2021mar"
ISSUE = pd.Timestamp("2021-03-25 22:00", tz="UTC")     # forecast issue time t0 (before the peak)
PRE_H, HORIZON = 30.0, 48.0                            # hours of history shown before t0; forecast horizon
HERE = pathlib.Path(__file__).parent
D, B = HERE / "data", HERE / "build"
DT = 0.25                                              # hours

q3 = pd.read_csv(D / f"inflow3_{TAG}_iv.csv", index_col=0, parse_dates=True).astype(float)
q3.index = pd.to_datetime(q3.index, utc=True)
q3 = q3[~q3.index.duplicated()].sort_index().interpolate(limit=8, limit_area="inside")
q = q3.sum(axis=1, min_count=3).dropna()
# flood thresholds for the combined inflow (see docstring)
hr = pd.read_csv(D / "inflow3_iv_hourly.csv", index_col=0, parse_dates=True).interpolate(limit=6)
hs = hr.sum(axis=1, min_count=3).dropna()
wy = hs.index.year + (hs.index.month >= 10)
am = hs.groupby(wy).agg(["max", "count"])
am = am[am["count"] > 7900]["max"]
Q_MEDIAN = float(am.median())
_x = np.log(am.drop(2024, errors="ignore"))
Q_MODERATE = float(np.exp(_x.mean() + 0.8416 * _x.std()))
print(f"inflow flood thresholds from {len(am)} water years: median flood {Q_MEDIAN:.0f}, moderate (5-yr) {Q_MODERATE:.0f} m3/s")
r = pd.read_csv(D / f"mrms_prcp_{CATCH}_{TAG}.csv", index_col=0, parse_dates=True).iloc[:, 0].astype(float)
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
area, dn, dist_dam = cr["area"].astype(float), cr["dn"], cr["dist"].astype(float)
n = area.size
ig = int(cr["out_i"])                                  # observed inflow is matched at the dam reach
upg = np.ones(n, bool)
dist = dist_dam
# local contributing area = total area minus that of direct upstream reaches
loc = area.copy()
for i in range(n):
    if dn[i] >= 0:
        loc[dn[i]] -= area[i]
loc = np.clip(loc, 0.05, None)
loc *= area[int(cr["out_i"])] / loc.sum()             # divergences double-count area
print(f"reaches {n}; summed local area {loc.sum():.0f} km2; dam area {area[ig]:.0f} km2")


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
    k = np.bincount(np.round(dist[upg] / c / DT).astype(int), weights=loc[upg]) / 3.6   # mm/h*km2 -> m3/s
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
order = np.argsort(-dist_dam)                                      # upstream first
Q = np.zeros((n, T.size))
for i in order:
    Q[i] += loc[i] / 3.6 * R
    # pass downstream with the translation lag of this reach's length
    if dn[i] >= 0:
        Q[dn[i]] += shift(Q[i], (dist_dam[i] - dist_dam[dn[i]]) / C)
QB = Q + base[None] * (area / area[ig])[:, None]
# nudge every reach by the dam reach's obs/model ratio, applied earlier by the reach's travel time, so
# the dam reach reproduces the observed inflow and upstream reaches lead it
ratio = np.clip(QOBS / np.maximum(QB[ig], 1e-3), 0.2, 5.0)
for i in range(n):
    QB[i] *= shift(ratio, -(dist_dam[i] - dist_dam[ig]) / C)
QN = (QB - QB.min(1, keepdims=True)) / (np.ptp(QB, 1, keepdims=True) + 1e-9)
io = int(cr["out_i"])
print(f"dam reach peak at +{T[QN[io].argmax()]:.1f} h (observed inflow +{T[ipk]:.1f} h)")

# ------------------------------------------------------------------ forecasts (one issue at t0, never revised)
VINH = D / "vinh" / "forecast.csv"
tt = T[T >= H_ISSUE - 1e-9]
lead = tt - H_ISSUE
FC_PLACEHOLDER = not VINH.exists()
if not FC_PLACEHOLDER:
    v = pd.read_csv(VINH, parse_dates=["time_utc"])
    vh = hrs(pd.DatetimeIndex(v.time_utc).tz_localize("UTC") if v.time_utc.dt.tz is None else pd.DatetimeIndex(v.time_utc))
    ML50, ML05, ML95, TVA = (np.interp(tt, vh, v[c].values) for c in ("ml_q50", "ml_q05", "ml_q95", "tva"))
else:
    # PLACEHOLDERS (replace with Vinh's data): ML close to the observed series, TVA lower and later.
    q0 = np.interp(H_ISSUE, T, QOBS)

    def fake(lag_h, gain, slow, tau):
        w = 1 - np.exp(-lead / tau)
        f = np.interp(tt - lag_h * w, T, QOBS) * (1 + (gain - 1) * w) * (1 + slow * (lead / HORIZON) ** 1.5 * w)
        return f + (q0 - f[0]) * np.exp(-lead / 2.0)

    ML50 = fake(0.75, 0.97, 0.03, 4.0)
    TVA = np.convolve(np.pad(fake(3.5, 0.80, 0.10, 6.0), 6, mode="edge"), np.ones(13) / 13, "valid")
    spread = 0.05 + 0.22 * (lead / HORIZON) ** 0.6 + 0.10 * np.clip((ML50 - q0) / (ML50.max() - q0), 0, 1)
    ML05, ML95 = ML50 * (1 - spread), ML50 * (1 + 1.25 * spread)
# chance of exceeding each flood category during the horizon: lognormal peak from the ML range
from scipy.stats import norm
ipf = int(ML50.argmax())
mu, sg = np.log(ML50[ipf]), (np.log(ML95[ipf]) - np.log(ML05[ipf])) / (2 * 1.645)
P_MEDIAN, P_MODERATE = (float(1 - norm.cdf((np.log(qt) - mu) / sg)) for qt in (Q_MEDIAN, Q_MODERATE))
FC_T = tt
print(f"forecasts ({'PLACEHOLDER' if FC_PLACEHOLDER else 'Vinh'}): ML peak {ML50[ipf]:.0f} m3/s at +{lead[ipf]:.1f} h, "
      f"TVA peak {TVA.max():.0f} m3/s at +{lead[TVA.argmax()]:.1f} h; P(median flood) {P_MEDIAN:.0%}, P(moderate) {P_MODERATE:.0%}")

pd.DataFrame({
    "time_utc": (T0 + pd.to_timedelta(T, unit="h")).strftime("%Y-%m-%d %H:%M"),
    "hours_after_issue": np.round(T - H_ISSUE, 2),
    "observed_streamflow_m3s": np.round(QOBS, 1),
    **{f"{k}_m3s": np.round(np.interp(T, FC_T, x, left=np.nan, right=np.nan), 1)
       for k, x in (("hydroorbit_ml_q50", ML50), ("hydroorbit_ml_q05", ML05), ("hydroorbit_ml_q95", ML95), ("tva", TVA))},
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
    FC_T=FC_T, ML50=ML50, ML05=ML05, ML95=ML95, TVA=TVA, FC_PLACEHOLDER=FC_PLACEHOLDER,
    P_MEDIAN=P_MEDIAN, P_MODERATE=P_MODERATE, Q_MEDIAN=Q_MEDIAN, Q_MODERATE=Q_MODERATE, clock_t=knots_t, clock_h=knots_h, rain_real=RAIN_REAL, C=C, KH=KH)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(T - H_ISSUE, QOBS, "k", label="observed inflow (3 NWIS gauges)")
ax.plot(T - H_ISSUE, qmod, "C1", label="routed model (map animation)")
ax.fill_between(FC_T - H_ISSUE, ML05, ML95, color="c", alpha=0.2)
ax.plot(FC_T - H_ISSUE, ML50, "c", label="HydroORBIT ML")
ax.plot(FC_T - H_ISSUE, TVA, "C1--", label="TVA")
ax.axhline(Q_MEDIAN, color="y", lw=0.8); ax.axhline(Q_MODERATE, color="orange", lw=0.8)
ax.axvline(0, color="r"); ax.legend()
ax2 = ax.twinx(); ax2.bar(T - H_ISSUE, RAIN, width=DT, color="b", alpha=0.3); ax2.set_ylim(RAIN.max() * 4, 0)
plt.savefig(B / "hydro_preview.png", dpi=70)
