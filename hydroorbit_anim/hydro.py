#!/usr/bin/env python3
"""Event hydrology for the animation (build/hydro.npz, data/rain_grid_*.npz).

Event: Feb 22-24, 2019 at USGS 03455000 (French Broad River near Newport, TN; 4,799 km2; 34 km above Douglas Dam).
One forecast issued at t0 = 2019-02-22 06:00 UTC, 48 h before the observed peak, never revised.

- Observed flow, observed rain, AI forecast (median, 10th and 90th percentile) and the TVA forecast all come
  from Vinh's file (Vinh_data/03455000_forecasts_2016_2025.nc). Values are 6-hour means / totals labelled with the
  END of the 6-hour window (valid time); the same convention is used for the flood levels below.
- Rain: observed 6-hour basin precipitation from Vinh's file, spread evenly over the six hours. There is no
  gridded rain for 2019 (MRMS starts 2020-10-14), so the SPATIAL pattern on the map is stylised: a smooth
  moving field scaled so its basin mean equals the observed rainfall at every hour.
- Channel animation: rain -> runoff (wetting-up coefficient) -> hillslope linear reservoir -> translation along
  NHDPlus flow distance. c, K and the coefficient are fitted so the modelled hydrograph at the gauge reach matches
  the observed flow; the pulse then continues downstream to Douglas Dam. Per-reach series are normalised for display.
- Flood levels at this gauge, from 6-hour annual maxima of the USGS record (water years 2008-2025):
  median flood = median annual peak; moderate flood = 5-year flood (20 % annual chance, lognormal fit without
  Hurricane Helene 2024).
- Chance of flooding: lognormal distribution fitted to the AI 10th/50th/90th percentiles at the forecast peak,
  evaluated at each flood level. These are derived from the AI quantiles, not model outputs.
"""
import pathlib
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage as ndi
from scipy.interpolate import PchipInterpolator
from scipy.signal import lfilter
from scipy.stats import norm

CATCH = "douglas"
TAG = "2019feb"
ISSUE = pd.Timestamp("2019-02-22 06:00", tz="UTC")     # forecast issue time t0 (before the peak)
PRE_H, HORIZON, POST_H = 30.0, 48.0, 72.0              # hours shown before t0; forecast horizon; hours shown after t0
HERE = pathlib.Path(__file__).parent
D, B = HERE / "data", HERE / "build"
(HERE / "private").mkdir(exist_ok=True)
DT = 0.25                                              # hours

# ------------------------------------------------------------------ Vinh's file
ds = xr.open_dataset(HERE / "Vinh_data" / "03455000_forecasts_2016_2025.nc")
df = ds[["observed_Q", "observed_Pr"]].to_dataframe().reset_index()
df["valid"] = pd.DatetimeIndex(df.forecast_time + pd.to_timedelta(df.lead_time, unit="h")).tz_localize("UTC")
obs = df.dropna(subset=["observed_Q"]).drop_duplicates("valid").set_index("valid").sort_index()
fc = ds.sel(forecast_time=ISSUE.tz_localize(None)).to_dataframe()          # one row per lead (6..48 h)
assert fc[["TVA_Q", "AI_Q10", "AI_Q50", "AI_Q90"]].notna().all().all(), "forecast has gaps"

T0 = ISSUE - pd.Timedelta(hours=PRE_H)                 # clock origin: hours since T0
H_ISSUE = PRE_H
H_END = PRE_H + POST_H
T = np.arange(0, H_END + DT / 2, DT)
hrs = lambda idx: ((idx - T0).total_seconds() / 3600).values
qe = obs.observed_Q[(obs.index >= T0 - pd.Timedelta(hours=6)) & (obs.index <= T0 + pd.Timedelta(hours=H_END + 6))]
gaps = qe.index.to_series().diff().dt.total_seconds().div(3600)
print(f"window {T0} -> {T0 + pd.Timedelta(hours=H_END)} ({H_END:.0f} h); issue at +{H_ISSUE:.0f} h; "
      f"obs records {len(qe)}, longest gap {gaps.max():.0f} h")
assert gaps.max() <= 6.0, "observed flow has a gap > 6 h"
QOBS = PchipInterpolator(hrs(qe.index), qe.values)(np.clip(T, hrs(qe.index)[0], hrs(qe.index)[-1])).astype(float)
ipk = int(QOBS.argmax())
print(f"observed peak {QOBS[ipk]:.0f} m3/s at {T0 + pd.Timedelta(hours=T[ipk])} (+{T[ipk] - H_ISSUE:.1f} h after issue)")

# flood levels from the long USGS record at this gauge
nw = pd.read_csv(D / "newport_iv_6h.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna()
wy = nw.index.year + (nw.index.month >= 10)
am = nw.groupby(wy).agg(["max", "count"])
am = am[am["count"] > 1200]["max"]                     # >= ~82 % of the 1,460 six-hour values in a water year
Q_MEDIAN = float(am.median())
_x = np.log(am.drop(2024, errors="ignore"))
Q_MODERATE = float(np.exp(_x.mean() + 0.8416 * _x.std()))
print(f"flood levels from {len(am)} water years ({am.index.min()}-{am.index.max()}): "
      f"median flood {Q_MEDIAN:.0f}, moderate (5-yr) {Q_MODERATE:.0f} m3/s")

# rain: 6-hour totals ending at the valid time -> mm/h for each hour of the window
pr = obs.observed_Pr.dropna()
hours = pd.date_range(T0.floor("h"), T0 + pd.Timedelta(hours=H_END), freq="h", tz="UTC")
rate = pd.Series(0.0, index=hours)
for v, p in pr.items():
    sel = (hours >= v - pd.Timedelta(hours=6)) & (hours < v)
    rate[sel] = p / 6.0
RAIN_HOURLY_T = hrs(hours)
RAIN_HOURLY = rate.values
RAIN = np.interp(T, RAIN_HOURLY_T + 0.5, RAIN_HOURLY, left=0, right=0)
rain_start = rate[rate > 0.5].index[rate[rate > 0.5].index >= T0 + pd.Timedelta(hours=12)].min()
print(f"rain (observed basin mean): total {RAIN.sum() * DT:.0f} mm, max {RAIN.max():.1f} mm/h, starts {rain_start}")

# ------------------------------------------------------------------ routing
cr = np.load(B / "catch_rivers.npz")
area, dn, dist_dam = cr["area"].astype(float), cr["dn"], cr["dist"].astype(float)
n = area.size
ig = int(cr["gauge_i"])                                # observed flow is matched at the gauge reach
upg = np.zeros(n, bool); upg[ig] = True                # reaches upstream of the gauge (for the fit)
for _ in range(n):
    new = upg | np.isin(dn, np.flatnonzero(upg))
    if new.sum() == upg.sum():
        break
    upg = new
dist = np.where(upg, dist_dam - dist_dam[ig], np.inf)  # flow distance to the gauge
# local contributing area = total area minus that of direct upstream reaches
loc = area.copy()
for i in range(n):
    if dn[i] >= 0:
        loc[dn[i]] -= area[i]
loc = np.clip(loc, 0.05, None)
loc *= area[int(cr["out_i"])] / loc.sum()             # divergences double-count area
print(f"reaches {n} ({upg.sum()} above the gauge); summed local area {loc.sum():.0f} km2; "
      f"dam area {area[int(cr['out_i'])]:.0f} km2, gauge area {area[ig]:.0f} km2")


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
    # translation: each reach's local runoff reaches the gauge after dist/c hours
    return np.bincount(np.round(dist[upg] / c / DT).astype(int), weights=loc[upg]) / 3.6   # mm/h*km2 -> m3/s


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
# nudge every reach by the gauge's obs/model ratio, shifted by the reach's travel time relative to the gauge,
# so the gauge reach reproduces the observed hydrograph and the pulse travels on to the dam
ratio = np.clip(QOBS / np.maximum(QB[ig], 1e-3), 0.2, 5.0)
for i in range(n):
    QB[i] *= shift(ratio, -(dist_dam[i] - dist_dam[ig]) / C)
QN = (QB - QB.min(1, keepdims=True)) / (np.ptp(QB, 1, keepdims=True) + 1e-9)
io = int(cr["out_i"])
print(f"gauge reach peak at +{T[QN[ig].argmax()]:.1f} h (observed +{T[ipk]:.1f} h); "
      f"dam reach peak at +{T[QN[io].argmax()]:.1f} h")

# ------------------------------------------------------------------ forecasts (one issue at t0, never revised)
tt = T[(T >= H_ISSUE - 1e-9) & (T <= H_ISSUE + HORIZON + 1e-9)]
lead = tt - H_ISSUE
q0 = float(np.interp(H_ISSUE, T, QOBS))                # the forecast starts from the flow observed at t0
leads = np.r_[0.0, fc.index.values.astype(float)]
curve = lambda col: PchipInterpolator(leads, np.r_[q0, fc[col].values.astype(float)])(lead)
ML50, ML10, ML90, TVA = (curve(c) for c in ("AI_Q50", "AI_Q10", "AI_Q90", "TVA_Q"))
ipf = int(ML50.argmax())
mu, sg = np.log(ML50[ipf]), (np.log(ML90[ipf]) - np.log(ML10[ipf])) / (2 * 1.2816)
P_MEDIAN, P_MODERATE = (float(1 - norm.cdf((np.log(qt) - mu) / sg)) for qt in (Q_MEDIAN, Q_MODERATE))
FC_T = tt
print(f"forecasts: AI peak {ML50[ipf]:.0f} m3/s at +{lead[ipf]:.1f} h (10-90 % {ML10[ipf]:.0f}-{ML90[ipf]:.0f}), "
      f"TVA peak {TVA.max():.0f} m3/s at +{lead[TVA.argmax()]:.1f} h, observed peak {QOBS[ipk]:.0f} m3/s at "
      f"+{T[ipk] - H_ISSUE:.1f} h; P(median flood) {P_MEDIAN:.0%}, P(moderate) {P_MODERATE:.0%}")

pd.DataFrame({
    "time_utc": (T0 + pd.to_timedelta(T, unit="h")).strftime("%Y-%m-%d %H:%M"),
    "hours_after_issue": np.round(T - H_ISSUE, 2),
    "observed_streamflow_m3s": np.round(QOBS, 1),
    **{f"{k}_m3s": np.round(np.interp(T, FC_T, x, left=np.nan, right=np.nan), 1)
       for k, x in (("hydroorbit_ai_q50", ML50), ("hydroorbit_ai_q10", ML10), ("hydroorbit_ai_q90", ML90), ("tva", TVA))},
    "basin_rain_mm_h": np.round(RAIN, 2),
}).to_csv(HERE / "private" / "hydrograph_data.csv", index=False)   # holds Vinh's forecasts: kept out of git

# ------------------------------------------------------------------ stylised rain field (shape of the MRMS grid file)
# Smooth cells drifting north-east, rescaled each hour so the mask mean equals the observed basin rain rate.
tpl = np.load(D / "mrms_grid_douglas_2021mar.npz")     # only the grid geometry (lat, lon, basin mask) is used
lat, lon, mask = tpl["lat"], tpl["lon"], tpl["mask"]
rng = np.random.default_rng(5)
big = ndi.gaussian_filter(rng.standard_normal((lat.size + 240, lon.size + 240)), 22, mode="wrap")
big = (big - big.mean()) / big.std()
small = ndi.gaussian_filter(rng.standard_normal(big.shape), 7, mode="wrap")
small = (small - small.mean()) / small.std()
field = 0.8 * big + 0.35 * small
grid = np.zeros((hours.size, lat.size, lon.size), np.float32)
for k in range(hours.size):
    off = k * 0.7                                      # cells per hour, towards the north-east
    f = ndi.shift(field, (off, -off), order=1, mode="wrap")[120:120 + lat.size, 120:120 + lon.size]
    pat = np.clip(np.exp(0.55 * f), 0.15, None)
    pat /= pat[mask].mean()
    grid[k] = (pat * rate.values[k]).astype(np.float32)
np.savez_compressed(D / f"rain_grid_{CATCH}_{TAG}.npz", rain=grid, lat=lat, lon=lon, mask=mask,
                    t_start_ns=hours.as_unit("ns").asi8)
print(f"rain grid {grid.shape}, basin mean check {grid[:, mask].mean():.3f} mm/h vs {rate.values.mean():.3f}")

# ------------------------------------------------------------------ video clock (PCHIP)
# 0-11 s static context; 11-18 s storm arrives and builds to the issue time; 18-27 s clock holds at t0 (forecast issued);
# 27-34.4 s the event plays out (rain, then slow through the peak); 34.4-40 s held at the end.
tp = T[ipk]
rain_h = float((rain_start - T0).total_seconds() / 3600)
knots_t = [11.0, 14.0, 18.0, 27.0, 28.4, 29.8, 31.6, 33.2, 34.4, 40.0]
knots_h = [rain_h - 3.0, H_ISSUE - 5.0, H_ISSUE, H_ISSUE, 50.0, 68.0, tp + 2.0, tp + 14.0, H_END, H_END]
assert np.all(np.diff(knots_t) > 0) and np.all(np.diff(knots_h) >= 0), knots_h
print("clock knots", [(a, round(b, 1)) for a, b in zip(knots_t, knots_h)])

np.savez_compressed(
    B / "hydro.npz", T=T, t0_ns=np.int64(T0.as_unit("ns").value), RAIN=RAIN, RAIN_HT=RAIN_HOURLY_T, RAIN_H=RAIN_HOURLY,
    QOBS=QOBS, QMOD=qmod, QN=QN.astype(np.float16), RUNOFF=R, H_END=H_END, H_ISSUE=H_ISSUE, HORIZON=HORIZON, ipk=ipk,
    FC_T=FC_T, ML50=ML50, ML10=ML10, ML90=ML90, TVA=TVA,
    P_MEDIAN=P_MEDIAN, P_MODERATE=P_MODERATE, Q_MEDIAN=Q_MEDIAN, Q_MODERATE=Q_MODERATE,
    clock_t=knots_t, clock_h=knots_h, C=C, KH=KH)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(T - H_ISSUE, QOBS, "k", label="observed (6-h mean, Newport)")
ax.plot(T - H_ISSUE, qmod, "C1", alpha=0.5, label="routed model (map animation)")
ax.fill_between(FC_T - H_ISSUE, ML10, ML90, color="c", alpha=0.2)
ax.plot(FC_T - H_ISSUE, ML50, "c", label="HydroORBIT AI")
ax.plot(FC_T - H_ISSUE, TVA, "C1--", label="TVA")
ax.axhline(Q_MEDIAN, color="y", lw=0.8); ax.axhline(Q_MODERATE, color="orange", lw=0.8)
ax.axvline(0, color="r"); ax.legend()
ax2 = ax.twinx(); ax2.bar(T - H_ISSUE, RAIN, width=DT, color="b", alpha=0.3); ax2.set_ylim(RAIN.max() * 4, 0)
plt.savefig(B / "hydro_preview.png", dpi=70)
