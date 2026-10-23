"""Illustrative storm response for the headwater catchment.

Routes hillslope runoff through the D8 tree with a translation + diffusion scheme,
giving discharge time series at every catchment cell (used to swell the rivers on screen)
and at the gauge (the hydrograph). Then builds 'observed' and rolling 'HydroORBIT' forecasts.
All numbers are synthetic, for visualisation only.
"""
import numpy as np
from numba import njit
from scipy.signal import lfilter
from storm import rain_at

d = np.load("terrain.npz")
dirs, acc, cmask = d["dirs"], d["acc"], d["cmask"]
gy, gx = d["gauge"]
H, W = acc.shape
DT = 0.1
T = np.arange(0, 96 + DT / 2, DT)
NT = T.size

cells = np.flatnonzero(cmask.ravel())
cy, cx = cells // W, cells % W
print("catchment cells", cells.size)

# ---- rainfall on every catchment cell (mm/h), chunked in time
rain = np.zeros((cells.size, NT), np.float32)
for i, t in enumerate(T):
    rain[:, i] = rain_at(t, cy, cx)
rain_mean = rain.mean(0)
cum = np.cumsum(rain, 1) * DT
# runoff coefficient grows as the soil wets up (simple SCS-flavoured curve)
coef = 0.08 + 0.62 * (1 - np.exp(-cum / 45.0))
excess = rain * coef
# hillslope linear reservoir, K = 2.5 h
a = DT / 2.0
b = DT / 16.0
runoff = (0.62 * lfilter([a], [1, -(1 - a)], excess, axis=1)
          + 0.38 * lfilter([b], [1, -(1 - b)], excess, axis=1)).astype(np.float32)  # mm/h

# ---- routing order inside the catchment (upstream -> downstream = descending acc is a valid topo order)
order = cells[np.argsort(acc.ravel()[cells], kind="stable")]
pos = -np.ones(H * W, np.int64)
pos[order] = np.arange(order.size)
runoff_o = runoff[np.searchsorted(cells, order)]

DY = np.array([-1, -1, 0, 1, 1, 1, 0, -1]); DX = np.array([0, 1, 1, 1, 0, -1, -1, -1])
DD = np.array([1, 1.4142, 1, 1.4142, 1, 1.4142, 1, 1.4142])
df = dirs.ravel()
down = np.full(order.size, -1, np.int64)
step = np.zeros(order.size)
for i, c in enumerate(order):
    k = df[c]
    if k < 0 or c == gy * W + gx:
        continue
    dn = (c // W + DY[k]) * W + c % W + DX[k]
    down[i] = pos[dn]
    step[i] = DD[k]

V = 52.0  # cells per hour in channels (hillslope delay handled by the reservoir)

@njit(cache=True)
def route(runoff_o, down, step, v, dt):
    n, nt = runoff_o.shape
    q = runoff_o.copy()                 # mm/h * cell area units, accumulated
    for i in range(n):
        j = down[i]
        if j < 0:
            continue
        s = step[i] / v / dt            # lag in time steps (fractional)
        m = int(s)
        f = s - m
        for t in range(nt):
            t0 = t - m
            v0 = q[i, t0] if t0 >= 0 else q[i, 0]
            v1 = q[i, t0 - 1] if t0 - 1 >= 0 else q[i, 0]
            q[j, t] += (1 - f) * v0 + f * v1
    return q

q = route(runoff_o, down, step, V, DT)
# units: cell = 100 m x 100 m = 0.01 km^2 ; Q[m3/s] = mm/h * km^2 / 3.6
CELL_KM2 = 0.01
Q = q * CELL_KM2 / 3.6
gi = pos[gy * W + gx]
base = 14.0 + 0.05 * np.cumsum(rain_mean) * DT   # slowly rising baseflow
Qtrue = Q[gi] + base
print("peak Q %.0f m3/s at %.1f h, area %.0f km2, rain total %.0f mm" %
      (Qtrue.max(), T[Qtrue.argmax()], cells.size * CELL_KM2, rain_mean.sum() * DT))

# ---- channel cells kept for display (normalised discharge per cell)
ch = acc.ravel()[order] >= 120
ch_cells = order[ch]
Qch = Q[ch]
Qch_norm = (Qch / (Qch.max(1, keepdims=True) + 1e-6)).astype(np.float16)
Qch_peak = Qch.max(1).astype(np.float32)

# ---- observed: truth + small autocorrelated gauge noise, hourly samples
rng = np.random.default_rng(3)
noise = lfilter([0.3], [1, -0.7], rng.normal(0, 1, NT))
Qobs = Qtrue * (1 + 0.025 * noise)

# ---- HydroORBIT rolling forecasts: issued every hour, 48 h horizon.
# Skill improves as the storm is observed: timing / magnitude errors shrink with issue time.
def forecast(t0, horizon=48.0):
    tt = np.arange(t0, min(t0 + horizon, T[-1]) + 1e-6, DT)
    lead = tt - t0
    w = 1 - np.exp(-lead / 9.0)
    # errors decay as the event is observed; early runs slightly late and low
    tp = T[Qtrue.argmax()]
    stage = np.clip((t0 - 2) / (tp - 6), 0, 1)
    shift = 2.6 * (1 - stage) ** 1.5 + 0.5          # hours late
    mag = -0.16 * (1 - stage) ** 1.2 + 0.035        # relative magnitude error
    qs = np.interp(tt - shift * w, T, Qtrue)
    qf = qs * (1 + mag * w)
    # start exactly on the last observation
    q0 = np.interp(t0, T, Qobs)
    qf = qf + (q0 - qf[0]) * np.exp(-lead / 3.0)
    spread = (0.06 + 0.30 * (1 - stage) + 0.10) * w
    return tt, qf, qf * (1 - spread), qf * (1 + spread * 1.15)

# a 'HydroORBIT 6-h lead' trace for the full-event summary
lead_trace = np.zeros(NT)
for i, t in enumerate(T):
    t0 = max(t - 6.0, 0)
    tt, qf, lo, hi = forecast(t0)
    lead_trace[i] = np.interp(t, tt, qf)

np.savez_compressed("hydro.npz", T=T, rain_mean=rain_mean, Qtrue=Qtrue, Qobs=Qobs,
                    lead_trace=lead_trace, ch_cells=ch_cells, Qch_norm=Qch_norm,
                    Qch_peak=Qch_peak, coef_mean=coef.mean(0))
import pickle
print("channel cells", ch_cells.size)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(T, Qobs, "k", label="obs")
ax.plot(T, lead_trace, "c", label="6h lead")
for t0 in (6, 14, 20, 28):
    tt, qf, lo, hi = forecast(t0)
    ax.fill_between(tt, lo, hi, alpha=0.15); ax.plot(tt, qf, lw=1)
ax2 = ax.twinx(); ax2.bar(T, rain_mean, width=DT, color="b", alpha=0.3); ax2.invert_yaxis(); ax2.set_ylim(40, 0)
ax.legend(); plt.savefig("hydro_preview.png", dpi=80)
