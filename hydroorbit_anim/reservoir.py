#!/usr/bin/env python3
"""Schematic reservoir + dam geometry at the gauge (build/reservoir.npz).

Not a real impoundment: the French Broad at Asheville is free-flowing. We flood-fill the valley floor
upstream of the gauge up to a fixed schematic pool level and put a dam across the valley at the gauge.
"""
import sys, pathlib
import numpy as np
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).parent
B = HERE / "build"
POOL_M = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0       # pool level above the gauge-cell terrain (m)
LEN_KM = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0        # reservoir length along the river
cd, rd, cr = np.load(B / "catch.npz"), np.load(B / "routing.npz"), np.load(B / "catch_rivers.npz")
z, xs, ys = cd["z"], cd["xs"], cd["ys"]
gx, gy = cd["gauge"]
dx = xs[1] - xs[0]
gc = int(round((gx - xs[0]) / dx)); gr = int(round((gy - ys[0]) / (ys[1] - ys[0])))
zg = float(z[gr - 2:gr + 3, gc - 2:gc + 3].min())               # river-bed level at the dam
level = zg + POOL_M / 1000
# flow distance (km) to gauge from the 90 m routing grid, sampled on the 60 m mesh grid
rx, ry = rd["xs"], rd["ys"]
fd = rd["fd"]
X, Y = np.meshgrid(xs, ys)
ci = np.clip(np.round((X - rx[0]) / (rx[1] - rx[0])).astype(int), 0, fd.shape[1] - 1)
ri = np.clip(np.round((Y - ry[0]) / (ry[1] - ry[0])).astype(int), 0, fd.shape[0] - 1)
fdm = fd[ri, ci]
cand = (z < level) & (fdm >= 0) & (fdm <= LEN_KM * 1.15)
lab, _ = ndi.label(cand)
pool = lab == lab[gr, gc] if lab[gr, gc] else np.zeros_like(cand)
# keep only cells within a corridor of the main river (valley floor, not side flats)
v, off = cr["v"], cr["off"]
big = np.flatnonzero(cr["order"] >= 4)
pts = np.concatenate([v[off[i]:off[i + 1], :2] for i in big])
d_river = ndi.distance_transform_edt(~np.isin(np.arange(1), [0])) if False else None
from scipy.spatial import cKDTree
dist_r, _ = cKDTree(pts).query(np.c_[X.ravel(), Y.ravel()])
pool &= dist_r.reshape(X.shape) < 0.5
pool = ndi.binary_opening(pool, iterations=1) | (pool & (np.hypot(X - gx, Y - gy) < 0.3))
lab, _ = ndi.label(pool)
pool = lab == lab[gr, gc]
print(f"pool level {level*1000:.0f} m (bed {zg*1000:.0f}); cells {pool.sum()}, area {pool.sum()*0.06*0.06:.2f} km2")
yy, xx = np.nonzero(pool)
print("extent x", xs[xx].min(), xs[xx].max(), " y", ys[yy].min(), ys[yy].max(), " gauge", gx, gy)
# river direction at the gauge (upstream -> downstream): from large-order polyline vertices near the gauge
near = pts[np.hypot(pts[:, 0] - gx, pts[:, 1] - gy) < 1.2]
u = np.linalg.svd(near - near.mean(0))[2][0]
ups = pool & True
mean_up = np.array([xs[xx].mean() - gx, ys[yy].mean() - gy])
if np.dot(u, mean_up) < 0: u = -u                                   # u points upstream
perp = np.array([-u[1], u[0]])
pool &= ((X - gx) * u[0] + (Y - gy) * u[1]) > 0.02                      # nothing downstream of the dam
pool &= fdm <= LEN_KM
lab, _ = ndi.label(pool)
near_g = np.argwhere((lab > 0) & (np.hypot(X - gx, Y - gy) < 0.5))[0]
pool = lab == lab[tuple(near_g)]
yy, xx = np.nonzero(pool)
print(f"trimmed pool: {pool.sum()*0.0036:.2f} km2, length ~{np.ptp(xs[xx]*u[0] + ys[yy]*u[1]):.1f} km")
# dam transect: walk both ways across the valley until terrain clears the pool level
def walk(sgn):
    for s in np.arange(0, 1.5, 0.03):
        px, py = np.array([gx, gy]) + sgn * s * perp
        c = int(round((px - xs[0]) / dx)); r = int(round((py - ys[0]) / (ys[1] - ys[0])))
        if z[r, c] > level + 0.006:
            return s
    return 1.5
sL, sR = walk(+1), walk(-1)
print(f"dam half-widths {sL*1000:.0f} m / {sR*1000:.0f} m; river dir {u}")
np.savez_compressed(B / "reservoir.npz", pool=pool, level=np.float32(level), bed=np.float32(zg), up=u, perp=perp,
                    sL=np.float32(sL), sR=np.float32(sR), gauge=np.array([gx, gy]))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
r0, r1, c0, c1 = max(yy.min() - 40, 0), yy.max() + 40, max(xx.min() - 40, 0), xx.max() + 40
fig, ax = plt.subplots(figsize=(9, 7))
ax.imshow(z[r0:r1, c0:c1], cmap="terrain", extent=[xs[c0], xs[c1], ys[r1], ys[r0]], vmin=zg, vmax=zg + 0.25)
ax.imshow(np.ma.masked_where(~pool[r0:r1, c0:c1], pool[r0:r1, c0:c1]), cmap="Blues", extent=[xs[c0], xs[c1], ys[r1], ys[r0]])
ax.plot([gx - sR * perp[0] * -1, gx + sL * perp[0]], [gy + sR * perp[1] * -1 * -1, gy + sL * perp[1]], "r-")
ax.plot(gx, gy, "ro"); plt.savefig(B / "reservoir_check.png", dpi=70)
