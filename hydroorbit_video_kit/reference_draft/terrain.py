"""Synthetic large river basin + headwater catchment for the HydroORBIT demo video.

Produces terrain.npz with a carved DEM, D8 flow routing, flow accumulation,
basin mask, the selected headwater catchment and its outlet (the "gauge").
"""
import numpy as np
from numba import njit
import heapq
from scipy import ndimage as ndi
from scipy.interpolate import splprep, splev

H, W = 1200, 2400
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- noise
def fbm(h, w, beta, seed):
    r = np.random.default_rng(seed)
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.rfftfreq(w)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1.0
    amp = k ** (-beta / 2.0)
    amp[0, 0] = 0
    ph = r.normal(size=(h, w // 2 + 1)) + 1j * r.normal(size=(h, w // 2 + 1))
    f = np.fft.irfft2(amp * ph, s=(h, w))
    f = (f - f.mean()) / f.std()
    return f

# ---------------------------------------------------------------- basin geometry
ctrl = np.array([
    (0.90, 0.16), (0.80, 0.25), (0.68, 0.37), (0.55, 0.52), (0.44, 0.66),
    (0.33, 0.78), (0.22, 0.76), (0.165, 0.60), (0.15, 0.42), (0.13, 0.24), (0.115, 0.12)])
tck, _ = splprep([ctrl[:, 0] * W, ctrl[:, 1] * H], s=0)
ss = np.linspace(0, 1, 4000)
sx, sy = splev(ss, tck)
stem = np.stack([sx, sy], 1)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
# nearest point on stem (coarse brute force in chunks)
from scipy.spatial import cKDTree
tree = cKDTree(stem)
d, idx = tree.query(np.stack([xx.ravel(), yy.ravel()], 1), workers=-1)
d = d.reshape(H, W)
s_of = ss[idx].reshape(H, W)          # 0 = headwaters, 1 = outlet

# basin half-width: wide upper basin, narrow lower basin
width = (0.25 - 0.17 * np.clip((s_of - 0.15) / 0.65, 0, 1) ** 0.8) * H
# side of the stem (SE side gets the high mountains in the upper basin)
tang = np.gradient(stem, axis=0)
tang /= np.linalg.norm(tang, axis=1, keepdims=True)
tx, ty = tang[idx, 0].reshape(H, W), tang[idx, 1].reshape(H, W)
cross = (xx - stem[idx, 0].reshape(H, W)) * ty - (yy - stem[idx, 1].reshape(H, W)) * tx
side_se = cross < 0  # right of flow direction

bnoise = ndi.gaussian_filter(fbm(H, W, 3.2, 11), 2) * 0.16
rel = d / width + bnoise
mask = rel < 1.0
# trim the tail beyond the outlet and the top of the headwaters
mask &= (s_of < 0.995)
lab, n = ndi.label(mask)
mask = lab == lab[int(stem[2000, 1]), int(stem[2000, 0])]
mask = ndi.binary_opening(mask, iterations=3)
mask = ndi.binary_fill_holes(mask)

# ---------------------------------------------------------------- elevation
n1 = fbm(H, W, 3.6, 21)
n1 = n1 - ndi.gaussian_filter(n1, 45); n1 /= n1.std()
n2 = fbm(H, W, 2.6, 22)
r3 = fbm(H, W, 3.0, 23); r3 = r3 - ndi.gaussian_filter(r3, 45); r3 /= r3.std()
ridged = 1 - np.abs(r3)
up = 1 - s_of                                   # upstream-ness
mtn = np.where(side_se, 1.0, 0.55) * np.clip((0.75 - s_of) / 0.5, 0, 1) ** 1.2
z = (650 * up                                   # regional slope to outlet
     + 600 * np.clip(rel, 0, 1.3) ** 1.3 * (0.35 + mtn)   # valley walls / mountains
     + 260 * n1 * (0.4 + mtn)
     + 120 * ridged * (0.3 + mtn)
     + 35 * n2)
z = z.astype(np.float64)
z -= z[mask].min()

# ---------------------------------------------------------------- hydrology kernels
DY = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
DX = np.array([0, 1, 1, 1, 0, -1, -1, -1])
DD = np.array([1, 1.4142, 1, 1.4142, 1, 1.4142, 1, 1.4142])

@njit(cache=True)
def priority_flood(z, mask, oy, ox):

    h, w = z.shape
    out = np.full((h, w), np.inf)
    closed = np.zeros((h, w), np.bool_)
    heap = [(z[oy, ox], oy, ox)]
    out[oy, ox] = z[oy, ox]
    closed[oy, ox] = True
    dy = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dx = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    while len(heap) > 0:
        e, y, x = heapq.heappop(heap)
        for k in range(8):
            ny, nx = y + dy[k], x + dx[k]
            if ny < 0 or nx < 0 or ny >= h or nx >= w:
                continue
            if closed[ny, nx] or not mask[ny, nx]:
                continue
            closed[ny, nx] = True
            ne = max(z[ny, nx], e + 1e-3)
            out[ny, nx] = ne
            heapq.heappush(heap, (ne, ny, nx))
    return out

@njit(cache=True)
def d8(zf, mask):
    h, w = zf.shape
    dirs = np.full((h, w), -1, np.int8)
    dy = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dx = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dd = np.array([1, 1.4142, 1, 1.4142, 1, 1.4142, 1, 1.4142])
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            best = 0.0
            bk = -1
            for k in range(8):
                ny, nx = y + dy[k], x + dx[k]
                if ny < 0 or nx < 0 or ny >= h or nx >= w or not mask[ny, nx]:
                    continue
                s = (zf[y, x] - zf[ny, nx]) / dd[k]
                if s > best:
                    best = s
                    bk = k
            dirs[y, x] = bk
    return dirs

@njit(cache=True)
def accumulate(order, dirs, w):
    n = dirs.size
    acc = np.ones(n)
    df = dirs.ravel()
    dy = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dx = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    for i in range(order.size):
        c = order[i]
        k = df[c]
        if k < 0:
            continue
        y, x = c // w, c % w
        acc[(y + dy[k]) * w + x + dx[k]] += acc[c]
    return acc

def route(z, mask, oy, ox):
    zf = priority_flood(z, mask, oy, ox)
    dirs = d8(zf, mask)
    zz = np.where(mask, zf, -np.inf).ravel()
    order = np.argsort(-zz, kind="stable")
    order = order[np.isfinite(zz[order])]
    acc = accumulate(order, dirs, W).reshape(H, W)
    acc[~mask] = 0
    return zf, dirs, acc, order

# outlet = last masked cell along the stem
oi = None
for i in range(len(stem) - 1, 0, -1):
    x, y = int(stem[i, 0]), int(stem[i, 1])
    if mask[y, x]:
        oi = (y, x)
        break
oy, ox = oi
z[oy, ox] = z[mask].min() - 5

# ---- drainage tree by Dijkstra from the outlet over a noisy cost field,
# ---- then steady-state stream-power elevations (slope ~ k * A^-0.5)
@njit(cache=True)
def dijkstra_tree(cost, mask, oy, ox):
    h, w = cost.shape
    dist = np.full((h, w), np.inf)
    dirs = np.full((h, w), -1, np.int8)
    done = np.zeros((h, w), np.bool_)
    order = np.empty(h * w, np.int64)
    no = 0
    dy = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dx = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dd = np.array([1, 1.4142, 1, 1.4142, 1, 1.4142, 1, 1.4142])
    dist[oy, ox] = 0.0
    heap = [(0.0, oy, ox)]
    while len(heap) > 0:
        e, y, x = heapq.heappop(heap)
        if done[y, x]:
            continue
        done[y, x] = True
        order[no] = y * w + x
        no += 1
        for k in range(8):
            ny, nx = y + dy[k], x + dx[k]
            if ny < 0 or nx < 0 or ny >= h or nx >= w or not mask[ny, nx] or done[ny, nx]:
                continue
            nd = e + dd[k] * 0.5 * (cost[y, x] + cost[ny, nx])
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                dirs[ny, nx] = (k + 4) % 8      # neighbour drains back to (y, x)
                heapq.heappush(heap, (nd, ny, nx))
    return dirs, order[:no]

@njit(cache=True)
def elevate(order, dirs, acc, kfield, w):
    n = dirs.size
    z = np.zeros(n)
    df = dirs.ravel()
    af = acc.ravel()
    kf = kfield.ravel()
    dy = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dx = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dd = np.array([1, 1.4142, 1, 1.4142, 1, 1.4142, 1, 1.4142])
    for i in range(order.size):          # downstream -> upstream
        c = order[i]
        k = df[c]
        if k < 0:
            continue
        y, x = c // w, c % w
        dn = (y + dy[k]) * w + x + dx[k]
        z[c] = z[dn] + dd[k] * kf[c] * af[c] ** -0.5
    return z

cn = ndi.gaussian_filter(fbm(H, W, 2.2, 31), 1.0)
cost = np.exp(0.55 * fbm(H, W, 3.0, 32) + 0.35 * cn) * (1 + 3.0 * np.clip(rel, 0, 1.5) ** 2)
dirs, order_dn = dijkstra_tree(cost, mask, oy, ox)
order = order_dn[::-1].copy()            # upstream -> downstream for accumulation
acc = accumulate(order, dirs, W).reshape(H, W)
acc[~mask] = 0
kfield = (2.2 + 9.0 * mtn + 2.5 * np.clip(rel, 0, 1)) * np.exp(0.25 * fbm(H, W, 3.2, 33))
z = elevate(order_dn, dirs, acc, kfield, W).reshape(H, W)
z += 6.0 * ndi.gaussian_filter(fbm(H, W, 1.6, 34), 0.7) * (0.3 + mtn)   # fine texture
z = np.where(mask, z, 0)
# ---------------------------------------------------------------- pick headwater catchment
# tributary mouth on the SE (mountain) side of the upper basin
cand = (acc > 30000) & (acc < 110000) & mask & (s_of < 0.55) & (s_of > 0.08)
cy, cx = np.nonzero(cand)
best = None
for y, x in zip(cy, cx):
    k = dirs[y, x]
    if k < 0:
        continue
    a_dn = acc[y + DY[k], x + DX[k]]
    if a_dn > 1.8 * acc[y, x]:
        score = acc[y, x]
        if best is None or score > best[0]:
            best = (score, y, x)
_, gy, gx = best
print("gauge", gy, gx, "area", acc[gy, gx])

# catchment mask: ascending elevation order, inherit from downstream
@njit(cache=True)
def upstream_mask(order, dirs, w, g):
    n = dirs.size
    m = np.zeros(n, np.bool_)
    m[g] = True
    df = dirs.ravel()
    dy = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dx = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    for i in range(order.size - 1, -1, -1):   # ascending elevation
        c = order[i]
        k = df[c]
        if k < 0 or c == g:
            continue
        y, x = c // w, c % w
        if m[(y + dy[k]) * w + x + dx[k]]:
            m[c] = True
    return m

cmask = upstream_mask(order, dirs, W, gy * W + gx).reshape(H, W)
ys, xs = np.nonzero(cmask)
print("catchment bbox", ys.min(), ys.max(), xs.min(), xs.max())

# flow distance to the gauge (cells), for travel-time lags
@njit(cache=True)
def flowdist(order, dirs, w, g, m):
    n = dirs.size
    fd = np.zeros(n)
    df = dirs.ravel()
    dy = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dx = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dd = np.array([1, 1.4142, 1, 1.4142, 1, 1.4142, 1, 1.4142])
    for i in range(order.size - 1, -1, -1):
        c = order[i]
        if not m[c] or c == g:
            continue
        k = df[c]
        y, x = c // w, c % w
        fd[c] = fd[(y + dy[k]) * w + x + dx[k]] + dd[k]
    return fd

fdist = flowdist(order, dirs, W, gy * W + gx, cmask.ravel()).reshape(H, W)

np.savez_compressed("terrain.npz", z=z.astype(np.float32), dirs=dirs, acc=acc.astype(np.float32),
                    mask=mask, cmask=cmask, gauge=np.array([gy, gx]), outlet=np.array([oy, ox]),
                    fdist=fdist.astype(np.float32), stem=stem.astype(np.float32))
print("saved")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ls = matplotlib.colors.LightSource(315, 40)
fig, ax = plt.subplots(figsize=(16, 8), dpi=80)
zz = np.where(mask, z, np.nan)
ax.imshow(ls.hillshade(np.nan_to_num(z), vert_exag=0.6), cmap="gray")
ax.imshow(np.where(mask, z, np.nan), cmap="terrain", alpha=0.5)
ax.imshow(np.ma.masked_less(np.log10(acc+1), 3.0), cmap="Blues", alpha=0.9, vmin=2, vmax=6)
ax.contour(cmask, [0.5], colors="r")
ax.plot(gx, gy, "r*", ms=15)
plt.savefig("preview.png", bbox_inches="tight")

