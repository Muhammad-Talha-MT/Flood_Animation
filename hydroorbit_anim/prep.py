#!/usr/bin/env python3
"""Turn the raw fetched layers into render-ready grids and polylines (build/).

All coordinates are km in EPSG:5070, shifted by ORIGIN so VTK works with small floats.
Elevations are stored in km (true scale); vertical exaggeration is applied at render time.
"""
import heapq, math, pathlib, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
from numba import njit
from rasterio import features
from rasterio.transform import Affine
from scipy import ndimage as ndi
from shapely.geometry import Point
warnings.filterwarnings("ignore")

SITE = "03451500"
HERE = pathlib.Path(__file__).parent
D, B = HERE / "data", HERE / "build"
B.mkdir(exist_ok=True)
CRS = 5070

basin = gpd.read_file(D / "tn_basin.geojson").to_crs(CRS)
catch = gpd.read_file(D / f"catch_{SITE}.geojson").to_crs(CRS)
gauge = gpd.read_file(D / f"gauge_{SITE}.geojson").to_crs(CRS)
BPOLY, CPOLY = basin.union_all(), catch.union_all()
GXY = np.array(gauge.geometry.iloc[0].coords[0])

bx0, by0, bx1, by1 = BPOLY.bounds
ORIGIN = np.array([(bx0 + bx1) / 2, (by0 + by1) / 2])
km = lambda xy: (np.asarray(xy) - ORIGIN) / 1000.0


def load_dem(path, res):
    """Read a DEM, fill nodata from neighbours, resample to `res` metres. Returns z (m), transform."""
    da = rioxarray.open_rasterio(path, masked=True).squeeze()
    if res is not None and abs(res - abs(da.rio.resolution()[0])) > 1:
        da = da.rio.reproject(CRS, resolution=res, resampling=5)  # average
    z = da.values.astype(np.float32)
    bad = ~np.isfinite(z) | (z < -500)
    if bad.any():
        idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
        z = z[tuple(idx)]
    return z, da.rio.transform(), bad


def grid_xy(tr, shape):
    """Cell-centre x, y vectors (metres) of a north-up raster."""
    h, w = shape
    xs = tr.c + tr.a * (np.arange(w) + 0.5)
    ys = tr.f + tr.e * (np.arange(h) + 0.5)
    return xs, ys


def rasterize(geom, tr, shape, buf=0.0):
    g = geom.buffer(buf) if buf else geom
    return features.rasterize([(g, 1)], out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)


def hillshade(z, cell, ex, az=315, alt=40):
    gy, gx = np.gradient(z * ex, cell)
    az, alt = math.radians(az), math.radians(alt)
    lx, ly, lz = math.cos(alt) * math.sin(az), math.cos(alt) * math.cos(az), math.sin(alt)
    # rows go north -> south, so +gy (row) is southwards
    n = np.sqrt(1 + gx ** 2 + gy ** 2)
    return np.clip((-gx * lx + gy * ly + lz) / n, 0, 1)


HYPSO_STOPS = np.array([0.0, 0.10, 0.25, 0.42, 0.60, 0.80, 1.0])
HYPSO_COLS = np.array([(70, 100, 66), (88, 120, 74), (122, 138, 86), (150, 140, 96),
                       (142, 124, 102), (160, 154, 146), (205, 202, 196)], np.float32)


def tint(z, zlo, zhi):
    zn = np.clip((z - zlo) / (zhi - zlo), 0, 1)
    return np.stack([np.interp(zn, HYPSO_STOPS, HYPSO_COLS[:, k]) for k in range(3)], -1)


def shade_rgb(z, cell, ex, zlo, zhi, ao_sigma):
    hs = 0.6 * hillshade(z, cell, ex, 315, 40) + 0.25 * hillshade(z, cell, ex, 220, 55) + 0.15
    relief = z - ndi.gaussian_filter(z, ao_sigma)
    ao = np.clip(0.75 + relief / (np.percentile(np.abs(relief), 95) + 1e-6) * 0.35, 0.35, 1.1)
    rgb = tint(z, zlo, zhi) * ((0.35 + 0.85 * hs) * ao)[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def lines_to_arrays(gdf, zfun):
    """Flatten LineStrings to vertex arrays (km) + offsets; z sampled from the display DEM."""
    xs, offs = [], [0]
    for g in gdf.geometry:
        if g.geom_type == "MultiLineString":
            g = max(g.geoms, key=lambda s: s.length)
        c = np.asarray(g.coords)[:, :2]
        xs.append(c)
        offs.append(offs[-1] + len(c))
    v = km(np.concatenate(xs))
    z = zfun(v[:, 0], v[:, 1])
    return np.column_stack([v, z]).astype(np.float32), np.array(offs, np.int64)


def sampler(z_km, tr, shape):
    """Bilinear z(x_km, y_km) on a grid with transform tr (metres)."""
    def f(xk, yk):
        X = xk * 1000 + ORIGIN[0]
        Y = yk * 1000 + ORIGIN[1]
        col = (X - tr.c) / tr.a - 0.5
        row = (Y - tr.f) / tr.e - 0.5
        return ndi.map_coordinates(z_km, [row, col], order=1, mode="nearest")
    return f


# =========================================================== basin overview
print("basin DEM")
zb, trb, badb = load_dem(D / "tn_dem_120m.tif", 480)
zb = ndi.gaussian_filter(zb, 0.6)
print("   grid", zb.shape, "cell", trb.a)
bxs, bys = grid_xy(trb, zb.shape)
bmask = rasterize(BPOLY, trb, zb.shape)
# signed distance (km) outside the basin, used for the soft fade at the basin edge
bout = ndi.distance_transform_edt(~bmask) * trb.a / 1000.0
cmask_b = rasterize(CPOLY, trb, zb.shape)
# texture at 240 m (2x mesh resolution)
zt, trt, _ = load_dem(D / "tn_dem_120m.tif", 240)
zlo, zhi = 60.0, 1900.0
tex_b = shade_rgb(ndi.gaussian_filter(zt, 0.5), 240, 6.0, zlo, zhi, 12)
bmask_t = rasterize(BPOLY, trt, zt.shape)
bout_t = ndi.distance_transform_edt(~bmask_t) * trt.a / 1000.0
np.savez_compressed(B / "basin.npz", z=(zb / 1000).astype(np.float32), xs=km(np.c_[bxs, bxs])[:, 0],
                    ys=km(np.c_[bys, bys])[:, 1], valid=~badb, mask=bmask, out_km=bout.astype(np.float32),
                    cmask=cmask_b, tex=tex_b, tex_xs=km(np.c_[grid_xy(trt, zt.shape)[0]] * np.ones((1, 2)))[:, 0],
                    tex_ys=km(np.c_[grid_xy(trt, zt.shape)[1]] * np.ones((1, 2)))[:, 1],
                    tex_out=bout_t.astype(np.float32), origin=ORIGIN)
zb_s = sampler(zb / 1000, trb, zb.shape)

print("basin flowlines")
fl = gpd.read_file(D / "tn_flowlines.gpkg").to_crs(CRS)
fl = fl[fl.geometry.notna() & ~fl.geometry.is_empty].reset_index(drop=True)
fl["in_catch"] = fl.geometry.representative_point().within(CPOLY)
fl = fl.sort_values("streamorde", ascending=False).reset_index(drop=True)
v, o = lines_to_arrays(fl, zb_s)
# distance of each line's first vertex from the main-stem mouth, for the "grow outward" reveal
np.savez_compressed(B / "basin_rivers.npz", v=v, off=o, order=fl.streamorde.values.astype(np.int16),
                    area=fl.totdasqkm.values.astype(np.float32), in_catch=fl.in_catch.values,
                    name=fl.gnis_name.fillna("").values.astype("U64"))
names = set(fl.gnis_name.dropna())
for n in ["Tennessee River", "Holston River", "French Broad River", "Little Tennessee River",
          "Hiwassee River", "Clinch River", "Elk River", "Duck River"]:
    print(f"   {'OK ' if n in names else 'MISSING'} {n}: {(fl.gnis_name == n).sum()} reaches")

# outlines (basin and catchment) as polylines draped on the basin DEM
def ring_arrays(poly, zfun, step=1000):
    g = poly if poly.geom_type == "Polygon" else max(poly.geoms, key=lambda p: p.area)
    ring = g.exterior.segmentize(step)
    v = km(np.asarray(ring.coords)[:, :2])
    return np.column_stack([v, zfun(v[:, 0], v[:, 1])]).astype(np.float32)

# =========================================================== catchment
print("catchment DEM")
zc30, trc30, badc = load_dem(D / f"catch_{SITE}_dem_30m.tif", None)
print("   30 m grid", zc30.shape)
zc_tex = ndi.gaussian_filter(zc30, 0.7)
cm30 = rasterize(CPOLY, trc30, zc30.shape)
zlo_c, zhi_c = np.percentile(zc30[cm30], 1), np.percentile(zc30[cm30], 99.7)
tex_c = shade_rgb(zc_tex, 30, 3.0, zlo_c - 150, zhi_c + 200, 40)
cxs30, cys30 = grid_xy(trc30, zc30.shape)

# 60 m mesh geometry
MESH_RES = 60
zc, trc, _ = load_dem(D / f"catch_{SITE}_dem_30m.tif", MESH_RES)
zc = ndi.gaussian_filter(zc, 0.8)
cxs, cys = grid_xy(trc, zc.shape)
c_valid = rasterize(CPOLY, trc, zc.shape, buf=3500)
cm = rasterize(CPOLY, trc, zc.shape)
cout = ndi.distance_transform_edt(~cm) * MESH_RES / 1000.0
cbnd = ndi.distance_transform_edt(cm) * MESH_RES / 1000.0
print("   mesh grid", zc.shape, "valid verts", c_valid.sum())
zc_s = sampler(zc / 1000, trc, zc.shape)
# basin-mesh hole where the catchment mesh takes over
hole = rasterize(CPOLY, trb, zb.shape, buf=2500)

# ---------- D8 on a 90 m grid, NHD streams burned in, gauge as the outlet
print("D8 routing grid")
zr, trr, _ = load_dem(D / f"catch_{SITE}_dem_30m.tif", 90)
rm = rasterize(CPOLY, trr, zr.shape)
cfl = gpd.read_file(D / f"catch_{SITE}_flowlines.gpkg").to_crs(CRS)
cfl = cfl[cfl.geometry.notna() & ~cfl.geometry.is_empty].reset_index(drop=True)
stream = features.rasterize([(g, 1) for g in cfl.geometry], out_shape=zr.shape, transform=trr,
                            fill=0, dtype="uint8", all_touched=True).astype(bool)
gr = int((GXY[1] - trr.f) / trr.e); gc = int((GXY[0] - trr.c) / trr.a)
# snap gauge to the largest-order stream cell nearby
big = features.rasterize([(g, 1) for g, o_ in zip(cfl.geometry, cfl.streamorde) if o_ >= cfl.streamorde.max()],
                         out_shape=zr.shape, transform=trr, fill=0, dtype="uint8", all_touched=True).astype(bool)
yy, xx = np.nonzero(big)
k = np.argmin((yy - gr) ** 2 + (xx - gc) ** 2)
print(f"   gauge snapped by {math.hypot(yy[k]-gr, xx[k]-gc)*90:.0f} m to main stem")
gr, gc = int(yy[k]), int(xx[k])
zburn = zr - np.where(stream, 40.0, 0.0) - 8.0 * np.clip(ndi.distance_transform_edt(~stream) * 90 / 1000.0, 0, 1) * 0
zburn = np.where(rm | stream, zburn, zburn)  # (kept full grid; outside cells are just not spawned on)

DY = np.array([-1, -1, 0, 1, 1, 1, 0, -1]); DX = np.array([0, 1, 1, 1, 0, -1, -1, -1])
DDIST = np.array([1, 1.4142, 1, 1.4142, 1, 1.4142, 1, 1.4142])


@njit(cache=True)
def priority_flood_d8(z, inside, orow, ocol):
    """Fill depressions from the outlet inward (only cells inside the catchment), D8 toward
    the outlet. Returns dirs (-1 = outlet/outside) and filled z."""
    h, w = z.shape
    dy = np.array([-1, -1, 0, 1, 1, 1, 0, -1]); dx = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dirs = -np.ones((h, w), np.int8)
    done = np.zeros((h, w), np.bool_)
    zf = z.copy()
    heap = [(z[orow, ocol], orow * w + ocol)]
    done[orow, ocol] = True
    while len(heap) > 0:
        zz, c = heapq.heappop(heap)
        r = c // w; cc = c % w
        for k in range(8):
            rn = r + dy[k]; cn = cc + dx[k]
            if rn < 0 or cn < 0 or rn >= h or cn >= w or done[rn, cn] or not inside[rn, cn]:
                continue
            done[rn, cn] = True
            if zf[rn, cn] < zz + 1e-3:
                zf[rn, cn] = zz + 1e-3
            dirs[rn, cn] = (k + 4) % 8          # neighbour drains back to (r, cc)
            heapq.heappush(heap, (zf[rn, cn], rn * w + cn))
    return dirs, zf


inside = ndi.binary_dilation(rm, iterations=2) | stream
dirs, _ = priority_flood_d8(zburn.astype(np.float64), inside, gr, gc)
# flow accumulation (cells) by sorting on distance-to-outlet
h, w = zr.shape
flat = np.arange(h * w)
dn = -np.ones(h * w, np.int64)
ok = dirs.ravel() >= 0
kk = dirs.ravel()[ok].astype(int)
dn[ok] = (flat[ok] // w + DY[kk]) * w + flat[ok] % w + DX[kk]
step = np.zeros(h * w); step[ok] = DDIST[kk] * 0.09        # km
# flow distance to the gauge (km): iterate pointer jumping
fd = np.zeros(h * w); nxt = dn.copy(); acc_d = step.copy()
fd = step.copy(); cur = dn.copy()
for _ in range(40):  # pointer doubling, 2^40 steps is plenty
    m = cur >= 0
    if not m.any():
        break
    fd[m] += acc_d[cur[m]] if False else 0
    break
# simple and exact: process cells in order of increasing distance via BFS from outlet
order = []
up = [[] for _ in range(0)]
from collections import deque
children = {}
src = np.flatnonzero(ok)
dst = dn[ok]
ordr = np.argsort(dst, kind="stable")
src_s, dst_s = src[ordr], dst[ordr]
starts = np.searchsorted(dst_s, np.arange(h * w))
ends = np.searchsorted(dst_s, np.arange(h * w), side="right")
fd = np.full(h * w, np.nan); fd[gr * w + gc] = 0.0
q = deque([gr * w + gc]); bfs = []
while q:
    c = q.popleft(); bfs.append(c)
    for s_ in src_s[starts[c]:ends[c]]:
        fd[s_] = fd[c] + step[s_]
        q.append(s_)
bfs = np.array(bfs)
acc = np.zeros(h * w)
acc[bfs] = 1
for c in bfs[::-1]:
    if dn[c] >= 0:
        acc[dn[c]] += acc[c]
print(f"   routed cells {bfs.size}, catchment cells {rm.sum()}, outlet acc {acc[gr*w+gc]:.0f}"
      f" ({acc[gr*w+gc]*0.0081:.0f} km2), max flow distance {np.nanmax(fd):.1f} km")
rxs, rys = grid_xy(trr, zr.shape)
np.savez_compressed(B / "routing.npz", dirs=dirs, acc=acc.reshape(h, w).astype(np.float32),
                    fd=np.nan_to_num(fd, nan=-1).reshape(h, w).astype(np.float32), mask=rm, stream=stream,
                    gauge_rc=np.array([gr, gc]), xs=km(np.c_[rxs, rxs])[:, 0], ys=km(np.c_[rys, rys])[:, 1],
                    z=(ndi.gaussian_filter(zr, 1.0) / 1000).astype(np.float32))
GKM = np.array([km(np.c_[rxs[gc], rys[gr]])[0, 0], km(np.c_[rxs[gc], rys[gr]])[0, 1]])

# ---------- catchment flowlines: topology + flow distance to gauge
print("catchment flowlines", len(cfl))
cfl["len_km"] = cfl.geometry.length / 1000
hs2i = {h_: i for i, h_ in enumerate(cfl.hydroseq)}
dn_i = np.array([hs2i.get(d_, -1) for d_ in cfl.dnhydroseq])
# outlet reach: the one the gauge sits on (closest to the snapped gauge point)
gpt = Point(GKM * 1000 + ORIGIN)
dist_g = cfl.geometry.distance(gpt)
out_i = int(dist_g.idxmin())
print(f"   gauge is {dist_g.min():.0f} m from reach {cfl.gnis_name[out_i]} (order {cfl.streamorde[out_i]})")
# distance from each reach's downstream end to the gauge
dist = np.full(len(cfl), np.nan); dist[out_i] = 0.0
changed = True
while changed:
    changed = False
    for i in range(len(cfl)):
        if np.isnan(dist[i]) and dn_i[i] >= 0 and not np.isnan(dist[dn_i[i]]):
            dist[i] = dist[dn_i[i]] + cfl.len_km[dn_i[i]]
            changed = True
miss = np.isnan(dist)
print(f"   reaches connected to gauge: {(~miss).sum()}/{len(cfl)}")
cfl = cfl[~miss].reset_index(drop=True)
dist = dist[~miss]
hs2i = {h_: i for i, h_ in enumerate(cfl.hydroseq)}
dn_i = np.array([hs2i.get(d_, -1) for d_ in cfl.dnhydroseq])
out_i = int(cfl.geometry.distance(gpt).idxmin())
dn_i[out_i] = -1
v, o = lines_to_arrays(cfl, zc_s)
# lines run upstream -> downstream in NHDPlus; per-vertex distance to gauge for the pulse
vd = np.zeros(len(v), np.float32)
for i in range(len(cfl)):
    seg = v[o[i]:o[i + 1], :2]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
    vd[o[i]:o[i + 1]] = dist[i] + (s[-1] - s)
np.savez_compressed(B / "catch_rivers.npz", v=v, off=o, order=cfl.streamorde.values.astype(np.int16),
                    area=cfl.totdasqkm.values.astype(np.float32), dn=dn_i, dist=dist.astype(np.float32),
                    len_km=cfl.len_km.values.astype(np.float32), vdist=vd,
                    name=cfl.gnis_name.fillna("").values.astype("U64"), out_i=out_i)

np.savez_compressed(B / "catch.npz", z=(zc / 1000).astype(np.float32), xs=km(np.c_[cxs, cxs])[:, 0],
                    ys=km(np.c_[cys, cys])[:, 1], valid=c_valid, mask=cm, out_km=cout.astype(np.float32),
                    in_km=cbnd.astype(np.float32), tex=tex_c, tex_xs=km(np.c_[cxs30, cxs30])[:, 0],
                    tex_ys=km(np.c_[cys30, cys30])[:, 1], basin_hole=hole, gauge=GKM.astype(np.float32),
                    gauge_z=np.float32(zc_s(GKM[:1], GKM[1:])[0]),
                    outline_b=ring_arrays(CPOLY, zb_s), outline_c=ring_arrays(CPOLY, zc_s, 300),
                    basin_outline=ring_arrays(BPOLY, zb_s, 3000))

# =========================================================== sanity plot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
states = gpd.read_file(f"zip://{D/'states.zip'}").to_crs(CRS)
inter = states[states.intersects(BPOLY.buffer(-2000))]
area = BPOLY.area / 1e6
print(f"basin area {area:,.0f} km2; states: {', '.join(sorted(inter.STUSPS))} ({len(inter)})")
with open(B / "facts.txt", "w") as f:
    f.write(f"basin_area_km2={area:.0f}\nn_states={len(inter)}\nstates={','.join(sorted(inter.STUSPS))}\n"
            f"catch_area_km2={CPOLY.area/1e6:.0f}\n")
fig, ax = plt.subplots(1, 2, figsize=(16, 7))
states.boundary.plot(ax=ax[0], color="0.6", lw=0.6)
basin.boundary.plot(ax=ax[0], color="k")
fl[fl.streamorde >= 5].plot(ax=ax[0], color="tab:blue", lw=0.5)
catch.plot(ax=ax[0], color="orange", alpha=0.6)
ax[0].set_xlim(bx0 - 1e5, bx1 + 1e5); ax[0].set_ylim(by0 - 1e5, by1 + 1e5); ax[0].set_title("Tennessee basin (HUC2 06)")
catch.boundary.plot(ax=ax[1], color="k")
cfl.plot(ax=ax[1], linewidth=0.3 + 0.4 * (cfl.streamorde - 1), color="tab:blue")
ax[1].plot(*gauge.geometry.iloc[0].coords[0], "o", color="orange", ms=10)
ax[1].set_title(f"Upstream basin of USGS {SITE} ({CPOLY.area/1e6:.0f} km2)")
plt.tight_layout(); plt.savefig(B / "sanity_map.png", dpi=70)
print("done")
