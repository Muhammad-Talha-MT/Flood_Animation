#!/usr/bin/env python3
"""HydroORBIT booth animation renderer (real terrain, 3D).

    python render.py --stills 2,6,10          # PNGs in stills/
    python render.py --out out/hydroorbit_booth.mp4 [--scale 2] [--frames 0:1200]

3D layer: PyVista/VTK off-screen (textured terrain meshes, perspective camera).
Everything that moves (rivers, particles, rain, clouds, gauge) is projected with the same
camera matrix and depth-tested against VTK's z-buffer, then drawn in screen space.
2D layer: Pillow (panel, captions, wordmark, end card), drawn at 2x and downsampled.
"""
import argparse, math, os, pathlib, subprocess, sys, time
import numpy as np
import pandas as pd
import pyvista as pv
import vtk
import cv2
from vtk.util import numpy_support as nps
from numba import njit
from scipy import ndimage as ndi
from scipy.interpolate import PchipInterpolator
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).parent
B = HERE / "build"
ap = argparse.ArgumentParser()
ap.add_argument("--stills", default="")
ap.add_argument("--out", default=str(HERE / "out" / "hydroorbit_booth.mp4"))
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--frames", default="")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--debug-cam", default="")
ARGS = ap.parse_args()

S = ARGS.scale
SW, SH = int(1920 * S), int(1080 * S)
FPS, DUR = ARGS.fps, 40.0
SITE = "03451500"

# palette
C_BG0, C_BG1 = np.array([7, 12, 24], np.float32), np.array([13, 22, 38], np.float32)
C_RIVER = np.array([79, 195, 247], np.float32)
C_FLOOD = np.array([189, 243, 255], np.float32)
C_TEAL, C_OBS, C_RAIN, C_AMBER = (64, 224, 208), (240, 244, 250), (96, 150, 255), (255, 200, 92)

# ================================================================== data
bd = np.load(B / "basin.npz")
cd = np.load(B / "catch.npz")
br = np.load(B / "basin_rivers.npz")
cr = np.load(B / "catch_rivers.npz")
rd = np.load(B / "routing.npz")
hd = np.load(B / "hydro.npz")
facts = dict(l.strip().split("=") for l in open(B / "facts.txt"))
mr = np.load(HERE / "data" / f"mrms_grid_{SITE}_2021mar.npz")
rv = np.load(B / "reservoir.npz")

T, H_END = hd["T"], float(hd["H_END"])
DT = T[1] - T[0]
QOBS, RAIN, QN = hd["QOBS"], hd["RAIN"], hd["QN"].astype(np.float32)
IPK = int(hd["ipk"])
H_ISSUE = float(hd["H_ISSUE"])
FC_T, FC_Q = hd["FC_T"], hd["FC_Q"]
T0 = pd.Timestamp(int(hd["t0_ns"]), tz="UTC")
RAIN_REAL = bool(hd["rain_real"])
TMAP = PchipInterpolator(hd["clock_t"], hd["clock_h"])
T_STORM0, T_STORM1 = float(hd["clock_t"][0]), float(hd["clock_t"][-1])
GAUGE = np.r_[cd["gauge"], cd["gauge_z"]].astype(np.float64)


def clamp(v, a=0.0, b=1.0): return max(a, min(b, v))
def ramp(t, a, b): return clamp((t - a) / (b - a))
def ease(u): u = clamp(u); return u * u * u * (u * (u * 6 - 15) + 10)
def lerp(a, b, u): return a + (b - a) * u
def sim_hours(t): return float(TMAP(clamp(t, T_STORM0, T_STORM1)))


# ================================================================== grids / samplers
class Grid:
    """Bilinear lookup of a raster given cell-centre coordinate vectors (km, ys descending)."""
    def __init__(self, xs, ys):
        self.x0, self.dx = xs[0], xs[1] - xs[0]
        self.y0, self.dy = ys[0], ys[1] - ys[0]

    def rc(self, x, y):
        return (y - self.y0) / self.dy, (x - self.x0) / self.dx

    def sample(self, A, x, y, order=1, cval=None):
        r, c = self.rc(np.asarray(x), np.asarray(y))
        mode = "nearest" if cval is None else "constant"
        return ndi.map_coordinates(A, [r, c], order=order, mode=mode, cval=0.0 if cval is None else cval)


GB = Grid(bd["xs"], bd["ys"])
GC = Grid(cd["xs"], cd["ys"])
GR = Grid(rd["xs"], rd["ys"])
ZB, ZC = bd["z"], cd["z"]
# distance (km) outside the catchment on the basin grid, for focus dimming
COUT_B = (ndi.distance_transform_edt(~bd["cmask"]) * GB.dx).astype(np.float32)
COUT_C = cd["out_km"]

# MRMS rain grid (lat/lon) -> world km via the catchment's own lon/lat <-> km fit
_mlat, _mlon = mr["lat"], mr["lon"]
_mt = mr["t_start_ns"]
MR_T = (pd.to_datetime(_mt, utc=True, unit="ns" if _mt.max() > 1e17 else "us") - T0).total_seconds().values / 3600
MR_RAIN = mr["rain"].astype(np.float32)
import pyproj
_tf = pyproj.Transformer.from_crs(4326, 5070, always_xy=True)
_ORIGIN = bd["origin"]
_lx0, _ly0 = _tf.transform(_mlon[0], _mlat[0])
_lx1, _ly1 = _tf.transform(_mlon[-1], _mlat[-1])
# small area: treat as affine in km (good to ~100 m over 1 deg)
_ll2km = lambda lon, lat: ((np.array(_tf.transform(lon, lat)).T - _ORIGIN) / 1000.0)
_cells = np.array(np.meshgrid(_mlon, _mlat))
_XY = _ll2km(_cells[0].ravel(), _cells[1].ravel())
MR_X = _XY[:, 0].reshape(_cells[0].shape)
MR_Y = _XY[:, 1].reshape(_cells[0].shape)
# least-squares affine km -> (row, col) for fast sampling
_A = np.c_[MR_X.ravel(), MR_Y.ravel(), np.ones(MR_X.size)]
_RC = np.linalg.lstsq(_A, np.c_[np.repeat(np.arange(_mlat.size), _mlon.size),
                                np.tile(np.arange(_mlon.size), _mlat.size)], rcond=None)[0]
MR_SM = ndi.gaussian_filter(MR_RAIN, (0, 2.5, 2.5))
RAIN_NORM = max(np.percentile(MR_SM, 99.7), 1.0)


def rain_field_at(th, x, y):
    """MRMS rain rate (mm/h) at hour th for world km arrays, time-interpolated, smoothed."""
    k = np.clip(np.interp(th - 0.5, MR_T, np.arange(MR_T.size)), 0, MR_T.size - 1.001)
    i = int(k); f = k - i
    rc = np.c_[np.ravel(x), np.ravel(y), np.ones(np.size(x))] @ _RC
    a = ndi.map_coordinates(MR_SM[i], rc.T, order=1, mode="nearest")
    b = ndi.map_coordinates(MR_SM[i + 1], rc.T, order=1, mode="nearest")
    return (a * (1 - f) + b * f).reshape(np.shape(x))


CUM = np.cumsum(MR_SM, 0)


def wet_at(th, x, y):
    k = int(np.clip(np.searchsorted(MR_T, th) - 1, 0, MR_T.size - 1))
    rc = np.c_[np.ravel(x), np.ravel(y), np.ones(np.size(x))] @ _RC
    c = ndi.map_coordinates(CUM[k], rc.T, order=1, mode="nearest").reshape(np.shape(x))
    dry = math.exp(-max(th - (T[IPK] + 4), 0) / 30.0)
    return np.clip(c / 260.0, 0, 1) * dry


# ================================================================== VTK scene
pv.global_theme.allow_empty_mesh = True
PL = pv.Plotter(off_screen=True, window_size=(SW, SH), lighting="none")
PL.set_background((0, 0, 0))
LIGHT = pv.Light(position=(-0.55, 0.65, 0.75), focal_point=(0, 0, 0), light_type="scene light",
                 intensity=1.05)
LIGHT.positional = False
PL.add_light(LIGHT)


def terrain_mesh(xs, ys, z, keep, tex, txs, tys):
    X, Y = np.meshgrid(xs, ys)
    g = pv.StructuredGrid(X.astype(np.float32), Y.astype(np.float32), z.astype(np.float32))
    dx, dy = txs[1] - txs[0], tys[1] - tys[0]
    u = (X - (txs[0] - dx / 2)) / (dx * len(txs))
    v = 1 - (Y - (tys[0] - dy / 2)) / (dy * len(tys))     # tys descending: v = 1 at the north edge
    g.active_texture_coordinates = np.c_[u.ravel(order="F"), v.ravel(order="F")].astype(np.float32)
    g.point_data["keep"] = keep.ravel(order="F").astype(np.uint8)
    m = g.threshold(0.5, scalars="keep").extract_surface().triangulate()
    m = m.compute_normals(cell_normals=False, split_vertices=False, auto_orient_normals=False)
    t = pv.Texture(np.ascontiguousarray(tex))         # pv.Texture flips numpy images itself
    t.interpolate = True
    t.mipmap = True
    return m, t


def add_terrain(m, t):
    a = PL.add_mesh(m, texture=t, ambient=0.38, diffuse=0.78, specular=0.0, smooth_shading=True)
    return a


print("building meshes...", flush=True)
_bkeep = bd["valid"] & (bd["out_km"] < 7.0)
_f = np.clip(bd["tex_out"] / 6.0, 0, 1)[..., None]
_f = _f * _f * (3 - 2 * _f)
TEX_B = (bd["tex"] * (1 - _f) + np.array([9, 15, 28]) * _f).astype(np.uint8)
MB_FULL, TB = terrain_mesh(bd["xs"], bd["ys"], ZB, _bkeep, TEX_B, bd["tex_xs"], bd["tex_ys"])
MB_HOLE, _ = terrain_mesh(bd["xs"], bd["ys"], ZB, _bkeep & ~cd["basin_hole"], TEX_B, bd["tex_xs"], bd["tex_ys"])
MC, TC = terrain_mesh(cd["xs"], cd["ys"], ZC, cd["valid"], cd["tex"], cd["tex_xs"], cd["tex_ys"])
A_BFULL, A_BHOLE, A_C = add_terrain(MB_FULL, TB), add_terrain(MB_HOLE, TB), add_terrain(MC, TC)


def build_reservoir():
    """Schematic water surface at the pool level and a simple concrete dam across the valley at the gauge."""
    pool, level, bed = rv["pool"], float(rv["level"]), float(rv["bed"])
    r_, c_ = np.nonzero(pool)
    pad = 6
    r0, r1 = max(r_.min() - pad, 0), r_.max() + pad + 1
    c0, c1 = max(c_.min() - pad, 0), c_.max() + pad + 1
    Z3 = 3                                         # 3x finer grid for a smooth shoreline
    sub = pool[r0:r1, c0:c1].astype(np.float32)
    sm = ndi.gaussian_filter(ndi.zoom(sub, Z3, order=1), 2.2)
    xs0, ys0 = cd["xs"][c0:c1], cd["ys"][r0:r1]
    xf = np.linspace(xs0[0] - (xs0[1] - xs0[0]) * 0.5, xs0[-1] + (xs0[1] - xs0[0]) * 0.5, sm.shape[1])
    yf = np.linspace(ys0[0] - (ys0[1] - ys0[0]) * 0.5, ys0[-1] + (ys0[1] - ys0[0]) * 0.5, sm.shape[0])
    X, Y = np.meshgrid(xf, yf)
    g = pv.StructuredGrid(X.astype(np.float32), Y.astype(np.float32), np.full(X.shape, level, np.float32))
    k = np.clip(ndi.distance_transform_edt(sm > 0.5) / 7.0, 0, 1)[..., None]
    col = (np.array([175, 225, 248]) * (1 - k) + np.array([28, 104, 164]) * k).astype(np.uint8)
    g.point_data["rgb"] = np.stack([col[..., i].ravel(order="F") for i in range(3)], 1)
    g.point_data["keep"] = (sm > 0.5).ravel(order="F").astype(np.uint8)
    water = g.threshold(0.5, scalars="keep", all_scalars=True).extract_surface(algorithm=None).triangulate()
    gx, gy = rv["gauge"]
    perp = rv["perp"]
    sL, sR = float(rv["sL"]) + 0.03, float(rv["sR"]) + 0.03
    half = (sL + sR) / 2
    ctr = np.array([gx, gy]) + (sL - sR) / 2 * perp
    top = level + 0.014
    box = pv.Box(bounds=(-half, half, -0.09, 0.09, bed - 0.008, top))
    box.rotate_z(math.degrees(math.atan2(perp[1], perp[0])), inplace=True)
    box.translate([ctr[0], ctr[1], 0], inplace=True)
    cx, cy = cd["xs"][c_].mean(), cd["ys"][r_].mean()
    return water, box, np.array([ctr[0], ctr[1], top]), np.array([cx, cy, level])


MW, MD, DAM_TOP, POOL_CTR = build_reservoir()
A_W = PL.add_mesh(MW, scalars="rgb", rgb=True, ambient=0.62, diffuse=0.45, specular=0.7, specular_power=20,
                  show_scalar_bar=False, smooth_shading=False)
A_D = PL.add_mesh(MD, color=(206, 208, 212), ambient=0.5, diffuse=0.75, specular=0.1, show_scalar_bar=False)
A_W.prop.opacity = 0.0
A_D.prop.opacity = 0.0
print(f"  basin {MB_FULL.n_points} pts, catchment {MC.n_points} pts", flush=True)
CAM = PL.camera
CAM.SetViewAngle(30.0)
CAM.SetViewUp(0, 0, 1)

# ================================================================== camera rig
B_FOCAL = np.array([20.0, -10.0, 0.0])
C_CENT = np.array([232.0, -16.0, 0.0])
VALLEY_A = np.array([228.0, -30.0, 0.0])     # upper main valley (south)
VALLEY_B = np.array([231.0, -4.0, 0.0])      # towards the outlet (north)
DAM_C = np.array([231.85, 7.2, 0.0])           # just upstream of the dam wall
DAM_F = np.array([232.6, 5.0, 0.0])           # between the dam and the middle of the reservoir


def keyed(t, keys):
    """Piecewise eased interpolation of parameter dicts between (time, dict) keys."""
    if t <= keys[0][0]:
        return dict(keys[0][1])
    for (ta, a), (tb, b) in zip(keys[:-1], keys[1:]):
        if t <= tb:
            u = ease((t - ta) / (tb - ta))
            out = {}
            for k in a:
                if k == "dist":
                    out[k] = a[k] ** (1 - u) * b[k] ** u
                else:
                    out[k] = lerp(np.asarray(a[k], float), np.asarray(b[k], float), u)
            return out
    return dict(keys[-1][1])


CAM_KEYS = [
    (0.0, dict(f=B_FOCAL, az=203.0, el=36.0, dist=860.0, wcx=0.0, wcy=-0.26, ex=10.0)),
    (5.0, dict(f=B_FOCAL + [20, 0, 0], az=219.0, el=38.0, dist=800.0, wcx=0.0, wcy=-0.22, ex=10.0)),
    (7.2, dict(f=C_CENT + [-10, 0, 0], az=232.0, el=40.0, dist=640.0, wcx=0.0, wcy=-0.02, ex=9.0)),
    (9.4, dict(f=C_CENT + [-4, 2, 0], az=290.0, el=56.0, dist=290.0, wcx=0.2, wcy=0.0, ex=6.0)),
    (11.2, dict(f=DAM_F, az=283.0, el=30.0, dist=40.0, wcx=0.30, wcy=0.0, ex=4.0)),
    (12.7, dict(f=DAM_C, az=196.0, el=24.0, dist=12.0, wcx=0.26, wcy=0.0, ex=4.0)),
    (13.6, dict(f=DAM_C, az=206.0, el=26.0, dist=13.0, wcx=0.26, wcy=0.0, ex=4.0)),
    (15.6, dict(f=C_CENT + [-2, 8, 0], az=318.0, el=38.0, dist=132.0, wcx=0.38, wcy=0.0, ex=4.0)),
    (20.0, dict(f=C_CENT + [-2, 8, 0], az=316.0, el=38.0, dist=126.0, wcx=0.38, wcy=0.0, ex=4.0)),
    (26.0, dict(f=VALLEY_B + [0, -4, 0], az=312.0, el=33.0, dist=100.0, wcx=0.38, wcy=0.0, ex=4.0)),
    (32.0, dict(f=C_CENT + [0, 4, 0], az=322.0, el=42.0, dist=140.0, wcx=0.38, wcy=0.0, ex=4.0)),
    (35.6, dict(f=DAM_F, az=283.0, el=28.0, dist=40.0, wcx=0.30, wcy=0.0, ex=4.0)),
    (38.0, dict(f=DAM_C, az=198.0, el=24.0, dist=12.5, wcx=0.26, wcy=0.0, ex=4.0)),
    (40.0, dict(f=DAM_C, az=208.0, el=26.0, dist=13.5, wcx=0.26, wcy=0.0, ex=4.0)),
]
# the 7 -> 12 s swoop: distance changes fastest in the middle (log-distance + ease)


def camera_params(t):
    p = keyed(t, CAM_KEYS)
    # focus point moves late in the swoop so the catchment stays framed while descending
    return p


def set_camera(p):
    f = np.asarray(p["f"], float).copy()
    ex = float(p["ex"])
    f[2] = (0.75 if p["dist"] < 300 else 0.35) * ex
    az, el = math.radians(float(p["az"])), math.radians(float(p["el"]))
    d = float(p["dist"])
    pos = f + d * np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
    CAM.SetFocalPoint(*f)
    CAM.SetPosition(*pos)
    CAM.SetViewUp(0, 0, 1)
    CAM.SetWindowCenter(float(p["wcx"]), float(p["wcy"]))
    for a in (A_BFULL, A_BHOLE, A_C, A_W, A_D):
        a.SetScale(1, 1, ex)
    PL.renderer.ResetCameraClippingRange()
    n, fa = CAM.GetClippingRange()
    CAM.SetClippingRange(max(n * 0.6, d * 0.02), fa * 1.2)


# ================================================================== projection / depth
class View:
    def __init__(self, ex):
        self.ex = ex
        m = CAM.GetCompositeProjectionTransformMatrix(SW / SH, -1, 1)
        self.M = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])
        self.Minv = np.linalg.inv(self.M)
        self.near, self.far = CAM.GetClippingRange()
        zb = vtk.vtkFloatArray()
        PL.ren_win.GetZbufferData(0, 0, SW - 1, SH - 1, zb)
        self.zbuf = np.flipud(nps.vtk_to_numpy(zb).reshape(SH, SW)).astype(np.float32)
        self.campos = np.array(CAM.GetPosition())

    def project(self, x, y, z):
        """world (km, true z) -> screen x, y, depth01, in-front mask."""
        P = np.stack([x, y, z * self.ex, np.ones_like(x)], 0)
        c = self.M @ P
        w = c[3]
        ok = w > 1e-6
        w = np.where(ok, w, 1)
        sx = (c[0] / w + 1) * 0.5 * SW
        sy = (1 - c[1] / w) * 0.5 * SH
        z01 = (c[2] / w + 1) * 0.5
        return sx, sy, z01, ok

    def lin(self, z01):
        n, f = self.near, self.far
        return 2 * n * f / (f + n - (2 * z01 - 1) * (f - n))

    def visible(self, sx, sy, z01, ok, tol=0.004, slack=1):
        ix = np.clip(sx.astype(np.int64), 0, SW - 1)
        iy = np.clip(sy.astype(np.int64), 0, SH - 1)
        zb = self.zbuf[iy, ix]
        if slack:
            for dy_, dx_ in ((-slack, 0), (slack, 0), (0, -slack), (0, slack)):
                zb = np.maximum(zb, self.zbuf[np.clip(iy + dy_, 0, SH - 1), np.clip(ix + dx_, 0, SW - 1)])
        inside = (sx >= -50) & (sx < SW + 50) & (sy >= -50) & (sy < SH + 50) & ok
        return inside & (self.lin(z01) <= self.lin(zb) * (1 + tol) + 0.02)

    def unproject(self, step):
        """world x, y, z(display) and linear depth for a subsampled pixel grid."""
        ys, xs = np.mgrid[step // 2:SH:step, step // 2:SW:step]
        z = self.zbuf[ys, xs]
        ndc = np.stack([(xs + 0.5) / SW * 2 - 1, 1 - (ys + 0.5) / SH * 2, z * 2 - 1, np.ones_like(z)], 0)
        P = np.tensordot(self.Minv, ndc, 1)
        P = P[:3] / P[3]
        return P[0], P[1], P[2], z

    def rays(self, step):
        ys, xs = np.mgrid[step // 2:SH:step, step // 2:SW:step]
        out = []
        for zz in (-1.0, 1.0):
            ndc = np.stack([(xs + 0.5) / SW * 2 - 1, 1 - (ys + 0.5) / SH * 2, np.full(xs.shape, zz), np.ones(xs.shape)], 0)
            P = np.tensordot(self.Minv, ndc, 1)
            out.append(P[:3] / P[3])
        return out


# ================================================================== raster helpers (numba)
@njit(cache=True)
def draw_segments(cov, col, x0, y0, x1, y1, wid, r, g, b, a):
    """Anti-aliased thick segments into a coverage layer (max coverage wins the colour)."""
    H, W = cov.shape
    for i in range(x0.size):
        hw = wid[i] * 0.5
        lo_x = int(max(min(x0[i], x1[i]) - hw - 2, 0)); hi_x = int(min(max(x0[i], x1[i]) + hw + 2, W - 1))
        lo_y = int(max(min(y0[i], y1[i]) - hw - 2, 0)); hi_y = int(min(max(y0[i], y1[i]) + hw + 2, H - 1))
        if hi_x < lo_x or hi_y < lo_y or hi_x - lo_x > 600 or hi_y - lo_y > 600:
            continue
        dx = x1[i] - x0[i]; dy = y1[i] - y0[i]
        L2 = dx * dx + dy * dy + 1e-9
        for py in range(lo_y, hi_y + 1):
            for px in range(lo_x, hi_x + 1):
                qx = px + 0.5 - x0[i]; qy = py + 0.5 - y0[i]
                u = (qx * dx + qy * dy) / L2
                if u < 0.0: u = 0.0
                if u > 1.0: u = 1.0
                ex_ = qx - u * dx; ey_ = qy - u * dy
                d = math.sqrt(ex_ * ex_ + ey_ * ey_)
                c = hw + 0.5 - d
                if hw < 0.5:
                    c = (1.0 - d) * hw * 2.0
                if c <= 0.0:
                    continue
                if c > 1.0: c = 1.0
                c *= a[i]
                if c > cov[py, px]:
                    cov[py, px] = c
                    col[py, px, 0] = r[i]; col[py, px, 1] = g[i]; col[py, px, 2] = b[i]


@njit(cache=True)
def add_segments(buf, x0, y0, x1, y1, wid, inten, cr_, cg_, cb_):
    """Additive anti-aliased segments (particles, rain)."""
    H, W = buf.shape[0], buf.shape[1]
    for i in range(x0.size):
        hw = wid[i] * 0.5
        lo_x = int(max(min(x0[i], x1[i]) - hw - 2, 0)); hi_x = int(min(max(x0[i], x1[i]) + hw + 2, W - 1))
        lo_y = int(max(min(y0[i], y1[i]) - hw - 2, 0)); hi_y = int(min(max(y0[i], y1[i]) + hw + 2, H - 1))
        if hi_x < lo_x or hi_y < lo_y or hi_x - lo_x > 300 or hi_y - lo_y > 300:
            continue
        dx = x1[i] - x0[i]; dy = y1[i] - y0[i]
        L2 = dx * dx + dy * dy + 1e-9
        for py in range(lo_y, hi_y + 1):
            for px in range(lo_x, hi_x + 1):
                qx = px + 0.5 - x0[i]; qy = py + 0.5 - y0[i]
                u = (qx * dx + qy * dy) / L2
                if u < 0.0: u = 0.0
                if u > 1.0: u = 1.0
                ex_ = qx - u * dx; ey_ = qy - u * dy
                d = math.sqrt(ex_ * ex_ + ey_ * ey_)
                c = hw + 0.5 - d
                if c <= 0.0:
                    continue
                if c > 1.0: c = 1.0
                c *= inten[i]
                buf[py, px, 0] += c * cr_[i]; buf[py, px, 1] += c * cg_[i]; buf[py, px, 2] += c * cb_[i]


def polyline_segments(v, off, ok_v):
    """Segment endpoints (vertex indices) for consecutive visible vertices of each line."""
    n = v.shape[0]
    a = np.arange(n - 1)
    same = np.ones(n - 1, bool)
    same[off[1:-1] - 1] = False          # no segment across line boundaries
    keep = same & ok_v[:-1] & ok_v[1:]
    return a[keep], a[keep] + 1


def line_id(off):
    lid = np.zeros(off[-1], np.int64)
    lid[off[1:-1]] = 1
    return np.cumsum(lid)


# ================================================================== rivers
BV, BOFF = br["v"].astype(np.float64), br["off"]
B_LID = line_id(BOFF)
B_ORDER = br["order"].astype(np.float32)
B_INC = br["in_catch"]
CV, COFF = cr["v"].astype(np.float64), cr["off"]
C_LID = line_id(COFF)
C_ORDER = cr["order"].astype(np.float32)
C_AREA = cr["area"].astype(np.float32)
C_OUT = int(cr["out_i"])
_pc = np.clip(np.round((CV[:, 0] - cd["xs"][0]) / (cd["xs"][1] - cd["xs"][0])).astype(int), 0, rv["pool"].shape[1] - 1)
_pr = np.clip(np.round((CV[:, 1] - cd["ys"][0]) / (cd["ys"][1] - cd["ys"][0])).astype(int), 0, rv["pool"].shape[0] - 1)
C_IN_POOL = ndi.binary_dilation(rv["pool"], iterations=1)[_pr, _pc]   # streams end at the shoreline
MAXO = B_ORDER.max()


def basin_river_layer(view, t, cov, col):
    # reveal: main stem first, then lower orders ("grows outward")
    u = 3.0 + (MAXO - 3.0) * (1 - ease(ramp(t, 0.4, 4.4)))
    vis = np.clip((B_ORDER - u) / 0.8 + 1, 0, 1)
    fade_c = ease(ramp(t, 9.5, 11.0))          # hand over to the detailed catchment network
    vis = vis * np.where(B_INC, 1 - fade_c, 1.0)
    dim = 1 - 0.55 * ease(ramp(t, 5.0, 7.5))
    vis_l = vis[B_LID]
    sx, sy, z01, ok = view.project(BV[:, 0], BV[:, 1], BV[:, 2])
    okv = view.visible(sx, sy, z01, ok, tol=0.01, slack=2) & (vis_l > 0.01)
    i0, i1 = polyline_segments(BV, BOFF, okv)
    if i0.size == 0:
        return
    lid = B_LID[i0]
    o = B_ORDER[lid]
    w = (0.35 + 0.42 * (o - 3)) * S * (1.0 if t < 9 else 0.8)
    q = np.clip((o - 3) / 5, 0, 1)
    rgb = C_RIVER[None] * (0.55 + 0.45 * q[:, None]) * dim
    alpha = (0.45 + 0.55 * q) * vis[lid]
    draw_segments(cov, col, sx[i0], sy[i0], sx[i1], sy[i1], w.astype(np.float32),
                  rgb[:, 0].astype(np.float32), rgb[:, 1].astype(np.float32), rgb[:, 2].astype(np.float32),
                  alpha.astype(np.float32))


def reach_q(th):
    """Normalised routed discharge per catchment reach (0..1) at event hour th."""
    k = np.clip(th / DT, 0, len(T) - 1.001)
    i = int(k); f = k - i
    return QN[:, i] * (1 - f) + QN[:, i + 1] * f


def catch_river_layer(view, t, th, cov, col, flood_on):
    reveal = ease(ramp(t, 9.0, 11.5))
    if reveal <= 0:
        return
    # small streams appear after the main stem
    lvl = 1 + (C_ORDER.max() - 1) * (1 - ease(ramp(t, 9.0, 11.8)))
    vis = np.clip((C_ORDER - lvl) / 0.7 + 1, 0, 1) * reveal
    q = reach_q(th) * flood_on if th > 0 else np.zeros_like(C_ORDER)
    # brightness: discharge, emphasised on big rivers; the flood front (rising) highlighted
    qm = q
    sx, sy, z01, ok = view.project(CV[:, 0], CV[:, 1], CV[:, 2] + 0.004)
    okv = view.visible(sx, sy, z01, ok, tol=0.003, slack=1) & (vis[C_LID] > 0.01) & ~C_IN_POOL
    i0, i1 = polyline_segments(CV, COFF, okv)
    if i0.size == 0:
        return None
    lid = C_LID[i0]
    base_w = 0.5 + 0.55 * (C_ORDER[lid] - 1)
    big = np.clip((C_ORDER[lid] - 1) / 4, 0, 1)                 # 0 for 1st order .. 1 for 5th
    w = (base_w * (1 + 1.2 * qm[lid] * (0.3 + 0.7 * big))) * S * view_zoom_scale(view)
    mix = (np.clip(qm[lid], 0, 1) * (0.25 + 0.6 * big))[:, None]
    rgb = C_RIVER[None] * (0.5 + 0.3 * big + 0.25 * qm[lid] * (1 - big))[:, None] * (1 - mix) + C_FLOOD[None] * mix
    alpha = np.clip(0.3 + 0.13 * C_ORDER[lid] + 0.3 * qm[lid] * (0.4 + 0.6 * big), 0, 1) * vis[lid]
    draw_segments(cov, col, sx[i0], sy[i0], sx[i1], sy[i1], w.astype(np.float32),
                  rgb[:, 0].astype(np.float32), rgb[:, 1].astype(np.float32), rgb[:, 2].astype(np.float32),
                  alpha.astype(np.float32))
    return q


def view_zoom_scale(view):
    # rivers a touch thicker when the camera is closer
    d = np.linalg.norm(view.campos - GAUGE * [1, 1, view.ex])
    return float(np.clip(110.0 / d, 0.7, 1.35))


# ================================================================== runoff particles (D8 on 90 m grid)
RDIRS = rd["dirs"].ravel()
RACC = rd["acc"].ravel()
RW = rd["dirs"].shape[1]
RMASK = rd["mask"].ravel()
RSTREAM = (RACC > 120) | rd["stream"].ravel()
RGAUGE = int(rd["gauge_rc"][0]) * RW + int(rd["gauge_rc"][1])
DYo = np.array([-1, -1, 0, 1, 1, 1, 0, -1]); DXo = np.array([0, 1, 1, 1, 0, -1, -1, -1])
DDo = np.array([1, 1.4142, 1, 1.4142, 1, 1.4142, 1, 1.4142])
CELLS = np.flatnonzero(RMASK & (RDIRS >= 0))
CELL_X = rd["xs"][CELLS % RW]
CELL_Y = rd["ys"][CELLS // RW]


CELL_RAIN = None


def cell_rain(th):
    """Rain (mm/h) at every catchment routing cell, cached per MRMS hour, time-interpolated."""
    global CELL_RAIN
    if CELL_RAIN is None:
        rc = (np.c_[CELL_X, CELL_Y, np.ones(CELL_X.size)] @ _RC).T
        CELL_RAIN = np.stack([ndi.map_coordinates(MR_SM[i], rc, order=1, mode="nearest")
                              for i in range(MR_T.size)]).astype(np.float32)
    k = np.clip(np.interp(th - 0.5, MR_T, np.arange(MR_T.size)), 0, MR_T.size - 1.001)
    i = int(k); f = k - i
    return CELL_RAIN[i] * (1 - f) + CELL_RAIN[i + 1] * f


class Particles:
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        self.c = np.zeros(0, np.int64); self.f = np.zeros(0); self.j = np.zeros((0, 2))
        self.age = np.zeros(0); self.prev = np.zeros((0, 3))

    def pos(self):
        k = np.maximum(RDIRS[self.c], 0)
        r = self.c // RW + DYo[k] * self.f
        c = self.c % RW + DXo[k] * self.f
        ch = RSTREAM[self.c]
        jw = np.where(ch, 0.12, 0.6)
        x = rd["xs"][0] + c * (rd["xs"][1] - rd["xs"][0]) + self.j[:, 0] * jw * 0.09
        y = rd["ys"][0] + r * (rd["ys"][1] - rd["ys"][0]) + self.j[:, 1] * jw * 0.09
        z = GC.sample(ZC, x, y) + 0.006
        return np.c_[x, y, z]

    def step(self, th, active):
        rng = self.rng
        rate = np.interp(th, T, RAIN) / max(RAIN.max(), 1e-3)
        n_new = int(260 * rate * active)
        if n_new > 0 and self.c.size < 14000:
            w = cell_rain(th) + 0.05
            c = rng.choice(CELLS, size=n_new, p=w / w.sum())
            self.c = np.r_[self.c, c]
            self.f = np.r_[self.f, rng.random(n_new)]
            self.j = np.r_[self.j, rng.uniform(-1, 1, (n_new, 2))]
            self.age = np.r_[self.age, np.zeros(n_new)]
            self.prev = np.r_[self.prev, np.full((n_new, 3), np.nan)]
        if self.c.size == 0:
            return
        self.prev = self.pos()
        self.age += 1
        ch = RSTREAM[self.c]
        spd = np.where(ch, 1.4 + 0.75 * np.log10(np.maximum(RACC[self.c], 1)), 0.45)
        self.f += spd / DDo[np.maximum(RDIRS[self.c], 0)]
        for _ in range(12):
            adv = (self.f >= 1) & (RDIRS[self.c] >= 0) & (self.c != RGAUGE)
            if not adv.any():
                break
            k = RDIRS[self.c[adv]]
            cc = self.c[adv]
            self.c[adv] = (cc // RW + DYo[k]) * RW + cc % RW + DXo[k]
            self.f[adv] -= 1
        alive = (self.c != RGAUGE) & (RDIRS[self.c] >= 0) & (self.age < 420)
        self.c, self.f, self.j, self.age, self.prev = (self.c[alive], self.f[alive], self.j[alive],
                                                        self.age[alive], self.prev[alive])

    def draw(self, view, buf, fade):
        if self.c.size == 0 or fade <= 0:
            return
        p = self.pos()
        pr = np.where(np.isnan(self.prev), p, self.prev)
        sx, sy, z01, ok = view.project(p[:, 0], p[:, 1], p[:, 2])
        px, py, _, ok2 = view.project(pr[:, 0], pr[:, 1], pr[:, 2])
        vis = view.visible(sx, sy, z01, ok, tol=0.004) & ok2
        ch = RSTREAM[self.c]
        inten = np.where(ch, 0.85, 0.5) * np.clip(self.age / 6, 0, 1) * fade
        m = vis
        add_segments(buf, px[m], py[m], sx[m], sy[m], np.full(m.sum(), 1.6 * S, np.float32),
                     inten[m].astype(np.float32),
                     np.full(m.sum(), 150, np.float32), np.full(m.sum(), 225, np.float32),
                     np.full(m.sum(), 255, np.float32))


# ================================================================== rain streaks + clouds
RNG_RAIN = np.random.default_rng(11)
NR = 11000
CX0, CX1 = cd["xs"][0] - 4, cd["xs"][-1] + 4
CY0, CY1 = cd["ys"][-1] - 4, cd["ys"][0] + 4
R_X = RNG_RAIN.uniform(CX0, CX1, NR)
R_Y = RNG_RAIN.uniform(CY0, CY1, NR)
R_H0 = RNG_RAIN.uniform(0, 1, NR)
R_SPD = RNG_RAIN.uniform(0.8, 1.2, NR)
R_KEEP = RNG_RAIN.random(NR)
R_ZG = GC.sample(ZC, R_X, R_Y)


def rain_layer(view, th, t, amount, buf):
    if amount <= 0.01:
        return
    r = rain_field_at(th, R_X, R_Y) / RAIN_NORM
    m = R_KEEP < np.clip(0.15 + 1.2 * r, 0, 1) * amount * (r > 0.03)
    if not m.any():
        return
    HT = 9.0                                  # display-km column of falling rain
    h = ((R_H0[m] - t * 0.9 * R_SPD[m]) % 1.0) * HT
    zt = R_ZG[m] + h / view.ex
    L = 0.85
    sx0, sy0, z0, ok0 = view.project(R_X[m], R_Y[m], zt)
    sx1, sy1, z1, ok1 = view.project(R_X[m] + 0.12, R_Y[m] + 0.05, zt + L / view.ex)
    vis = view.visible(sx0, sy0, z0, ok0, tol=0.0, slack=0) & ok1
    n = vis.sum()
    add_segments(buf, sx1[vis], sy1[vis], sx0[vis], sy0[vis], np.full(n, 1.2 * S, np.float32),
                 np.full(n, 0.42, np.float32), np.full(n, 120, np.float32), np.full(n, 165, np.float32),
                 np.full(n, 255, np.float32))


def _fbm(n=512, seed=3):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n), np.float32)
    for o, amp in ((8, 1.0), (16, 0.5), (32, 0.25), (64, 0.125)):
        g = rng.standard_normal((o, o)).astype(np.float32)
        up = ndi.zoom(np.tile(g, (2, 2)), n * 2 / (2 * o), order=3, mode="grid-wrap")[:n, :n]
        out += amp * up
    out = (out - out.mean()) / out.std()
    return out


NOISE = _fbm()
CLOUD_H = 2.8                                   # true km above datum, x exaggeration on screen
WIND = np.array([0.55, 0.9])                    # km per event-hour (display drift)


def cloud_density(x, y, th, t):
    u = (x - WIND[0] * (22.0 + t) * 0.9) / 18.0
    v = (y - WIND[1] * (22.0 + t) * 0.9) / 18.0
    nz = ndi.map_coordinates(NOISE, [(v * 64) % 512, (u * 64) % 512], order=1, mode="grid-wrap")
    r = rain_field_at(th, x, y) / RAIN_NORM
    d = np.clip((0.55 * nz + 1.6 * r - 0.35), 0, 1)
    return d * d * (3 - 2 * d)


# ================================================================== frame compositing
_gy = np.linspace(0, 1, SH, dtype=np.float32)[:, None]
BG = (C_BG0[None] * (1 - _gy) + C_BG1[None] * _gy).astype(np.float32)      # (SH, 3)
_vy, _vx = np.mgrid[0:SH, 0:SW].astype(np.float32)
VIG = np.clip(1 - 0.30 * (((_vx - SW / 2) / (SW / 2)) ** 2 + ((_vy - SH / 2) / (SH / 2)) ** 2) ** 1.5,
              0.5, 1)[..., None].astype(np.float32)
del _vy, _vx


def upsample(a, step=None):
    """Smooth (bilinear) upsample of a coarse per-pixel field to full res."""
    return cv2.resize(np.ascontiguousarray(a, np.float32), (SW, SH), interpolation=cv2.INTER_LINEAR)


def blur(a, sigma):
    return cv2.GaussianBlur(np.ascontiguousarray(a, np.float32), (0, 0), sigma)


def disc(img, cx, cy, R, col, a):
    x0, x1 = int(max(cx - R - 2, 0)), int(min(cx + R + 3, SW))
    y0, y1 = int(max(cy - R - 2, 0)), int(min(cy + R + 3, SH))
    if x1 <= x0 or y1 <= y0: return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    al = np.clip(R + 0.5 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2), 0, 1) * a
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - al[..., None]) + np.array(col)[None, None] * al[..., None]


def ring(img, cx, cy, R, wpx, col, a):
    if a <= 0: return
    x0, x1 = int(max(cx - R - 3, 0)), int(min(cx + R + 4, SW))
    y0, y1 = int(max(cy - R - 3, 0)), int(min(cy + R + 4, SH))
    if x1 <= x0 or y1 <= y0: return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    al = np.clip(1 - np.abs(rr - R) / wpx, 0, 1) * a
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - al[..., None]) + np.array(col)[None, None] * al[..., None]


def glow(img, layer, sigma, gain):
    k = 4
    small = layer[::k, ::k]
    g = ndi.gaussian_filter(small, (sigma / k, sigma / k, 0))
    img += gain * upsample(g, k)[..., :3] if g.ndim == 3 else 0


# ================================================================== 2D overlay
FONT_DIR = "/usr/share/fonts/opentype/inter/"


def font(name, size): return ImageFont.truetype(FONT_DIR + name, int(size))


SS = 2  # overlay supersampling
FS = S * SS
F = {k: font(n, s * FS) for k, (n, s) in {
    "brand": ("Inter-SemiBold.otf", 32), "brand2": ("Inter-Regular.otf", 24),
    "title": ("Inter-SemiBold.otf", 76), "title2": ("Inter-SemiBold.otf", 64), "title3": ("Inter-SemiBold.otf", 52),
    "sub": ("Inter-Medium.otf", 34), "foot": ("Inter-Medium.otf", 22),
    "dam": ("Inter-SemiBold.otf", 30), "lbl": ("Inter-SemiBold.otf", 26), "lbl2": ("Inter-Medium.otf", 21),
    "ptitle": ("Inter-SemiBold.otf", 30), "leg": ("Inter-Medium.otf", 22), "tick": ("Inter-Medium.otf", 22),
    "axis": ("Inter-Medium.otf", 21), "chip": ("Inter-Bold.otf", 25), "ann": ("Inter-Medium.otf", 21),
    "pk": ("Inter-SemiBold.otf", 24), "note": ("Inter-Medium.otf", 20)}.items()}

CAPTIONS = [
    (0.4, 5.0, "HydroORBIT: Hydropower inflow forecasting", "Tennessee River Basin | hydropower inflow testbed"),
    (5.6, 10.8, "Streams drain into the reservoir", "Schematic reservoir and dam at the outlet gauge"),
    (11.8, 17.6, "Storm rainfall generates runoff", "March 25, 2021 · French Broad River basin"),
    (18.4, 26.6, "Forecast issued before the peak", "HydroORBIT predicts the next 48 hours"),
    (27.4, 33.6, "Predicted vs. observed streamflow", "The forecast is not revised as data arrive"),
    (34.4, 39.3, "HydroORBIT: AI-Powered Hydropower Inflow Forecasting",
     "Predicting when and how much water arrives to support hydropower and water management"),
]
FOOTNOTE = "Conceptual illustration · synthetic forecast · schematic reservoir and dam"


def draw_caption(d, t):
    for a, b, title, sub in CAPTIONS:
        ca = ease(ramp(t, a, a + 0.4)) * (1 - ease(ramp(t, b - 0.4, b)))
        if ca <= 0:
            continue
        dy = 14 * FS * (1 - ease(ramp(t, a, a + 0.4)))
        cx = SW * SS / 2
        ty = (SH - 190 * S) * SS + dy
        fk = "title" if len(title) <= 26 else "title2" if len(title) <= 42 else "title3"
        for ox, oy, al in ((0, 3, 0.5), (0, 1.5, 0.4)):
            d.text((cx + ox * FS, ty + oy * FS), title, font=F[fk], fill=(0, 0, 0, int(255 * al * ca)), anchor="ms")
        d.text((cx, ty), title, font=F[fk], fill=(255, 255, 255, int(255 * ca)), anchor="ms")
        bw = (0.74 if len(sub) > 60 else 0.60) * SW * SS
        by0 = ty + 24 * FS
        d.rounded_rectangle([cx - bw / 2, by0, cx + bw / 2, by0 + 60 * FS], radius=30 * FS, fill=(6, 12, 24, int(165 * ca)))
        d.text((cx, by0 + 30 * FS), sub, font=F["sub"], fill=(230, 238, 248, int(255 * ca)), anchor="mm")
    # data-status footnote, always on from the first caption to the end
    fa = ease(ramp(t, 1.0, 1.8)) * (1 - ease(ramp(t, 39.4, 39.9)))
    d.text((SW * SS / 2, (SH - 40 * S) * SS), FOOTNOTE, font=F["foot"], fill=(255, 214, 140, int(235 * fa)), anchor="mm")


# ------------------------------------------------------------------ hydrograph panel
PW, PH = 700, 690
PX, PY = 1920 - PW - 36, 112
QMAX = 800.0
MINUS = "\u2212"


def panel_layer(t, th, alpha):
    w, h = int(PW * FS), int(PH * FS)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    u = FS
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=22 * u, fill=(9, 16, 30, 228), outline=(255, 255, 255, 46), width=int(u))
    m = 26 * u
    d.text((m, 20 * u), "Streamflow at Asheville gauge", font=F["ptitle"], fill=(244, 247, 252, 255))
    d.text((w - m, 31 * u), "USGS 03451500", font=F["note"], fill=(150, 162, 182, 255), anchor="rm")
    ly = 70 * u
    d.line([m, ly + 12 * u, m + 34 * u, ly + 12 * u], fill=C_OBS + (255,), width=int(5 * u))
    d.text((m + 44 * u, ly), "Observed", font=F["leg"], fill=(222, 228, 238, 255))
    lx2 = m + 190 * u
    d.line([lx2, ly + 12 * u, lx2 + 34 * u, ly + 12 * u], fill=C_TEAL + (255,), width=int(5 * u))
    d.text((lx2 + 44 * u, ly), "HydroORBIT (synthetic)", font=F["leg"], fill=(222, 228, 238, 255))
    x0, x1 = 100 * u, w - 26 * u
    rs0, rs1 = 112 * u, 158 * u                      # rain strip (own axis, mm/h)
    y0, y1 = 232 * u, h - 138 * u

    def X(ha): return x0 + (x1 - x0) * ha / H_END
    def Y(q): return y1 - (y1 - y0) * min(q, QMAX) / QMAX

    # rain strip, hanging from the top, with its own axis
    RMAX = 15.0
    d.text((x0 - 12 * u, rs0), "0", font=F["tick"], fill=(150, 160, 178, 255), anchor="rm")
    d.text((x0 - 12 * u, rs1), f"{RMAX:.0f}", font=F["tick"], fill=(150, 160, 178, 255), anchor="rm")
    d.line([x0, rs0, x1, rs0], fill=(255, 255, 255, 40), width=int(u))
    d.text((x1, rs1 + 4 * u), "Rain (mm/h), basin mean", font=F["note"], fill=C_RAIN + (255,), anchor="rt")
    for ha, rv_ in zip(hd["RAIN_HT"], hd["RAIN_H"]):
        if ha > th or ha >= H_END:
            break
        if rv_ < 0.1:
            continue
        d.rectangle([X(ha) + 0.4 * u, rs0, X(ha + 1) - 0.4 * u, rs0 + (rs1 - rs0) * min(rv_, RMAX) / RMAX], fill=C_RAIN + (210,))
    # discharge grid + ticks
    d.text((m, y0 - 26 * u), "Streamflow (m\u00b3/s)", font=F["axis"], fill=(170, 180, 198, 255), anchor="lm")
    for q in (0, 200, 400, 600):
        yy = Y(q)
        d.line([x0, yy, x1, yy], fill=(255, 255, 255, 30), width=int(u))
        d.text((x0 - 12 * u, yy), f"{q}", font=F["tick"], fill=(170, 180, 198, 255), anchor="rm")
    for hr in range(-24, 49, 12):
        xx = X(hr + H_ISSUE)
        d.line([xx, y1, xx, y1 + 7 * u], fill=(170, 180, 198, 200), width=int(2 * u))
        lab = "0" if hr == 0 else (f"+{hr}" if hr > 0 else f"{MINUS}{-hr}")
        d.text((xx, y1 + 12 * u), lab, font=F["tick"], fill=(200, 208, 222, 255), anchor="mt")
    d.text(((x0 + x1) / 2, y1 + 46 * u), "Hours after forecast issue (t\u2080)", font=F["axis"], fill=(170, 180, 198, 255), anchor="mt")
    t0l = T0 + pd.Timedelta(hours=H_ISSUE)
    d.text(((x0 + x1) / 2, y1 + 74 * u), f"t\u2080 = {t0l:%b} {t0l.day}, {t0l.year} {t0l:%H:%M} UTC", font=F["note"],
           fill=(150, 162, 182, 255), anchor="mt")

    ia = ease(ramp(t, 17.4, 18.4))                    # issue marker appears when the clock reaches t0
    fprog = ease(ramp(t, 18.6, 22.0))                 # forecast curve draws in once, then never changes
    # forecast (fixed since t0)
    if fprog > 0:
        sel = (FC_T - H_ISSUE) <= fprog * (H_END - H_ISSUE) + 1e-9
        pts = [(X(a), Y(b)) for a, b in zip(FC_T[sel], FC_Q[sel])]
        if len(pts) > 1:
            d.line(pts, fill=C_TEAL + (255,), width=int(5 * u), joint="curve")
    # observed (only what has happened)
    kk = T <= min(th, H_END)
    if kk.sum() > 2:
        d.line([(X(a), Y(b)) for a, b in zip(T[kk][::2], QOBS[kk][::2])], fill=C_OBS + (255,), width=int(5 * u), joint="curve")
        lx, ly_ = X(T[kk][-1]), Y(QOBS[kk][-1])
        r_ = 6 * u
        d.ellipse([lx - r_, ly_ - r_, lx + r_, ly_ + r_], fill=C_OBS + (255,))
    # issue-time marker
    if ia > 0:
        xi = X(H_ISSUE)
        d.line([xi, y0 - 6 * u, xi, y1], fill=C_AMBER + (int(255 * ia),), width=int(4 * u))
        lab = "Forecast issued (t\u2080)"
        tw = d.textlength(lab, font=F["chip"])
        d.rounded_rectangle([xi - tw / 2 - 14 * u, y0 + 4 * u, xi + tw / 2 + 14 * u, y0 + 44 * u], radius=10 * u,
                            fill=C_AMBER + (int(255 * ia),))
        d.text((xi, y0 + 24 * u), lab, font=F["chip"], fill=(20, 24, 34, int(255 * ia)), anchor="mm")
        d.text((xi - 12 * u, y0 + 66 * u), "Known at t\u2080", font=F["ann"], fill=(222, 228, 238, int(235 * ia)), anchor="rm")
        d.text((xi + 12 * u, y0 + 66 * u), "Forecast horizon: 48 h", font=F["ann"], fill=C_TEAL + (int(255 * ia),), anchor="lm")
    # predicted peak (from the fixed forecast)
    pa = ease(ramp(t, 22.0, 23.0))
    if pa > 0:
        ip = int(np.argmax(FC_Q))
        px_, py_ = X(FC_T[ip]), Y(FC_Q[ip])
        r_ = 8 * u
        d.ellipse([px_ - r_, py_ - r_, px_ + r_, py_ + r_], fill=C_TEAL + (int(255 * pa),), outline=(255, 255, 255, int(255 * pa)), width=int(2 * u))
        pk_y = Y(QOBS[IPK]) + 112 * u
        d.line([px_, py_ + 9 * u, px_, pk_y - 12 * u], fill=C_TEAL + (int(170 * pa),), width=int(2 * u))
        d.text((px_ + 16 * u, pk_y), "Predicted peak", font=F["pk"], fill=C_TEAL + (int(255 * pa),), anchor="lm")
        d.ellipse([px_ + 2 * u, pk_y - 5 * u, px_ + 12 * u, pk_y + 5 * u], fill=C_TEAL + (int(255 * pa),)) if False else None
    # observed peak, only once it has happened
    if th > T[IPK] + 0.3:
        oa = ease(ramp(th, T[IPK] + 0.3, T[IPK] + 2.5))
        px_, py_ = X(T[IPK]), Y(QOBS[IPK])
        r_ = 9 * u
        d.line([px_, py_, px_, y1], fill=C_OBS + (int(110 * oa),), width=int(2 * u))
        d.ellipse([px_ - r_, py_ - r_, px_ + r_, py_ + r_], outline=C_OBS + (int(255 * oa),), width=int(3 * u))
        d.text((px_ + 12 * u, py_ + 72 * u), "Observed peak", font=F["pk"], fill=C_OBS + (int(255 * oa),), anchor="lm")
    d.text((w / 2, h - 22 * u), "Conceptual illustration \u00b7 synthetic forecast", font=F["note"], fill=(255, 214, 140, 255), anchor="mm")
    im = im.resize((int(PW * S), int(PH * S)), Image.LANCZOS)
    if alpha < 1:
        a = np.asarray(im).copy()
        a[..., 3] = (a[..., 3] * alpha).astype(np.uint8)
        im = Image.fromarray(a)
    return im


# ================================================================== renderer
class Renderer:
    def __init__(self):
        self.parts = Particles()
        self.last_t = None

    def advance_particles(self, t):
        """Step the particle system once per video frame up to time t."""
        if self.last_t is None:
            self.last_t = T_STORM0 - 1 / FPS
        while self.last_t < t - 1e-6:
            self.last_t += 1 / FPS
            tt = self.last_t
            if tt >= T_STORM0:
                self.parts.step(sim_hours(tt), ramp(tt, T_STORM0, T_STORM0 + 0.8))

    def frame(self, t):
        p = camera_params(t)
        ex = float(p["ex"])
        th = sim_hours(t)
        # which terrain: crossfade basin -> catchment mesh during the swoop
        xf = ramp(t, 8.6, 9.6)
        if 0 < xf < 1:
            img_a, view_a = self.render_3d(p, False, t)
            img_b, view = self.render_3d(p, True, t)
            img = img_a * (1 - xf) + img_b * xf
        else:
            img, view = self.render_3d(p, xf >= 1, t)
        mode_catch = xf >= 0.5

        # ---- screen-space terrain effects (half resolution fields)
        step = 4
        wx, wy, wz, zz = view.unproject(step)
        terr = zz < 1.0
        dimfac = np.ones_like(wx, dtype=np.float32)
        focus = ease(ramp(t, 5.0, 7.5))
        if focus > 0:
            dout = GB.sample(COUT_B, wx, wy)
            if mode_catch:
                dout = np.minimum(dout, GC.sample(COUT_C, wx, wy, cval=99.0))
            k = np.clip(dout / (3.0 if mode_catch else 6.0), 0, 1)
            dimfac = 1 - focus * 0.74 * k
        desat = 1 - dimfac
        storm = ramp(t, T_STORM0, T_STORM0 + 1.0) * (1 - ramp(t, 34, 36))
        shade = np.ones_like(wx, dtype=np.float32)
        wet = np.zeros_like(wx, dtype=np.float32)
        cloud_scr = None
        if storm > 0 and mode_catch:
            sun = np.array([0.45, -0.35])          # shadow offset per km height
            cd_sh = cloud_density(wx + sun[0] * CLOUD_H * 4, wy + sun[1] * CLOUD_H * 4, th, t)
            shade = 1 - 0.42 * cd_sh * storm
            wet = wet_at(th, wx, wy).astype(np.float32) * storm
            # clouds: intersect view rays with a plane CLOUD_H above datum (display units)
            (n0, n1) = view.rays(step)
            zc_disp = (CLOUD_H + 1.6) * ex
            s_ = (zc_disp - n0[2]) / (n1[2] - n0[2] + 1e-9)
            hx = n0[0] + s_ * (n1[0] - n0[0]); hy = n0[1] + s_ * (n1[1] - n0[1])
            dens = cloud_density(hx, hy, th, t)
            # only where the cloud plane is in front of the terrain
            front = (wz > zc_disp) == False
            cloud_scr = (dens * storm * (s_ > 0)).astype(np.float32)
        lum_k = (dimfac * shade * (1 - 0.22 * wet)).astype(np.float32)
        lum_k = np.where(terr, lum_k, 1.0)
        desat = np.where(terr, desat, 0.0)
        wet = np.where(terr, wet, 0.0)
        lum_full = upsample(lum_k, step)[..., None]
        des_full = upsample(desat.astype(np.float32), step)[..., None]
        wet_full = upsample(wet, step)[..., None]
        gray = img.mean(-1, keepdims=True)
        img = img * (1 - 0.7 * des_full) + gray * 0.7 * des_full
        img = img * lum_full
        img = img + wet_full * np.array([4, 10, 26], np.float32) * 0.8
        # atmospheric depth fade toward the background
        zfull = view.zbuf
        terr_full = zfull < 1.0
        lin = view.lin(zfull)
        dmin = np.percentile(lin[terr_full], 2) if terr_full.any() else 1
        fog = np.clip((lin / max(dmin, 1e-3) - 1.0) * 0.35, 0, 0.55)[..., None] * terr_full[..., None]
        img = img * (1 - fog) + BG[:, None, :] * fog

        # ---- rivers
        cov = np.zeros((SH, SW), np.float32)
        col = np.zeros((SH, SW, 3), np.float32)
        if t < 11.5:
            basin_river_layer(view, t, cov, col)
        flood_on = ramp(t, T_STORM0 - 0.5, T_STORM0 + 0.5)
        q = None
        if mode_catch:
            q = catch_river_layer(view, t, th, cov, col, flood_on)
        riv = col * cov[..., None]
        img = img * (1 - cov[..., None]) + riv
        glow_gain = 0.55 + 0.45 * (np.interp(th, T, QOBS) / QOBS.max() if mode_catch and t > T_STORM0 else 0)
        small = riv[::4, ::4]
        g = blur(small, 2.2) + 0.5 * blur(small, 7)
        img += glow_gain * 0.6 * upsample(g, 4)

        # ---- catchment outline (amber) while selecting it
        ol = ease(ramp(t, 4.8, 5.8)) * (1 - ease(ramp(t, 10.5, 12.0)))
        if ol > 0:
            ob = cd["outline_b"].astype(np.float64)
            sx, sy, z01, ok = view.project(ob[:, 0], ob[:, 1], ob[:, 2] + 0.02)
            okv = ok & view.visible(sx, sy, z01, ok, tol=0.02, slack=3)
            i0 = np.flatnonzero(okv[:-1] & okv[1:])
            ocov = np.zeros((SH, SW), np.float32); ocol = np.zeros((SH, SW, 3), np.float32)
            pul = 0.8 + 0.2 * math.sin(t * 5.0)
            n = i0.size
            draw_segments(ocov, ocol, sx[i0], sy[i0], sx[i0 + 1], sy[i0 + 1], np.full(n, 2.4 * S, np.float32),
                          np.full(n, 255, np.float32), np.full(n, 200, np.float32), np.full(n, 92, np.float32),
                          np.full(n, ol * pul, np.float32))
            layer = ocol * ocov[..., None]
            img = img * (1 - ocov[..., None]) + layer
            sm = blur(layer[::4, ::4], 3)
            img += 1.2 * upsample(sm, 4)

        # ---- particles + rain (additive)
        add = np.zeros((SH, SW, 3), np.float32)
        if t >= T_STORM0 and mode_catch:
            self.advance_particles(t)
            self.parts.draw(view, add, 1 - ramp(t, 33.5, 35.5))
        if storm > 0 and mode_catch:
            rain_layer(view, th, t, storm, add)
        # outlet glow tied to observed discharge at the gauge
        gsx, gsy, gz, gok = view.project(GAUGE[:1], GAUGE[1:2], GAUGE[2:] + 0.01)
        gsx, gsy = float(gsx[0]), float(gsy[0])
        if mode_catch and t > T_STORM0:
            qn = (np.interp(th, T, QOBS) / QOBS.max()) ** 1.5
            R = 11 * S * (0.6 + 0.8 * qn)
            ys_, xs_ = np.ogrid[0:SH, 0:SW]
            x0_, x1_ = int(max(gsx - 4 * R, 0)), int(min(gsx + 4 * R, SW))
            y0_, y1_ = int(max(gsy - 4 * R, 0)), int(min(gsy + 4 * R, SH))
            if x1_ > x0_ and y1_ > y0_:
                yy, xx = np.mgrid[y0_:y1_, x0_:x1_]
                gl = np.exp(-((xx - gsx) ** 2 + (yy - gsy) ** 2) / (2 * R * R)) * qn * 0.55
                add[y0_:y1_, x0_:x1_] += gl[..., None] * C_RIVER[None, None]
        img += add
        if cloud_scr is not None:
            cfull = upsample(cloud_scr, step)[..., None]
            ccol = np.array([62, 70, 86], np.float32)
            img = img * (1 - 0.6 * cfull) + ccol * 0.6 * cfull

        # ---- bloom on highlights
        small = img[::4, ::4]
        br_ = np.clip(small - 195, 0, None)
        br_ = blur(br_, 5)
        img += 0.35 * upsample(br_, 4)
        img *= VIG

        # ---- gauge marker
        ga = ease(ramp(t, 10.2, 11.4))
        if ga > 0:
            pul = (t * 1.1) % 1.0
            ring(img, gsx, gsy, (8 + 14 * pul) * S, 1.6 * S, (255, 210, 110), ga * (1 - pul) * 0.8)
            ring(img, gsx, gsy, 7 * S, 1.4 * S, (255, 255, 255), ga)
            disc(img, gsx, gsy, 3.4 * S, (255, 200, 92), ga)

        img = np.clip(img, 0, 255).astype(np.uint8)
        marks = {}
        if mode_catch:
            for k_, pt_ in (("dam", DAM_TOP), ("pool", POOL_CTR)):
                px_, py_, _, _ = view.project(pt_[:1], pt_[1:2], pt_[2:])
                marks[k_] = (float(px_[0]), float(py_[0]))
        return self.overlay(img, t, th, ga, gsx, gsy, marks)

    def render_3d(self, p, use_catch, t):
        ro = ease(ramp(t, 9.2, 10.8)) if use_catch else 0.0
        A_W.prop.opacity = 0.94 * ro
        A_D.prop.opacity = ro
        A_W.SetVisibility(ro > 0)
        A_D.SetVisibility(ro > 0)
        A_BFULL.SetVisibility(not use_catch)
        A_BHOLE.SetVisibility(use_catch)
        A_C.SetVisibility(use_catch)
        set_camera(p)
        PL.ren_win.Render()
        pix = vtk.vtkUnsignedCharArray()
        PL.ren_win.GetPixelData(0, 0, SW - 1, SH - 1, 0, pix, 0)
        rgb = np.flipud(nps.vtk_to_numpy(pix).reshape(SH, SW, -1)[..., :3]).astype(np.float32)
        view = View(float(p["ex"]))
        bgm = (view.zbuf >= 1.0)[..., None]
        img = np.where(bgm, np.broadcast_to(BG[:, None, :], rgb.shape), rgb)
        return img, view

    def overlay(self, img, t, th, ga, gsx, gsy, marks):
        pim = Image.fromarray(img).convert("RGBA")
        ov = Image.new("RGBA", (SW * SS, SH * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)

        def shadow_text(xy, txt, font, fill, anchor, a):
            x0_, y0_, x1_, y1_ = d.textbbox(xy, txt, font=font, anchor=anchor)
            d.rounded_rectangle([x0_ - 9 * FS, y0_ - 5 * FS, x1_ + 9 * FS, y1_ + 7 * FS], radius=9 * FS, fill=(6, 12, 24, int(135 * a)))
            d.text((xy[0], xy[1] + 2 * FS), txt, font=font, fill=(0, 0, 0, int(190 * a)), anchor=anchor)
            d.text(xy, txt, font=font, fill=fill[:3] + (int(255 * a),), anchor=anchor)

        def on_screen(x, y): return 120 < x < SW - 120 and 70 < y < SH - 260
        if ga > 0 and marks:
            if on_screen(*marks["dam"]):
                dx_, dy_ = marks["dam"][0] * SS, marks["dam"][1] * SS
                shadow_text((dx_ - 22 * FS, dy_ - 6 * FS), "Dam", F["dam"], (255, 255, 255), "rm", ga)
            if on_screen(*marks["pool"]):
                px_, py_ = marks["pool"][0] * SS, marks["pool"][1] * SS
                shadow_text((px_ + 30 * FS, py_), "Reservoir", F["lbl"], (210, 236, 255), "lm", ga)
            if on_screen(gsx, gsy):
                shadow_text((gsx * SS, gsy * SS + 30 * FS), "USGS gauge", F["lbl2"], (255, 222, 160), "ma", ga)
        d.text((48 * FS, 38 * FS), "HydroORBIT", font=F["brand"], fill=(255, 255, 255, 240))
        d.text((48 * FS + d.textlength("HydroORBIT", font=F["brand"]) + 14 * FS, 47 * FS), "streamflow forecasting",
               font=F["brand2"], fill=(170, 182, 200, 225))
        draw_caption(d, t)
        ov = ov.resize((SW, SH), Image.LANCZOS)
        pim.alpha_composite(ov)
        pa = ease(ramp(t, 11.4, 12.4))
        if pa > 0:
            panel = panel_layer(t, th, pa)
            off = int(40 * S * (1 - pa))
            pim.alpha_composite(panel, (int(PX * S) + off, int(PY * S)))
        arr = np.asarray(pim.convert("RGB"), np.float32)
        fin = ease(ramp(t, 0, 0.6)) * (1 - ease(ramp(t, DUR - 0.6, DUR)))
        if fin < 1:
            arr = arr * fin + (BG[:, None, :] * VIG) * (1 - fin)
        return arr.astype(np.uint8)


# ================================================================== main
if __name__ == "__main__":
    if ARGS.debug_cam:
        for tt in (float(v) for v in ARGS.debug_cam.split(",")):
            p = camera_params(tt); set_camera(p); PL.render()
            v = View(float(p["ex"]))
            for name, pt in (("centre", C_CENT), ("gauge", GAUGE)):
                z = GC.sample(ZC, [pt[0]], [pt[1]])
                sx, sy, _, _ = v.project(np.array([pt[0]]), np.array([pt[1]]), z)
                print(f"t={tt:5.1f} {name:6s} screen=({sx[0]:6.0f},{sy[0]:6.0f})  focal={np.round(CAM.GetFocalPoint(), 1)}"
                      f"  pos={np.round(CAM.GetPosition(), 1)}")
        sys.exit()
    R = Renderer()
    if ARGS.stills:
        (HERE / "stills").mkdir(exist_ok=True)
        for tt in sorted(float(v) for v in ARGS.stills.split(",")):
            t0 = time.time()
            im = R.frame(tt)
            print(f"t={tt:5.1f}  th={sim_hours(tt):6.1f} h  particles={R.parts.c.size:5d}  {time.time()-t0:.2f}s", flush=True)
            Image.fromarray(im).save(HERE / "stills" / f"still_{tt:05.1f}.png")
        sys.exit()
    n = int(DUR * FPS)
    a, b = (0, n)
    if ARGS.frames:
        a, b = (int(v) for v in ARGS.frames.split(":"))
    import imageio_ffmpeg
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{SW}x{SH}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                           "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", ARGS.out],
                          stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(a, b):
        ff.stdin.write(R.frame(i / FPS).tobytes())
        if i % 30 == 0:
            el = time.time() - t0
            print(f"frame {i}/{b}  {el:.0f}s  ({el / max(i - a + 1, 1):.2f} s/frame)", flush=True)
    ff.stdin.close(); ff.wait()
    print("done", ARGS.out, f"{time.time() - t0:.0f}s")
