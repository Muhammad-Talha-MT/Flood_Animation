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
FPS = ARGS.fps
T_INTRO = 7.0                      # CONUS opening; the basin story (all other timings) starts after it
DUR_MAIN = 47.5                    # basin story 0-38.6 s, pull back out to CONUS, then the title moves to the centre
DUR = T_INTRO + DUR_MAIN
CATCH = "douglas"                  # zoom catchment: drainage above Douglas Dam

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
mr = np.load(HERE / "data" / f"rain_grid_{CATCH}_2019feb.npz")   # observed basin-mean rain, stylised pattern (hydro.py)
dm = np.load(B / "dams.npz")

T, H_END = hd["T"], float(hd["H_END"])
H_SHOW = float(hd["H_SHOW"])                      # panel x axis and observed line end here (t0 + 48 h)
DT = T[1] - T[0]
QOBS, RAIN, QN = hd["QOBS"], hd["RAIN"], hd["QN"].astype(np.float32)
IPK = int(hd["ipk"])
H_ISSUE = float(hd["H_ISSUE"])
FC_T, ML50, ML10, ML90, TVA = hd["FC_T"], hd["ML50"], hd["ML10"], hd["ML90"], hd["TVA"]
HORIZON = float(hd["HORIZON"])
P_MEDIAN, P_MODERATE = float(hd["P_MEDIAN"]), float(hd["P_MODERATE"])
Q_MEDIAN, Q_MODERATE = float(hd["Q_MEDIAN"]), float(hd["Q_MODERATE"])
T0 = pd.Timestamp(int(hd["t0_ns"]), tz="UTC")
TMAP = PchipInterpolator(hd["clock_t"], hd["clock_h"])
T_STORM0, T_STORM1 = float(hd["clock_t"][0]), float(hd["clock_t"][-1])
INFLOW = np.c_[cd["inflow_xy"], cd["inflow_z"]].astype(np.float64)    # USGS reservoir-inflow gauges
DAM = np.r_[cd["dam_xy_true"], cd["dam_z"]].astype(np.float64)
N_DAMS = int(dm["xy"].shape[0])


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
# CONUS map (prep_conus.py): flat textured plane just below the basin terrain, same km frame
cu = np.load(B / "conus.npz")
_ux = np.linspace(cu["xs"][0], cu["xs"][-1], 64)
_uy = np.linspace(cu["ys"][0], cu["ys"][-1], 48)
MU, TU = terrain_mesh(_ux, _uy, np.full((48, 64), -0.03), np.ones((48, 64), bool), cu["tex"], cu["xs"], cu["ys"])
A_US = PL.add_mesh(MU, texture=TU, ambient=1.0, diffuse=0.0, specular=0.0)


print(f"  basin {MB_FULL.n_points} pts, catchment {MC.n_points} pts", flush=True)
CAM = PL.camera
CAM.SetViewAngle(30.0)
CAM.SetViewUp(0, 0, 1)

# ================================================================== camera rig
B_FOCAL = np.array([20.0, -10.0, 0.0])
_cm = cd["mask"]
C_CENT = np.array([cd["xs"][np.nonzero(_cm)[1]].mean(), cd["ys"][np.nonzero(_cm)[0]].mean(), 0.0])
DAM_C = np.array([DAM[0], DAM[1], 0.0])
INFLOW_C = np.r_[INFLOW[:, :2].mean(0), 0.0]


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


W = 0.30                                        # window-centre shift while the hydrograph panel is on the right
HEAD = INFLOW_C + [-6, -4, 0]                   # toward the Newport gauge, between the headwaters and the lake
US_C = np.r_[cu["centre"], 0.0]
CAM_KEYS = [
    (0.0, dict(f=B_FOCAL + [-40, 10, 0], az=196.0, el=44.0, dist=1060.0, wcx=0.0, wcy=-0.04, ex=10.0)),
    (5.0, dict(f=B_FOCAL + [-10, 10, 0], az=208.0, el=46.0, dist=1000.0, wcx=0.0, wcy=-0.03, ex=10.0)),
    (7.2, dict(f=C_CENT + [-20, 0, 0], az=240.0, el=42.0, dist=660.0, wcx=0.0, wcy=-0.04, ex=8.0)),
    (9.6, dict(f=C_CENT + [-14, 0, 0], az=278.0, el=46.0, dist=340.0, wcx=0.0, wcy=0.0, ex=5.0)),
    (11.2, dict(f=C_CENT + [-14, -2, 0], az=284.0, el=40.0, dist=305.0, wcx=0.05, wcy=0.0, ex=4.0)),
    (20.0, dict(f=C_CENT + [-8, -4, 0], az=292.0, el=39.0, dist=290.0, wcx=W, wcy=0.0, ex=4.0)),
    (26.0, dict(f=HEAD, az=298.0, el=35.0, dist=225.0, wcx=W, wcy=0.0, ex=4.0)),
    (32.0, dict(f=C_CENT + [-10, 0, 0], az=290.0, el=40.0, dist=300.0, wcx=W, wcy=0.0, ex=4.0)),
    (35.6, dict(f=DAM_C + [14, -2, 0], az=298.0, el=30.0, dist=85.0, wcx=W, wcy=0.0, ex=4.0)),
    (38.0, dict(f=DAM_C + [7, -2, 0], az=308.0, el=27.0, dist=50.0, wcx=W, wcy=0.0, ex=4.0)),
    (38.6, dict(f=DAM_C + [7, -2, 0], az=311.0, el=27.0, dist=51.0, wcx=W, wcy=0.0, ex=4.0)),
    # pull back out: Douglas -> Tennessee basin -> contiguous US (title centred at the top)
    (41.0, dict(f=B_FOCAL + [-10, 10, 0], az=214.0, el=46.0, dist=1150.0, wcx=0.0, wcy=-0.03, ex=10.0)),
    (43.6, dict(f=US_C + [0, 200, 0], az=184.0, el=78.0, dist=6400.0, wcx=0.0, wcy=0.0, ex=10.0)),
    (47.5, dict(f=US_C + [0, 200, 0], az=182.0, el=80.0, dist=6800.0, wcx=0.0, wcy=0.0, ex=10.0)),
]
INTRO_KEYS = [
    (0.0, dict(f=US_C + [0, 60, 0], az=180.0, el=82.0, dist=5900.0, wcx=0.0, wcy=0.0, ex=10.0)),
    (2.8, dict(f=US_C + [60, 30, 0], az=184.0, el=76.0, dist=5500.0, wcx=0.0, wcy=0.0, ex=10.0)),
    (T_INTRO, CAM_KEYS[0][1]),
]


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
    for a in (A_BFULL, A_BHOLE, A_C, A_US):
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
MAXO = B_ORDER.max()


B_VPATH = br["vpath"].astype(np.float64)


def basin_river_layer(view, t, cov, col):
    # reveal: main stem first, then lower orders ("grows outward")
    u = 3.0 + (MAXO - 3.0) * (1 - ease(ramp(t, 0.4, 4.4)))
    vis = np.clip((B_ORDER - u) / 0.8 + 1, 0, 1)
    fade_c = ease(ramp(t, 9.5, 11.0)) * (1 - ease(ramp(t, 39.8, 40.6)))   # catchment network takes over, then hands back
    vis = vis * np.where(B_INC, 1 - fade_c, 1.0)
    dim = 1 - 0.55 * ease(ramp(t, 5.0, 7.5)) * (1 - ease(ramp(t, 40.0, 41.4)))
    vis_l = vis[B_LID]
    sx, sy, z01, ok = view.project(BV[:, 0], BV[:, 1], BV[:, 2])
    okv = view.visible(sx, sy, z01, ok, tol=0.01, slack=2) & (vis_l > 0.01)
    i0, i1 = polyline_segments(BV, BOFF, okv)
    if i0.size == 0:
        return
    lid = B_LID[i0]
    o = B_ORDER[lid]
    w = (0.35 + 0.42 * (o - 3)) * S * (1.0 if t < 9 or t > 40 else 0.8)
    q = np.clip((o - 3) / 5, 0, 1)
    # flow: brightness waves travelling downstream (toward smaller distance-to-mouth)
    ph = (B_VPATH[i0] + 55.0 * t) / 38.0
    wave = 0.5 + 0.5 * np.cos(2 * np.pi * ph)
    wave = wave ** 6
    lift = (0.25 + 0.75 * q) * wave * (1 - 0.6 * ease(ramp(t, 5.0, 7.5)))
    rgb = (C_RIVER[None] * (0.55 + 0.45 * q[:, None]) * (1 - lift[:, None]) + C_FLOOD[None] * lift[:, None]) * dim
    alpha = (0.45 + 0.55 * q) * vis[lid] * (1 + 0.4 * wave)
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
    okv = view.visible(sx, sy, z01, ok, tol=0.003, slack=1) & (vis[C_LID] > 0.01)
    i0, i1 = polyline_segments(CV, COFF, okv)
    if i0.size == 0:
        return None
    lid = C_LID[i0]
    base_w = 0.5 + 0.55 * (C_ORDER[lid] - 1)
    big = np.clip((C_ORDER[lid] - 1) / 4, 0, 1)                 # 0 for 1st order .. 1 for 5th
    w = (base_w * (1 + 0.9 * qm[lid] * (0.3 + 0.7 * big))) * S * view_zoom_scale(view)
    mix = (np.clip(qm[lid], 0, 1) * (0.15 + 0.35 * big))[:, None]
    rgb = C_RIVER[None] * (0.5 + 0.3 * big + 0.25 * qm[lid] * (1 - big))[:, None] * (1 - mix) + C_FLOOD[None] * mix
    alpha = np.clip(0.3 + 0.13 * C_ORDER[lid] + 0.3 * qm[lid] * (0.4 + 0.6 * big), 0, 1) * vis[lid]
    draw_segments(cov, col, sx[i0], sy[i0], sx[i1], sy[i1], w.astype(np.float32),
                  rgb[:, 0].astype(np.float32), rgb[:, 1].astype(np.float32), rgb[:, 2].astype(np.float32),
                  alpha.astype(np.float32))
    return q


def view_zoom_scale(view):
    # rivers a touch thicker when the camera is closer
    d = np.linalg.norm(view.campos - np.array(CAM.GetFocalPoint()))
    return float(np.clip(260.0 / d, 0.7, 1.6))


# ================================================================== flow streaks along the river network
class FlowStreaks:
    """Stateless streaks sliding downstream along polylines (NHDPlus vertex order is upstream ->
    downstream). Position at video time t is (s0 + speed * t) mod line length, so any frame can be
    rendered on its own (chunked rendering stays consistent)."""
    def __init__(self, V, OFF, LID, order, n, seed, min_order):
        seg = np.hypot(*np.diff(V[:, :2], axis=0).T)
        seg[OFF[1:-1] - 1] = 0.0                                 # no length across line boundaries
        arc = np.r_[0.0, np.cumsum(seg)]
        self.V, self.arc = V, arc
        self.start = arc[OFF[:-1]]
        self.length = arc[OFF[1:] - 1] - self.start
        w = np.where(order >= min_order, self.length * (order - min_order + 1.0) ** 1.6, 0.0)
        rng = np.random.default_rng(seed)
        self.line = rng.choice(order.size, size=n, p=w / w.sum())
        self.s0 = rng.random(n) * self.length[self.line]
        self.spd = rng.uniform(0.75, 1.25, n)
        self.order = order[self.line].astype(np.float32)
        self.off = OFF

    def at(self, s):
        """xyz at arc position s (absolute along the concatenated arc)."""
        k = np.clip(np.searchsorted(self.arc, s, side="right") - 1, 0, self.arc.size - 2)
        f = np.clip((s - self.arc[k]) / np.maximum(self.arc[k + 1] - self.arc[k], 1e-9), 0, 1)[:, None]
        return self.V[k] * (1 - f) + self.V[k + 1] * f

    def draw(self, view, buf, t, speed_kms, tail_km, inten, width, gain=None, zlift=0.0):
        L = self.length[self.line]
        ok = L > tail_km * 1.5
        s = np.mod(self.s0 + self.spd * speed_kms * t, np.maximum(L, 1e-6))
        s = np.clip(s, tail_km * self.spd + 1e-3, None)
        head = self.at(self.start[self.line] + s)
        tail = self.at(self.start[self.line] + s - tail_km * self.spd)
        hx, hy, hz, hok = view.project(head[:, 0], head[:, 1], head[:, 2] + zlift)
        tx, ty, _, tok = view.project(tail[:, 0], tail[:, 1], tail[:, 2] + zlift)
        vis = view.visible(hx, hy, hz, hok, tol=0.006, slack=2) & tok & ok
        edge = np.sin(np.pi * np.clip(s / np.maximum(L, 1e-6), 0, 1)) ** 0.5   # fade at line ends
        a = inten * edge * (1.0 if gain is None else gain)
        m = vis & (a > 0.01)
        n = int(m.sum())
        if n == 0:
            return
        add_segments(buf, tx[m], ty[m], hx[m], hy[m], (width * (0.7 + 0.12 * self.order[m])).astype(np.float32),
                     a[m].astype(np.float32), np.full(n, 170, np.float32), np.full(n, 235, np.float32),
                     np.full(n, 255, np.float32))


_tt = np.arange(0, DUR + 1, 1 / 60)
_spd = 3.0 * (1 + 2.2 * np.interp([float(TMAP(clamp(x, T_STORM0, T_STORM1))) for x in _tt], T, QOBS) / QOBS.max())
_cum = np.r_[0, np.cumsum(_spd[:-1]) / 60]
def STREAM_CLOCK(t): return float(np.interp(t, _tt, _cum))      # km of travel at video time t


B_STREAKS = FlowStreaks(BV, BOFF, B_LID, B_ORDER, 5200, 21, 4)
C_STREAKS = FlowStreaks(CV, COFF, C_LID, C_ORDER, 6500, 22, 2)


# ================================================================== hydropower dams (NID)
DAM_XY = dm["xy"].astype(np.float64)
DAM_ZB, DAM_ZC = dm["zb"].astype(np.float64), dm["zc"].astype(np.float64)
DAM_R = np.clip(np.log10(np.maximum(dm["storage"], 100)) - 1.5, 0.6, 5.5) / 5.5     # 0..1 marker size
DAM_FOCUS = dm["focus"]
C_DAM = (235, 52, 52)
_in_c = GC.sample(cd["valid"].astype(np.float32), DAM_XY[:, 0], DAM_XY[:, 1], cval=0.0) > 0.5
# dams inside the drainage above Douglas (the focus dam sits on its edge)
_in_focus = (GC.sample(cd["mask"].astype(np.float32), DAM_XY[:, 0], DAM_XY[:, 1], cval=0.0) > 0.3) | DAM_FOCUS


def dam_layer(img, view, t, mode_catch, alpha):
    """Red points at NID hydropower dams; the focus dam (Douglas) pulses once zoomed in."""
    if alpha <= 0:
        return
    z = np.where(mode_catch & _in_c, DAM_ZC, DAM_ZB) + (0.01 if mode_catch else 0.03)
    sx, sy, z01, ok = view.project(DAM_XY[:, 0], DAM_XY[:, 1], z)
    vis = view.visible(sx, sy, z01, ok, tol=0.02, slack=3)
    # staggered pop-in by storage (largest first)
    rank = np.argsort(np.argsort(-dm["storage"])) / max(N_DAMS - 1, 1)
    for i in np.flatnonzero(vis):
        a = alpha * ease(ramp(t, 1.4 + 2.2 * rank[i], 1.8 + 2.2 * rank[i]))
        if not _in_focus[i]:
            a *= 1 - ease(ramp(t, 5.4, 7.0)) * (1 - ease(ramp(t, 40.4, 41.4)))   # focus drainage only, until the pull-back
        if a <= 0:
            continue
        R = (3.2 + 4.2 * DAM_R[i]) * S * (1.25 if mode_catch else 1.0)
        if DAM_FOCUS[i] and mode_catch:
            R *= 1.5
            pul = (t * 0.9) % 1.0
            ring(img, sx[i], sy[i], R + (6 + 20 * pul) * S, 2.0 * S, (255, 90, 90), a * (1 - pul) * 0.9)
        disc(img, sx[i], sy[i], R + 1.6 * S, (20, 6, 6), a * 0.85)
        disc(img, sx[i], sy[i], R, C_DAM, a)
        disc(img, sx[i] - R * 0.25, sy[i] - R * 0.3, R * 0.35, (255, 170, 160), a * 0.6)


# ================================================================== runoff particles (D8 on 180 m grid)
RDIRS = rd["dirs"].ravel()
RACC = rd["acc"].ravel()
RW = rd["dirs"].shape[1]
RMASK = rd["mask"].ravel()
RSTREAM = (RACC > 120) | rd["stream"].ravel()
RGAUGE = int(rd["outlet_rc"][0]) * RW + int(rd["outlet_rc"][1])     # routing outlet (the dam)
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
        cs = rd["xs"][1] - rd["xs"][0]
        x = rd["xs"][0] + c * cs + self.j[:, 0] * jw * cs
        y = rd["ys"][0] + r * (rd["ys"][1] - rd["ys"][0]) + self.j[:, 1] * jw * cs
        z = GC.sample(ZC, x, y) + 0.006
        return np.c_[x, y, z]

    def step(self, th, active):
        rng = self.rng
        rate = np.interp(th, T, RAIN) / max(RAIN.max(), 1e-3)
        n_new = int(420 * rate * active)
        if n_new > 0 and self.c.size < 22000:
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
        add_segments(buf, px[m], py[m], sx[m], sy[m], np.full(m.sum(), 1.4 * S, np.float32),
                     inten[m].astype(np.float32),
                     np.full(m.sum(), 150, np.float32), np.full(m.sum(), 225, np.float32),
                     np.full(m.sum(), 255, np.float32))


# ================================================================== rain streaks + clouds
RNG_RAIN = np.random.default_rng(11)
NR = 40000
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
    m = R_KEEP < np.clip(0.08 + 0.8 * r, 0, 0.75) * amount * (r > 0.03)
    if not m.any():
        return
    HT = 12.0                                 # display-km column of falling rain
    h = ((R_H0[m] - t * 0.9 * R_SPD[m]) % 1.0) * HT
    zt = R_ZG[m] + h / view.ex
    L = 2.0
    sx0, sy0, z0, ok0 = view.project(R_X[m], R_Y[m], zt)
    sx1, sy1, z1, ok1 = view.project(R_X[m] + 0.28, R_Y[m] + 0.12, zt + L / view.ex)
    vis = view.visible(sx0, sy0, z0, ok0, tol=0.0, slack=0) & ok1
    n = vis.sum()
    add_segments(buf, sx1[vis], sy1[vis], sx0[vis], sy0[vis], np.full(n, 1.2 * S, np.float32),
                 np.full(n, 0.30, np.float32), np.full(n, 120, np.float32), np.full(n, 165, np.float32),
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
    u = (x - WIND[0] * (22.0 + t) * 2.2) / 42.0
    v = (y - WIND[1] * (22.0 + t) * 2.2) / 42.0
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
    "title": ("Inter-SemiBold.otf", 32), "sub": ("Inter-Medium.otf", 22),
    "maplbl": ("Inter-SemiBold.otf", 50), "maplbl2": ("Inter-Medium.otf", 26),
    "ptitle": ("Inter-SemiBold.otf", 27), "leg": ("Inter-Medium.otf", 20), "tick": ("Inter-Medium.otf", 19),
    "axis": ("Inter-Medium.otf", 19), "chip": ("Inter-Bold.otf", 21), "note": ("Inter-Medium.otf", 17)}.items()}

# ------------------------------------------------------------------ text: full-screen map, text block top-left
# All times in this section are VIDEO time (tv), including the CONUS opening.
TX = 48                            # left margin of the top-left text block (1080p units)
C_TVA = (255, 152, 72)
C_LAKE = (46, 118, 178)
FACTS_AREA = int(round(float(facts["basin_area_km2"]), -3))
CATCH_AREA = float(facts["catch_area_km2"])
TI = T_INTRO

CAPTIONS = [  # (start, end, title, subtitle)
    (TI + 11.6, TI + 18.0, "Storm rainfall generates runoff", "February 22–24, 2019"),
    (TI + 18.2, TI + 26.8, "Forecast issued before the peak", "HydroORBIT and TVA forecasts, next 48 hours"),
    (TI + 27.0, TI + 33.8, "Predicted vs. observed streamflow", "Observed flow at the Newport gauge"),
]
# once the pull-back has reached the contiguous US (story 43.6 s), the top-left title slides to the top centre
T_MOVE0, T_MOVE1 = TI + 43.9, TI + 45.2

LEGEND_ITEMS = [  # (key, label, in, out); rows stack and slide as items come and go
    ("river", "River network", TI + 11.4, TI + 38.2),
    ("dam", "Hydropower dam", TI + 11.4, TI + 38.2),
    ("lake", "Reservoir", TI + 11.4, TI + 38.2),
    ("focus", "Douglas Dam", TI + 11.4, TI + 38.2),
    ("gauge", "Newport gauge", TI + 11.4, TI + 38.2),
]
CAP_Y, LEG_Y = 104, 192

# soft shading behind the text: top-left corner (always) and a top band (end title)
_yy, _xx = np.mgrid[0:SH, 0:SW].astype(np.float32) / S
SHADE_TL = (0.62 * np.clip(1 - np.hypot(_xx / 900.0, _yy / 520.0), 0, 1) ** 1.2)[..., None].astype(np.float32)
SHADE_TOP = (0.55 * np.clip(1 - _yy / 330.0, 0, 1) ** 1.3)[..., None].astype(np.float32)
del _yy, _xx


def shadow_text(d, xy, txt, fnt, col, a, anchor="la"):
    d.text((xy[0], xy[1] + 2 * FS), txt, font=fnt, fill=(0, 0, 0, int(170 * a)), anchor=anchor)
    d.text(xy, txt, font=fnt, fill=col[:3] + (int(255 * a),), anchor=anchor)


def draw_caption(d, tv):
    for a, b, title, sub in CAPTIONS:
        ca = ease(ramp(tv, a, a + 0.4)) * (1 - ease(ramp(tv, b - 0.4, b)))
        if ca <= 0:
            continue
        x = (TX + 14 * (1 - ease(ramp(tv, a, a + 0.4)))) * FS
        shadow_text(d, (x, CAP_Y * FS), title, F["title"], (255, 255, 255), ca)
        if sub:
            shadow_text(d, (x, (CAP_Y + 42) * FS), sub, F["sub"], (214, 224, 238), ca)


def draw_legend(d, tv):
    u = FS
    acc = 0.0
    for key, label, t_in, t_out in LEGEND_ITEMS:
        a = ease(ramp(tv, t_in, t_in + 0.5)) * (1 - ease(ramp(tv, t_out - 0.4, t_out)))
        if a <= 0:
            continue
        x = TX * u
        y = (LEG_Y + 30 * acc) * u
        acc += a
        cx, cy = x + 12 * u, y + 13 * u
        A = int(255 * a)
        if key in ("dam", "focus"):
            r = 7 * u
            if key == "focus":
                d.ellipse([cx - 12 * u, cy - 12 * u, cx + 12 * u, cy + 12 * u], outline=(255, 90, 90, A), width=int(2 * u))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=C_DAM + (A,), outline=(20, 6, 6, A), width=int(1.5 * u))
        elif key == "lake":
            d.rounded_rectangle([cx - 12 * u, cy - 8 * u, cx + 12 * u, cy + 8 * u], radius=5 * u, fill=C_LAKE + (A,),
                                outline=(120, 190, 230, A), width=int(1.5 * u))
        elif key in ("river", "river_us"):
            d.line([cx - 13 * u, cy + 5 * u, cx - 3 * u, cy - 2 * u, cx + 13 * u, cy - 5 * u], fill=tuple(int(c) for c in C_RIVER) + (A,),
                   width=int(4 * u), joint="curve")
        elif key == "state":
            d.line([cx - 13 * u, cy, cx + 13 * u, cy], fill=(150, 165, 188, A), width=int(2 * u))
        elif key == "outline":
            d.rounded_rectangle([cx - 12 * u, cy - 9 * u, cx + 12 * u, cy + 9 * u], radius=4 * u, outline=C_AMBER + (A,), width=int(3 * u))
        elif key == "gauge":
            d.ellipse([cx - 10 * u, cy - 10 * u, cx + 10 * u, cy + 10 * u], outline=(255, 255, 255, A), width=int(2 * u))
            d.ellipse([cx - 5 * u, cy - 5 * u, cx + 5 * u, cy + 5 * u], fill=C_AMBER + (A,))
        shadow_text(d, (x + 34 * u, cy), label, F["leg"], (226, 232, 242), a, anchor="lm")


def draw_map_labels(d, labels):
    """Centred labels above a highlighted area: (text, x, y_baseline, font key, colour, alpha) in 1080p px."""
    for txt, x, y, fk, col, a in labels:
        if a > 0:
            shadow_text(d, (x * FS, y * FS), txt, F[fk], col, a, anchor="ms")


def draw_title(d, tv):
    """'HydroORBIT  Hydropower inflow forecasting': top-left for the whole video, larger than the scene text below
    it; at the end the same line slides to the top centre and grows."""
    m = ease(ramp(tv, T_MOVE0, T_MOVE1))
    sc = 1 + 0.4 * m
    f1 = font("Inter-SemiBold.otf", 46 * sc * FS)
    f2 = font("Inter-Regular.otf", 40 * sc * FS)
    t1, t2 = "HydroORBIT", "Hydropower inflow forecasting"
    w1, gap, w2 = d.textlength(t1, font=f1), 16 * sc * FS, d.textlength(t2, font=f2)
    x = TX * FS * (1 - m) + (SW * SS / 2 - (w1 + gap + w2) / 2) * m
    y = (78 * (1 - m) + 140 * m) * FS
    shadow_text(d, (x, y), t1, f1, (255, 255, 255), 0.97, anchor="ls")
    shadow_text(d, (x + w1 + gap, y), t2, f2, (206, 216, 232), 0.95, anchor="ls")
    return m


# ------------------------------------------------------------------ hydrograph panel (right side, only while shown)
PW, PH = 650, 676
PX, PY = 1920 - PW - 36, 110
QMAX = 1500.0
MINUS = "−"


def dashed(d, pts, fill, width, on, off):
    """Polyline with a dash pattern measured along its length."""
    acc, draw_on, seg = 0.0, True, [pts[0]]
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        L = math.hypot(xb - xa, yb - ya)
        pos = 0.0
        while pos < L:
            lim = (on if draw_on else off) - acc
            stepl = min(lim, L - pos)
            pos += stepl; acc += stepl
            pt = (xa + (xb - xa) * pos / max(L, 1e-9), ya + (yb - ya) * pos / max(L, 1e-9))
            if draw_on:
                seg.append(pt)
            if acc >= (on if draw_on else off) - 1e-9:
                if draw_on and len(seg) > 1:
                    d.line(seg, fill=fill, width=width, joint="curve")
                draw_on = not draw_on; acc = 0.0; seg = [pt]
    if draw_on and len(seg) > 1:
        d.line(seg, fill=fill, width=width, joint="curve")


def panel_layer(t, th, alpha):
    w, h = int(PW * FS), int(PH * FS)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    u = FS
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=20 * u, fill=(40, 60, 90, 46), outline=(255, 255, 255, 46), width=int(u))
    m = 22 * u
    d.text((m, 16 * u), "Inflow to Douglas Reservoir", font=F["ptitle"], fill=(244, 247, 252, 255))
    # legend: two rows
    ly = 62 * u
    def leg(x, y, kind, txt):
        if kind == "obs":
            d.line([x, y + 11 * u, x + 30 * u, y + 11 * u], fill=C_OBS + (255,), width=int(4 * u))
        elif kind == "ml":
            d.line([x, y + 11 * u, x + 30 * u, y + 11 * u], fill=C_TEAL + (255,), width=int(4 * u))
        elif kind == "band":
            d.rectangle([x, y + 3 * u, x + 30 * u, y + 19 * u], fill=C_TEAL + (70,))
        elif kind == "tva":
            dashed(d, [(x, y + 11 * u), (x + 30 * u, y + 11 * u)], C_TVA + (255,), int(4 * u), 9 * u, 5 * u)
        d.text((x + 38 * u, y + 11 * u), txt, font=F["leg"], fill=(222, 228, 238, 255), anchor="lm")
    leg(m, ly, "obs", "Observed")
    leg(m + 170 * u, ly, "ml", "HydroORBIT forecast")
    leg(m, ly + 30 * u, "tva", "TVA forecast")
    leg(m + 170 * u, ly + 30 * u, "band", "HydroORBIT 10–90% range")
    x0, x1 = 70 * u, w - 22 * u
    rs0, rs1 = 172 * u, 252 * u                      # rain strip (own axis, mm/h), tall enough to read the bursts
    y0, y1 = 302 * u, h - 100 * u

    def X(ha): return x0 + (x1 - x0) * ha / H_SHOW
    def Y(q): return y1 - (y1 - y0) * min(q, QMAX) / QMAX

    RMAX = 4.0
    d.line([x0, rs0, x1, rs0], fill=(255, 255, 255, 40), width=int(u))
    d.text((x0 - 10 * u, rs0 + 2 * u), "Rain", font=F["note"], fill=C_RAIN + (255,), anchor="ra")
    d.line([x0, rs1, x1, rs1], fill=(255, 255, 255, 18), width=int(u))
    d.text((x0 - 10 * u, rs1), f"{RMAX:.0f} mm/h", font=F["note"], fill=(150, 160, 178, 255), anchor="rd")
    for ha, rv_ in zip(hd["RAIN_HT"], hd["RAIN_H"]):
        if ha > th or ha >= H_SHOW:
            break
        if rv_ < 0.1:
            continue
        d.rectangle([X(ha) + 0.4 * u, rs0, X(ha + 1) - 0.4 * u, rs0 + (rs1 - rs0) * min(rv_, RMAX) / RMAX], fill=C_RAIN + (210,))
    d.text((x0 - 10 * u, y0 - 26 * u), "Streamflow (m³/s)", font=F["note"], fill=(170, 180, 198, 255), anchor="lm")
    for q in (0, 500, 1000, 1500):
        yy = Y(q)
        d.line([x0, yy, x1, yy], fill=(255, 255, 255, 26), width=int(u))
        d.text((x0 - 10 * u, yy), f"{q}", font=F["tick"], fill=(170, 180, 198, 255), anchor="rm")
    for hr in range(-24, 49, 12):
        xx = X(hr + H_ISSUE)
        d.line([xx, y1, xx, y1 + 6 * u], fill=(170, 180, 198, 200), width=int(2 * u))
        lab = "0" if hr == 0 else (f"+{hr}" if hr > 0 else f"{MINUS}{-hr}")
        d.text((xx, y1 + 10 * u), lab, font=F["tick"], fill=(200, 208, 222, 255), anchor="mt")
    t0l = T0 + pd.Timedelta(hours=H_ISSUE)
    d.text(((x0 + x1) / 2, y1 + 38 * u), f"Hours after forecast issue (t₀ = {t0l:%b} {t0l.day}, {t0l.year} {t0l:%H:%M} UTC)",
           font=F["note"], fill=(170, 180, 198, 255), anchor="mt")
    # NWS flood categories at this gauge (AVLN7)
    fa = ease(ramp(t, 12.4, 13.4))
    for qv, lab, c in ((Q_MEDIAN, "Median flood", (255, 214, 102)), (Q_MODERATE, "Moderate flood", (255, 130, 90))):
        if fa <= 0:
            break
        yy = Y(qv)
        dashed(d, [(x0, yy), (x1, yy)], c + (int(200 * fa),), int(2 * u), 7 * u, 6 * u)
        d.text((x0 + 8 * u, yy - 4 * u), lab, font=F["note"], fill=c + (int(255 * fa),), anchor="ld")   # empty history side

    ia = ease(ramp(t, 17.4, 18.4))                    # issue marker appears when the clock reaches t0
    fprog = ease(ramp(t, 18.6, 22.0))                 # forecasts draw in once, then never change
    if fprog > 0:
        sel = (FC_T - H_ISSUE) <= fprog * HORIZON + 1e-9
        if sel.sum() > 1:
            band = Image.new("RGBA", im.size, (0, 0, 0, 0))
            ImageDraw.Draw(band).polygon([(X(a_), Y(b_)) for a_, b_ in zip(FC_T[sel], ML90[sel])] +
                                         [(X(a_), Y(b_)) for a_, b_ in zip(FC_T[sel][::-1], ML10[sel][::-1])],
                                         fill=C_TEAL + (64,))
            im.alpha_composite(band)
            dashed(d, [(X(a_), Y(b_)) for a_, b_ in zip(FC_T[sel], TVA[sel])], C_TVA + (255,), int(4 * u), 12 * u, 7 * u)
            d.line([(X(a_), Y(b_)) for a_, b_ in zip(FC_T[sel], ML50[sel])], fill=C_TEAL + (255,), width=int(5 * u), joint="curve")
    # observed (only what has happened)
    kk = T <= min(th, H_SHOW)
    if kk.sum() > 2:
        d.line([(X(a_), Y(b_)) for a_, b_ in zip(T[kk][::2], QOBS[kk][::2])], fill=C_OBS + (255,), width=int(5 * u), joint="curve")
        lx, ly_ = X(T[kk][-1]), Y(QOBS[kk][-1])
        r_ = 6 * u
        d.ellipse([lx - r_, ly_ - r_, lx + r_, ly_ + r_], fill=C_OBS + (255,))
    if ia > 0:
        xi = X(H_ISSUE)
        d.line([xi, y0 - 4 * u, xi, y1], fill=C_AMBER + (int(255 * ia),), width=int(3 * u))
        lab = "Forecast issued (t₀)"
        tw = d.textlength(lab, font=F["chip"])
        cy_ = Y(0.93 * QMAX)                          # centred on the line, above the forecast band
        d.rounded_rectangle([xi - tw / 2 - 12 * u, cy_ - 17 * u, xi + tw / 2 + 12 * u, cy_ + 17 * u], radius=9 * u,
                            fill=C_AMBER + (int(255 * ia),))
        d.text((xi, cy_), lab, font=F["chip"], fill=(20, 24, 34, int(255 * ia)), anchor="mm")
    # chance of exceeding each flood category, from the HydroORBIT forecast range
    pa = ease(ramp(t, 22.0, 23.0))
    if pa > 0:
        # one row under the legend: "Chance of flooding  [Minor 64%]  [Moderate 1%]"
        yy = 136 * u
        d.text((m, yy), "Chance of flooding", font=F["leg"], fill=(200, 210, 226, int(255 * pa)), anchor="lm")
        xx = m + d.textlength("Chance of flooding", font=F["leg"]) + 14 * u
        for lab, pv, c in (("Median", P_MEDIAN, (255, 214, 102)), ("Moderate", P_MODERATE, (255, 130, 90))):
            txt = f"{lab}  {pv:.0%}"
            tw = d.textlength(txt, font=F["chip"])
            d.rounded_rectangle([xx, yy - 15 * u, xx + tw + 20 * u, yy + 15 * u], radius=8 * u,
                                fill=c + (int(46 * pa),), outline=c + (int(220 * pa),), width=int(1.5 * u))
            d.text((xx + 10 * u, yy), txt, font=F["chip"], fill=c + (int(255 * pa),), anchor="lm")
            xx += tw + 32 * u
    # observed peak, only once it has happened
    if th > T[IPK] + 0.3:
        oa = ease(ramp(th, T[IPK] + 0.3, T[IPK] + 2.5))
        px_, py_ = X(T[IPK]), Y(QOBS[IPK])
        r_ = 9 * u
        d.ellipse([px_ - r_, py_ - r_, px_ + r_, py_ + r_], outline=C_OBS + (int(255 * oa),), width=int(3 * u))
    im = im.resize((int(PW * S), int(PH * S)), Image.LANCZOS)
    if alpha < 1:
        a = np.asarray(im).copy()
        a[..., 3] = (a[..., 3] * alpha).astype(np.uint8)
        im = Image.fromarray(a)
    return im


def US_FADE(t):
    """CONUS plane opacity during the basin story: gone once the focus moves to the French Broad, back for the pull-back."""
    return max(1 - ease(ramp(t, 4.6, 7.0)), ease(ramp(t, 40.6, 42.4)))


def outline_glow(img, view, ring_v, a, width):
    """Amber glowing outline draped along a ring of 3D vertices."""
    sx, sy, z01, ok = view.project(ring_v[:, 0], ring_v[:, 1], ring_v[:, 2] + 0.02)
    okv = ok & view.visible(sx, sy, z01, ok, tol=0.02, slack=3)
    i0 = np.flatnonzero(okv[:-1] & okv[1:])
    ocov = np.zeros((SH, SW), np.float32); ocol = np.zeros((SH, SW, 3), np.float32)
    n = i0.size
    draw_segments(ocov, ocol, sx[i0], sy[i0], sx[i0 + 1], sy[i0 + 1], np.full(n, width * S, np.float32),
                  np.full(n, 255, np.float32), np.full(n, 200, np.float32), np.full(n, 92, np.float32),
                  np.full(n, a, np.float32))
    layer = ocol * ocov[..., None]
    img = img * (1 - ocov[..., None]) + layer
    return img + 1.2 * upsample(blur(layer[::4, ::4], 3), 4)


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

    def frame(self, tv):
        """Video time tv: CONUS opening first, then the basin story on its own clock (t = tv - T_INTRO)."""
        self.tv = tv
        if tv < T_INTRO:
            return self.frame_intro(tv)
        return self.frame_main(tv - T_INTRO)

    def frame_intro(self, tv):
        self.labels = []
        p = keyed(tv, INTRO_KEYS)
        # plain CONUS map first; the basin terrain and its outline appear together at the highlight
        img, view = self.render_3d(p, False, tv, us=1.0, basin=ease(ramp(tv, 2.8, 3.4)))
        ol = ease(ramp(tv, 2.8, 3.4))
        if ol > 0:
            img = outline_glow(img, view, cd["basin_outline"].astype(np.float64), ol * (0.8 + 0.2 * math.sin(tv * 4.0)), 2.6)
        img *= VIG
        img = np.clip(img, 0, 255).astype(np.uint8)
        return self.overlay(img, tv, 0.0, intro=True)

    def label_anchor(self, view, ring_v):
        """Screen-space (centre x, top y) of a 3D outline, in 1080p px; None if it is off screen."""
        sx, sy, z01, ok = view.project(ring_v[:, 0], ring_v[:, 1], ring_v[:, 2])
        m = ok & (sx > -200) & (sx < SW + 200) & (sy > -200) & (sy < SH + 200)
        if m.sum() < 10:
            return None
        return (float(np.clip((sx[m].min() + sx[m].max()) / 2 / S, 330, 1590)), float(sy[m].min() / S))

    def frame_main(self, t):
        self.labels = []
        p = camera_params(t)
        ex = float(p["ex"])
        th = sim_hours(t)
        # which terrain: crossfade basin -> catchment mesh during the swoop
        xf = ramp(t, 8.6, 9.6) * (1 - ramp(t, 39.7, 40.3))     # catchment mesh between the swoop and the pull-back
        if 0 < xf < 1:
            img_a, view_a = self.render_3d(p, False, t, us=US_FADE(t))
            img_b, view = self.render_3d(p, True, t)
            img = img_a * (1 - xf) + img_b * xf
        else:
            img, view = self.render_3d(p, xf >= 1, t, us=0.0 if xf >= 1 else US_FADE(t))
        mode_catch = xf >= 0.5

        # ---- screen-space terrain effects (half resolution fields)
        step = 4
        wx, wy, wz, zz = view.unproject(step)
        terr = zz < 1.0
        dimfac = np.ones_like(wx, dtype=np.float32)
        focus = ease(ramp(t, 5.0, 7.5)) * (1 - ease(ramp(t, 40.0, 41.4)))
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
        basin_river_layer(view, t, cov, col)
        flood_on = ramp(t, T_STORM0 - 0.5, T_STORM0 + 0.5)
        q = None
        if mode_catch:
            q = catch_river_layer(view, t, th, cov, col, flood_on)
        riv = col * cov[..., None]
        img = img * (1 - cov[..., None]) + riv
        # flowing streaks along the network: whole basin first, then the catchment (driven by routed flow)
        fl_add = np.zeros((SH, SW, 3), np.float32)
        sa = ease(ramp(t, 1.0, 3.0))
        if not mode_catch:
            B_STREAKS.draw(view, fl_add, t, 16.0, 7.0,
                           0.55 * sa * (1 - 0.45 * ease(ramp(t, 5.0, 7.5)) * (1 - ease(ramp(t, 40.0, 41.4)))), 1.5 * S)
        else:
            qr = reach_q(th) * flood_on if th > 0 else np.zeros(C_ORDER.size, np.float32)
            gain = (0.35 + 0.7 * qr[C_STREAKS.line]) * ease(ramp(t, 9.4, 11.0))
            C_STREAKS.draw(view, fl_add, STREAM_CLOCK(t), 1.0, 1.1, 0.6 * gain, 1.3 * S * view_zoom_scale(view),
                           zlift=0.006)
        img += fl_add
        img += 0.35 * upsample(blur(fl_add[::4, ::4], 2.0), 4)
        glow_gain = 0.55 + 0.45 * (np.interp(th, T, QOBS) / QOBS.max() if mode_catch and t > T_STORM0 else 0)
        glow_gain *= 0.55
        small = riv[::4, ::4]
        g = blur(small, 2.2) + 0.5 * blur(small, 7)
        img += glow_gain * 0.6 * upsample(g, 4)

        # ---- Tennessee basin outline (amber), carried over from the CONUS opening
        bo = 1 - ease(ramp(t, 4.4, 5.4))
        if bo > 0:
            img = outline_glow(img, view, cd["basin_outline"].astype(np.float64), bo * (0.8 + 0.2 * math.sin((t + T_INTRO) * 4.0)), 2.6)

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
        # glow where the rivers enter the reservoir, tied to the observed inflow
        gsx, gsy, gz, gok = view.project(DAM[:1], DAM[1:2], DAM[2:] + 0.01)
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
        br_ = np.clip(small - 215, 0, None)
        br_ = blur(br_, 5)
        img += 0.35 * upsample(br_, 4)
        img *= VIG

        # ---- hydropower dams (red points), then the three inflow gauges (amber)
        dam_layer(img, view, t, mode_catch, 1.0)
        ga = ease(ramp(t, 10.2, 11.4))
        if ga > 0 and mode_catch:
            isx, isy, iz, iok = view.project(INFLOW[:, 0], INFLOW[:, 1], INFLOW[:, 2] + 0.006)
            ivis = view.visible(isx, isy, iz, iok, tol=0.02, slack=3)
            for k_ in np.flatnonzero(ivis):
                x_, y_ = float(isx[k_]), float(isy[k_])
                pul = (t * 1.1 + 0.33 * k_) % 1.0
                ring(img, x_, y_, (14 + 26 * pul) * S, 2.6 * S, (255, 210, 110), ga * (1 - pul) * 0.95)
                disc(img, x_, y_, 13 * S, (10, 14, 24), ga * 0.7)
                ring(img, x_, y_, 12.5 * S, 2.4 * S, (255, 255, 255), ga)
                disc(img, x_, y_, 7.5 * S, (255, 200, 92), ga)

        # ---- names on the map, centred above the highlighted area
        la = ease(ramp(t, 0.3, 0.8)) * (1 - ease(ramp(t, 5.45, 5.85)))
        if la > 0:
            anc = self.label_anchor(view, cd["basin_outline"].astype(np.float64))
            if anc:
                y = max(anc[1] - 54, 190)
                self.labels.append(("Tennessee River Basin", anc[0], y, "maplbl", (255, 255, 255), la))
                ra = la * ease(ramp(t, 1.4, 1.9))
                self.labels.append(("Red dots: Hydropower dam", anc[0], y + 36, "maplbl2", (255, 170, 160), ra))
        ld = ease(ramp(t, 5.9, 6.4)) * (1 - ease(ramp(t, 10.9, 11.4)))
        if ld > 0:
            anc = self.label_anchor(view, cd["outline_c" if mode_catch else "outline_b"].astype(np.float64))
            if anc:
                self.labels.append(("Douglas Hydropower Reservoir", anc[0], max(anc[1] - 30, 190), "maplbl",
                                    (255, 255, 255), ld))
        img = np.clip(img, 0, 255).astype(np.uint8)
        return self.overlay(img, t, th)

    def render_3d(self, p, use_catch, t, us=0.0, basin=1.0):
        A_US.SetVisibility(us > 0.01)
        A_US.prop.opacity = us
        A_BFULL.prop.opacity = basin           # < 1 only in the CONUS opening, before the basin is highlighted
        A_BFULL.SetVisibility(not use_catch and basin > 0.01)
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

    def overlay(self, img, t, th, intro=False):
        """t: story time (panel); self.tv: video time (captions, legend, titles)."""
        tv = self.tv
        e_a = ease(ramp(tv, T_MOVE0, T_MOVE1))
        arr = img.astype(np.float32)
        shade = np.maximum(SHADE_TL, SHADE_TOP * e_a)
        arr = arr * (1 - shade) + BG[:, None, :] * shade
        pim = Image.fromarray(arr.clip(0, 255).astype(np.uint8)).convert("RGBA")
        ov = Image.new("RGBA", (SW * SS, SH * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        draw_title(d, tv)
        draw_map_labels(d, getattr(self, "labels", []))
        draw_caption(d, tv)
        draw_legend(d, tv)
        ov = ov.resize((SW, SH), Image.LANCZOS)
        pim.alpha_composite(ov)
        pa = 0.0 if intro else ease(ramp(t, 11.4, 12.4)) * (1 - ease(ramp(t, 38.2, 39.0)))
        if pa > 0:
            panel = panel_layer(t, th, pa)
            off = int(40 * S * (1 - pa))
            pim.alpha_composite(panel, (int(PX * S) + off, int(PY * S)))
        arr = np.asarray(pim.convert("RGB"), np.float32)
        fin = ease(ramp(tv, 0, 0.6)) * (1 - ease(ramp(tv, DUR - 0.6, DUR)))
        if fin < 1:
            arr = arr * fin + (BG[:, None, :] * VIG) * (1 - fin)
        return arr.astype(np.uint8)


# ================================================================== main
if __name__ == "__main__":
    if ARGS.debug_cam:
        for tt in (float(v) for v in ARGS.debug_cam.split(",")):
            p = camera_params(tt); set_camera(p); PL.render()
            v = View(float(p["ex"]))
            for name, pt in (("centre", C_CENT), ("dam", DAM)):
                z = GC.sample(ZC, [pt[0]], [pt[1]])
                sx, sy, _, _ = v.project(np.array([pt[0]]), np.array([pt[1]]), z)
                print(f"t={tt:5.1f} {name:6s} screen=({sx[0]:6.0f},{sy[0]:6.0f})  focal={np.round(CAM.GetFocalPoint(), 1)}"
                      f"  pos={np.round(CAM.GetPosition(), 1)}")
        sys.exit()
    R = Renderer()
    if ARGS.stills:
        (HERE / "private" / "stills").mkdir(parents=True, exist_ok=True)     # frames show Vinh's forecasts: kept out of git
        for tt in sorted(float(v) for v in ARGS.stills.split(",")):
            t0 = time.time()
            im = R.frame(tt)
            print(f"t={tt:5.1f}  th={sim_hours(tt):6.1f} h  particles={R.parts.c.size:5d}  {time.time()-t0:.2f}s", flush=True)
            Image.fromarray(im).save(HERE / "private" / "stills" / f"still_{tt:05.1f}.png")
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
