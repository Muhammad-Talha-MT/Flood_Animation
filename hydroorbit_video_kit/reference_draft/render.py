"""HydroORBIT booth animation renderer.

python3 render.py [--frames a:b] [--stills t1,t2,...] [--out file.mp4]
"""
import sys, math, subprocess, argparse
import numpy as np
from numba import njit, prange
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from storm import rain_at

SW, SH, FPS = 1920, 1080, 30
DUR = 40.0
EXZ = 0.15            # metres -> vertical cells (vertical exaggeration)
FONT = "/usr/share/fonts/opentype/inter/"

# ============================================================ data
td = np.load("terrain.npz")
hd = np.load("hydro.npz")
Z = td["z"].astype(np.float32)
ZG = ndi.gaussian_filter(Z, 1.6).astype(np.float32)   # geometry (smoother than shading)
DH, DW = Z.shape
ACC = td["acc"]
DIRS = td["dirs"]
BM = td["mask"]
CM = td["cmask"]
GY, GX = [int(v) for v in td["gauge"]]

T = hd["T"]; DT = T[1] - T[0]
RAIN = hd["rain_mean"]; QTRUE = hd["Qtrue"]; QOBS = hd["Qobs"]; LEAD = hd["lead_trace"]

# ============================================================ static DEM layers
def build_layers():
    zs = ndi.gaussian_filter(Z, 0.8) * EXZ
    gy_, gx_ = np.gradient(ndi.gaussian_filter(zs, 0.6))
    def hill(az, alt):
        az, alt = math.radians(az), math.radians(alt)
        lx, ly, lz = math.cos(alt) * math.sin(az), -math.cos(alt) * math.cos(az), math.sin(alt)
        nz = 1.0 / np.sqrt(1 + gx_**2 + gy_**2)
        return np.clip((-gx_ * lx - gy_ * ly + lz) * nz, 0, 1)
    hs = 0.62 * hill(315, 38) + 0.25 * hill(250, 55) + 0.13 * hill(20, 65)
    relief = zs - ndi.gaussian_filter(zs, 10)
    ao = np.clip(0.5 + relief / 12.0, 0, 1)
    zn = np.clip(Z / 330.0, 0, 1)
    stops = np.array([0, .12, .3, .5, .72, .9, 1.0])
    cols = np.array([(78, 112, 74), (104, 134, 82), (150, 152, 98), (160, 134, 92),
                     (140, 118, 98), (168, 160, 150), (215, 212, 205)], np.float32)
    rgb = np.stack([np.interp(zn, stops, cols[:, k]) for k in range(3)], -1)
    shade = (0.30 + 0.95 * hs) * (0.80 + 0.30 * ao)
    rgb = rgb * shade[..., None]
    rgb = np.clip(rgb, 0, 255).astype(np.float32)

    chmask = (ACC >= 120) & BM
    chd, (iy, ix) = ndi.distance_transform_edt(~chmask, return_indices=True)
    ch_flat = np.flatnonzero(chmask.ravel())
    lut = -np.ones(DH * DW, np.int32)
    lut[ch_flat] = np.arange(ch_flat.size)
    nn = lut[(iy * DW + ix).ravel()].reshape(DH, DW).astype(np.int32)

    bnd = CM ^ ndi.binary_erosion(CM)
    bd = ndi.distance_transform_edt(~bnd)
    dc = ndi.distance_transform_edt(~CM)
    L = np.stack([Z, rgb[..., 0], rgb[..., 1], rgb[..., 2], chd, bd, dc,
                  BM.astype(np.float32), CM.astype(np.float32), ZG], -1).astype(np.float32)
    return L, nn, ch_flat

print("building layers...", flush=True)
L, NN, CH = build_layers()
CH_ACC = ACC.ravel()[CH]
CH_W0 = (0.45 + 0.0042 * CH_ACC ** 0.5).astype(np.float32)          # half-width, cells
# map hydro channel series onto the basin channel list
hc = hd["ch_cells"]
pos = -np.ones(DH * DW, np.int64); pos[CH] = np.arange(CH.size)
HC_IDX = pos[hc]
QCH = hd["Qch_norm"].astype(np.float32)
QCH_PEAK = hd["Qch_peak"]
CH_INC = np.zeros(CH.size, bool); CH_INC[HC_IDX] = True
# flood swelling strength scales with how big the peak is
FLOOD_GAIN = np.clip(np.log10(QCH_PEAK + 1) / 2.6, 0.35, 1.0).astype(np.float32)

# coarse grid for rain / wetness fields
CS = 8
cyy, cxx = np.mgrid[0:DH // CS + 1, 0:DW // CS + 1].astype(np.float32) * CS
print("cumulative rain...", flush=True)
CT = np.arange(0, 96.01, 0.5)
_r = np.stack([rain_at(t, cyy, cxx) for t in CT])
CUM = (np.cumsum(_r, 0) * 0.5).astype(np.float32)
RMAX = max(_r.max(), 1e-3)

def rain_field(th):
    return (rain_at(th, cyy, cxx) / RMAX).astype(np.float32)

def wet_field(th):
    k = np.clip(th / 0.5, 0, len(CT) - 1.001)
    i = int(k); f = k - i
    c = CUM[i] * (1 - f) + CUM[i + 1] * f
    dry = math.exp(-max(th - 30.0, 0) / 22.0)
    return (np.clip(c / 55.0, 0, 1) * dry).astype(np.float32)

# background gradient + vignette
_gy = np.linspace(0, 1, SH)[:, None]
BG = (np.array([7, 12, 24])[None] * (1 - _gy) + np.array([14, 24, 40])[None] * _gy).astype(np.float32)
_vy, _vx = np.mgrid[0:SH, 0:SW]
VIG = (1 - 0.32 * (((_vx - SW / 2) / (SW / 2)) ** 2 + ((_vy - SH / 2) / (SH / 2)) ** 2) ** 1.4).astype(np.float32)
VIG = np.clip(VIG, 0.55, 1)[..., None]

# ============================================================ terrain kernel
@njit(inline="always")
def bil(L, y, x, k):
    y0 = int(y); x0 = int(x)
    fy = y - y0; fx = x - x0
    return ((L[y0, x0, k] * (1 - fx) + L[y0, x0 + 1, k] * fx) * (1 - fy)
            + (L[y0 + 1, x0, k] * (1 - fx) + L[y0 + 1, x0 + 1, k] * fx) * fy)

@njit(inline="always")
def bil2(A, y, x):
    y0 = int(y); x0 = int(x)
    if y0 < 0: y0 = 0
    if x0 < 0: x0 = 0
    if y0 >= A.shape[0] - 1: y0 = A.shape[0] - 2
    if x0 >= A.shape[1] - 1: x0 = A.shape[1] - 2
    fy = min(max(y - y0, 0.0), 1.0); fx = min(max(x - x0, 0.0), 1.0)
    return ((A[y0, x0] * (1 - fx) + A[y0, x0 + 1] * fx) * (1 - fy)
            + (A[y0 + 1, x0] * (1 - fx) + A[y0 + 1, x0 + 1] * fx) * fy)

@njit(parallel=True, fastmath=True, cache=True)
def render_terrain(out, bg, L, nn, ch_w, ch_vis, ch_q, rainc, wetc, P):
    cx, cy, s, ct, st, ex, ox, oy, ylo, yhi = P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9]
    nr = int(P[10]); dim_out = P[11]; focus = P[12]; outline = P[13]; csz = P[14]; rain_on = P[15]
    sh, sw = out.shape[0], out.shape[1]
    dh, dw = L.shape[0], L.shape[1]
    pxw = 1.0 / s
    for j in prange(sw):
        x = cx + (j + 0.5 - ox) / s
        ybuf = sh
        for rr_ in range(sh):
            out[rr_, j, 0] = bg[rr_, 0]; out[rr_, j, 1] = bg[rr_, 1]; out[rr_, j, 2] = bg[rr_, 2]
        if x >= 0.0 and x < dw - 1.001:
            prev = -1
            for i in range(nr):
                y = yhi - (yhi - ylo) * i / (nr - 1)
                if y < 0.0 or y >= dh - 1.001:
                    continue
                z = bil(L, y, x, 9)
                sy = oy + ((y - cy) * ct - z * ex * st) * s
                isy = int(math.ceil(sy))
                if isy < 0:
                    isy = 0
                pv = prev
                prev = isy
                bm = bil(L, y, x, 7)
                if bm < 0.02:
                    continue
                if focus > 0.0:
                    dcv = bil(L, y, x, 6)
                    f = 1.0 - dcv / 95.0
                    if f < 0.0:
                        f = 0.0
                    f = f * f * (3 - 2 * f)
                    bm = bm * (1.0 - focus + focus * f)
                    if bm < 0.02:
                        continue
                if pv < 0:
                    pv = isy + 1
                end = ybuf
                if pv < end:
                    end = pv
                if end < isy + 1:
                    end = isy + 1
                if end > ybuf:
                    end = ybuf
                if isy >= end:
                    continue
                # ---------------- colour
                r = bil(L, y, x, 1); g = bil(L, y, x, 2); b = bil(L, y, x, 3)
                cmv = bil(L, y, x, 8)
                if rain_on > 0.0:
                    w = bil2(wetc, y / csz, x / csz)
                    k = 1.0 - 0.20 * w
                    r = r * k * 0.92 + 8 * w; g = g * k * 0.96 + 10 * w; b = b * k + 22 * w
                    rr = bil2(rainc, y / csz, x / csz) * rain_on
                    k2 = 1.0 - 0.30 * rr
                    r = r * k2 + 18 * rr; g = g * k2 + 26 * rr; b = b * k2 + 44 * rr
                # rivers
                chd = bil(L, y, x, 4)
                yi = int(y + 0.5); xi = int(x + 0.5)
                cid = nn[yi, xi]
                if cid >= 0:
                    vis = ch_vis[cid]
                    if vis > 0.0:
                        q = ch_q[cid]
                        wd = ch_w[cid]
                        if wd < 0.75 * pxw:
                            wd = 0.75 * pxw
                        a = (wd - chd) / pxw + 0.5
                        if a < 0.0: a = 0.0
                        if a > 1.0: a = 1.0
                        a *= vis
                        rc = 52 + 120 * q; gc = 128 + 112 * q; bc = 190 + 62 * q
                        r = r * (1 - a) + rc * a; g = g * (1 - a) + gc * a; b = b * (1 - a) + bc * a
                        hw = 2.5 * pxw + wd * (0.6 + 2.2 * q)
                        hv = 1.0 - (chd - wd) / hw
                        if hv > 0.0 and chd > wd - pxw:
                            if hv > 1.0: hv = 1.0
                            hv = hv * hv * vis * (0.18 + 0.55 * q)
                            r += 40 * hv; g += 120 * hv; b += 170 * hv
                # outside-catchment dimming
                if dim_out > 0.0:
                    o = dim_out * (1.0 - cmv)
                    lum = 0.3 * r + 0.59 * g + 0.11 * b
                    r = (r * (1 - 0.6 * o) + lum * 0.6 * o) * (1 - 0.55 * o)
                    g = (g * (1 - 0.6 * o) + lum * 0.6 * o) * (1 - 0.55 * o)
                    b = (b * (1 - 0.6 * o) + lum * 0.6 * o) * (1 - 0.50 * o)
                # catchment outline
                if outline > 0.0:
                    bdv = bil(L, y, x, 5)
                    lw = (1.0 + 2.2 * outline) * pxw
                    r += 22 * cmv * outline * (1 - st); g += 20 * cmv * outline * (1 - st); b += 8 * cmv * outline * (1 - st)
                    ol = 1.0 - bdv / lw
                    if ol > 0.0:
                        ol = min(ol, 1.0) * outline
                        r = r * (1 - ol) + 255 * ol; g = g * (1 - ol) + 214 * ol; b = b * (1 - ol) + 120 * ol
                    gl = 1.0 - bdv / (7.0 * pxw)
                    if gl > 0.0:
                        gl = gl * gl * outline * 0.35
                        r += 90 * gl; g += 60 * gl; b += 10 * gl
                for rr_ in range(isy, end):
                    out[rr_, j, 0] = r * bm + bg[rr_, 0] * (1 - bm)
                    out[rr_, j, 1] = g * bm + bg[rr_, 1] * (1 - bm)
                    out[rr_, j, 2] = b * bm + bg[rr_, 2] * (1 - bm)
                if isy < ybuf:
                    ybuf = isy
                if ybuf <= 0:
                    break

# ============================================================ timeline helpers
def clamp(v, a=0.0, b=1.0): return max(a, min(b, v))
def ramp(t, a, b): return clamp((t - a) / (b - a))
def ease(u): u = clamp(u); return u * u * u * (u * (u * 6 - 15) + 10)
def lerp(a, b, u): return a + (b - a) * u

ys_, xs_ = np.nonzero(BM)
BASIN_C = ((ys_.min() + ys_.max()) / 2, (xs_.min() + xs_.max()) / 2)
S0 = SW * 0.86 / (xs_.max() - xs_.min())
ys_, xs_ = np.nonzero(CM)
CATCH_C = ((ys_.min() + ys_.max()) / 2, (xs_.min() + xs_.max()) / 2)
S1 = 1000.0 / (xs_.max() - xs_.min())

T_STORM0, T_STORM1, H_END = 12.5, 33.5, 72.0
from scipy.interpolate import PchipInterpolator
_TMAP = PchipInterpolator([T_STORM0, 14.0, 28.0, T_STORM1], [0.0, 6.0, 34.0, H_END])
def sim_hours(t):
    return float(_TMAP(clamp(t, T_STORM0, T_STORM1)))

def camera(t):
    """returns cy, cx, s, tilt(rad), ox, oy"""
    s0 = S0 * (1 + 0.035 * ramp(t, 0, 6))
    u = ease(ramp(t, 6.0, 11.2))
    s1 = S1
    s = s0 ** (1 - u) * s1 ** u
    k = (1 - s0 / s) / (1 - s0 / s1)
    cy = lerp(BASIN_C[0], CATCH_C[0], k)
    cx = lerp(BASIN_C[1], CATCH_C[1], k)
    v = ease(ramp(t, 8.8, 12.6))
    tilt = math.radians(lerp(0, 55, v))
    ox = lerp(SW / 2, 655, ease(ramp(t, 9.5, 12.8)))
    oy = lerp(SH / 2, 600, v)
    cy = cy + 10 * v
    # slow push during the storm, gentle pull back at the summary
    w = ease(ramp(t, 12.8, 33.5))
    s *= 1 + 0.07 * w
    tilt += math.radians(4) * w
    e = ease(ramp(t, 33.5, 39.5))
    s *= 1 - 0.06 * e
    return cy, cx, s, tilt, ox, oy

# ============================================================ particles (runoff)
DYo = np.array([-1, -1, 0, 1, 1, 1, 0, -1]); DXo = np.array([0, 1, 1, 1, 0, -1, -1, -1])
DDo = np.array([1, 1.4142, 1, 1.4142, 1, 1.4142, 1, 1.4142])
CCELLS = np.flatnonzero(CM.ravel())
CCY, CCX = CCELLS // DW, CCELLS % DW
DIRF = DIRS.ravel()
ACCF = ACC.ravel()
GAUGE_F = GY * DW + GX

class Particles:
    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)
        self.c = np.zeros(0, np.int64); self.f = np.zeros(0); self.j = np.zeros((0, 2)); self.age = np.zeros(0)
        self.trail = np.zeros((SH, SW), np.float32)

    def step(self, th, dth):
        rng = self.rng
        # spawn proportional to runoff generation
        rr = rain_at(th, CCY, CCX)
        tot = rr.sum()
        n_new = int(min(260, tot * 0.0042))
        if n_new > 0 and self.c.size < 9000:
            p = rr / tot
            c = rng.choice(CCELLS, size=n_new, p=p)
            self.c = np.concatenate([self.c, c])
            self.f = np.concatenate([self.f, rng.random(n_new)])
            self.j = np.concatenate([self.j, rng.uniform(-0.45, 0.45, (n_new, 2))])
            self.age = np.concatenate([self.age, np.zeros(n_new)])
        if self.c.size == 0:
            return
        self.age += 1
        for _ in range(3):
            ch = ACCF[self.c] >= 120
            speed = np.where(ch, 3.6, 0.55) / 3.0
            k = DIRF[self.c]
            dd = DDo[np.maximum(k, 0)]
            self.f += speed / dd
            adv = (self.f >= 1) & (k >= 0) & (self.c != GAUGE_F)
            while adv.any():
                kk = DIRF[self.c[adv]]
                cc = self.c[adv]
                self.c[adv] = (cc // DW + DYo[kk]) * DW + cc % DW + DXo[kk]
                self.f[adv] -= 1
                k = DIRF[self.c]
                adv = (self.f >= 1) & (k >= 0) & (self.c != GAUGE_F)
        alive = (self.c != GAUGE_F) & (self.age < 900)
        self.c, self.f, self.j, self.age = self.c[alive], self.f[alive], self.j[alive], self.age[alive]

    def world(self):
        k = np.maximum(DIRF[self.c], 0)
        y = self.c // DW + DYo[k] * self.f
        x = self.c % DW + DXo[k] * self.f
        ch = ACCF[self.c] >= 120
        wj = np.where(ch, np.minimum(0.25 + 0.002 * ACCF[self.c] ** 0.5, 1.2), 1.0)
        return y + self.j[:, 0] * wj, x + self.j[:, 1] * wj, ch

def project(y, x, z, cam):
    cy, cx, s, tilt, ox, oy = cam
    sx = ox + (x - cx) * s
    sy = oy + ((y - cy) * math.cos(tilt) - z * EXZ * math.sin(tilt)) * s
    return sx, sy

def zat(y, x):
    return ndi.map_coordinates(ZG, [np.atleast_1d(y), np.atleast_1d(x)], order=1, mode="nearest")

def splat(buf, sx, sy, w):
    ok = (sx >= 0) & (sx < SW - 1) & (sy >= 0) & (sy < SH - 1)
    sx, sy, w = sx[ok], sy[ok], w[ok]
    x0 = sx.astype(np.int64); y0 = sy.astype(np.int64)
    fx = sx - x0; fy = sy - y0
    np.add.at(buf, (y0, x0), w * (1 - fx) * (1 - fy))
    np.add.at(buf, (y0, x0 + 1), w * fx * (1 - fy))
    np.add.at(buf, (y0 + 1, x0), w * (1 - fx) * fy)
    np.add.at(buf, (y0 + 1, x0 + 1), w * fx * fy)

# ============================================================ rain streaks
def rain_streaks(th, cam, amount, rng):
    buf = np.zeros((SH, SW), np.float32)
    if amount <= 0.01:
        return buf
    cy, cx, s, tilt, ox, oy = cam
    n = 9000
    xs = rng.uniform(CATCH_C[1] - 330, CATCH_C[1] + 330, n)
    ys = rng.uniform(CATCH_C[0] - 200, CATCH_C[0] + 230, n)
    r = rain_at(th, ys, xs) / (RMAX * 0.42 + 1e-6)
    keep = rng.random(n) < np.clip(r, 0, 1) * 0.75 * amount
    xs, ys = xs[keep], ys[keep]
    if xs.size == 0:
        return buf
    h = rng.uniform(0, 70, xs.size)
    zg = zat(ys, xs)
    sx = ox + (xs - cx) * s
    sy = oy + ((ys - cy) * math.cos(tilt) - (zg * EXZ + h) * math.sin(tilt)) * s
    ln = rng.uniform(16, 30, xs.size)
    for q in np.linspace(0, 1, 12):
        splat(buf, sx - 0.28 * ln * q, sy + ln * q, np.full(xs.size, 0.10 + 0.1 * q, np.float32))
    return ndi.gaussian_filter(buf, 0.6)

# ============================================================ fonts / text
def font(name, size): return ImageFont.truetype(FONT + name, size)
F = {
    "brand": font("Inter-SemiBold.otf", 30),
    "cap": font("Inter-Medium.otf", 30),
    "end": font("Inter-SemiBold.otf", 96),
    "tag": font("Inter-Regular.otf", 36),
    "foot": font("Inter-Regular.otf", 20),
    "lbl": font("Inter-Medium.otf", 21),
}
SS = 2
PF = {k: font(n, s * SS) for k, (n, s) in {
    "title": ("Inter-SemiBold.otf", 24), "leg": ("Inter-Medium.otf", 17), "tick": ("Inter-Regular.otf", 15),
    "small": ("Inter-Medium.otf", 15), "now": ("Inter-SemiBold.otf", 17), "metric": ("Inter-SemiBold.otf", 26),
    "mlab": ("Inter-Regular.otf", 15), "tag": ("Inter-Regular.otf", 13)}.items()}

CAPTIONS = [
    (0.6, 5.6, "A large river basin"),
    (7.0, 11.4, "Into a headwater catchment"),
    (13.6, 19.4, "Storm rainfall generates runoff"),
    (19.6, 24.4, "Tributaries converge, streamflow rises"),
    (24.6, 33.2, "Peak passes, flow recedes"),
    (34.0, 37.4, "HydroORBIT vs. observed at the outlet"),
]

# ============================================================ hydrograph panel
PX, PY, PW, PH = 1238, 150, 646, 640
TEAL = (64, 224, 208); OBS = (240, 244, 250); RAINC = (96, 150, 255); AMBER = (255, 200, 92)
QMAX = 700.0

def forecast(t0, horizon=48.0):
    tt = np.arange(t0, min(t0 + horizon, T[-1]) + 1e-6, DT)
    lead = tt - t0
    w = 1 - np.exp(-lead / 9.0)
    tp = T[QTRUE.argmax()]
    stage = np.clip((t0 - 2) / (tp - 6), 0, 1)
    shift = 2.6 * (1 - stage) ** 1.5 + 0.5
    mag = -0.16 * (1 - stage) ** 1.2 + 0.035
    qs = np.interp(tt - shift * w, T, QTRUE)
    qf = qs * (1 + mag * w)
    q0 = np.interp(t0, T, QOBS)
    qf = qf + (q0 - qf[0]) * np.exp(-lead / 3.0)
    spread = (0.06 + 0.30 * (1 - stage) + 0.10) * w
    return tt, qf, qf * (1 - spread), qf * (1 + spread * 1.15)

def draw_panel(t, th, alpha):
    w, h = PW * SS, PH * SS
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=22 * SS, fill=(10, 18, 32, 214), outline=(255, 255, 255, 34), width=SS)
    m = 26 * SS
    d.text((m, 22 * SS), "Streamflow at the outlet", font=PF["title"], fill=(238, 242, 248, 255))
    # legend
    ly = 64 * SS
    d.line([m, ly + 10 * SS, m + 26 * SS, ly + 10 * SS], fill=OBS + (255,), width=3 * SS)
    d.text((m + 34 * SS, ly), "Observed", font=PF["leg"], fill=(205, 212, 224, 255))
    lx2 = m + 150 * SS
    d.rectangle([lx2, ly + 2 * SS, lx2 + 26 * SS, ly + 18 * SS], fill=TEAL + (60,))
    d.line([lx2, ly + 10 * SS, lx2 + 26 * SS, ly + 10 * SS], fill=TEAL + (255,), width=3 * SS)
    d.text((lx2 + 34 * SS, ly), "HydroORBIT forecast", font=PF["leg"], fill=(205, 212, 224, 255))
    # plot box
    x0, x1 = 78 * SS, w - 30 * SS
    y0, y1 = 128 * SS, h - 92 * SS
    def X(hh): return x0 + (x1 - x0) * hh / H_END
    def Y(q): return y1 - (y1 - y0) * min(q, QMAX * 1.05) / QMAX
    for q in (0, 200, 400, 600):
        yy = Y(q)
        d.line([x0, yy, x1, yy], fill=(255, 255, 255, 22), width=SS)
        d.text((x0 - 12 * SS, yy), f"{q}", font=PF["tick"], fill=(150, 160, 178, 255), anchor="rm")
    d.text((x0 - 12 * SS, y0 - 12 * SS), "m³/s", font=PF["tick"], fill=(150, 160, 178, 255), anchor="rm")
    for hh in range(0, 73, 12):
        d.text((X(hh), y1 + 12 * SS), f"{hh}", font=PF["tick"], fill=(150, 160, 178, 255), anchor="mt")
    d.text((x1, y1 + 34 * SS), "hours", font=PF["tick"], fill=(120, 130, 150, 255), anchor="rt")
    # rainfall bars hanging from top
    rscale = (y1 - y0) * 0.28 / max(RAIN.max(), 1e-3)
    hrs = np.arange(0, 72, 1.0)
    for hh in hrs:
        if hh > th:
            break
        rv = RAIN[(T >= hh) & (T < hh + 1)].mean()
        if rv < 0.05:
            continue
        d.rectangle([X(hh) + SS, y0, X(hh + 1) - SS, y0 + rv * rscale], fill=RAINC + (150,))
    if RAIN[T <= th].max(initial=0) > 0.3:
        d.text((x1 - 4 * SS, y0 + 6 * SS), "rainfall", font=PF["small"], fill=RAINC + (220,), anchor="ra")

    summary = ease(ramp(t, 33.6, 34.8))
    plume_a = 1 - summary
    # forecast plume ahead of 'now'
    if th < H_END and plume_a > 0.01:
        tt, qf, lo, hi = forecast(th)
        sel = tt <= H_END
        tt, qf, lo, hi = tt[sel], qf[sel], lo[sel], hi[sel]
        poly = [(X(a), Y(b)) for a, b in zip(tt[::3], hi[::3])] + [(X(a), Y(b)) for a, b in zip(tt[::-3], lo[::-3])]
        band = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(band).polygon(poly, fill=TEAL + (int(62 * plume_a),))
        im.alpha_composite(band)
        pts = [(X(a), Y(b)) for a, b in zip(tt[::2], qf[::2])]
        if len(pts) > 1:
            d.line(pts, fill=TEAL + (int(255 * plume_a),), width=3 * SS, joint="curve")
        # forecast peak marker
        ip = int(np.argmax(qf))
        if 3 < ip < len(qf) - 3 and qf[ip] > 60:
            px_, py_ = X(tt[ip]), Y(qf[ip])
            r_ = 6 * SS
            d.ellipse([px_ - r_, py_ - r_, px_ + r_, py_ + r_], fill=TEAL + (int(255 * plume_a),), outline=(255, 255, 255, int(220 * plume_a)), width=SS)
            d.text((px_ + 10 * SS, py_ - 12 * SS), "forecast peak", font=PF["small"], fill=TEAL + (int(240 * plume_a),), anchor="ls")
    # HydroORBIT past (6-h lead) - faint during the run, full at summary
    k = T <= min(th, H_END)
    if k.sum() > 2:
        la = 0.45 + 0.55 * summary
        pts = [(X(a), Y(b)) for a, b in zip(T[k][::2], LEAD[k][::2])]
        d.line(pts, fill=TEAL + (int(255 * la),), width=int((2 + summary) * SS), joint="curve")
        pts = [(X(a), Y(b)) for a, b in zip(T[k][::2], QOBS[k][::2])]
        d.line(pts, fill=OBS + (255,), width=3 * SS, joint="curve")
        # observed hourly dots
        for hh in np.arange(0, min(th, H_END) + 1e-6, 3.0):
            q = np.interp(hh, T, QOBS)
            r_ = 2.6 * SS
            d.ellipse([X(hh) - r_, Y(q) - r_, X(hh) + r_, Y(q) + r_], fill=OBS + (255,))
    # observed peak marker once passed
    ipk = int(np.argmax(QOBS[T <= H_END]))
    if th > T[ipk] + 1.5:
        a2 = int(255 * ramp(th, T[ipk] + 1.5, T[ipk] + 4))
        px_, py_ = X(T[ipk]), Y(QOBS[ipk])
        d.line([px_, py_, px_, y1], fill=AMBER + (int(a2 * 0.5),), width=SS)
        r_ = 7 * SS
        d.ellipse([px_ - r_, py_ - r_, px_ + r_, py_ + r_], outline=AMBER + (a2,), width=2 * SS)
        d.text((px_ - 12 * SS, py_ - 4 * SS), "observed peak", font=PF["small"], fill=AMBER + (a2,), anchor="rs")
    # now cursor
    if th <= H_END and summary < 1:
        ca = int(255 * (1 - summary))
        xn = X(th)
        d.line([xn, y0 - 6 * SS, xn, y1], fill=(255, 255, 255, int(ca * 0.55)), width=SS)
        lab = f"now  +{th:4.1f} h"
        d.rounded_rectangle([xn - 52 * SS, y1 + 40 * SS, xn + 52 * SS, y1 + 66 * SS], radius=8 * SS, fill=(255, 255, 255, int(ca * 0.12)))
        d.text((xn, y1 + 53 * SS), lab, font=PF["now"], fill=(255, 255, 255, ca), anchor="mm")
    # summary metrics
    if summary > 0:
        sa = int(255 * summary)
        ipl = int(np.argmax(LEAD[T <= H_END]))
        dq = (LEAD[ipl] - QOBS[ipk]) / QOBS[ipk] * 100
        dtp = T[ipl] - T[ipk]
        px_, py_ = X(T[ipl]), Y(LEAD[ipl])
        r_ = 6 * SS
        d.ellipse([px_ - r_, py_ - r_, px_ + r_, py_ + r_], fill=TEAL + (sa,), outline=(255, 255, 255, sa), width=SS)
        bx = x0 + (x1 - x0) * 0.52
        by = y0 + (y1 - y0) * 0.36
        d.text((bx, by), f"{dq:+.0f}%", font=PF["metric"], fill=(255, 255, 255, sa))
        d.text((bx, by + 34 * SS), "peak magnitude", font=PF["mlab"], fill=(170, 180, 198, sa))
        d.text((bx + 140 * SS, by), f"{dtp:+.1f} h", font=PF["metric"], fill=(255, 255, 255, sa))
        d.text((bx + 140 * SS, by + 34 * SS), "peak timing", font=PF["mlab"], fill=(170, 180, 198, sa))
        d.text((bx, by + 58 * SS), "6-h lead forecast vs. observed", font=PF["tag"], fill=(130, 140, 160, sa))
    d.text((w - 22 * SS, h - 16 * SS), "Illustrative data", font=PF["tag"], fill=(120, 130, 150, 200), anchor="rb")
    im = im.resize((PW, PH), Image.LANCZOS)
    if alpha < 1:
        a = np.asarray(im).copy()
        a[..., 3] = (a[..., 3] * alpha).astype(np.uint8)
        im = Image.fromarray(a)
    return im

# ============================================================ frame
def ring(img, cx, cy, R, wpx, col, a):
    if a <= 0: return
    x0, x1 = int(max(cx - R - 3, 0)), int(min(cx + R + 4, SW))
    y0, y1 = int(max(cy - R - 3, 0)), int(min(cy + R + 4, SH))
    if x1 <= x0 or y1 <= y0: return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    al = np.clip(1 - np.abs(rr - R) / wpx, 0, 1) * a
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - al[..., None]) + np.array(col)[None, None] * al[..., None]

def disc(img, cx, cy, R, col, a):
    x0, x1 = int(max(cx - R - 2, 0)), int(min(cx + R + 3, SW))
    y0, y1 = int(max(cy - R - 2, 0)), int(min(cy + R + 3, SH))
    if x1 <= x0 or y1 <= y0: return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    al = np.clip(R + 0.5 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2), 0, 1) * a
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - al[..., None]) + np.array(col)[None, None] * al[..., None]

class Renderer:
    def __init__(self):
        self.out = np.zeros((SH, SW, 3), np.float32)
        self.parts = Particles()
        self.rng = np.random.default_rng(5)
        self.last_th = 0.0

    def frame(self, t):
        cam = camera(t)
        cy, cx, s, tilt, ox, oy = cam
        th = sim_hours(t)
        ct, st = math.cos(tilt), math.sin(tilt)
        zmax = 360 * EXZ
        ylo = cy - oy / (s * ct) - 2
        yhi = cy + (SH - oy) / (s * ct) + zmax * math.tan(tilt) + 2
        ylo, yhi = max(ylo, 0), min(yhi, DH - 1.01)
        nr = int(min(3200, max(400, (yhi - ylo) * s * ct * 1.7)))
        # river reveal (main stem first, then tributaries)
        u = ease(ramp(t, 0.6, 4.6))
        thr = 10 ** (lerp(6.1, 3.0, u) - 0.95 * ease(ramp(t, 7.0, 10.5)))
        vis = (np.clip((np.log10(CH_ACC) - math.log10(thr)) / 0.35, 0, 1)
               * (0.5 + 0.5 * np.clip((np.log10(CH_ACC) - 2.0) / 2.5, 0, 1))).astype(np.float32)
        # flood response in catchment channels
        qn = np.zeros(CH.size, np.float32)
        if th > 0:
            ti = min(th / DT, len(T) - 1.001)
            i0 = int(ti); f = ti - i0
            qn[HC_IDX] = (QCH[:, i0] * (1 - f) + QCH[:, i0 + 1] * f) * FLOOD_GAIN
        flood_on = ramp(t, 12.0, 13.0) * (1 - 0.0 * ramp(t, 36, 38))
        qn *= flood_on
        width = CH_W0 * (1 + 1.5 * qn)
        rain_on = ramp(t, 12.0, 13.2)
        rainc = rain_field(th) if rain_on > 0 else np.zeros_like(cyy)
        wetc = wet_field(th) if rain_on > 0 else np.zeros_like(cyy)
        dim_out = max(ease(ramp(t, 6.8, 9.8)), 0.45 * ease(ramp(t, 4.3, 5.3)))
        focus = ease(ramp(t, 9.0, 12.4))
        outline = 0.9 * ease(ramp(t, 4.4, 5.4)) * (0.75 + 0.25 * math.cos((t - 4.4) * 4.0) if t < 7 else 1) \
            * (1 - 0.8 * ease(ramp(t, 12, 13.5)))
        P = np.array([cx, cy, s, ct, st, EXZ, ox, oy, ylo, yhi, nr, dim_out, focus, outline, CS, rain_on], np.float64)
        render_terrain(self.out, BG, L, NN, width, vis, qn, rainc, wetc, P)
        img = self.out.copy()

        # runoff particles
        if t >= T_STORM0:
            self.parts.step(th, th - self.last_th)
            self.parts.trail *= 0.70
            if self.parts.c.size:
                py, pxw, ch = self.parts.world()
                zz = zat(py, pxw) + 0.6 / EXZ
                sx, sy = project(py, pxw, zz, cam)
                splat(self.parts.trail, sx, sy, np.where(ch, 0.75, 0.32).astype(np.float32))
            glow = ndi.gaussian_filter(self.parts.trail, 0.8)
            fade = 1 - ramp(t, 35, 37.5)
            img += np.clip(glow, 0, 2.5)[..., None] * np.array([120, 210, 255], np.float32) * fade
        self.last_th = th

        # rain streaks
        if rain_on > 0 and 0 < th < 40:
            rb = rain_streaks(th, cam, rain_on, self.rng)
            img += np.clip(rb, 0, 1.5)[..., None] * np.array([170, 195, 230], np.float32)

        # bloom
        small = img[::4, ::4]
        lum = small.mean(-1, keepdims=True)
        br = np.clip(small - 150, 0, None) * np.clip((lum - 120) / 80, 0, 1)
        br = ndi.gaussian_filter(br, (6, 6, 0))
        bl = np.repeat(np.repeat(br, 4, 0), 4, 1)[:SH, :SW]
        img += 0.55 * bl
        img *= VIG

        # gauge marker + leader line
        gz = ZG[GY, GX]
        gsx, gsy = project(GY, GX, gz, cam)
        ga = ease(ramp(t, 10.5, 12.0)) * (1 - ease(ramp(t, 37.0, 38.2)))
        if ga > 0:
            pul = (t * 1.1) % 1.0
            ring(img, gsx, gsy, 9 + 22 * pul, 1.6, (255, 210, 110), ga * (1 - pul) * 0.9)
            ring(img, gsx, gsy, 8.5, 1.4, (255, 255, 255), ga)
            disc(img, gsx, gsy, 4.2, (255, 200, 92), ga)

        img = np.clip(img, 0, 255).astype(np.uint8)
        pim = Image.fromarray(img).convert("RGBA")
        ov = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        # end card dimming
        endk = ease(ramp(t, 37.0, 38.2))
        if ga > 0:
            pa = ease(ramp(t, 11.6, 12.8)) * (1 - endk)
            d.text((gsx - 4, gsy + 18), "Outlet gauge", font=F["lbl"], fill=(255, 225, 160, int(235 * ga)), anchor="ma")
        # brand
        ba = 1 - endk
        d.text((60, 52), "HydroORBIT", font=F["brand"], fill=(255, 255, 255, int(235 * ba)))
        d.text((60 + d.textlength("HydroORBIT", font=F["brand"]) + 14, 58), "streamflow forecasting",
               font=font("Inter-Regular.otf", 24), fill=(170, 182, 200, int(220 * ba)))
        # captions
        for a, b, txt in CAPTIONS:
            ca = ramp(t, a, a + 0.5) * (1 - ramp(t, b - 0.5, b))
            if ca > 0:
                dy = 8 * (1 - ease(ramp(t, a, a + 0.6)))
                d.text((60, SH - 78 + dy), txt, font=F["cap"], fill=(240, 244, 250, int(240 * ca)))
                d.rectangle([60, SH - 90 + dy, 60 + 46, SH - 87 + dy], fill=(255, 200, 92, int(240 * ca)))
        pim.alpha_composite(ov)
        # hydrograph panel
        pa = ease(ramp(t, 11.6, 12.8)) * (1 - endk)
        if pa > 0:
            panel = draw_panel(t, th, pa)
            off = int(30 * (1 - ease(ramp(t, 11.6, 12.8))))
            pim.alpha_composite(panel, (PX + off, PY))
        # end card
        if endk > 0:
            dim = Image.new("RGBA", (SW, SH), (6, 10, 20, int(205 * endk)))
            pim.alpha_composite(dim)
            e = ImageDraw.Draw(pim)
            ta = int(255 * ease(ramp(t, 37.5, 38.6)))
            e.text((SW / 2, SH / 2 - 40), "HydroORBIT", font=F["end"], fill=(255, 255, 255, ta), anchor="mm")
            e.text((SW / 2, SH / 2 + 48), "Forecasting how rivers respond to every storm", font=F["tag"],
                   fill=(190, 205, 225, ta), anchor="mm")
            e.rectangle([SW / 2 - 40, SH / 2 + 98, SW / 2 + 40, SH / 2 + 101], fill=(255, 200, 92, ta))
            e.text((SW / 2, SH - 60), "Illustrative animation · synthetic terrain and data", font=F["foot"],
                   fill=(120, 132, 152, ta), anchor="mm")
        # fade in / out for seamless loop
        fin = ease(ramp(t, 0, 0.6)) * (1 - ease(ramp(t, DUR - 0.5, DUR)))
        arr = np.asarray(pim.convert("RGB"), np.float32)
        if fin < 1:
            arr = arr * fin + BG[:, None, :] * (1 - fin) * VIG
        return arr.astype(np.uint8)

# ============================================================ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", default="")
    ap.add_argument("--out", default="hydroorbit_demo.mp4")
    ap.add_argument("--fps", type=int, default=FPS)
    a = ap.parse_args()
    R = Renderer()
    if a.stills:
        ts = [float(v) for v in a.stills.split(",")]
        # run particles forward so stills during the storm look right
        tcur = 0.0
        import time
        for tt in ts:
            if tt > T_STORM0:
                for tp in np.arange(max(tcur, T_STORM0), tt, 1 / a.fps):
                    R.parts.step(sim_hours(tp), 0)
            tcur = tt
            t0 = time.time()
            im = R.frame(tt)
            print(f"t={tt} {time.time()-t0:.2f}s", flush=True)
            Image.fromarray(im).save(f"still_{tt:05.1f}.png")
        sys.exit()
    n = int(DUR * a.fps)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{SW}x{SH}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                           "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out],
                          stdin=subprocess.PIPE)
    import time
    t0 = time.time()
    for i in range(n):
        im = R.frame(i / a.fps)
        ff.stdin.write(im.tobytes())
        if i % 60 == 0:
            print(f"frame {i}/{n}  {time.time()-t0:.0f}s", flush=True)
    ff.stdin.close(); ff.wait()
    print("done", time.time() - t0)
