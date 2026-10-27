#!/usr/bin/env python3
"""Static CONUS map for the opening shot (build/conus.npz).

A flat textured plane in the same EPSG:5070 km frame as the 3D scene (same ORIGIN as prep.py), so the
camera can fly from the whole country straight down into the Tennessee River Basin mesh.

- Rivers: USGS small-scale (1:1,000,000) streams, streaml010g, Strahler order >= 4.
  Downloaded + extracted to data/conus/ by:
    curl -O https://prd-tnm.s3.amazonaws.com/StagedProducts/Small-scale/data/Hydrography/streaml010g.shp_nt00885.tar.gz
- States: Census cartographic boundaries 2022, 1:20M (data/states.zip, fetched for prep.py).
- The Tennessee River Basin outline is drawn by render.py (3D, animated), not baked in here.
"""
import pathlib, warnings
import numpy as np
import geopandas as gpd
import cv2
warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).parent
D, B = HERE / "data", HERE / "build"
CRS = 5070
PX = 1.25                          # km per texture pixel
SS = 2                             # supersampling for anti-aliasing
MIN_ORDER = 4
ORIGIN = np.load(B / "basin.npz")["origin"]

states = gpd.read_file(f"zip://{D / 'states.zip'}").to_crs(CRS)
states = states[~states.STUSPS.isin(["AK", "HI", "PR", "VI", "GU", "MP", "AS"])]
conus = states.union_all()
x0, y0, x1, y1 = conus.bounds
pad = 450e3
x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
W, H = int((x1 - x0) / 1000 / PX), int((y1 - y0) / 1000 / PX)
print(f"texture {W} x {H} at {PX} km/px")
s = SS / (PX * 1000)


def to_px(xy):
    xy = np.asarray(xy)[:, :2]
    return np.c_[(xy[:, 0] - x0) * s, (y1 - xy[:, 1]) * s]


def rings(geom):
    for p in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
        yield p.exterior
        yield from p.interiors


def lines(geom):
    for g in (geom.geoms if geom.geom_type.startswith("Multi") else [geom]):
        yield g


def draw_lines(img, geoms, col, width, alpha):
    layer = np.zeros(img.shape[:2], np.uint8)
    for g in geoms:
        for ln in lines(g):
            pts = np.round(to_px(ln.coords) * 16).astype(np.int32)       # 4 fractional bits
            if len(pts) > 1:
                cv2.polylines(layer, [pts], False, 255, max(1, int(round(width * SS))), cv2.LINE_AA, shift=4)
    a = (layer.astype(np.float32) / 255 * alpha)[..., None]
    img[:] = img * (1 - a) + np.array(col, np.float32) * a


img = np.zeros((H * SS, W * SS, 3), np.float32)
img[:] = (9, 15, 28)                                      # background, as the 3D scene
land = np.zeros(img.shape[:2], np.uint8)
polys = [np.round(to_px(r.coords) * 16).astype(np.int32) for st in states.geometry for r in rings(st)]
cv2.fillPoly(land, polys, 255, cv2.LINE_AA, shift=4)
a = (land.astype(np.float32) / 255)[..., None]
img[:] = img * (1 - a) + np.array([24, 34, 50], np.float32) * a

print("rivers...")
st = gpd.read_file(next((D / "conus").glob("streaml010g.shp")), columns=["Strahler", "State", "Feature"],
                   where=f"Strahler >= {MIN_ORDER}")
st = st[~st.State.isin(["AK", "HI", "PR", "VI"])].to_crs(CRS)
print(f"   reaches, order >= {MIN_ORDER}: {len(st)}")
for o in sorted(st.Strahler.unique()):
    sel = st[st.Strahler == o]
    k = min((o - MIN_ORDER) / 5, 1)
    col = np.array([60, 140, 200]) * (1 - k) + np.array([110, 210, 255]) * k
    draw_lines(img, sel.geometry, col, 0.8 + 0.55 * (o - MIN_ORDER), 0.55 + 0.45 * k)
print("states...")
draw_lines(img, [r for g in states.geometry for r in rings(g)], (150, 165, 188), 0.9, 0.55)
draw_lines(img, list(rings(conus)), (170, 184, 204), 1.3, 0.75)

tex = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).clip(0, 255).astype(np.uint8)
xs = (x0 + (np.arange(W) + 0.5) * PX * 1000 - ORIGIN[0]) / 1000
ys = (y1 - (np.arange(H) + 0.5) * PX * 1000 - ORIGIN[1]) / 1000
cx, cy = conus.centroid.x, conus.centroid.y
np.savez_compressed(B / "conus.npz", tex=tex, xs=xs.astype(np.float32), ys=ys.astype(np.float32),
                    centre=np.array([(cx - ORIGIN[0]) / 1000, (cy - ORIGIN[1]) / 1000], np.float32),
                    extent=np.array([(x0 - ORIGIN[0]) / 1000, (x1 - ORIGIN[0]) / 1000,
                                     (y0 - ORIGIN[1]) / 1000, (y1 - ORIGIN[1]) / 1000], np.float32))
cv2.imwrite(str(B / "conus_preview.png"), cv2.cvtColor(cv2.resize(tex, (W // 2, H // 2)), cv2.COLOR_RGB2BGR))
print("done")
