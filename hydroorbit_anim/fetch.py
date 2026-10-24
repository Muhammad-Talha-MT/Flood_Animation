#!/usr/bin/env python3
"""Fetch all real data for the HydroORBIT booth animation into data/.

Adapted from hydroorbit_video_kit/fetch_tennessee_data.py: same basin layers, but the
zoom catchment is the gauge's own NLDI upstream basin, and the discharge/rain windows
are the chosen event. Each step is skipped if its output already exists.
"""
import argparse, pathlib, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--site", default="03451500")
ap.add_argument("--tag", default="2021mar", help="event tag used in output file names")
ap.add_argument("--start", default="2021-03-19")
ap.add_argument("--end", default="2021-04-03")
a = ap.parse_args()

import pygeohydro as gh
import py3dep

D = pathlib.Path(__file__).parent / "data"
D.mkdir(exist_ok=True)
CRS = 5070
KEEP = ["comid", "gnis_name", "streamorde", "totdasqkm", "reachcode", "hydroseq", "dnhydroseq", "geometry"]


def step(name):
    p = D / name
    if p.exists():
        print("  have", name)
        return None
    print("  fetch", name, flush=True)
    return p


# WBD from the USGS staged HU2 06 shapefile (the WBD map service times out on HUC2 queries)
WBD_ZIP = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/Shape/WBD_06_HU2_Shape.zip"
if p := step("tn_basin.geojson"):
    gpd.read_file(f"zip+{WBD_ZIP}!Shape/WBDHU2.shp").to_file(p, driver="GeoJSON")
if p := step("tn_huc4.geojson"):
    gpd.read_file(f"zip+{WBD_ZIP}!Shape/WBDHU4.shp").to_file(p, driver="GeoJSON")

# NHDPlusV2 VPU 06 from the EPA snapshot (the USGS WaterData geoserver returns 403 and NLDI
# times out). Downloaded + extracted to data/nhdplus06/ by:
#   curl -O https://dmap-data-commons-ow.s3.amazonaws.com/NHDPlusV21/Data/NHDPlusMS/NHDPlus06/NHDPlusV21_MS_06_{NHDSnapshot_09,NHDPlusAttributes_10,NHDPlusCatchment_05}.7z
#   python -m py7zr x <file> data/nhdplus06/
NP = D / "nhdplus06" / "NHDPlusMS" / "NHDPlus06"


def nhd_flowlines():
    fl = gpd.read_file(NP / "NHDSnapshot/Hydrography/NHDFlowline.shp")
    fl.columns = [c.lower() if c != "geometry" else c for c in fl.columns]
    vaa = gpd.read_file(NP / "NHDPlusAttributes/PlusFlowlineVAA.dbf", ignore_geometry=True)
    vaa.columns = [c.lower() for c in vaa.columns]
    fl = fl.merge(vaa[["comid", "streamorde", "totdasqkm", "hydroseq", "dnhydroseq", "pathlength", "lengthkm"]],
                  on="comid")
    fl = fl[fl.ftype != "Coastline"]
    # NHDFlowline geometries are 3D (zero z); drop z
    from shapely import force_2d
    fl["geometry"] = force_2d(fl.geometry)
    return fl


KEEP = KEEP + ["pathlength", "lengthkm"]
if (p := step("tn_flowlines.gpkg")) or not (D / f"catch_{a.site}.geojson").exists():
    fl = nhd_flowlines()
    if p:
        f = fl[fl.streamorde >= 3]
        gpd.GeoDataFrame(f[[c for c in KEEP if c in f.columns]]).to_file(p, driver="GPKG")
        print("    basin flowlines (order >= 3)", len(f))

    # gauge location from NWIS; snap to the nearest named reach of the river
    info = gh.NWIS().get_info({"site": a.site})
    gpt = gpd.GeoSeries.from_xy(info.dec_long_va, info.dec_lat_va, crs=4269).to_crs(fl.crs)
    river = info.station_nm.iloc[0].split(" At ")[0].split(" at ")[0].upper()
    cand = fl[fl.gnis_name.fillna("").str.upper() == river]
    d = cand.distance(gpt.iloc[0])
    g0 = cand.loc[d.idxmin()]
    print(f"    gauge reach comid {g0.comid}, {d.min():.0f} m from gauge, {g0.totdasqkm:.0f} km2")
    # everything upstream: follow dnhydroseq links back from the gauge reach
    dn = fl.set_index("hydroseq").dnhydroseq
    up = {g0.hydroseq}
    children = fl.groupby("dnhydroseq").hydroseq.apply(list).to_dict()
    stack = [g0.hydroseq]
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in up:
                up.add(c); stack.append(c)
    cf = fl[fl.hydroseq.isin(up)]
    print("    upstream reaches", len(cf))
    gpd.GeoDataFrame(cf[[c for c in KEEP if c in cf.columns]]).to_file(D / f"catch_{a.site}_flowlines.gpkg", driver="GPKG")
    cat = gpd.read_file(NP / "NHDPlusCatchment/Catchment.shp")
    cat.columns = [c.lower() if c != "geometry" else c for c in cat.columns]
    cat = cat[cat.featureid.isin(cf.comid)]
    poly = cat.to_crs(CRS).buffer(1).union_all().buffer(-1)
    gpd.GeoDataFrame({"site": [a.site]}, geometry=[poly], crs=CRS).to_file(D / f"catch_{a.site}.geojson", driver="GeoJSON")
    gpt.to_frame("geometry").assign(site=a.site).to_file(D / f"gauge_{a.site}.geojson", driver="GeoJSON")
    print(f"    upstream basin {poly.area/1e6:.0f} km2 (NHDPlus totdasqkm at gauge reach {g0.totdasqkm:.0f})")


def save_dem(gdf, buf_m, res, path):
    geom = gdf.to_crs(CRS).buffer(buf_m).to_crs(4326).union_all()
    dem = py3dep.get_dem(geom, res, crs=4326)
    dem = dem.rio.reproject(CRS, resolution=res)
    dem = dem.where(np.isfinite(dem), -9999).round().astype("int16")
    dem.rio.write_nodata(-9999, inplace=True)
    dem.rio.to_raster(path, compress="deflate", predictor=2, tiled=True)
    print("   ", path.name, dem.shape)


if p := step("tn_dem_120m.tif"):
    save_dem(gpd.read_file(D / "tn_basin.geojson"), 8000, 120, p)
if p := step(f"catch_{a.site}_dem_30m.tif"):
    save_dem(gpd.read_file(D / f"catch_{a.site}.geojson"), 4000, 30, p)

if p := step(f"gauge_{a.site}_{a.tag}_iv.csv"):
    q = gh.NWIS().get_streamflow(a.site, (a.start, a.end), freq="iv")   # m3/s, UTC
    q.to_csv(p)

if p := step(f"mrms_prcp_{a.site}_{a.tag}.csv"):
    # Hourly NOAA MRMS MultiSensor QPE (Pass 2, radar + gauges, ~1 km) from the public AWS
    # archive, cropped to the catchment. (NLDAS-2 data rods were retired in 2025.)
    # Each file is the 1-h accumulation ending at its timestamp.
    import gzip, io, tempfile, urllib.request
    from concurrent.futures import ThreadPoolExecutor
    import pygrib
    from rasterio import features
    from rasterio.transform import from_origin
    cpoly = gpd.read_file(D / f"catch_{a.site}.geojson").to_crs(4326).union_all()
    lo0, la0, lo1, la1 = cpoly.buffer(0.15).bounds
    hours = pd.date_range(a.start, a.end, freq="h", tz="UTC")
    MR = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass2_00.00"

    def download(t):
        url = f"{MR}/{t:%Y%m%d}/MRMS_MultiSensor_QPE_01H_Pass2_00.00_{t:%Y%m%d-%H%M%S}.grib2.gz"
        return urllib.request.urlopen(url, timeout=120).read()

    def decode(gz):
        # crop indices from the grid definition (latlons() on the CONUS grid costs ~400 MB)
        with tempfile.NamedTemporaryFile(suffix=".grib2") as f:
            f.write(gzip.decompress(gz)); f.flush()
            g = pygrib.open(f.name).message(1)
            la_first, lo_first = g["latitudeOfFirstGridPointInDegrees"], g["longitudeOfFirstGridPointInDegrees"]
            di, dj = g["iDirectionIncrementInDegrees"], g["jDirectionIncrementInDegrees"]
            lats = la_first - dj * np.arange(g["Nj"])
            lons = (lo_first - 360 if lo_first > 180 else lo_first) + di * np.arange(g["Ni"])
            r = np.flatnonzero((lats >= la0) & (lats <= la1))
            c = np.flatnonzero((lons >= lo0) & (lons <= lo1))
            v = np.asarray(g.values)[r[0]:r[-1] + 1, c[0]:c[-1] + 1]
            return np.clip(v, 0, None).astype(np.float32), lats[r[0]:r[-1] + 1], lons[c[0]:c[-1] + 1]

    with ThreadPoolExecutor(6) as ex:
        res = [decode(gz) for gz in ex.map(download, hours)]
    grid = np.stack([x[0] for x in res])
    lat, lon = res[0][1], res[0][2]
    if lat[0] < lat[-1]:
        grid, lat = grid[:, ::-1], lat[::-1]
    tr = from_origin(lon[0] - 0.005, lat[0] + 0.005, 0.01, 0.01)
    m = features.rasterize([(cpoly, 1)], out_shape=grid.shape[1:], transform=tr).astype(bool)
    s = pd.Series(grid[:, m].mean(1), index=hours - pd.Timedelta(hours=1), name="prcp_mm_h")  # hour start
    s.to_csv(p)
    np.savez_compressed(D / f"mrms_grid_{a.site}_{a.tag}.npz", rain=grid, lat=lat, lon=lon,
                        t_start_ns=(hours - pd.Timedelta(hours=1)).as_unit("ns").asi8, mask=m)
    print(f"    MRMS basin-mean total {s.sum():.0f} mm, max {s.max():.1f} mm/h at {s.idxmax()}")
print("done")
