#!/usr/bin/env python3
"""Fetch everything the HydroORBIT booth animation needs, in one go.

Run on any machine with normal internet (laptop, Bridges-2 login node, zgx server):

    pip install pygeohydro pynhd py3dep rioxarray
    python fetch_tennessee_data.py

It writes hydroorbit_anim_data.zip (about 20-40 MB). Upload that zip to the chat.

Contents
  tn_basin.geojson          WBD HUC2 06 (Tennessee River Basin) boundary
  tn_huc4.geojson           HUC4 subregions 0601-0604
  tn_flowlines.gpkg         NHDPlusV2 flowlines, stream order >= 3, whole basin
  tn_dem_120m.tif           3DEP DEM of the basin, 120 m, int16 metres (EPSG:5070)
  fb_huc8.geojson           French Broad-Holston HUC8 06010105 boundary
  fb_flowlines.gpkg         NHDPlusV2 flowlines, all orders, French Broad HUC8
  fb_dem_30m.tif            3DEP DEM of the French Broad HUC8, 30 m, int16 (EPSG:5070)
  gauge_<site>_iv.csv       USGS instantaneous discharge (m3/s) for the outlet gauge
"""
import argparse, zipfile, pathlib, warnings
import numpy as np
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--site", default="03451500", help="USGS gauge for the hydrograph (default: French Broad R at Asheville, NC)")
ap.add_argument("--start", default="2020-01-01")
ap.add_argument("--end", default="2025-12-31")
ap.add_argument("--out", default="hydroorbit_anim_data")
a = ap.parse_args()

import geopandas as gpd
import pygeohydro as gh
import pynhd
import py3dep

out = pathlib.Path(a.out); out.mkdir(exist_ok=True)
CRS = 5070

print("1/7 basin boundary (WBD HUC2 06)")
basin = gh.WBD("huc2").byids("huc2", "06")
basin.to_file(out / "tn_basin.geojson", driver="GeoJSON")
gh.WBD("huc4").byids("huc4", ["0601", "0602", "0603", "0604"]).to_file(out / "tn_huc4.geojson", driver="GeoJSON")

print("2/7 French Broad HUC8 06010105")
fb = gh.WBD("huc8").byids("huc8", "06010105")
fb.to_file(out / "fb_huc8.geojson", driver="GeoJSON")

wd = pynhd.WaterData("nhdflowline_network")
keep = ["comid", "gnis_name", "streamorde", "totdasqkm", "reachcode", "geometry"]

print("3/7 basin flowlines (order >= 3), by HUC4")
parts = []
for h4 in ["0601", "0602", "0603", "0604"]:
    f = wd.byfilter(f"streamorde >= 3 AND reachcode LIKE '{h4}%'")
    print("   ", h4, len(f))
    parts.append(f[[c for c in keep if c in f.columns]])
import pandas as pd
fl = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
fl.to_file(out / "tn_flowlines.gpkg", driver="GPKG")

print("4/7 French Broad flowlines (all orders)")
ff = wd.byfilter("reachcode LIKE '06010105%'")
ff[[c for c in keep if c in ff.columns]].to_file(out / "fb_flowlines.gpkg", driver="GPKG")

def save_dem(gdf, buf_m, res, path):
    geom = gdf.to_crs(CRS).buffer(buf_m).to_crs(4326).iloc[0]
    dem = py3dep.get_dem(geom, res, crs=4326)
    dem = dem.rio.reproject(CRS, resolution=res)
    dem = dem.where(np.isfinite(dem), -9999).round().astype("int16")
    dem.rio.write_nodata(-9999, inplace=True)
    dem.rio.to_raster(path, compress="deflate", predictor=2, tiled=True)
    print("   ", path, dem.shape)

print("5/7 basin DEM 120 m (takes a few minutes)")
save_dem(basin, 5000, 120, out / "tn_dem_120m.tif")

print("6/7 French Broad DEM 30 m")
save_dem(fb, 2000, 30, out / "fb_dem_30m.tif")

print(f"7/7 USGS discharge at {a.site}")
nwis = gh.NWIS()
info = nwis.get_info({"site": a.site})
print("   ", info[["site_no", "station_nm"]].to_string(index=False))
q = nwis.get_streamflow(a.site, (a.start, a.end), freq="iv")   # m3/s
q.to_csv(out / f"gauge_{a.site}_iv.csv")
info.drop(columns="geometry", errors="ignore").to_csv(out / f"gauge_{a.site}_info.csv", index=False)

zp = pathlib.Path(a.out + ".zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.iterdir()):
        z.write(p, p.name)
print(f"done -> {zp}  ({zp.stat().st_size / 1e6:.1f} MB). Upload this zip to the chat.")
