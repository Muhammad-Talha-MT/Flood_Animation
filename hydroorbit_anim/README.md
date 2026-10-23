# HydroORBIT booth animation: Hurricane Helene, French Broad at Asheville

40 s, 1920×1080, 30 fps. The event is Hurricane Helene (Sept 2024) at USGS 03451500.

## Run order

```
python fetch.py     # data/      (WBD, NHDPlusV2 VPU 06, 3DEP, NWIS, MRMS)
python prep.py      # build/     grids, textures, D8 routing, river polylines, sanity_map.png
python hydro.py     # build/hydro.npz, hydro_preview.png
python render.py --stills 2,6,10,14,20,24,30,35,38.5     # stills/
python render.py --out out/hydroorbit_booth.mp4          # add --scale 2 for a 4K master
```

The NHDPlusV2 archives must be downloaded and extracted into `data/nhdplus06/` first.
`fetch.py` has the commands in a comment. Headless rendering needs `libosmesa6` or `libegl1`.

## What is real and what is illustrative

| Element | Source |
|---|---|
| Basin, catchment outline | USGS WBD HU2 06; gauge basin = union of NHDPlusV2 catchments upstream of the gauge reach (2,447 km²) |
| Rivers | NHDPlusV2 flowlines and VAA (stream order, drainage area, hydroseq) |
| Terrain | USGS 3DEP (120 m basin, 30 m catchment), 10× / 4× vertical exaggeration |
| Observed discharge | USGS NWIS instantaneous values, 03451500 (peak 3,200 m³/s, 27 Sep 21:38 UTC) |
| Rain (panel bars, rain/cloud/particle timing and placement) | NOAA MRMS MultiSensor QPE Pass 2, hourly, about 1 km |
| River glow | Simple routing: SCS-CN excess, fast and slow reservoirs, translation along NHDPlus flow distance. Fitted and nudged so the outlet matches the observed series |
| Particles, rain streaks, clouds | Stylized, but driven by MRMS rain in time and space |
| **HydroORBIT forecast and the two summary numbers** | **Illustrative**, derived from the observed series. Not HydroORBIT output |
