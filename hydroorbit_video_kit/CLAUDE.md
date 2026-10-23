# HydroORBIT booth animation — build brief for Claude Code

You are building a 35–45 second animation for the HydroORBIT demo at the DOE booth.
Work through this file top to bottom without waiting for approval between steps,
except where it says **ASK THE USER**. Keep a task list, render stills to check your
work, and deliver an MP4 at the end.

The goal is not just to show water moving. It is to show **how HydroORBIT forecasts the
evolving hydrologic response to a storm**: rainfall → runoff → tributaries converging →
streamflow rising, peaking and receding, with a synchronized hydrograph comparing
HydroORBIT predictions to observed discharge at the catchment outlet.

---

## 0. Before you start

1. **ASK THE USER** these three things in one message, then continue with the defaults
   for anything they skip:
   - **Storm event.** Which storm, and at which gauge? Default gauge: USGS 03451500,
     French Broad River at Asheville, NC. Default event: the largest peak in the gauge
     record from 2020 onward, excluding Hurricane Helene (Sept 2024). Helene caused
     deaths in the Asheville area, so use it only if the user explicitly chooses it.
   - **HydroORBIT predictions.** Do they have a CSV of HydroORBIT forecasts for that
     event? Expected columns: `time, q_pred` (m³/s), and ideally quantile columns such
     as `q05, q50, q95` and/or an `issue_time` column for rolling forecasts. If there is
     no file, build an illustrative forecast (section 3) and label it
     **"Illustrative forecast"** on screen. Never present invented numbers as
     HydroORBIT results.
   - **Camera reference.** They liked the movement in a LinkedIn video you can't open.
     Ask for a one-line description, or for the clip or screenshots to be dropped in
     `reference/`. If they don't answer, use the camera plan in section 4.
2. Make a project folder `hydroorbit_anim/` with `data/`, `build/`, `stills/`, `out/`.
3. Set up Python 3.11+ in a fresh venv or conda env and install:
   ```
   pip install pygeohydro pynhd py3dep rioxarray geopandas shapely pyproj numpy scipy \
               pandas matplotlib pillow pyvista imageio imageio-ffmpeg numba
   ```
   You also need `ffmpeg` on PATH (or use the binary from `imageio-ffmpeg`).
   On a headless Linux server, PyVista needs off-screen rendering: try
   `pip install vtk-osmesa` or use `pyvista.start_xvfb()` with `xvfb` installed.

---

## 1. Data (all real, no synthetic terrain)

Run `fetch_tennessee_data.py` (included in this kit) or do the same steps yourself:

| Layer | Source | Notes |
|---|---|---|
| Tennessee River Basin boundary | WBD HUC2 `06` via `pygeohydro.WBD("huc2")` | Also fetch HUC4 0601–0604. |
| Basin river network | NHDPlusV2 flowlines via `pynhd.WaterData("nhdflowline_network")`, CQL `streamorde >= 3 AND reachcode LIKE '060X%'` per HUC4 | Keep `streamorde`, `totdasqkm`, `gnis_name`. |
| Basin DEM | 3DEP via `py3dep.get_dem(geom, 120)` → EPSG:5070 | 90–120 m is enough for the overview. |
| Zoom catchment | WBD HUC8 `06010105` (French Broad–Holston), or the upstream basin of the chosen gauge via `pynhd.NLDI().get_basins(site)` | Prefer the gauge's own upstream basin, so the outlet is the gauge. |
| Catchment network | NHDPlusV2 flowlines, all orders, inside that basin | Use `NLDI().navigate_byid("nwissite", "USGS-<site>", "upstreamTributaries", source="flowlines", distance=1000)` |
| Catchment DEM | 3DEP 30 m (10 m if the machine copes) | Smooth lightly for geometry. |
| Observed discharge | `pygeohydro.NWIS().get_streamflow(site, (start, end), freq="iv")` (m³/s) | Window: about 1–2 days before rain starts to about 3–4 days after the peak. |
| Rainfall timing | Use the user's forcing data if offered (ERA5/ERA5-Land basin mean). Otherwise derive a plausible hyetograph that ends before the rising limb, and label rain as illustrative. | Only the timing on screen must be honest. |

Sanity checks before rendering. Print each result and fix anything that fails:
- The basin polygon looks like the Tennessee basin: a long NE–SW valley from SW Virginia
  through East Tennessee, dipping into north Alabama, then running north through west
  Tennessee to Paducah, KY. Plot it with state outlines and look at the PNG.
- The flowlines include the Tennessee main stem, plus the Holston, French Broad, Little
  Tennessee, Hiwassee, Clinch, Elk and Duck rivers.
- The gauge point sits on the catchment network, within one flowline of the outlet.
- Observed discharge has no gaps longer than 2 hours in the event window; interpolate
  short gaps.

---

## 2. Visual style

- **Truly 3D from the first frame.** Use a perspective camera at all times, with terrain
  as a lit, shaded 3D mesh using vertical exaggeration: about 8–12× for the whole basin,
  3–5× for the catchment. No flat top-down map scene.
- Dark, cinematic background: deep navy to near-black gradient with a slight vignette.
- Terrain coloring: muted natural hypsometric tint (greens → tans → grey highs) with
  strong hillshade or ambient occlusion. Out-of-focus areas are desaturated and darkened.
- Rivers: glowing cyan lines on the terrain surface, width ∝ `streamorde` or
  `sqrt(totdasqkm)`. During the storm, catchment rivers brighten and thicken with
  routed discharge. The flood pulse must visibly travel downstream to the outlet.
- Rain: falling streaks plus a moving dark cloud-shadow over the terrain.
- Runoff: bright particles that start on hillslopes, flow down D8 directions from the
  DEM, join channels, speed up and converge to the outlet.
- Outlet gauge: amber pulsing ring with a label, always visible after the zoom.
- Palette tokens: background `#070C18`, river `#4FC3F7`, flood highlight `#BDF3FF`,
  HydroORBIT `#40E0D0`, observed `#F0F4FA`, rain `#6096FF`, highlight/gauge `#FFC85C`.

### Captions (important: the user said the last ones were too small and off to the side)
- One large centered **scene title** at each scene change: about 64–76 px at 1080p,
  semibold, white, with a subtle drop shadow. Fade and slide in over 0.4 s, hold
  2–3 s, fade out.
- Under it, or as a centered lower-third, one short line of 26–30 px secondary text
  on a soft translucent dark band spanning about 60% of the width.
- At most 6 words in a title and 10 in a subline. No paragraphs.
- Font: Inter, Source Sans 3 or IBM Plex Sans (download from Google Fonts if missing).
- Keep a small "HydroORBIT" wordmark top-left the whole time.

---

## 3. Hydrology on screen

1. **Observed hydrograph:** the real NWIS series for the chosen gauge.
2. **HydroORBIT:**
   - If the user gave a CSV: plot it exactly. For rolling forecasts, at each animation
     time t show the latest forecast issued at or before t as a plume extending 24–48 h
     ahead, with its quantile band. Otherwise show the single forecast trace and band.
   - If not, build an illustrative rolling forecast from the observed series. Early
     issues are slightly late and low, with a wide band. They converge on the observed
     peak as the storm is observed. Label it "Illustrative forecast" in the panel.
3. **Channel animation:** route a simple runoff field through the catchment network.
   Use translation along flow distance plus a linear reservoir, with runoff timed to
   the hyetograph. Scale it so the outlet curve's shape and timing match the observed
   hydrograph (normalize per reach). This drives river glow and width, so the map and
   the chart rise and fall together.
4. **Hydrograph panel** (right side, about 34% of the width, translucent dark card):
   - Rainfall bars hanging from the top.
   - Observed line with dots up to "now".
   - HydroORBIT plume ahead of "now", with a "forecast peak" marker.
   - A vertical "now" cursor with a time label.
   - An "observed peak" marker once the peak has passed.
   - Axes in m³/s and hours (or real dates).
   - The panel is drawn every frame and is synchronized to the same clock as the map.
5. **Summary beat:** the full event, HydroORBIT vs. observed, with two big numbers:
   peak magnitude error (%) and peak timing error (h). Compute both from the data.
   Only call them HydroORBIT skill if the forecast is real.

---

## 4. Storyboard (40 s at 30 fps = 1200 frames; adjust ±5 s if needed)

Map video time to event time non-linearly: slow through the rising limb and peak,
faster through the recession. Use a monotone (PCHIP) curve.

| Time | Scene | Camera | Title / subline |
|---|---|---|---|
| 0–6 s | Whole Tennessee River Basin in 3D. The river network grows outward from the main stem (stream order high → low). | High oblique view from the southwest, slow orbit about 20° and slight push-in. | **Tennessee River Basin** / "105,000 km² · 7 states" (compute area from the polygon) |
| 5–7 s | Chosen catchment outline glows amber; everything else dims. | Orbit eases toward the catchment. | — |
| 7–12 s | Fly down into the catchment. A 30 m DEM crossfades in, detailed network appears, gauge marker appears. | Smooth swoop: fast in the middle, eased at both ends; ends low and oblique, about 35–45° above horizon, outlet at front-left. | **Into the French Broad** (use the real name) / "Headwater catchment and outlet gauge" |
| 12–20 s | Storm arrives: clouds, rain streaks, cloud shadow. Hillslope particles start. Panel slides in. HydroORBIT plume already forecasts the coming peak. | Slow lateral dolly along the main valley. | **The storm arrives** / "HydroORBIT forecasts the response before the peak" |
| 20–26 s | Tributaries converge; flood pulse travels down the main stem; outlet glows; hydrograph climbs to the peak. | Gentle push toward the outlet, slight low pass. | **Streamflow peaks** / "Tributaries converge at the outlet" |
| 26–33 s | Rain clears, rivers dim, recession on the hydrograph. | Slow pull back and rise. | **Flow recedes** / "Forecast tracks the full recession" |
| 33–37 s | Summary on the panel: full traces plus the two error numbers. | Hold, slight drift. | **HydroORBIT vs. observed** |
| 37–40 s | End card: "HydroORBIT", one-line tagline, DOE/team logos if the user supplies them, data credits (USGS NHDPlus, 3DEP, NWIS). Fade to the first frame's background so it loops cleanly. | — | — |

If the user describes a different camera move (section 0), replace the camera column
with theirs and keep the rest.

---

## 5. How to build it

Recommended pipeline (robust on any machine):

1. **`prep.py`:**
   - Reproject everything to EPSG:5070.
   - Clip and downsample the basin DEM to about 1600 × 900 for the overview mesh.
   - Clip the 30 m catchment DEM.
   - Rasterize flowlines to a stream mask, and compute D8 directions and flow
     distance to the gauge on the catchment DEM. pysheds or your own numba code both work.
   - Save everything as `.npz` plus GeoPackages.
2. **`hydro.py`:**
   - Load the observed series and the HydroORBIT CSV (or build the illustrative one).
   - Build the hyetograph and the per-reach routed discharge (section 3).
   - Save the video clock mapping as well.
3. **`render.py`**, two layers composited per frame:
   - **3D layer:** PyVista/VTK offscreen. Draw terrain as `StructuredGrid` meshes with
     per-vertex RGB (hypsometric tint × hillshade, with wetness darkening updated per
     frame). Draw rivers as tubes or polylines lifted slightly above the surface, with
     per-segment color and width updated per frame. Draw particles and rain as point
     sprites. Animate the camera with `plotter.camera.position/focal_point/up`,
     interpolated with ease-in-out splines between the storyboard keyframes. Render with
     `screenshot()` at 1920×1080 (or 3840×2160 for a 4K master).
   - **2D overlay:** Pillow, drawn at 2× and downsampled for antialiasing. It holds the
     hydrograph panel, captions, wordmark, gauge label and end card.
   - Composite, then pipe raw frames to ffmpeg:
     `-c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -movflags +faststart`.
   - If Blender is installed and the user wants more polish, you may instead build the
     3D layer in Blender (`bpy`, displaced plane, Eevee) and keep the same overlay step.
     PyVista is the default because it is quick to set up.
4. Use the draft code in `reference_draft/` (the earlier 2.5D version) for reusable
   pieces: the hydrograph panel (`draw_panel`), the particle advection, rain streaks,
   bloom, the PCHIP time mapping and the ffmpeg piping. Its terrain is synthetic. Do
   not reuse its terrain.

Performance: the target is under 1 s per frame. Precompute anything static: meshes,
colors and the stream mask.

---

## 6. Check your work (do this; don't skip it)

1. Before the full render, render stills at t = 2, 6, 10, 14, 20, 24, 30, 35, 38.5 s.
   Look at every one and fix:
   - the basin shape is wrong or cut off;
   - captions are small, overlapping or off-center;
   - the panel covers the outlet;
   - terrain looks flat, spiky or striped;
   - rivers float above or sink under the terrain;
   - text is unreadable on the background.
2. Confirm that the hydrograph's "now" time, the river glow and the particles agree. The
   outlet should glow brightest at the observed peak time.
3. Render the full video, then check it with ffprobe: duration 35–45 s, 1920×1080,
   30 fps.
4. Make a 3×3 contact sheet of frames and look at it once more.
5. Deliver:
   - `out/hydroorbit_booth.mp4`
   - `out/hydroorbit_booth_4k.mp4` (optional)
   - `out/contact_sheet.png`
   - one paragraph listing what is real data and what is illustrative.

---

## 7. Honesty rules for the final video

- Every map feature comes from real data (WBD, NHDPlus, 3DEP).
- The observed hydrograph is real NWIS data, with credit in the end card.
- HydroORBIT numbers are shown only if they come from the user's CSV. Otherwise the
  panel and the end card say "Illustrative forecast".
- Any stylized rain or runoff animation is fine, but its timing must match the event.
