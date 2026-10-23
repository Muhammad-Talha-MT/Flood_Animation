Earlier draft (synthetic terrain, 2.5D). Reuse the hydrograph panel, particles, rain,
bloom, time mapping and ffmpeg piping from render.py. Do not reuse the terrain.
Run order of the draft: terrain.py -> hydro.py -> render.py (needs numba, scipy, pillow).
