"""Shared storm definition: a band of moving rain cells crossing the catchment SW -> NE."""
import numpy as np

# rain cells: (start x, start y, peak time h, peak intensity mm/h, sigma cells, duration sigma h)
# positions are DEM cells; motion is common to all cells
VX, VY = 34.0, -9.0          # cells per hour (~3.4 km/h at 100 m cells... visual scale)
CELLS = [
    (1990, 380, 10.0, 12.0, 80, 2.2),
    (2040, 300, 13.0, 20.0, 70, 2.0),
    (2100, 400, 15.5, 14.0, 90, 2.4),
    (2060, 330, 18.0, 30.0, 75, 1.8),
    (2120, 360, 19.5, 18.0, 85, 2.0),
    (2150, 290, 22.0, 12.0, 85, 2.2),
    (2200, 380, 24.5, 7.0, 95, 2.4),
]
STRAT = (2080, 340, 17.5, 2.5, 260, 7.5)   # wide light stratiform shield


def rain_at(t, ys, xs):
    """Rain rate (mm/h) at hour t for broadcastable arrays of DEM rows ys and cols xs."""
    r = np.zeros(np.broadcast(np.asarray(ys), np.asarray(xs)).shape)
    for (x0, y0, tp, a, sg, dur) in CELLS + [STRAT]:
        amp = 0.42 * a * np.exp(-0.5 * ((t - tp) / dur) ** 2)
        if amp < 0.02:
            continue
        cx = x0 + VX * (t - tp) + 0.0
        cy = y0 + VY * (t - tp)
        r = r + amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sg * sg))
    return r
