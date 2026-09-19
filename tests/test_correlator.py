"""The v6 correlator: a bar moving +x on the hex lattice drives the +x
channel and not the -x one, and vice versa; no data needed."""

import numpy as np

from noflyzone.brain_connectome import lattice_neighbours, reichardt
from noflyzone.retina import HexRetina


def bar_frames(xy, spacing, direction):
    xs = np.linspace(-0.6, 0.6, 10)
    if direction < 0:
        xs = xs[::-1]
    return [(np.abs(xy[:, 0] - x) <= 1.2 * spacing).astype(np.float32) for x in xs]


def energy(frames, nb):
    prev_on = None
    tot = 0.0
    for k in range(1, len(frames)):
        on = np.clip(frames[k] - frames[k - 1], 0, None)
        if prev_on is not None:
            tot += float(reichardt(on, prev_on, nb).sum())
        prev_on = on
    return tot


def test_reichardt_prefers_its_direction():
    r = HexRetina()
    xy = r.centers_xy.astype(np.float64)
    from scipy.spatial import cKDTree
    spacing = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    nb = lattice_neighbours(xy)
    assert (nb["px"] >= 0).mean() > 0.8 and (nb["py"] >= 0).mean() > 0.8
    right = bar_frames(xy, spacing, +1)
    left = bar_frames(xy, spacing, -1)
    assert energy(right, nb["px"]) > 3 * energy(right, nb["mx"])
    assert energy(left, nb["mx"]) > 3 * energy(left, nb["px"])
    # a horizontal bar has no vertical preference to speak of
    assert abs(energy(right, nb["py"]) - energy(right, nb["my"])) < 0.5 * energy(right, nb["px"])
