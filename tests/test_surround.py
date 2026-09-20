"""The centre-surround front end prefers a small object to a large one of
the same luminance, which the raw transient does not."""

import numpy as np

from noflyzone.brain_connectome import SURROUND_RADIUS, centre_surround, surround_matrix
from noflyzone.retina import HexRetina


def test_small_disc_beats_large_disc_after_surround():
    ret = HexRetina(zoom=2.0)
    xy = ret.centers_xy
    spacing = np.median(np.sort(np.linalg.norm(xy[:, None] - xy[None], axis=-1), axis=1)[:, 1])
    S = surround_matrix(xy, SURROUND_RADIUS)
    disc = lambda r: (np.linalg.norm(xy, axis=1) <= r * spacing).astype(np.float32)
    small, large = disc(2.0), disc(8.0)
    centre = np.linalg.norm(xy, axis=1) < 1.0 * spacing
    assert large[centre].mean() == small[centre].mean() == 1.0          # raw: the centre cannot tell them apart
    cs_small, cs_large = centre_surround(small, S), centre_surround(large, S)
    assert cs_large[np.linalg.norm(xy, axis=1) < 4 * spacing].max() == 0.0   # a uniform interior is silent
    assert cs_small[centre].mean() > 0.4                                # the small disc's centre stays lit
    assert cs_small[centre].mean() > cs_large[centre].mean()            # size tuning, per ommatidium
    # the large disc still speaks through its edge, as an expanding edge does in a loom
    assert cs_large.sum() > 0 and cs_large.max() > 0.3
