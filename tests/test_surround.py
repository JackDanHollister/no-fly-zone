"""The centre-surround front end prefers a small object to a large one of
the same luminance, which the raw transient does not."""

import numpy as np

from noflyzone.brain_connectome import SURROUND_RADIUS, centre_surround, surround_matrix, wide_field_opponent
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


def test_wide_field_opponency_keeps_a_lone_disc_and_silences_a_loom_edge():
    ret = HexRetina(zoom=2.0)
    xy = ret.centers_xy
    spacing = np.median(np.sort(np.linalg.norm(xy[:, None] - xy[None], axis=-1), axis=1)[:, 1])
    S = surround_matrix(xy, SURROUND_RADIUS)
    disc = lambda r: (np.linalg.norm(xy, axis=1) <= r * spacing).astype(np.float32)
    small = centre_surround(disc(2.0), S)
    loom_edge = centre_surround(disc(9.0), S)                 # a big disc: only its edge ring survives the surround
    assert loom_edge.sum() > small.sum()                      # after the surround the loom's edge still wins in total
    wf_small, wf_edge = wide_field_opponent(small), wide_field_opponent(loom_edge)
    assert wf_small.sum() > 0.5 * small.sum()                 # the lone disc keeps most of its drive
    assert wf_edge.sum() < 0.2 * loom_edge.sum()              # the ring everywhere suppresses itself
    assert wf_small.sum() > wf_edge.sum()
