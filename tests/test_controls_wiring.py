"""The wiring controls are what they claim: a rewiring keeps every cell's
output and the left/right structure it is supposed to keep."""

import numpy as np
import pytest

from noflyzone.brain_lif import ShiuBrain, annotations


def column_totals(brain):
    return np.asarray(abs(brain.W).sum(axis=0)).ravel()


@pytest.mark.data
def test_rewired_ipsi_preserves_output_and_hemispheres():
    measured = ShiuBrain(aminergic="zero")
    control = ShiuBrain(ablate="rewired_ipsi", aminergic="zero")
    # every presynaptic cell keeps its total synaptic output (count x sign)
    assert np.allclose(column_totals(control), column_totals(measured), rtol=1e-5)
    ann = annotations().drop_duplicates("root_id").set_index("root_id")
    side = ann["side"].reindex(measured.root_ids).fillna("na").astype(str).to_numpy()
    m, c = measured.W.tocoo(), control.W.tocoo()
    for s in ("left", "right"):
        onto_m = abs(m.data[side[m.row] == s]).sum()
        onto_c = abs(c.data[side[c.row] == s]).sum()
        assert abs(onto_c - onto_m) / onto_m < 1e-5, s
    # but the targets themselves moved
    same = (m.row == c.row).mean() if m.nnz == c.nnz else 0.0
    assert same < 0.05
    assert control.meta["ablate"] == "rewired_ipsi"
    assert set(control.meta["rewired_groups"]) >= {"left", "right"}


@pytest.mark.data
def test_weight_scale_scales_every_weight():
    base = ShiuBrain(aminergic="zero")
    scaled = ShiuBrain(aminergic="zero", weight_scale=0.7)
    assert np.allclose(scaled.W.data, base.W.data * np.float32(0.7))
    assert scaled.meta["weight_scale"] == 0.7
