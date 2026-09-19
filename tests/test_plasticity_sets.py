"""The plastic edge sets are what they claim, the controls are size-matched
and disjoint, and learned weights round-trip through the brain."""

import numpy as np
import pytest

from noflyzone.brain_backend import make_brain
from noflyzone.brain_lif import annotations
from noflyzone.plasticity import MBPlasticity, VISUAL_CLASSES


@pytest.mark.data
@pytest.mark.cuda
def test_descending_edge_sets_and_controls(tmp_path):
    brain = make_brain("cuda", aminergic="zero", sensory_input="external_only")
    ann = annotations().drop_duplicates("root_id").set_index("root_id").reindex(brain.root_ids)
    sc = ann["super_class"].fillna("").astype(str).to_numpy()
    vis = MBPlasticity(brain, dt=1 / 15, edge_set="vis_dn")
    assert vis.eligibility == "hebbian" and vis.scale == "relative" and vis.elig_cap == 50.0
    assert np.all(sc[vis.post] == "descending")
    assert np.all(np.isin(sc[vis.pre], VISUAL_CLASSES))
    assert vis.n_edges > 20000
    dn = MBPlasticity(brain, dt=1 / 15, edge_set="dn_in")
    assert np.all(sc[dn.post] == "descending") and dn.n_edges > vis.n_edges
    ctrl = MBPlasticity(brain, dt=1 / 15, edge_set="vis_dn_shuffled", seed=3)
    assert ctrl.n_edges == vis.n_edges
    assert not np.any(sc[ctrl.post] == "descending")
    assert not np.any(np.isin(sc[ctrl.pre], VISUAL_CLASSES))
    dn_ctrl = MBPlasticity(brain, dt=1 / 15, edge_set="dn_in_shuffled", seed=3)
    assert dn_ctrl.n_edges == dn.n_edges
    assert not np.any(sc[dn_ctrl.post] == "descending")
    # the mushroom-body rule keeps its 2026-09-14 defaults
    mb = MBPlasticity(brain, dt=1 / 15, edge_set="kc_mbon")
    assert mb.eligibility == "presynaptic" and mb.scale == "absolute" and not np.isfinite(mb.elig_cap)
    # a learned weight file installs into the brain, and rejects the wrong set
    w = vis.w0 * 1.5
    np.savez(tmp_path / "w.npz", pos=vis.pos, w0=vis.w0, w=w, pre=vis.pre, post=vis.post)
    vis.load_weights(tmp_path / "w.npz")
    assert np.allclose(brain.W.data[vis.pos], w)
    with pytest.raises(ValueError):
        dn.load_weights(tmp_path / "w.npz")


@pytest.mark.data
@pytest.mark.cuda
def test_relative_rule_is_bounded_and_signed():
    brain = make_brain("cuda", aminergic="zero", sensory_input="external_only")
    pl = MBPlasticity(brain, dt=1 / 15, edge_set="vis_dn", eta=1.0)
    counts = np.zeros((2, brain.n), np.int32)
    active = np.zeros(brain.n, bool)
    active[pl.pre[:100]] = True; active[pl.post[:100]] = True
    counts[:, active] = 5
    coactive = active[pl.pre] & active[pl.post]     # every edge with both ends active is eligible
    pl.trace_pre[:] = 0; pl.trace_post[:] = 0
    pl.da_mean = -1000.0                      # a centred dopamine term far above its mean
    pl.step(counts, kills=np.array([1, 0]), hits=np.array([0, 0]))
    live = pl.w0 != 0        # aminergic edges carry no fast weight and are inert under the bounds
    ratio = np.where(live, pl.mag / np.where(live, np.abs(pl.w0), 1.0), 1.0)
    assert ratio.max() <= 2.0 + 1e-6 and ratio.min() >= 0.1 - 1e-6
    assert np.all(np.sign(pl.w) == np.sign(pl.w0))   # the rule scales, never flips
    assert np.all(ratio[coactive & live] > 1.0)       # rewarded co-active edges strengthened
    assert np.all(ratio[~coactive] == 1.0)            # nothing else touched
    assert live.sum() > 0.9 * live.size
