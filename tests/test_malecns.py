"""The MaleCNS v1.0 conversion loads through the same code path as FlyWire
and reproduces the counts pinned in data/malecns/cache/manifest.json."""

import json
from pathlib import Path

import numpy as np
import pytest

from noflyzone.brain_lif import DATASETS, ShiuBrain, annotations

CACHE = Path(__file__).resolve().parents[1] / "data" / "malecns" / "cache"

pytestmark = pytest.mark.skipif(not (CACHE / "manifest.json").exists(),
                                reason="MaleCNS not converted (scripts/build_malecns.py)")


@pytest.mark.data
def test_malecns_counts_and_vocabulary():
    manifest = json.loads((CACHE / "manifest.json").read_text())
    ann = annotations("malecns")
    assert len(ann) == manifest["neurons"] == 166700
    assert set(ann["super_class"]) >= {"descending", "sensory", "sensory_ascending", "optic"}
    assert (ann["cell_type"] == "R1-6").sum() == manifest["cell_counts"]["R1-6"]
    assert (ann["cell_type"] == "pIP10").sum() == 2
    assert (ann["cell_type"] == "").any()  # untyped bodies are empty strings, not NaN


@pytest.mark.data
def test_malecns_loads_and_escapes():
    brain = ShiuBrain(version="malecns", aminergic="zero", sensory_input="external_only")
    manifest = json.loads((CACHE / "manifest.json").read_text())
    assert brain.n == manifest["neurons"]
    assert brain.W.nnz <= manifest["edges_directed_pairs"]
    assert brain.meta["version"] == "malecns"
    brain.reset()
    assert brain.run(0.05).sum() == 0
    lc4 = brain.indices_for_type("LC4")
    gf = np.concatenate([brain.indices_for_type("DNp01", side=s) for s in ("left", "right")])
    assert lc4.size > 100 and gf.size == 2
    brain.reset(); brain.set_poisson({int(i): 100.0 for i in lc4})
    assert brain.run(0.2)[gf].max() > 0
    assert DATASETS["malecns"][1].exists()
