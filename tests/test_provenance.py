"""Every artefact carries the commit, the settings and the brain that made it."""

import json

import numpy as np

from noflyzone import provenance


def test_stamp_has_the_parts_and_a_stable_id():
    a = provenance.stamp(args={"seed": 1, "path": object()}, note="x")
    b = provenance.stamp(args={"seed": 1, "path": object()})
    assert a["git"]["commit"] is None or len(a["git"]["commit"]) == 40
    assert "numpy" in a["packages"] and a["python"] and a["host"]
    assert a["note"] == "x" and a["args"]["seed"] == 1
    assert a["id"] == b["id"]                       # same code, same settings, same id
    assert a["id"] != provenance.stamp(args={"seed": 2})["id"]


def test_dump_json_dict_inline_and_list_sidecar(tmp_path):
    p = provenance.dump_json(tmp_path / "a.json", {"x": 1}, args={"k": "v"})
    d = json.loads(p.read_text())
    assert d["x"] == 1 and d["provenance"]["args"]["k"] == "v"
    p = provenance.dump_json(tmp_path / "b.json", [1, 2, 3])
    assert json.loads(p.read_text()) == [1, 2, 3]
    assert "id" in json.loads((tmp_path / "b.provenance.json").read_text())


def test_savez_and_attach(tmp_path):
    provenance.savez(tmp_path / "w.npz", args={"eta": 2e-5}, w=np.arange(3))
    z = np.load(tmp_path / "w.npz")
    assert z["w"].tolist() == [0, 1, 2]
    assert json.loads(str(z["provenance"]))["args"]["eta"] == 2e-5
    d = provenance.attach({"net": 1}, update=7)
    assert d["net"] == 1 and d["provenance"]["update"] == 7
