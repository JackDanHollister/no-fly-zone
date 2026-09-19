"""Convert the MaleCNS v1.0 flat connectome (Janelia/Google, CC-BY 4.0) into
the three files brain_lif.ShiuBrain reads for any dataset: an index CSV
(one row per neuron, fixing the index order), a signed connectivity parquet
(Presynaptic_Index, Postsynaptic_Index, "Excitatory x Connectivity" = sign x
synapse count) and an annotation TSV in the FlyWire annotation vocabulary
(root_id, cell_type, side, super_class, pos_x/y/z, top_nt, known_nt).

Decisions, all recorded in data/malecns/cache/manifest.json:
- neurons kept: every body with a non-empty superclass (166,700; the
  photoreceptors are not all "Traced", so status is not used);
- edges: every (pre, post) pair between kept bodies, duplicates summed,
  no synapse-count threshold (Shiu-compatible), pairs involving fragments
  dropped;
- sign per presynaptic body from consensus_nt: acetylcholine +1; gaba,
  glutamate, histamine -1; dopamine, octopamine, serotonin +1 (Shiu
  convention; `aminergic="zero"` removes them at load); unclear +1, counted;
- super_class mapped onto the FlyWire vocabulary the code tests against
  ("descending", "sensory", "sensory_ascending", ...); the photoreceptor
  type "R1-R6" is renamed "R1-6" so the injection maps apply unchanged;
- side from somaSide (L/R/M -> left/right/center), rootSide when soma is
  missing, else "na"; positions are somaLocation voxels x 8 nm.
Inputs are verified against pinned SHA-256 hashes before anything is read.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc

DATA = Path(__file__).resolve().parents[1] / "data" / "malecns"
CACHE = DATA / "cache"
FILES = {
    "annotations": ("body-annotations-male-cns-v1.0-minconf-0.5.feather",
                    "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2"),
    "neurotransmitters": ("body-neurotransmitters-male-cns-v1.0.feather",
                          "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621"),
    "weights": ("connectome-weights-male-cns-v1.0-minconf-0.5.feather",
                "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1"),
}
SUPERCLASS_MAP = {
    "descending_neuron": "descending", "descending_neuron_tbc": "descending",
    "ascending_neuron": "ascending", "visual_projection": "visual_projection",
    "visual_projection_tbc": "visual_projection", "visual_centrifugal": "visual_centrifugal",
    "ol_intrinsic": "optic", "cb_intrinsic": "central", "vnc_intrinsic": "vnc_intrinsic",
    "ol_sensory": "sensory", "cb_sensory": "sensory", "vnc_sensory": "sensory",
    "cb_sensory_tbc": "sensory", "vnc_sensory_tbc": "sensory", "sensory_descending": "sensory",
    "sensory_ascending": "sensory_ascending", "sensory_ascending_tbc": "sensory_ascending",
    "vnc_motor": "motor", "cb_motor": "motor", "vnc_efferent": "efferent", "cb_efferent": "efferent",
    "efferent_ascending": "efferent", "efferent_descending": "efferent",
    "cb_endocrine": "endocrine", "vnc_endocrine": "endocrine", "ENS": "endocrine", "vnc_tbc": "vnc_intrinsic",
}
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
        "dopamine": 1, "octopamine": 1, "serotonin": 1}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    for key, (name, digest) in FILES.items():
        got = sha256(DATA / name)
        assert got == digest, f"{name}: hash {got} != pinned {digest}"
        print(f"verified {name}")
    CACHE.mkdir(exist_ok=True)

    ann = pf.read_table(DATA / FILES["annotations"][0]).to_pandas()
    keep = ann["superclass"].notna() & (ann["superclass"].astype(str) != "")
    ann = ann[keep].copy()
    ann = ann.drop_duplicates("bodyId").sort_values("bodyId").reset_index(drop=True)
    n = len(ann)
    print(f"neurons kept: {n}")

    nt = pf.read_table(DATA / FILES["neurotransmitters"][0], columns=["body", "consensus_nt", "ground_truth"]).to_pandas()
    nt = nt.drop_duplicates("body").set_index("body")
    top = nt["consensus_nt"].reindex(ann["bodyId"]).fillna("unclear").astype(str).str.lower().to_numpy()
    known = nt["ground_truth"].reindex(ann["bodyId"]).fillna("").astype(str).str.lower().to_numpy()

    side_raw = ann["somaSide"].astype("string").fillna(ann["rootSide"].astype("string")).fillna("")
    side = side_raw.map({"L": "left", "R": "right", "M": "center"}).fillna("na").to_numpy()
    pos = np.full((n, 3), np.nan)
    has = ann["somaLocation"].notna().to_numpy()
    pos[has] = np.stack(ann.loc[has, "somaLocation"].to_numpy()) * 8.0
    cell_type = ann["type"].astype("string").fillna("").to_numpy().astype(object)
    cell_type = np.where(cell_type == "R1-R6", "R1-6", cell_type)
    super_class = ann["superclass"].astype(str).map(SUPERCLASS_MAP).fillna("other").to_numpy()
    out_ann = pd.DataFrame({
        "root_id": ann["bodyId"].to_numpy(np.int64), "cell_type": cell_type, "side": side,
        "super_class": super_class, "pos_x": pos[:, 0], "pos_y": pos[:, 1], "pos_z": pos[:, 2],
        "top_nt": top, "known_nt": known, "flywire_type": ann["flywireType"].astype("string").fillna("").to_numpy(),
        "hex1": ann["assignedOlHex1"].to_numpy(), "hex2": ann["assignedOlHex2"].to_numpy(),
    })
    out_ann.to_csv(CACHE / "annotations_malecns.tsv", sep="\t", index=False)
    pd.DataFrame({"root_id": out_ann["root_id"]}).to_csv(CACHE / "Completeness_malecns.csv", index=False)

    ids = out_ann["root_id"].to_numpy(np.int64)
    sign = np.array([SIGN.get(t, 1) for t in top], np.int8)
    n_unclear = int((~np.isin(top, list(SIGN))).sum())
    reader = ipc.open_file(pa.memory_map(str(DATA / FILES["weights"][0])))
    pres, posts, ws = [], [], []
    total = 0
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        pre = b.column("body_pre").to_numpy(); post = b.column("body_post").to_numpy(); w = b.column("weight").to_numpy()
        total += len(pre)
        ip = np.searchsorted(ids, pre); jp = np.searchsorted(ids, post)
        ip = np.minimum(ip, n - 1); jp = np.minimum(jp, n - 1)
        ok = (ids[ip] == pre) & (ids[jp] == post)
        pres.append(ip[ok].astype(np.int32)); posts.append(jp[ok].astype(np.int32)); ws.append(w[ok].astype(np.int64))
        if (i + 1) % 500 == 0:
            print(f"  batch {i + 1}/{reader.num_record_batches}", flush=True)
    pre = np.concatenate(pres); post = np.concatenate(posts); w = np.concatenate(ws)
    print(f"weight rows {total}; between kept bodies {len(pre)}")
    df = pd.DataFrame({"pre": pre, "post": post, "w": w}).groupby(["pre", "post"], sort=True, as_index=False)["w"].sum()
    signed = (df["w"].to_numpy(np.int64) * sign[df["pre"].to_numpy()]).astype(np.int64)
    con = pd.DataFrame({"Presynaptic_Index": df["pre"].to_numpy(np.int64),
                        "Postsynaptic_Index": df["post"].to_numpy(np.int64),
                        "Excitatory x Connectivity": signed})
    con.to_parquet(CACHE / "Connectivity_malecns.parquet", index=False)
    manifest = {
        "source_files": {k: {"name": v[0], "sha256": v[1]} for k, v in FILES.items()},
        "neurons": n, "edges_directed_pairs": int(len(con)), "synapses": int(df["w"].sum()),
        "pairs_weight_1": int((df["w"] == 1).sum()), "unclear_transmitter_bodies": n_unclear,
        "sign_convention": SIGN, "superclass_map": SUPERCLASS_MAP,
        "super_class_counts": {k: int(v) for k, v in pd.Series(super_class).value_counts().items()},
        "side_counts": {k: int(v) for k, v in pd.Series(side).value_counts().items()},
        "renamed_types": {"R1-R6": "R1-6"}, "position_units": "nm (somaLocation voxels x 8)",
        "cell_counts": {t: int((cell_type == t).sum()) for t in
                        ["Mi1", "Tm1", "Tm2", "Tm3", "Tm4", "Tm9", "T3", "R1-6", "LC4", "LPLC2", "LC11",
                         "DNa02", "DNg100", "DNp01", "DNp07", "DNb05", "DNg52", "DNb06", "DNp12", "pIP10", "MN9"]},
    }
    (CACHE / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("superclass_map", "sign_convention")}, indent=1))


if __name__ == "__main__":
    main()
