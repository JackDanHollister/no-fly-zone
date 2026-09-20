"""Structural audit of the FlyWire v783 graph as loaded by brain_lif: who
feeds the readout descending neurons and the lobula relays, and how many
synaptic hops they are from each retina stage.

For each target type: top presynaptic cell types by summed synapse count
(with net sign), and the share of input synapses by super_class. Then BFS hop
distances from source stages to targets over edges with >= k synapses, once
over all edges and once over excitatory edges only (an inhibitory-only path
cannot propagate spikes in a LIF). Output: out/audit_dn_inputs.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from noflyzone.brain_lif import SHIU, ShiuBrain, annotations
from noflyzone.provenance import dump_json

OUT = Path(__file__).resolve().parent.parent / "out"
T4 = ["T4a", "T4b", "T4c", "T4d"]
T5 = ["T5a", "T5b", "T5c", "T5d"]
TARGETS = ["DNa02", "DNg100", "DNp01", "oviDN*", "DNa01", "DNa03", "DNa10",
           "DNae002", "DNb05", "DNp04", "DNp11", "LC10a", "LC11", "LC4",
           "LPLC2", "LC6", "T4a", "T5a", "HSE"]
SOURCES = {"R1-6": ["R1-6"], "L1+L2+L3": ["L1", "L2", "L3"],
           "Mi1+Tm3": ["Mi1", "Tm3"], "Tm1+Tm2+Tm4+Tm9": ["Tm1", "Tm2", "Tm4", "Tm9"],
           "T4+T5": T4 + T5, "LC10a": ["LC10a"], "LC11": ["LC11"],
           "LC4": ["LC4"], "LPLC2": ["LPLC2"]}
HOP_TARGETS = ["DNa02", "DNg100", "DNp01", "oviDN*", "LC10a", "LC11", "LC4",
               "LPLC2", "T4a", "T5a"]


def main() -> None:
    brain = ShiuBrain()
    n = brain.n
    ann = annotations().drop_duplicates("root_id").set_index("root_id")
    ann = ann.reindex(brain.root_ids)
    ct = ann["cell_type"].astype(str).to_numpy()
    side = ann["side"].astype(str).to_numpy()
    sclass = ann["super_class"].astype(str).to_numpy()
    con = pd.read_parquet(SHIU / "Connectivity_783.parquet",
                          columns=["Presynaptic_Index", "Postsynaptic_Index",
                                   "Excitatory x Connectivity"])
    pre = con["Presynaptic_Index"].to_numpy(np.int64)
    post = con["Postsynaptic_Index"].to_numpy(np.int64)
    w = con["Excitatory x Connectivity"].to_numpy(np.float32)
    cnt = np.abs(w)

    def cells(pattern, s=None):
        if pattern.endswith("*"):
            m = np.char.startswith(ct.astype(str), pattern[:-1])
        else:
            m = ct == pattern
        if s is not None:
            m &= side == s
        return np.flatnonzero(m)

    report = {"targets": {}, "hops": {}}
    for t in TARGETS:
        for s in (["left", "right"] if t in ("DNa02", "LC10a", "LC11", "HSE") else [None]):
            idx = cells(t, s)
            if idx.size == 0:
                continue
            m = np.isin(post, idx)
            sub = pd.DataFrame({"pre_type": ct[pre[m]], "pre_side": side[pre[m]],
                                "pre_class": sclass[pre[m]], "cnt": cnt[m], "w": w[m]})
            total = sub["cnt"].sum()
            by_type = sub.groupby("pre_type").agg(syn=("cnt", "sum"), net=("w", "sum"),
                                                  cells=("cnt", "size")).sort_values("syn", ascending=False)
            by_class = (sub.groupby("pre_class")["cnt"].sum() / total).sort_values(ascending=False)
            ipsi = float(sub.loc[sub["pre_side"] == (s or ""), "cnt"].sum() / total) if s else None
            name = f"{t}_{s[0].upper()}" if s else t
            report["targets"][name] = {
                "n_cells": int(idx.size), "input_synapses": float(total),
                "ipsilateral_share": ipsi,
                "by_super_class": {k: round(float(v), 3) for k, v in by_class.items()},
                "top_pre_types": [{"type": k, "syn": float(r.syn), "net_sign": float(r.net),
                                   "edges": int(r.cells)} for k, r in by_type.head(20).iterrows()],
            }
            top = ", ".join(f"{k} {r.syn:.0f}{'+' if r.net > 0 else '-'}"
                            for k, r in by_type.head(12).iterrows())
            cls = ", ".join(f"{k} {v:.0%}" for k, v in by_class.head(4).items())
            print(f"\n{name}: {idx.size} cells, {total:.0f} input synapses"
                  + (f", ipsilateral {ipsi:.0%}" if ipsi is not None else ""))
            print(f"  classes: {cls}")
            print(f"  top pre types: {top}")

    # hop distances
    print("\nBFS hops (edges >= k synapses); 'exc' = excitatory edges only")
    hop_rows = []
    for k in (1, 3):
        keep = cnt >= k
        adj_all = sparse.csr_matrix((np.ones(keep.sum(), np.int8), (pre[keep], post[keep])),
                                    shape=(n, n))
        keep_exc = keep & (w > 0)
        adj_exc = sparse.csr_matrix((np.ones(keep_exc.sum(), np.int8),
                                     (pre[keep_exc], post[keep_exc])), shape=(n, n))
        tgt_sets = {t: cells(t) for t in HOP_TARGETS}
        for sname, stypes in SOURCES.items():
            src = np.flatnonzero(np.isin(ct, stypes))
            for label, adj in (("all", adj_all), ("exc", adj_exc)):
                reached = np.zeros(n, bool)
                reached[src] = True
                frontier = reached.copy()
                first = {t: (0 if reached[i].any() else None) for t, i in tgt_sets.items()}
                for hop in range(1, 8):
                    nxt = (adj.T @ frontier.astype(np.int32)) > 0
                    frontier = nxt & ~reached
                    reached |= nxt
                    for t, i in tgt_sets.items():
                        if first[t] is None and reached[i].any():
                            first[t] = hop
                    if not frontier.any():
                        break
                hop_rows.append({"k": k, "edges": label, "source": sname, "hops": first})
                report["hops"][f"k{k}_{label}_{sname}"] = first
    hdr = "k  edges source            " + "".join(f"{t:>8}" for t in HOP_TARGETS)
    print(hdr)
    for r in hop_rows:
        cells_ = "".join(f"{(r['hops'][t] if r['hops'][t] is not None else '-'):>8}" for t in HOP_TARGETS)
        print(f"{r['k']}  {r['edges']:5s} {r['source']:18s}{cells_}")
    OUT.mkdir(exist_ok=True)
    dump_json(OUT / "audit_dn_inputs.json", report, brain=brain)
    print(f"\nwrote {OUT / 'audit_dn_inputs.json'}")


if __name__ == "__main__":
    main()
