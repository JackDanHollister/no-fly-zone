"""Compare learned brains synapse by synapse (same edge set): which changes
separate one group from another, by postsynaptic and presynaptic type.

Usage: python scripts/compare_learned_pairs.py --a TAG TAG --b TAG TAG
(tags of out/plasticity_<tag>_weights.npz). Writes out/compare_<a>_vs_<b>.json.
"""

import argparse
import json

import numpy as np
import pandas as pd

from noflyzone.brain_lif import ShiuBrain, annotations
from noflyzone.provenance import dump_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--min-n", type=int, default=30)
    args = ap.parse_args()
    ann = annotations().drop_duplicates("root_id").set_index("root_id")
    brain = ShiuBrain(aminergic="zero")
    ct = ann["cell_type"].reindex(brain.root_ids).fillna("").astype(str).to_numpy()
    Z = {t: np.load(f"out/plasticity_{t}_weights.npz") for t in args.a + args.b}
    pos = Z[args.a[0]]["pos"]
    assert all(np.array_equal(Z[t]["pos"], pos) for t in Z), "different edge sets"
    w0 = Z[args.a[0]]["w0"]; live = w0 != 0
    R = {t: np.log(np.abs(Z[t]["w"][live]) / np.abs(w0[live])) for t in Z}
    pre, post = Z[args.a[0]]["pre"][live], Z[args.a[0]]["post"][live]
    A = np.mean([R[t] for t in args.a], axis=0); B = np.mean([R[t] for t in args.b], axis=0)
    C = np.corrcoef(np.stack([R[t] for t in Z]))
    print("correlation of log-ratios:", {f"{u}~{v}": round(float(C[i, j]), 2) for i, u in enumerate(Z) for j, v in enumerate(Z) if i < j})
    df = pd.DataFrame({"post": ct[post], "pre": ct[pre], "a": A, "b": B, "diff": A - B})
    out = {}
    for key in ("post", "pre"):
        g = df.groupby(key).agg(n=("diff", "size"), a=("a", "mean"), b=("b", "mean"), diff=("diff", "mean"))
        g = g[g.n >= args.min_n].sort_values("diff")
        print(f"\nby {key}synaptic type, mean log-ratio, groups a={args.a} b={args.b}; most negative and most positive differences:")
        print(pd.concat([g.head(6), g.tail(8)]).round(3).to_string())
        out[key] = {k: {c: float(v[c]) for c in ("n", "a", "b", "diff")} for k, v in g.iterrows()}
    dump_json(f"out/compare_{'+'.join(args.a)}_vs_{'+'.join(args.b)}.json", {"correlation": C.tolist(), "tags": list(Z), **out}, args=args)


if __name__ == "__main__":
    main()
