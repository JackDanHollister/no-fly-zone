"""What a plasticity run did to the wiring: per postsynaptic cell type, how
many synapses moved, which way, and by how much, for the readout pools first.

Usage: python scripts/audit_learned_weights.py out/plasticity_<tag>_weights.npz [...]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from noflyzone.brain_lif import ShiuBrain, annotations

POOLS = ("DNa02", "DNg100", "DNp01", "oviDN", "DNp04", "DNb05", "DNp02", "DNp20", "DNp07", "DNg33", "DNp12")


def main() -> None:
    root_ids = ShiuBrain.root_ids_only() if hasattr(ShiuBrain, "root_ids_only") else None
    ann = annotations().drop_duplicates("root_id").set_index("root_id")
    for path in sys.argv[1:]:
        z = np.load(path)
        w0, w, post, pre = z["w0"], z["w"], z["post"], z["pre"]
        if root_ids is None:
            from noflyzone.brain_lif import ShiuBrain as SB
            root_ids = SB(aminergic="zero").root_ids
        ct = ann["cell_type"].reindex(root_ids).fillna("").astype(str).to_numpy()
        side = ann["side"].reindex(root_ids).fillna("").astype(str).to_numpy()
        live = w0 != 0
        ratio = np.where(live, w / np.where(live, w0, 1.0), 1.0)
        df = pd.DataFrame({"post": ct[post], "side": side[post], "pre": ct[pre],
                           "ratio": ratio, "live": live, "moved": live & (ratio != 1.0),
                           "up": ratio > 1.0, "down": ratio < 1.0,
                           "abs_w0": np.abs(w0), "abs_w": np.abs(w)})
        print(f"\n{Path(path).name}: {df.moved.sum():,} of {df.live.sum():,} live synapses moved "
              f"({df.up.sum():,} up, {df.down.sum():,} down); mean ratio of moved "
              f"{df.ratio[df.moved].mean():.2f}, total |w| {df.abs_w.sum() / df.abs_w0.sum():.3f} x measured")
        g = df.groupby("post").agg(n=("ratio", "size"), moved=("moved", "sum"), up=("up", "sum"),
                                   down=("down", "sum"), mean_ratio=("ratio", "mean"),
                                   input_w0=("abs_w0", "sum"), input_w=("abs_w", "sum"))
        g["input_x"] = g.input_w / g.input_w0
        g = g.sort_values("moved", ascending=False)
        print("readout pools (total synaptic input relative to measured):")
        for t in POOLS:
            if t in g.index:
                r = g.loc[t]
                print(f"  {t:8s} {int(r.moved):5d}/{int(r.n):5d} moved, {int(r.up):4d} up {int(r.down):4d} down, "
                      f"input x{r.input_x:.3f}")
        print("most changed postsynaptic types:")
        print(g.head(12)[["n", "moved", "up", "down", "input_x"]].to_string())
        print("most changed presynaptic types (by count moved):")
        print(df[df.moved].groupby("pre").agg(moved=("moved", "size"), mean_ratio=("ratio", "mean"))
              .sort_values("moved", ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
