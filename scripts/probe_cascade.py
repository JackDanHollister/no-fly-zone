"""Where does visual drive die? Stimulate one cell type (left eye) with a
Poisson rate, run the whole brain for a short window on the chosen backend,
and report the mean rate of every downstream stage plus the descending
neurons that respond, split by side so lateralisation is visible.

Reads nothing from the game; this is the connectome alone. Output:
out/probe_cascade.json and a table on stdout.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from noflyzone.brain_backend import BACKENDS, describe, make_brain
from noflyzone.brain_lif import DT, annotations

OUT = Path(__file__).resolve().parent.parent / "out"

T4 = ["T4a", "T4b", "T4c", "T4d"]
T5 = ["T5a", "T5b", "T5c", "T5d"]
STIM_SETS = [
    ("R1-6",), ("R7", "R8"), ("L1", "L2", "L3"), ("L1",), ("L2",), ("L3",),
    ("Mi1",), ("Tm3",), ("Mi1", "Tm3"), ("Tm1", "Tm2"),
    ("Tm1", "Tm2", "Tm4", "Tm9"), ("Mi1", "Tm3", "Tm1", "Tm2", "Tm4", "Tm9"),
    tuple(T4), tuple(T5), tuple(T4 + T5),
    ("LC4",), ("LPLC2",), ("LC6",), ("LC10a",), ("LC11",), ("LC12",),
    ("LC15",), ("LC16",), ("LC17",), ("LC18",), ("LC21",), ("LC22",),
    ("HSE", "HSN", "HSS"),
]
MONITOR = ["L1", "L2", "L3", "L4", "L5", "C2", "C3", "T1",
           "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9",
           "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
           "HSE", "HSN", "HSS", "VS1", "VS2", "VS3",
           "LC4", "LC6", "LC9", "LC10a", "LC11", "LC12", "LC15", "LC16",
           "LC17", "LC18", "LC21", "LC22", "LPLC1", "LPLC2", "LPLC4", "LLPC1",
           "PFL3", "DNa01", "DNa02", "DNa03", "DNa10", "DNg100",
           "DNp01", "DNp02", "DNp03", "DNp06", "DNp09", "DNp11",
           "oviDNa_a", "oviDNb"]
CHAIN = ["L1", "L2", "L3", "Mi1", "Tm3", "Tm1", "Tm2", "T4a", "T5a", "HSE",
         "LC4", "LPLC2", "LC10a", "LC11", "DNa02", "DNg100", "DNp01"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, default="auto")
    ap.add_argument("--rates", type=float, nargs="+", default=[150.0, 300.0])
    ap.add_argument("--seconds", type=float, default=0.3)
    ap.add_argument("--stim", nargs="*", default=None,
                    help="override stimulus sets, e.g. Tm5f 'T2+T3' LC9")
    ap.add_argument("--out", default="probe_cascade.json")
    args = ap.parse_args()
    stim_sets = [tuple(s.split("+")) for s in args.stim] if args.stim else STIM_SETS

    brain = make_brain(args.backend)
    print(f"backend {describe(brain)}")
    ann = annotations().drop_duplicates("root_id").set_index("root_id")
    ann = ann.reindex(brain.root_ids)
    ct = ann["cell_type"].astype(str).to_numpy()
    side = ann["side"].astype(str).to_numpy()
    sclass = ann["super_class"].astype(str).to_numpy()

    def cells(types, s=None):
        m = np.isin(ct, list(types))
        if s is not None:
            m &= side == s
        return np.flatnonzero(m)

    mon = {t: {s: cells([t], s) for s in ("left", "right")} for t in MONITOR}
    dn_mask = sclass == "descending"
    dn_types = sorted(set(ct[dn_mask]))
    dn_idx = {t: {s: cells([t], s) for s in ("left", "right")} for t in dn_types}
    steps = round(args.seconds / DT)
    results = []
    t_all = time.time()
    hdr = "stim".ljust(28) + "Hz ".rjust(5) + "".join(f"{c:>7}" for c in CHAIN) + "  DN>5Hz  top DN types (L/R Hz)"
    print(hdr)
    for stim in stim_sets:
        src = cells(stim, "left")
        if src.size == 0:
            print(f"{'+'.join(stim):28s} no left cells found")
            continue
        for rate in args.rates:
            brain.reset()
            brain.set_poisson({int(i): rate for i in src})
            counts = brain.run(args.seconds)
            hz = counts / args.seconds
            row = {"stim": "+".join(stim), "rate": rate, "n_src": int(src.size),
                   "src_mean_hz": float(hz[src].mean()),
                   "monitor": {t: {s: float(hz[mon[t][s]].mean()) if mon[t][s].size else None
                                   for s in ("left", "right")} for t in MONITOR},
                   "n_active": int((counts > 0).sum())}
            dn_rates = []
            for t in dn_types:
                l = float(hz[dn_idx[t]["left"]].mean()) if dn_idx[t]["left"].size else 0.0
                r = float(hz[dn_idx[t]["right"]].mean()) if dn_idx[t]["right"].size else 0.0
                if max(l, r) > 0:
                    dn_rates.append((t, l, r))
            dn_rates.sort(key=lambda x: -max(x[1], x[2]))
            row["dn"] = [{"type": t, "left": l, "right": r} for t, l, r in dn_rates[:25]]
            row["n_dn_types_over_5hz"] = sum(1 for _, l, r in dn_rates if max(l, r) > 5)
            results.append(row)
            chain = "".join(f"{(row['monitor'][c]['left'] or 0):7.1f}" for c in CHAIN)
            top = ", ".join(f"{t} {l:.0f}/{r:.0f}" for t, l, r in dn_rates[:5])
            print(f"{row['stim']:28s}{rate:5.0f}{chain}  {row['n_dn_types_over_5hz']:6d}  {top}",
                  flush=True)
    print(f"\n{len(results)} runs in {time.time() - t_all:.0f} s")
    OUT.mkdir(exist_ok=True)
    (OUT / args.out).write_text(json.dumps(
        {"seconds": args.seconds, "backend": describe(brain), "chain": CHAIN,
         "results": results}, indent=1))
    print(f"wrote {OUT / args.out}")


if __name__ == "__main__":
    main()
