"""DN screen: which descending-neuron types lateralise consistently with the
eye a small moving object is shown to, and which respond to loom but not to
small objects? Uses the game's injection path (a named map) and the synthetic
patterns from probe_frontend, so the answer is specific to what the loop can
actually deliver. Output: out/screen_dns_<map>.json plus tables.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_frontend import DT_GAME, stimuli  # noqa: E402

from noflyzone.brain_backend import BACKENDS, describe, make_brain  # noqa: E402
from noflyzone.brain_connectome import ConnectomeBrain  # noqa: E402
from noflyzone.brain_lif import annotations  # noqa: E402
from noflyzone.retina import HexRetina  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"
CURRENT = ["DNa02", "DNg100", "DNp01", "oviDNa_a", "oviDNb", "DNa01", "DNa03",
           "DNa10", "DNb05", "DNp04", "DNp11", "DNae002"]


def run(brain, frames, eye, xy):
    dark = np.zeros(len(xy), np.float32)
    brain.reset()
    total = np.zeros(brain.brain.n, np.int64)
    for lum in frames:
        lit, unlit = np.stack([lum, lum], 1), np.stack([dark, dark], 1)
        drives = {"left": lit, "right": unlit} if eye == "left" else {"left": unlit, "right": lit}
        brain.step(drives, DT_GAME)
        total += brain.window_counts
    return total / (len(frames) * DT_GAME)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, default="auto")
    ap.add_argument("--injection", nargs="+", default=["v3"])
    args = ap.parse_args()
    retina = HexRetina()
    xy = retina.centers_xy.astype(np.float64)
    from scipy.spatial import cKDTree
    spacing = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    core = make_brain(args.backend)
    ann = annotations().drop_duplicates("root_id").set_index("root_id").reindex(core.root_ids)
    ct = ann["cell_type"].astype(str).to_numpy()
    side = ann["side"].astype(str).to_numpy()
    dn = ann["super_class"].astype(str).to_numpy() == "descending"
    types = sorted(set(ct[dn]))
    idx = {t: {s: np.flatnonzero((ct == t) & (side == s)) for s in ("left", "right")} for t in types}
    stims = stimuli(xy, spacing)
    small = stims["small_disc_LtoR"] + stims["small_disc_RtoL"]
    conds = {"dark": ([np.zeros(len(xy), np.float32)] * 12, "left"),
             "small@L": (small, "left"), "small@R": (small, "right"),
             "loom@L": (stims["loom"], "left"), "loom@R": (stims["loom"], "right"),
             "bar@L": (stims["bar_LtoR"] + stims["bar_RtoL"], "left"),
             "bar@R": (stims["bar_LtoR"] + stims["bar_RtoL"], "right")}
    for inj in args.injection:
        brain = ConnectomeBrain(retina=retina, brain=core, injection=inj)
        print(f"\n########## injection {inj} | {describe(core)}")
        hz = {c: run(brain, f, e, xy) for c, (f, e) in conds.items()}
        rate = lambda t, s, c: float(hz[c][idx[t][s]].mean()) if idx[t][s].size else np.nan
        rows = []
        for t in types:
            if not (idx[t]["left"].size and idx[t]["right"].size):
                continue
            r = {c: {s: rate(t, s, c) for s in ("left", "right")} for c in conds}
            ipsi = (r["small@L"]["left"] + r["small@R"]["right"]) / 2
            contra = (r["small@L"]["right"] + r["small@R"]["left"]) / 2
            dark = (r["dark"]["left"] + r["dark"]["right"]) / 2
            li = (ipsi - contra) / (ipsi + contra + 1e-9)
            consistent = np.sign(r["small@L"]["left"] - r["small@L"]["right"]) == \
                np.sign(r["small@R"]["right"] - r["small@R"]["left"]) != 0
            loom = (r["loom@L"]["left"] + r["loom@L"]["right"] + r["loom@R"]["left"] + r["loom@R"]["right"]) / 4
            smallr = (ipsi + contra) / 2
            bar_ipsi = (r["bar@L"]["left"] + r["bar@R"]["right"]) / 2
            bar_contra = (r["bar@L"]["right"] + r["bar@R"]["left"]) / 2
            rows.append({"type": t, "n": int(idx[t]["left"].size + idx[t]["right"].size),
                         "dark": dark, "small_ipsi": ipsi, "small_contra": contra, "LI": li,
                         "consistent": bool(consistent), "loom": loom, "small": smallr,
                         "loom_sel": loom / (smallr + 1.0), "bar_ipsi": bar_ipsi,
                         "bar_contra": bar_contra, "raw": r})
        resp = [x for x in rows if max(x["small_ipsi"], x["small_contra"]) - x["dark"] > 5 and x["consistent"]]
        score = lambda x: abs(x["LI"]) * max(x["small_ipsi"], x["small_contra"])
        ipsi_t = sorted([x for x in resp if x["LI"] > 0.3], key=score, reverse=True)[:12]
        contra_t = sorted([x for x in resp if x["LI"] < -0.3], key=score, reverse=True)[:12]
        hdr = f"{'type':12s}{'n':>3} {'dark':>6} {'ipsi':>7} {'contra':>7} {'LI':>6} {'bar ipsi/contra':>16} {'loom':>6}"
        for label, lst in (("IPSILATERAL small-object DN types (turn-toward candidates if ipsi = same side as object)", ipsi_t),
                           ("CONTRALATERAL small-object DN types", contra_t)):
            print(f"\n{label}\n{hdr}")
            for x in lst:
                print(f"{x['type']:12s}{x['n']:3d} {x['dark']:6.1f} {x['small_ipsi']:7.1f} {x['small_contra']:7.1f} "
                      f"{x['LI']:+6.2f} {x['bar_ipsi']:7.1f}/{x['bar_contra']:<7.1f} {x['loom']:6.1f}")
        esc = sorted([x for x in rows if x["loom"] - x["dark"] > 20 and x["loom_sel"] > 3],
                     key=lambda x: x["loom"], reverse=True)[:12]
        print(f"\nLOOM-selective DN types (loom > 3x small object)\n{'type':12s}{'n':>3} {'dark':>6} {'small':>7} {'loom':>7} {'sel':>6}")
        for x in esc:
            print(f"{x['type']:12s}{x['n']:3d} {x['dark']:6.1f} {x['small']:7.1f} {x['loom']:7.1f} {x['loom_sel']:6.1f}")
        print(f"\ncurrent and candidate readouts\n{hdr}")
        for x in rows:
            if x["type"] in CURRENT:
                print(f"{x['type']:12s}{x['n']:3d} {x['dark']:6.1f} {x['small_ipsi']:7.1f} {x['small_contra']:7.1f} "
                      f"{x['LI']:+6.2f} {x['bar_ipsi']:7.1f}/{x['bar_contra']:<7.1f} {x['loom']:6.1f}"
                      + ("" if x["consistent"] else "  (inconsistent across eyes)"))
        OUT.mkdir(exist_ok=True)
        (OUT / f"screen_dns_{inj}.json").write_text(json.dumps(
            {"injection": inj, "rows": rows}, indent=1, default=float))
    print("\nwrote out/screen_dns_<map>.json")


if __name__ == "__main__":
    main()
