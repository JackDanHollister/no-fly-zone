"""Motion probe: can the connectome's own T4/T5 wiring compute direction?

Bars are moved across the LEFT eye in four directions through a named
injection map, and every T4 and T5 subtype, the HS/VS tangential cells and
the descending neurons are read out. The fly's T4a-d and T5a-d each prefer
one of four directions, and that preference is made by the spatial offsets
of their columnar inputs (Mi1/Tm3 versus Mi4/Mi9 for T4), which are in the
wiring; what the wiring cannot supply is the input kinetics, which is what
injection map v5 adds. Because the retina-to-column mapping is a PCA
heuristic whose orientation is uncertain, the test is orientation-free:
do the a and b subtypes prefer OPPOSITE horizontal directions, and c and d
opposite vertical ones, with the a/b and c/d pairs split along different
axes? A random rewiring inside the hemisphere keeps the cell types but not
the offsets, so it should not.

Output: out/probe_motion_<dataset>_<brain>_<map>.json and a table.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_frontend import DT_GAME  # noqa: E402

from noflyzone.brain_backend import BACKENDS, describe, make_brain  # noqa: E402
from noflyzone.brain_connectome import ConnectomeBrain  # noqa: E402
from noflyzone.retina import HexRetina  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"
T4 = ["T4a", "T4b", "T4c", "T4d"]
T5 = ["T5a", "T5b", "T5c", "T5d"]
TANGENTIAL = ["HSE", "HSN", "HSS", "VS1", "VS2", "VS3", "VS4", "H2", "CH", "LPi3-4", "LPi4-3"]
DIRS = ["LtoR", "RtoL", "DtoU", "UtoD"]


def parse(token):
    m = re.fullmatch(r"(measured|rewired|rewired_ipsi)(?:@(\d+))?(?:x([\d.]+))?", token)
    if not m:
        raise SystemExit(f"bad brain token {token!r}")
    name, seed, scale = m.groups()
    return {"ablate": "none" if name == "measured" else name,
            "ablate_seed": int(seed or 0), "weight_scale": float(scale or 1.0)}


def bars(xy, spacing, n=18, width=1.5, dark_frames=4):
    dark = np.zeros(len(xy), np.float32)
    xs = np.linspace(-0.9, 0.9, n)
    vbar = lambda x: (np.abs(xy[:, 0] - x) <= width * spacing).astype(np.float32)
    hbar = lambda y: (np.abs(xy[:, 1] - y) <= width * spacing).astype(np.float32)
    return {
        "LtoR": [dark] * dark_frames + [vbar(x) for x in xs] + [dark] * 2,
        "RtoL": [dark] * dark_frames + [vbar(x) for x in xs[::-1]] + [dark] * 2,
        "DtoU": [dark] * dark_frames + [hbar(y) for y in xs] + [dark] * 2,
        "UtoD": [dark] * dark_frames + [hbar(y) for y in xs[::-1]] + [dark] * 2,
    }


def run(brain, frames, xy, both_eyes=False):
    dark = np.zeros(len(xy), np.float32)
    brain.reset()
    total = np.zeros(brain.brain.n, np.int64)
    for lum in frames:
        lit = np.stack([lum, lum], 1)
        drives = {"left": lit, "right": lit if both_eyes else np.stack([dark, dark], 1)}
        brain.step(drives, DT_GAME)
        total += brain.window_counts
    return total / (len(frames) * DT_GAME)


def dsi(a, b):
    return (a - b) / (a + b + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, default="auto")
    ap.add_argument("--dataset", default="783")
    ap.add_argument("--brains", nargs="+", default=["measured", "rewired_ipsi@0"])
    ap.add_argument("--injection", nargs="+", default=["v4", "v5"])
    ap.add_argument("--aminergic", default="zero")
    ap.add_argument("--sensory", default="external_only")
    ap.add_argument("--speeds", type=int, nargs="+", default=[18],
                    help="frames per sweep (fewer = faster bar)")
    ap.add_argument("--yaw", action="store_true",
                    help="drive BOTH eyes with the same sweep (the screen rotating past the "
                         "fly, yaw optic flow) and report left/right descending rates: the "
                         "optomotor steering signal")
    args = ap.parse_args()
    retina = HexRetina()
    xy = retina.centers_xy.astype(np.float64)
    from scipy.spatial import cKDTree
    spacing = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))

    table = []
    for token in args.brains:
        core = make_brain(args.backend, version=args.dataset, aminergic=args.aminergic,
                          sensory_input=args.sensory, **parse(token))
        ann = core.annotations().drop_duplicates("root_id").set_index("root_id").reindex(core.root_ids)
        ct = ann["cell_type"].astype(str).to_numpy(); side = ann["side"].astype(str).to_numpy()
        dn = ann["super_class"].astype(str).to_numpy() == "descending"
        left = lambda t: np.flatnonzero((ct == t) & (side == "left"))
        dn_types = sorted(t for t in set(ct[dn]) if t and left(t).size)
        right = lambda t: np.flatnonzero((ct == t) & (side == "right"))
        for inj in args.injection:
            brain = ConnectomeBrain(retina=retina, brain=core, injection=inj)
            if args.yaw:
                for n in args.speeds:
                    st = bars(xy, spacing, n=n)
                    hz = {d: run(brain, st[d], xy, both_eyes=True) for d in ("LtoR", "RtoL")}
                    print(f"\n########## YAW {args.dataset} {token} {inj} n={n}: {describe(core)}")
                    print(f"{'type':10s} {'L:LtoR':>7} {'L:RtoL':>7} {'R:LtoR':>7} {'R:RtoL':>7} {'yaw_L':>7} {'yaw_R':>7} {'flip':>5}")
                    rows = {}
                    for t in ["HSE", "HSN", "HSS", "H2", "DNa02", "DNg41", "DNp15", "DNa15", "DNae002", "DNae010", "DNa10", "DNa16", "DNg100", "DNp01"] + dn_types:
                        if t in rows or not (left(t).size and right(t).size):
                            continue
                        r = {(s_, d): float(hz[d][(left if s_ == "L" else right)(t)].mean()) for s_ in "LR" for d in ("LtoR", "RtoL")}
                        yl, yr = dsi(r[("L", "LtoR")], r[("L", "RtoL")]), dsi(r[("R", "LtoR")], r[("R", "RtoL")])
                        peak = max(r.values())
                        # the steering signal: does (R - L) flip sign between the two yaw directions?
                        d1 = r[("R", "LtoR")] - r[("L", "LtoR")]; d2 = r[("R", "RtoL")] - r[("L", "RtoL")]
                        flip = bool(d1 * d2 < 0 and min(abs(d1), abs(d2)) > 0.25 * max(peak, 1e-9))
                        rows[t] = {**{f"{k[0]}:{k[1]}": v for k, v in r.items()}, "yaw_L": yl, "yaw_R": yr, "peak": peak, "flip": flip}
                    shown = [t for t in ["HSE", "HSN", "HSS", "H2", "DNa02", "DNg41", "DNp15", "DNa15", "DNae002", "DNae010", "DNa10", "DNa16", "DNg100", "DNp01"] if t in rows]
                    flips = sorted([t for t in rows if t in dn_types and rows[t]["flip"] and rows[t]["peak"] > 5],
                                   key=lambda t: -rows[t]["peak"])
                    for t in shown + [t for t in flips if t not in shown][:10]:
                        r = rows[t]
                        print(f"{t:10s} {r['L:LtoR']:7.1f} {r['L:RtoL']:7.1f} {r['R:LtoR']:7.1f} {r['R:RtoL']:7.1f} {r['yaw_L']:+7.2f} {r['yaw_R']:+7.2f} {str(r['flip']):>5s}")
                    print(f"descending types whose right-minus-left rate flips sign with yaw direction (peak > 5 Hz): {len(flips)}: {flips[:15]}")
                    (OUT / f"probe_yaw_{args.dataset}_{token}_{inj}_n{n}.json").write_text(json.dumps(rows, indent=1, default=float))
                del brain
                continue
            for n in args.speeds:
                st = bars(xy, spacing, n=n)
                hz = {d: run(brain, st[d], xy) for d in DIRS}
                rate = lambda t, d: float(hz[d][left(t)].mean()) if left(t).size else float("nan")
                rows = {}
                for t in T4 + T5 + TANGENTIAL + dn_types:
                    if not left(t).size:
                        continue
                    r = {d: rate(t, d) for d in DIRS}
                    h = dsi(r["LtoR"], r["RtoL"]); v = dsi(r["DtoU"], r["UtoD"])
                    rows[t] = {**r, "DSI_h": h, "DSI_v": v, "n": int(left(t).size), "peak": max(r.values())}
                # the orientation-free test
                def opposite(a, b, key):
                    return rows[a][key] * rows[b][key] < 0 and min(abs(rows[a][key]), abs(rows[b][key])) > 0.2
                verdict = {}
                for fam in (T4, T5):
                    a, b, c, d = fam
                    ab_h = opposite(a, b, "DSI_h"); ab_v = opposite(a, b, "DSI_v")
                    cd_h = opposite(c, d, "DSI_h"); cd_v = opposite(c, d, "DSI_v")
                    verdict[fam[0][:2]] = {"ab_opposite_h": ab_h, "ab_opposite_v": ab_v,
                                           "cd_opposite_h": cd_h, "cd_opposite_v": cd_v,
                                           "four_directions": (ab_h and cd_v) or (ab_v and cd_h)}
                label = f"{args.dataset} {token} {inj} n={n}"
                print(f"\n########## {label}: {describe(core)}")
                print(f"{'type':8s} {'n':>5} {'LtoR':>7} {'RtoL':>7} {'DtoU':>7} {'UtoD':>7} {'DSI_h':>7} {'DSI_v':>7}")
                for t in T4 + T5 + [x for x in TANGENTIAL if x in rows]:
                    if t in rows:
                        r = rows[t]
                        print(f"{t:8s} {r['n']:5d} {r['LtoR']:7.1f} {r['RtoL']:7.1f} {r['DtoU']:7.1f} {r['UtoD']:7.1f} {r['DSI_h']:+7.2f} {r['DSI_v']:+7.2f}")
                print("verdict:", json.dumps(verdict))
                top = sorted([t for t in dn_types if t in rows and rows[t]["peak"] > 5],
                             key=lambda t: max(abs(rows[t]["DSI_h"]), abs(rows[t]["DSI_v"])), reverse=True)[:8]
                print("most direction-selective DN types (peak > 5 Hz):",
                      [(t, round(rows[t]["DSI_h"], 2), round(rows[t]["DSI_v"], 2), round(rows[t]["peak"], 1)) for t in top])
                n_ds_dn = sum(1 for t in dn_types if t in rows and rows[t]["peak"] > 5
                              and max(abs(rows[t]["DSI_h"]), abs(rows[t]["DSI_v"])) > 1 / 3)
                print(f"direction-selective DN types (|DSI| > 1/3, peak > 5 Hz): {n_ds_dn}")
                table.append({"dataset": args.dataset, "brain": token, "injection": inj, "frames": n,
                              "verdict": verdict, "n_ds_dn": n_ds_dn, "rows": rows})
                OUT.mkdir(exist_ok=True)
                (OUT / f"probe_motion_{args.dataset}_{token}_{inj}_n{n}.json").write_text(
                    json.dumps(table[-1], indent=1, default=float))
            del brain
        del core
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    print("\n" + "=" * 90)
    print(f"{'dataset':8s} {'brain':18s} {'map':4s} {'n':>3} {'T4 a/b opp':>11} {'T4 c/d opp':>11} {'T4 4-dir':>9} {'T5 a/b opp':>11} {'T5 c/d opp':>11} {'T5 4-dir':>9} {'DS DNs':>7}")
    for r in table:
        v4_, v5_ = r["verdict"]["T4"], r["verdict"]["T5"]
        f = lambda v: ("h" if v["ab_opposite_h"] else "") + ("v" if v["ab_opposite_v"] else "") or "-"
        g = lambda v: ("h" if v["cd_opposite_h"] else "") + ("v" if v["cd_opposite_v"] else "") or "-"
        print(f"{r['dataset']:8s} {r['brain']:18s} {r['injection']:4s} {r['frames']:3d} {f(v4_):>11s} {g(v4_):>11s} {str(v4_['four_directions']):>9s} "
              f"{f(v5_):>11s} {g(v5_):>11s} {str(v5_['four_directions']):>9s} {r['n_ds_dn']:7d}")


if __name__ == "__main__":
    main()
