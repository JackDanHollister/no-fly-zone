"""What does the game's injection path deliver? Push synthetic luminance
patterns (left eye only) through ConnectomeBrain.step exactly as the loop
does — R1-6 luminance plus Mi1/Tm1/Tm2 ON/OFF transients — and report, per
game tick, the mean rate of each visual stage and the readout DNs.

Stimuli: a static disc appearing then vanishing (three sizes), a looming
disc, a small disc moving left-to-right and right-to-left, and a wide bar
sweeping each way. Output: out/probe_frontend.json and a table on stdout.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from noflyzone.brain_backend import BACKENDS, describe, make_brain
from noflyzone.brain_connectome import ConnectomeBrain
from noflyzone.brain_lif import annotations
from noflyzone.retina import HexRetina

OUT = Path(__file__).resolve().parent.parent / "out"
DT_GAME = 1.0 / 15.0
WATCH = ["Mi1", "Tm1", "Tm2", "T4a", "T4b", "T5a", "T5b", "HSE", "HSN", "HSS",
         "LC4", "LPLC2", "LC6", "LC10a", "LC11", "LC15", "LC17"]
WATCH_R = ["HSE", "LC4", "LC11"]  # right-side copies, for lateralisation


def stimuli(xy: np.ndarray, spacing: float) -> dict[str, list[np.ndarray]]:
    """name -> list of per-tick luminance arrays (n_omm,) for the left eye."""
    d = lambda c, r: (np.linalg.norm(xy - c, axis=1) <= r * spacing).astype(np.float32)
    dark = np.zeros(len(xy), np.float32)
    out = {}
    for r in (1.5, 3.0, 6.0, 99.0):
        out[f"disc_r{r:g}"] = [dark] * 2 + [d((0, 0), r)] * 6 + [dark] * 4
    out["loom"] = [dark] * 2 + [d((0, 0), r) for r in np.linspace(1, 14, 10)] + [dark] * 3
    xs = np.linspace(-0.9, 0.9, 18)
    out["small_disc_LtoR"] = [dark] + [d((x, 0), 2.0) for x in xs] + [dark]
    out["small_disc_RtoL"] = [dark] + [d((x, 0), 2.0) for x in xs[::-1]] + [dark]
    bar = lambda x: ((np.abs(xy[:, 0] - x) <= 1.5 * spacing)).astype(np.float32)
    out["bar_LtoR"] = [dark] + [bar(x) for x in xs] + [dark]
    out["bar_RtoL"] = [dark] + [bar(x) for x in xs[::-1]] + [dark]
    out["full_flash"] = [dark] * 2 + [np.ones(len(xy), np.float32)] * 3 + [dark] * 3
    return out


def right_eye(name: str) -> bool:
    return name.endswith("@R")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, default="auto")
    ap.add_argument("--injection", default="v0")
    ap.add_argument("--out", default="probe_frontend.json")
    args = ap.parse_args()
    retina = HexRetina()
    xy = retina.centers_xy.astype(np.float64)
    from scipy.spatial import cKDTree
    dd, _ = cKDTree(xy).query(xy, k=2)
    spacing = float(np.median(dd[:, 1]))
    brain = ConnectomeBrain(retina=retina, brain=make_brain(args.backend),
                            injection=args.injection)
    print(f"backend {describe(brain.brain)} | {brain.describe()} | spacing {spacing:.3f}")
    ann = annotations().drop_duplicates("root_id").set_index("root_id")
    ann = ann.reindex(brain.brain.root_ids)
    ct = ann["cell_type"].astype(str).to_numpy()
    side = ann["side"].astype(str).to_numpy()
    watch = {t: np.flatnonzero((ct == t) & (side == "left")) for t in WATCH}
    watch.update({f"{t}_R": np.flatnonzero((ct == t) & (side == "right"))
                  for t in WATCH_R})
    dark = np.zeros(len(xy), np.float32)

    stims = stimuli(xy, spacing)
    for k in ("small_disc_LtoR", "small_disc_RtoL", "loom"):
        stims[f"{k}@R"] = stims[k]
    results = {}
    for name, frames in stims.items():
        brain.reset()
        ticks = []
        for lum in frames:
            lit, unlit = np.stack([lum, lum], 1), np.stack([dark, dark], 1)
            drives = ({"left": unlit, "right": lit} if right_eye(name)
                      else {"left": lit, "right": unlit})
            dn = brain.step(drives, DT_GAME)
            hz = brain.window_counts / DT_GAME
            row = {t: float(hz[i].mean()) if i.size else 0.0 for t, i in watch.items()}
            row.update({f"DN:{k}": float(v) for k, v in dn.items()})
            row["spiking"] = int((brain.window_counts > 0).sum())
            row["lit_omm"] = int(lum.sum())
            ticks.append(row)
        results[name] = ticks
        peak = {t: max(r[t] for r in ticks) for t in watch}
        dnpk = {k: max(r[f"DN:{k}"] for r in ticks) for k in dn}
        top = ", ".join(f"{t} {v:.0f}" for t, v in sorted(peak.items(), key=lambda kv: -kv[1])[:7] if v > 0)
        print(f"{name:18s} lit<= {max(r['lit_omm'] for r in ticks):3d}  spiking<= {max(r['spiking'] for r in ticks):6,}"
              f"  peak DN: " + " ".join(f"{k} {v:.0f}" for k, v in dnpk.items())
              + f"\n{'':18s} peak stages: {top or 'none'}", flush=True)
    OUT.mkdir(exist_ok=True)
    print("\nlateralisation (peak Hz, sum over ticks in brackets):")
    for name in ("small_disc_LtoR", "small_disc_RtoL", "small_disc_LtoR@R",
                 "small_disc_RtoL@R", "loom", "loom@R"):
        t = results[name]
        f = lambda k: f"{max(r[k] for r in t):4.0f} [{sum(r[k] for r in t):5.0f}]"
        print(f"  {name:20s} DNa02 L {f('DN:DNa02_L')} R {f('DN:DNa02_R')}"
              f" | LC11 L {f('LC11')} R {f('LC11_R')} | HSE L {f('HSE')} R {f('HSE_R')}"
              f" | LC4 L {f('LC4')} R {f('LC4_R')} | DNp01 {f('DN:DNp01')}")
    (OUT / args.out).write_text(json.dumps(
        {"injection": args.injection, "results": results}, indent=1))
    print(f"wrote {OUT / args.out}")


if __name__ == "__main__":
    main()
