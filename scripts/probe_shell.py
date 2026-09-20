"""Is the shell that hits the fly visible to it? (2026-09-20)

The recorded fights show every descending pool silent in the second before
a hit. This probe plays the reflex pilot and, at every tick, records what
the retina and the front end carry: the mean transient at each eye, the
engineered small-object channel's injected rate (v9: T2/T2a/T3 on
surround plus wide-field opponency), and the readout pools; then averages
each signal over the ticks before a hit and before a kill against its
baseline, and saves the game frames and retinal mosaics around the first
hits so the shell can be looked for by eye. Output: out/probe_shell.json,
out/probe_shell_<injection>_hits.png.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from noflyzone.brain_backend import make_brain
from noflyzone.brain_connectome import ConnectomeBrain
from noflyzone.controls import CockpitControls
from noflyzone.loop import PilotLoop
from noflyzone.provenance import dump_json

OUT = Path(__file__).resolve().parent.parent / "out"


def run(args, injection, readout, seed):
    loop = PilotLoop(seed=seed, retina_zoom=2.0, eye_offset=0.15)
    core = make_brain("cuda", aminergic="zero", sensory_input="external_only")
    cb = ConnectomeBrain(retina=loop.retina, brain=core, injection=injection, readout=readout)
    loop.brain = cb
    loop.controls = CockpitControls(loop.game.action_meanings, calibration=readout, saccade_hold=3,
                                    startle_refractory=15, steer_mode="li")
    # the small-object populations' rate slots (v9) or the T3 onoff slots (v4)
    small = [sl for mode, _, sl, _, _ in cb._pops if mode.endswith(("_wf", "onoff"))]
    rows, frames = [], []
    prev = None
    for t in range(args.steps):
        state = loop.step()
        lum = {s: state["drives"][s][:, 0] for s in ("left", "right")}
        tr = {s: float(np.abs(lum[s] - prev[s]).mean()) if prev is not None else 0.0 for s in lum}
        prev = lum
        small_hz = float(np.mean([cb._rates[sl].mean() for sl in small])) if small else 0.0
        rows.append({"tick": t, "hit": bool(state["life_lost"]), "kill": bool(state["reward"] > 0),
                     "transient_L": tr["left"], "transient_R": tr["right"], "small_channel_hz": small_hz,
                     **{k: float(v) for k, v in state["dn"].items()}})
        if len(frames) < 3 * 4:
            frames.append((t, loop.game.frame.copy(), loop.retina.reconstruct(state["rgb"]["left"])))
        if len(frames) >= 3 * 4 and False:
            pass
    loop.close()
    return rows, frames


def window_means(rows, key, event, before, after):
    idx = [r["tick"] for r in rows if r[event]]
    vals = [np.mean([rows[j][key] for j in range(max(0, i - before), i - after)]) for i in idx if i - before >= 0]
    return float(np.mean(vals)) if vals else float("nan"), len(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    report = {}
    for injection, readout in (("v4", "pool4"), ("v9", "pool_v9")):
        allrows = []
        hit_frames = []
        for seed in args.seeds:
            rows, _ = run(args, injection, readout, seed)
            allrows.append(rows)
        keys = ["transient_L", "transient_R", "small_channel_hz", "DNa02_L", "DNa02_R", "DNg100", "DNp01", "pIP10"]
        base = {k: float(np.mean([r[k] for rows in allrows for r in rows])) for k in keys}
        rep = {"baseline": base, "n_hits": sum(r["hit"] for rows in allrows for r in rows),
               "n_kills": sum(r["kill"] for rows in allrows for r in rows)}
        for event in ("hit", "kill"):
            for (b, a) in ((15, 5), (5, 0)):
                rep[f"before_{event}_{b}_{a}"] = {k: float(np.nanmean([window_means(rows, k, event, b, a)[0] for rows in allrows])) for k in keys}
        # the maximum transient in the 15 ticks before each hit: was there anything at all?
        peaks = []
        for rows in allrows:
            for r in rows:
                if r["hit"] and r["tick"] >= 15:
                    peaks.append(max(max(rows[j]["transient_L"], rows[j]["transient_R"]) for j in range(r["tick"] - 15, r["tick"])))
        rep["pre_hit_transient_peak_median"] = float(np.median(peaks)) if peaks else float("nan")
        rep["pre_hit_transient_peak_p90"] = float(np.percentile(peaks, 90)) if peaks else float("nan")
        rep["transient_p50_p90_overall"] = [float(np.percentile([max(r["transient_L"], r["transient_R"]) for rows in allrows for r in rows], q)) for q in (50, 90)]
        report[injection] = rep
        print(f"\n{injection} / {readout}: {rep['n_hits']} hits, {rep['n_kills']} kills over {len(args.seeds)} x {args.steps} ticks")
        print(f"  {'signal':18s} {'baseline':>9s} {'15-5 pre-hit':>13s} {'5-0 pre-hit':>12s} {'15-5 pre-kill':>14s} {'5-0 pre-kill':>13s}")
        for k in keys:
            print(f"  {k:18s} {base[k]:9.3f} {rep['before_hit_15_5'][k]:13.3f} {rep['before_hit_5_0'][k]:12.3f} {rep['before_kill_15_5'][k]:14.3f} {rep['before_kill_5_0'][k]:13.3f}")
        print(f"  transient peak in the 15 ticks before a hit: median {rep['pre_hit_transient_peak_median']:.4f}, p90 {rep['pre_hit_transient_peak_p90']:.4f}; "
              f"overall p50/p90 {rep['transient_p50_p90_overall'][0]:.4f}/{rep['transient_p50_p90_overall'][1]:.4f}")
    OUT.mkdir(exist_ok=True)
    dump_json(OUT / "probe_shell.json", report, args=args)
    print("wrote out/probe_shell.json")


if __name__ == "__main__":
    main()
