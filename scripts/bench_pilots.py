"""Same-seed pilot benchmark: the connectome against its controls.

Conditions: connectome (measured wiring), rewired (every edge's target
permuted; same degrees, counts and signs), silenced (all weights zero), and
the hand-written reflex stub. Same Battlezone seeds, same injection map,
readout map, zoom and motor program for all connectome conditions. Reports
score, lives lost, fire and movement counts, the firing regime (mean spiking
neurons per tick, to spot regime mismatch in the rewired control) and the
correlation between the steering signal and the retina's left-right
luminance. Output: out/bench_pilots.json.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from noflyzone.brain_backend import BACKENDS, describe, make_brain
from noflyzone.brain_connectome import ConnectomeBrain
from noflyzone.controls import CockpitControls
from noflyzone.loop import PilotLoop

OUT = Path(__file__).resolve().parent.parent / "out"
MOVES = ("UP", "DOWN", "LEFT", "RIGHT")


def run_one(loop, brain, args, calibration, condition, seed):
    loop.brain = brain
    loop.controls = CockpitControls(loop.game.action_meanings, calibration=calibration,
                                    saccade_hold=args.saccade_hold,
                                    startle_refractory=args.startle_refractory,
                                    steer_mode=args.steer_mode)
    names = loop.game.action_meanings
    acts, spiking, diffs, lrs, lives, fire = [], [], [], [], 0, 0
    t0 = time.time()
    for _ in range(args.steps):
        state = loop.step()
        name = names[loop.controls.to_action()]
        acts.append(name)
        lives += int(state["life_lost"])
        fire += int(loop.controls.fire)
        wc = getattr(brain, "window_counts", None)
        if wc is not None:
            spiking.append(int((wc > 0).sum()))
        if not loop.controls.suppressed:
            diffs.append(state["dn"]["DNa02_R"] - state["dn"]["DNa02_L"])
            lrs.append(float(state["drives"]["right"][:, 0].mean()
                             - state["drives"]["left"][:, 0].mean()))
    c = Counter(acts)
    diffs, lrs = np.array(diffs), np.array(lrs)
    act_mask = np.abs(diffs) > 0
    corr = float(np.corrcoef(diffs[act_mask], lrs[act_mask])[0, 1]) if act_mask.sum() > 3 else float("nan")
    row = {"condition": condition, "seed": seed, "reward": float(loop.total_reward),
           "lives_lost": lives, "fire_ticks": fire,
           "moves": sum(v for k, v in c.items() if any(m in k for m in MOVES)),
           "turns": sum(v for k, v in c.items() if "LEFT" in k or "RIGHT" in k),
           "noop": c.get("NOOP", 0), "mean_spiking": float(np.mean(spiking)) if spiking else float("nan"),
           "steer_active_ticks": int(act_mask.sum()), "steer_lr_corr": corr,
           "wall_s": time.time() - t0, "actions": dict(c)}
    print(f"  {condition:10s} seed {seed}: reward {row['reward']:5.0f} lives-lost {lives} fire {fire:3d} "
          f"moves {row['moves']:3d} turns {row['turns']:3d} noop {row['noop']:3d} "
          f"spiking/tick {row['mean_spiking']:8.0f} steer-corr {corr:+.2f} ({row['wall_s']:.0f} s)", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, default="auto")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--injection", default="v4")
    ap.add_argument("--readout", default="pool4")
    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--eye-offset", type=float, default=0.15)
    ap.add_argument("--saccade-hold", type=int, default=3)
    ap.add_argument("--startle-refractory", type=int, default=15)
    ap.add_argument("--steer-mode", choices=["diff", "li"], default="li")
    ap.add_argument("--sensory", choices=["free", "external_only"], default="free")
    ap.add_argument("--signs", choices=["predicted", "known_first"], default="predicted")
    ap.add_argument("--aminergic", choices=["excitatory", "zero"], default="excitatory")
    ap.add_argument("--conditions", nargs="+",
                    default=["connectome", "rewired", "silenced", "stub"])
    ap.add_argument("--out", default="bench_pilots.json")
    args = ap.parse_args()
    rows = []
    for cond in args.conditions:
        core = None
        if cond != "stub":
            ablate = "none" if cond == "connectome" else cond
            core = make_brain(args.backend, ablate=ablate, sensory_input=args.sensory,
                              signs=args.signs, aminergic=args.aminergic)
            print(f"\n{cond}: {describe(core)}")
        else:
            print("\nstub: hand-written ReflexBrain, calibration v0")
        for seed in args.seeds:
            loop = PilotLoop(seed=seed, retina_zoom=args.zoom,
                             eye_offset=args.eye_offset)
            if cond == "stub":
                brain, calibration = loop.brain, "v0"
            else:
                brain = ConnectomeBrain(retina=loop.retina, brain=core,
                                        injection=args.injection, readout=args.readout)
                calibration = args.readout
            rows.append(run_one(loop, brain, args, calibration, cond, seed))
            loop.close()
    print("\nmeans over seeds")
    for cond in args.conditions:
        rs = [r for r in rows if r["condition"] == cond]
        m = lambda k: float(np.nanmean([r[k] for r in rs]))
        print(f"  {cond:10s} reward {m('reward'):6.0f}  lives-lost {m('lives_lost'):.1f}  fire {m('fire_ticks'):5.1f}  "
              f"moves {m('moves'):5.1f}  turns {m('turns'):4.1f}  spiking/tick {m('mean_spiking'):8.0f}  steer-corr {m('steer_lr_corr'):+.2f}")
    OUT.mkdir(exist_ok=True)
    (OUT / args.out).write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
    print(f"wrote {OUT / args.out}")


if __name__ == "__main__":
    main()
