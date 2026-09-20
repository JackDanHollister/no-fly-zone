"""Same-seed pilot benchmark: the connectome against its controls.

Conditions: connectome (measured wiring), rewired (every edge's target
permuted; same degrees, counts and signs; --ablate-seeds gives several
independent rewirings and --rewired-scales gain-matched variants), silenced
(all weights zero), and the hand-written reflex stub. Same Battlezone seeds,
same injection map, readout map, zoom and motor program for all connectome
conditions. Battlezone ignores the gym seed (ALE's RNG does not touch it
with sticky actions off), so each seed instead starts the episode with a
seeded burst of 15-60 random joystick ticks before measurement begins, and
the brain is reset at the start of every run; without this, ten seeds of a
deterministic pilot are ten copies of one episode (the 2026-09-14
benchmarks had this flaw). Reports score, lives lost, fire and movement counts, the firing
regime (mean spiking neurons per tick, to spot regime mismatch in the rewired
control), and the steering-correctness metric: on every look tick where the
retina's transient energy is clearly on one side and the steer pools cross
the steering gate, does the sign of the lateralisation index match the side
of the stimulus? Balanced over left and right targets, so a pilot that always
turns one way scores 0.5. Output: out/bench_pilots.json.
"""

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from noflyzone.brain_backend import BACKENDS, describe, make_brain
from noflyzone.brain_connectome import ConnectomeBrain, lattice_neighbours, reichardt
from noflyzone.controls import CockpitControls
from noflyzone.loop import PilotLoop

OUT = Path(__file__).resolve().parent.parent / "out"
MOVES = ("UP", "DOWN", "LEFT", "RIGHT")
TARGET_LI_MIN = 0.3   # retina transient energy must be this lateralised to count as a side
TARGET_ENERGY_MIN = 0.5  # and this large in summed |delta luminance| units
# motion-signed steering: the screen's horizontal motion on each look tick from a
# Reichardt correlator over the hex lattice (both eyes), and whether the steer
# sign matches it; the score a motion readout ("yaw") is meant to be judged by
MOTION_LI_MIN = 0.3      # (rightward - leftward) / total correlator energy
MOTION_ENERGY_MIN = 1e-4 # total correlator energy over ON and OFF transients, both eyes


WARMUP_MIN, WARMUP_MAX = 15, 60


def run_one(loop, brain, args, calibration, condition, seed):
    loop.brain = brain
    brain.reset()
    # seeded random warm-up so the ten episodes are ten different games
    rng = np.random.default_rng(1000 + seed)
    warmup = int(rng.integers(WARMUP_MIN, WARMUP_MAX + 1))
    for _ in range(warmup):
        loop.sense(int(rng.integers(loop.game.n_actions)))
    loop.total_reward = 0.0
    loop.controls = CockpitControls(loop.game.action_meanings, calibration=calibration,
                                    saccade_hold=args.saccade_hold,
                                    startle_refractory=args.startle_refractory,
                                    steer_mode=args.steer_mode)
    names = loop.game.action_meanings
    acts, spiking, diffs, lrs, lives, fire = [], [], [], [], 0, 0
    # steering correctness: (target side, pool LI, steer gate crossed) per look tick
    events = []
    # motion-signed: (screen motion sign, pool LI, gated) per look tick
    motion_events = []
    nb = lattice_neighbours(loop.retina.centers_xy)
    m_tots, m_lis, e_lis = [], [], []
    prev_lum = None
    prev_tr = None      # last tick's ON/OFF transients per eye, the correlator's delayed arm
    kills = 0
    t0 = time.time()
    for _ in range(args.steps):
        state = loop.step()
        name = names[loop.controls.to_action()]
        acts.append(name)
        lives += int(state["life_lost"])
        kills += int(state["reward"] > 0)
        fire += int(loop.controls.fire)
        wc = getattr(brain, "window_counts", None)
        if wc is not None:
            spiking.append(int((wc > 0).sum()))
        lum = {s: state["drives"][s][:, 0] for s in ("left", "right")}
        energy = {s: float(np.abs(lum[s] - prev_lum[s]).sum()) for s in lum} \
            if prev_lum is not None else {"left": 0.0, "right": 0.0}
        # the same computation as injection map v6: ON and OFF transients,
        # correlated with their own previous tick along +x and -x
        tr = ({s: np.clip(lum[s] - prev_lum[s], 0.0, None) for s in lum},
              {s: np.clip(prev_lum[s] - lum[s], 0.0, None) for s in lum}) \
            if prev_lum is not None else None
        e_right = e_left = 0.0
        if tr is not None and prev_tr is not None:
            for ch in (0, 1):
                for s in lum:
                    e_right += float(reichardt(tr[ch][s], prev_tr[ch][s], nb["px"]).sum())
                    e_left += float(reichardt(tr[ch][s], prev_tr[ch][s], nb["mx"]).sum())
        m_tot = e_right + e_left
        m_li = (e_right - e_left) / m_tot if m_tot > 0 else 0.0
        # convention check: the transient energy's horizontal centroid should
        # move the way the correlator says
        if tr is not None:
            w = np.concatenate([tr[0][s] + tr[1][s] for s in ("left", "right")])
            xs = np.concatenate([loop.retina.centers_xy[:, 0]] * 2)
            cx = float((w * xs).sum() / w.sum()) if w.sum() > 0 else float("nan")
        else:
            cx = float("nan")
        prev_tr = tr
        prev_lum = lum
        if state["episode_end"]:
            prev_lum = None
            prev_tr = None
        if not loop.controls.suppressed:
            left, right = state["dn"]["DNa02_L"], state["dn"]["DNa02_R"]
            diffs.append(right - left)
            lrs.append(float(state["drives"]["right"][:, 0].mean()
                             - state["drives"]["left"][:, 0].mean()))
            e_tot = energy["left"] + energy["right"]
            e_li = (energy["right"] - energy["left"]) / e_tot if e_tot > 0 else 0.0
            total = left + right
            pool_li = (right - left) / total if total > 0 else 0.0
            gated = total >= loop.controls.steer_min_hz
            if e_tot >= TARGET_ENERGY_MIN and abs(e_li) >= TARGET_LI_MIN:
                events.append((np.sign(e_li), pool_li, gated))
            if m_tot >= MOTION_ENERGY_MIN and abs(m_li) >= MOTION_LI_MIN:
                motion_events.append((np.sign(m_li), pool_li, gated))
            m_tots.append(m_tot); m_lis.append(m_li); e_lis.append(cx)
    c = Counter(acts)
    diffs, lrs = np.array(diffs), np.array(lrs)
    act_mask = np.abs(diffs) > 0
    corr = float(np.corrcoef(diffs[act_mask], lrs[act_mask])[0, 1]) if act_mask.sum() > 3 else float("nan")
    steer = steering_summary(events)
    motion = {f"motion_{k}": v for k, v in steering_summary(motion_events).items()}
    m_lis, e_lis, m_tots = np.array(m_lis), np.array(e_lis), np.array(m_tots)
    d_eli = np.diff(e_lis, prepend=e_lis[:1]) if e_lis.size else e_lis
    mv = (m_tots >= MOTION_ENERGY_MIN) & np.isfinite(d_eli)
    motion["motion_convention_corr"] = (float(np.corrcoef(m_lis[mv], d_eli[mv])[0, 1])
                                        if mv.sum() > 3 and m_lis[mv].std() > 0 and d_eli[mv].std() > 0 else float("nan"))
    motion["motion_energy_p50_p90"] = [float(np.percentile(m_tots, 50)), float(np.percentile(m_tots, 90)),
                                       float(np.percentile(m_tots, 99))] if m_tots.size else [0.0, 0.0, 0.0]
    row = {"condition": condition, "seed": seed, "reward": float(loop.total_reward),
           "kills": kills, "lives_lost": lives, "fire_ticks": fire,
           "moves": sum(v for k, v in c.items() if any(m in k for m in MOVES)),
           "turns": sum(v for k, v in c.items() if "LEFT" in k or "RIGHT" in k),
           "noop": c.get("NOOP", 0), "mean_spiking": float(np.mean(spiking)) if spiking else float("nan"),
           "steer_active_ticks": int(act_mask.sum()), "steer_lr_corr": corr,
           "warmup_ticks": warmup, "wall_s": time.time() - t0, "actions": dict(c), **steer, **motion,
           "motion_events": [(float(a), float(b), bool(c)) for a, b, c in motion_events]}
    print(f"  {condition:16s} seed {seed}: kills {kills:2d} lives-lost {lives} fire {fire:3d} "
          f"moves {row['moves']:3d} turns {row['turns']:3d} "
          f"spiking/tick {row['mean_spiking']:6.0f} | targets L/R {steer['targets_left']:3d}/{steer['targets_right']:3d} "
          f"responded {steer['responded_frac']:.2f} correct L/R {steer['correct_left']:.2f}/{steer['correct_right']:.2f} "
          f"balanced {steer['balanced_acc']:.2f} li-corr {steer['li_corr']:+.2f} ({row['wall_s']:.0f} s)", flush=True)
    print(f"  {'':16s}         motion L/R {motion['motion_targets_left']:3d}/{motion['motion_targets_right']:3d} "
          f"correct L/R {motion['motion_correct_left']:.2f}/{motion['motion_correct_right']:.2f} "
          f"balanced {motion['motion_balanced_acc']:.2f} li-corr {motion['motion_li_corr']:+.2f} "
          f"convention {motion['motion_convention_corr']:+.2f} energy p50/p90/p99 "
          f"{motion['motion_energy_p50_p90'][0]:.2e}/{motion['motion_energy_p50_p90'][1]:.2e}/{motion['motion_energy_p50_p90'][2]:.2e}", flush=True)
    return row


def steering_summary(events) -> dict:
    """Balanced steering correctness from (target side, pool LI, gated)
    look-tick events. correct_<side>: fraction of gated events with a target
    on that side whose pool LI has the target's sign; balanced_acc is their
    mean (0.5 = no side information, 1.0 = always toward); responded_frac is
    the fraction of lateralised-target ticks where the pools crossed the
    steering gate at all; li_corr is the correlation between the retina's
    energy side and the pool LI over all such ticks."""
    out = {"targets_left": 0, "targets_right": 0, "responded_frac": float("nan"),
           "correct_left": float("nan"), "correct_right": float("nan"),
           "balanced_acc": float("nan"), "li_corr": float("nan"), "n_gated": 0,
           "n_gated_left": 0, "n_gated_right": 0,
           "events": [(float(a), float(b), bool(c)) for a, b, c in events]}
    if not events:
        return out
    side = np.array([e[0] for e in events]); li = np.array([e[1] for e in events])
    gated = np.array([e[2] for e in events], bool)
    out["targets_left"], out["targets_right"] = int((side < 0).sum()), int((side > 0).sum())
    out["responded_frac"] = float(gated.mean())
    out["n_gated"] = int(gated.sum())
    accs = []
    for name, sgn in (("correct_left", -1), ("correct_right", 1)):
        m = gated & (side == sgn)
        out[f"n_gated_{name[8:]}"] = int(m.sum())
        if m.sum():
            out[name] = float((np.sign(li[m]) == sgn).mean())
            accs.append(out[name])
    if len(accs) == 2:
        out["balanced_acc"] = float(np.mean(accs))
    if gated.sum() > 3 and li[gated].std() > 0:
        out["li_corr"] = float(np.corrcoef(side[gated], li[gated])[0, 1])
    return out


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
    ap.add_argument("--dataset", default="783", help="brain_lif.DATASETS key: 783 (FlyWire) or malecns")
    ap.add_argument("--ablate-seeds", type=int, nargs="+", default=[0],
                    help="independent rewirings; each becomes its own rewired condition")
    ap.add_argument("--rewired-scales", type=float, nargs="+", default=[1.0],
                    help="weight scales for the rewired brain (gain-matched controls)")
    ap.add_argument("--calibration", default=None,
                    help="controller calibration for every condition (default: the condition's readout); "
                         "'adaptive' scales spans and the steer gate to each pool's own running peak")
    ap.add_argument("--out", default="bench_pilots.json")
    args = ap.parse_args()
    # condition tokens: name[@ablate_seed][xweight_scale][:readout]; a bare
    # "rewired" expands over --ablate-seeds and --rewired-scales
    conditions = []
    for cond in args.conditions:
        if cond == "rewired":
            for aseed in args.ablate_seeds:
                for scale in args.rewired_scales:
                    label = "rewired" + (f"@{aseed}" if len(args.ablate_seeds) > 1 or aseed else "") \
                        + (f"x{scale:g}" if scale != 1.0 else "")
                    conditions.append((label, "rewired", aseed, scale, args.readout))
        elif cond == "stub":
            conditions.append(("stub", None, 0, 1.0, "v0"))
        else:
            m = re.fullmatch(r"(\w+?)(?:@(\d+))?(?:x([\d.]+))?(?::(\w+))?", cond)
            if not m:
                raise SystemExit(f"bad condition token {cond!r}")
            name, aseed, scale, readout = m.groups()
            conditions.append((cond, "none" if name == "connectome" else name,
                               int(aseed or 0), float(scale or 1.0), readout or args.readout))
    args.expanded_conditions = [c[0] for c in conditions]
    rows = []
    for cond, ablate, aseed, scale, readout in conditions:
        core = None
        if ablate is not None:
            core = make_brain(args.backend, version=args.dataset, ablate=ablate, ablate_seed=aseed,
                              sensory_input=args.sensory, signs=args.signs,
                              aminergic=args.aminergic, weight_scale=scale)
            print(f"\n{cond}: {describe(core)} readout {readout}")
        else:
            print("\nstub: hand-written ReflexBrain, calibration v0")
        for seed in args.seeds:
            loop = PilotLoop(seed=seed, retina_zoom=args.zoom,
                             eye_offset=args.eye_offset)
            if cond == "stub":
                brain, calibration = loop.brain, "v0"
            else:
                brain = ConnectomeBrain(retina=loop.retina, brain=core,
                                        injection=args.injection, readout=readout)
                calibration = args.calibration or readout
            rows.append(run_one(loop, brain, args, calibration, cond, seed))
            loop.close()
    print("\nmeans over seeds")
    import warnings
    for cond in args.expanded_conditions:
        rs = [r for r in rows if r["condition"] == cond]
        def m(k):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(np.nanmean([r[k] for r in rs]))
        n_seed = len(rs)
        # pooled over all seeds' gated events, so a side with few events per seed still counts
        ev = [e for r in rs for e in r["events"] if e[2]]
        pooled = {}
        for name, sgn in (("L", -1), ("R", 1)):
            k = [e for e in ev if e[0] == sgn]
            pooled[name] = (sum(np.sign(e[1]) == sgn for e in k) / len(k) if k else float("nan"), len(k))
        pooled_bal = float(np.mean([pooled["L"][0], pooled["R"][0]]))
        print(f"  {cond:16s} pooled gated events L {pooled['L'][1]:3d} R {pooled['R'][1]:3d}: "
              f"correct L {pooled['L'][0]:.2f} R {pooled['R'][0]:.2f} balanced {pooled_bal:.2f}")
        mev = [e for r in rs for e in r["motion_events"] if e[2]]
        mp = {}
        for name, sgn in (("L", -1), ("R", 1)):
            k = [e for e in mev if e[0] == sgn]
            mp[name] = (sum(np.sign(e[1]) == sgn for e in k) / len(k) if k else float("nan"), len(k))
        print(f"  {cond:16s} pooled motion-gated events L {mp['L'][1]:3d} R {mp['R'][1]:3d}: "
              f"steer with motion L {mp['L'][0]:.2f} R {mp['R'][0]:.2f} "
              f"balanced {float(np.mean([mp['L'][0], mp['R'][0]])):.2f}  convention {m('motion_convention_corr'):+.2f}")
        print(f"  {cond:16s} kills {m('kills'):4.1f}  lives-lost {m('lives_lost'):.1f}  fire {m('fire_ticks'):5.1f}  "
              f"moves {m('moves'):5.1f}  turns {m('turns'):4.1f}  spiking/tick {m('mean_spiking'):6.0f}  "
              f"| targets/ep {(m('targets_left') + m('targets_right')):5.1f} responded {m('responded_frac'):.2f}  "
              f"correct L {m('correct_left'):.2f} R {m('correct_right'):.2f} balanced {m('balanced_acc'):.2f} "
              f"li-corr {m('li_corr'):+.2f}  (n={n_seed})")
    OUT.mkdir(exist_ok=True)
    (OUT / args.out).write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
    print(f"wrote {OUT / args.out}")


if __name__ == "__main__":
    main()
