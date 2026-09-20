"""M7: let the reflex pilot play while its brain learns from kills and hits,
against the pre-registered controls.

--rule on      the three-factor rule changes the chosen edge set
--rule lr0     the same run with the rule computed but never applied
--edge-set     kc_mbon (the mushroom body, 2026-09-14), dn_in (every input
               to a descending neuron), vis_dn (visual inputs to descending
               neurons), or <set>_shuffled for the size-matched control on
               edges touching neither end
--init-weights out/plasticity_<tag>_weights.npz  start from learned weights;
               with --rule lr0 that is a frozen evaluation of a learned brain

Same seeds and settings across conditions. Logs one line per episode
(kills, lives lost, weight drift, dopamine statistics) to
out/plasticity_<tag>.jsonl and saves the learned weights.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from noflyzone.controls import CockpitControls
from noflyzone.plasticity import EDGE_SETS, MBPlasticity
from noflyzone.rl_env import FlySquad
from noflyzone.provenance import dump_json, savez

OUT = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["on", "lr0"], default="on")
    ap.add_argument("--edge-set", choices=list(EDGE_SETS), default="kc_mbon")
    ap.add_argument("--eligibility", choices=["presynaptic", "hebbian"], default=None,
                    help="default: presynaptic for kc_mbon, hebbian for the descending sets")
    ap.add_argument("--scale", choices=["absolute", "relative"], default=None,
                    help="default: absolute for kc_mbon, relative for the descending sets")
    ap.add_argument("--init-weights", default=None)
    ap.add_argument("--edge-seed", type=int, default=None,
                    help="seed of a shuffled edge set (default: --seed); lets a learned "
                         "shuffled set be evaluated on fresh game seeds")
    ap.add_argument("--elig-cap", type=float, default=None,
                    help="default: none for absolute scale, 50 for relative")
    ap.add_argument("--readout", default="pool4")
    ap.add_argument("--calibration", default=None,
                    help="controller calibration (default: same as --readout); 'adaptive' follows "
                         "each pool's running peak so a gain change does not break the thresholds")
    ap.add_argument("--teaching", choices=["both", "kills", "hits"], default="both",
                    help="which game events teach: kills pulse PAM, hits pulse PPL1, or only one of them")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=20, help="per fly")
    ap.add_argument("--episode-ticks", type=int, default=900)
    ap.add_argument("--eta", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"{args.rule}_{args.edge_set}_s{args.seed}"
    calibration = args.calibration or args.readout
    # a squad whose Poisson set also holds the dopamine populations
    from noflyzone.brain_lif import annotations
    from noflyzone.brain_backend import make_brain
    probe = make_brain("cpu") if False else None  # indices come from annotations below
    squad = FlySquad(batch=args.batch, seeds=[args.seed * 100 + i for i in range(args.batch)],
                     episode_ticks=args.episode_ticks, features="brain",
                     extra_sites=None)
    ann = annotations().drop_duplicates("root_id").set_index("root_id").reindex(squad.brain.root_ids)
    ct = ann["cell_type"].fillna("").astype(str).to_numpy()
    pam = np.flatnonzero(np.char.startswith(ct.astype(str), "PAM"))
    ppl1 = np.flatnonzero(np.char.startswith(ct.astype(str), "PPL1"))
    squad.close()
    squad = FlySquad(batch=args.batch, seeds=[args.seed * 100 + i for i in range(args.batch)],
                     episode_ticks=args.episode_ticks, features="brain",
                     readout=args.readout, extra_sites={"pam": pam, "ppl1": ppl1})
    plast = MBPlasticity(squad.brain, dt=squad.dt, eta=args.eta, rule=args.rule,
                         edge_set=args.edge_set,
                         seed=args.seed if args.edge_seed is None else args.edge_seed,
                         eligibility=args.eligibility, scale=args.scale, elig_cap=args.elig_cap)
    if args.init_weights:
        plast.load_weights(args.init_weights)
        print(f"loaded weights from {args.init_weights}", flush=True)
    print(f"{tag}: {plast.n_edges} plastic edges ({plast.edge_set}, {plast.eligibility}, {plast.scale}), "
          f"{pam.size} PAM + {ppl1.size} PPL1 teaching cells, "
          f"{args.batch} flies x {args.episodes} episodes of {args.episode_ticks} ticks", flush=True)
    controls = [CockpitControls(squad.action_names, calibration=calibration, saccade_hold=3,
                                startle_refractory=15, steer_mode="li") for _ in range(args.batch)]
    OUT.mkdir(exist_ok=True)
    log = open(OUT / f"plasticity_{tag}.jsonl", "w")
    squad.reset()
    done_eps = 0; ep_index = np.zeros(args.batch, int)
    t0 = time.time(); diag_acc = []
    target = args.batch * args.episodes
    while done_eps < target:
        actions = np.array([c.to_action() for c in controls])
        # teaching pulses go in before the tick they should act on
        rates = plast.teaching_rates()
        squad.set_teaching("pam", rates["pam"])
        squad.set_teaching("ppl1", rates["ppl1"])
        obs, rew, done, infos = squad.step(actions)
        kills = np.array([i["kill"] for i in infos]); hits = np.array([i["life_lost"] for i in infos])
        if args.teaching == "kills":
            hits = np.zeros_like(hits)
        elif args.teaching == "hits":
            kills = np.zeros_like(kills)
        d = plast.step(squad.last_counts, kills, hits); diag_acc.append(d)
        for i, c in enumerate(controls):
            c.update(squad.last_dn[i], squad.dt)
        for i in np.flatnonzero(done):
            e = infos[i]["episode"]; ep_index[i] += 1; done_eps += 1
            recent = diag_acc[-args.episode_ticks:]
            row = {"fly": int(i), "episode": int(ep_index[i]), "global_episode": int(done_eps),
                   "kills": int(e["kills"]), "lives_lost": int(e["lives_lost"]), "ticks": int(e["ticks"]),
                   "drift": float(d["drift"]), "frac_changed": float(d["frac_changed"]),
                   "pam_hz": float(np.mean([x["pam_hz"] for x in recent])),
                   "ppl1_hz": float(np.mean([x["ppl1_hz"] for x in recent])),
                   "da_centred": float(np.mean([x["da_centred"] for x in recent])),
                   "wall_s": time.time() - t0}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(f"ep {done_eps:3d}/{target} fly {i} #{ep_index[i]:2d}: kills {e['kills']} lives {e['lives_lost']} "
                  f"| drift {d['drift']:.4f} changed {d['frac_changed']:.3f} | PAM {row['pam_hz']:.1f} PPL1 {row['ppl1_hz']:.1f} Hz", flush=True)
            if infos[i]["game_over"]:
                controls[i] = CockpitControls(squad.action_names, calibration=calibration, saccade_hold=3,
                                              startle_refractory=15, steer_mode="li")
    savez(OUT / f"plasticity_{tag}_weights.npz", args=args, brain=squad.brain,
          pos=plast.pos, w0=plast.w0, w=plast.w, pre=plast.pre, post=plast.post)
    if calibration == "adaptive":
        peaks = {k: float(np.mean([c.peak[k] for c in controls])) for k in controls[0].peak}
        dump_json(OUT / f"plasticity_{tag}_peaks.json", peaks, args=args, brain=squad.brain)
        print("adaptive spans at the end (Hz, mean over flies): " +
              "  ".join(f"{k} {v:.1f}" for k, v in peaks.items()))
    print(f"done in {time.time() - t0:.0f} s; weights -> out/plasticity_{tag}_weights.npz")
    squad.close()


if __name__ == "__main__":
    main()
