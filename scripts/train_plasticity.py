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
--dopamine events  a kill pulses PAM and a hit pulses PPL1 (2026-09-19)
--dopamine rpe     the teaching signal is a reward-prediction error from a
               linear critic over the pooled descending/visual-projection
               features (TD(0), learned online): PAM fires for a better-than-
               predicted outcome, PPL1 for a worse one, in proportion. A hit
               that a loom already predicted teaches nothing; an escape that
               averts a predicted hit is rewarded (2026-09-20)

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
    ap.add_argument("--dopamine", choices=["events", "rpe"], default="events")
    ap.add_argument("--latch", action="store_true",
                    help="apply a reinforcer's pulse to the eligibility as it stood the tick before the "
                         "event, not to the brain's response to the event's own flash (2026-09-20)")
    ap.add_argument("--critic-lr", type=float, default=0.5,
                    help="normalised-LMS TD(0) step: a hit moves the prediction of its state by this "
                         "fraction of the error (1e-3 unnormalised left the critic at zero, 2026-09-20)")
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--rpe-hz", type=float, default=100.0, help="teaching rate at |RPE| = one running RMS")
    ap.add_argument("--critic", choices=["population", "structured"], default="structured",
                    help="population: 907 pooled features; structured: the readout pools' own rates "
                         "(loom, steer L/R, drive) and their 1 s low-passes, the fly's own loom detectors "
                         "as the predictor of the hit (2026-09-20)")
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
                         eligibility=args.eligibility, scale=args.scale, elig_cap=args.elig_cap,
                         latch=args.latch)
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
    obs = squad.reset()
    # the critic: V(x) = w . x over the squad's pooled features, standardised
    # by a running mean and variance, TD(0)
    def structured_obs(lp_state):
        """Per fly: the readout pools' rates this tick and their 1 s low-passes."""
        keys = ("DNp01", "DNa02_L", "DNa02_R", "DNg100", "pIP10")
        raw = np.array([[squad.last_dn[i].get(k, 0.0) for k in keys] for i in range(args.batch)])
        a = 1.0 - np.exp(-squad.dt / 1.0)
        lp_state[:] = lp_state + a * (raw - lp_state)
        return np.c_[np.log1p(raw), np.log1p(lp_state)]
    lp_state = np.zeros((args.batch, 5))
    n_crit = squad.n_features if args.critic == "population" else 10
    critic_w = np.zeros(n_crit + 1, np.float64)
    feat_mean = np.zeros(n_crit); feat_var = np.ones(n_crit); feat_n = 1e-3
    def standardise(o):
        nonlocal feat_mean, feat_var, feat_n
        bm, bv, bc = o.mean(0), o.var(0), o.shape[0]
        d_ = bm - feat_mean; tot = feat_n + bc
        feat_mean = feat_mean + d_ * bc / tot
        feat_var = (feat_var * feat_n + bv * bc + d_ ** 2 * feat_n * bc / tot) / tot
        feat_n = tot
        return np.c_[(o - feat_mean) / np.sqrt(feat_var + 1e-6), np.ones(o.shape[0])]
    rpe_rms = 0.1        # the reward scale; adapts with a 1,000-tick memory
    x_prev = standardise(obs if args.critic == "population" else structured_obs(lp_state))
    rpe_log = []
    hit_rpe = []         # (global tick, mean delta at flies that were hit): does the critic learn to expect them?
    done_eps = 0; ep_index = np.zeros(args.batch, int)
    t0 = time.time(); diag_acc = []
    target = args.batch * args.episodes
    while done_eps < target:
        actions = np.array([c.to_action() for c in controls])
        # teaching goes in before the tick it should act on
        if args.dopamine == "events":
            rates = plast.teaching_rates()
            squad.set_teaching("pam", rates["pam"])
            squad.set_teaching("ppl1", rates["ppl1"])
        obs, rew, done, infos = squad.step(actions)
        kills = np.array([i["kill"] for i in infos]); hits = np.array([i["life_lost"] for i in infos])
        if args.teaching == "kills":
            hits = np.zeros_like(hits)
        elif args.teaching == "hits":
            kills = np.zeros_like(kills)
        if args.dopamine == "rpe":
            # one critic for the squad (shared weights, pooled experience), one
            # prediction error per fly; the teaching rate for the next tick is
            # the squad's mean error in units of its running RMS
            x = standardise(obs if args.critic == "population" else structured_obs(lp_state))
            r = kills.astype(float) - 0.5 * hits.astype(float)
            v_prev = x_prev @ critic_w; v_next = np.where(done, 0.0, x @ critic_w)
            delta = r + args.gamma * v_next - v_prev
            # normalised LMS, one update per fly: the prediction for that fly's
            # state moves by critic_lr times its error
            critic_w += (args.critic_lr * (delta[:, None] * x_prev) / (1.0 + np.sum(x_prev ** 2, axis=1))[:, None]).sum(axis=0)
            if hits.any():
                hit_rpe.append((len(rpe_log), float(delta[hits.astype(bool)].mean())))
            rpe_rms = 0.999 * rpe_rms + 0.001 * float(np.sqrt(np.mean(delta ** 2)) + 1e-6)
            d_mean = float(delta.mean()) / rpe_rms
            squad.set_teaching("pam", args.rpe_hz * float(np.clip(d_mean, 0.0, 2.0)))
            squad.set_teaching("ppl1", args.rpe_hz * float(np.clip(-d_mean, 0.0, 2.0)))
            x_prev = x
            rpe_log.append(d_mean)
            kills = np.zeros_like(kills); hits = np.zeros_like(hits)   # no event pulses on top
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
                   "rpe_mean": float(np.mean(rpe_log[-args.episode_ticks:])) if rpe_log else None,
                   "rpe_abs": float(np.mean(np.abs(rpe_log[-args.episode_ticks:]))) if rpe_log else None,
                   "critic_norm": float(np.linalg.norm(critic_w)),
                   "hit_rpe": float(np.mean([v for _, v in hit_rpe[-20:]])) if hit_rpe else None,
                   "wall_s": time.time() - t0}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(f"ep {done_eps:3d}/{target} fly {i} #{ep_index[i]:2d}: kills {e['kills']} lives {e['lives_lost']} "
                  f"| drift {d['drift']:.4f} changed {d['frac_changed']:.3f} | PAM {row['pam_hz']:.1f} PPL1 {row['ppl1_hz']:.1f} Hz", flush=True)
            if infos[i]["game_over"]:
                controls[i] = CockpitControls(squad.action_names, calibration=calibration, saccade_hold=3,
                                              startle_refractory=15, steer_mode="li")
    savez(OUT / f"plasticity_{tag}_weights.npz", args=args, brain=squad.brain,
          pos=plast.pos, w0=plast.w0, w=plast.w, pre=plast.pre, post=plast.post,
          critic_w=critic_w)
    if calibration == "adaptive":
        peaks = {k: float(np.mean([c.peak[k] for c in controls])) for k in controls[0].peak}
        dump_json(OUT / f"plasticity_{tag}_peaks.json", peaks, args=args, brain=squad.brain)
        print("adaptive spans at the end (Hz, mean over flies): " +
              "  ".join(f"{k} {v:.1f}" for k, v in peaks.items()))
    print(f"done in {time.time() - t0:.0f} s; weights -> out/plasticity_{tag}_weights.npz")
    squad.close()


if __name__ == "__main__":
    main()
