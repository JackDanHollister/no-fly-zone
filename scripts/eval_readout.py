"""Evaluate trained readout heads on fresh seeds, greedily, against a random
policy and the reflex pilot, all on the same seeds and tick budget.
Output: out/eval_readout.json and a table.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_readout import ActorCritic  # noqa: E402

from noflyzone.controls import CockpitControls  # noqa: E402
from noflyzone.rl_env import FlySquad  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"


def run(squad, policy, ticks):
    obs = squad.reset()
    kills = np.zeros(squad.B); lives = np.zeros(squad.B)
    for t in range(ticks):
        a = policy(obs, squad)
        obs, rew, done, infos = squad.step(a)
        kills += [i["kill"] for i in infos]; lives += [i["life_lost"] for i in infos]
    per = ticks / 900
    return kills / per, lives / per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104])
    ap.add_argument("--ticks", type=int, default=1800)
    ap.add_argument("--ckpts", nargs="+", default=["out/readout_connectome_brain_s0.pt",
                                                   "out/readout_rewired_brain_s0.pt",
                                                   "out/readout_connectome_retina_s0.pt"])
    args = ap.parse_args()
    dev = torch.device("cuda")
    results = {}
    B = len(args.seeds)
    # baselines on the measured brain
    squad = FlySquad(batch=B, seeds=args.seeds, features="brain", episode_ticks=900)
    rng = np.random.default_rng(0)
    k, l = run(squad, lambda o, s: rng.integers(0, s.n_actions, size=s.B), args.ticks)
    results["random policy"] = (k, l); print(f"random policy        kills {k.mean():.2f} ± {k.std():.2f}  lives {l.mean():.2f} ± {l.std():.2f}  per 900 ticks over {B} seeds", flush=True)
    ctrls = [CockpitControls(squad.action_names, calibration="pool4", saccade_hold=3, startle_refractory=15, steer_mode="li") for _ in range(B)]
    def reflex(o, s):
        acts = np.array([c.to_action() for c in ctrls])
        return acts
    obs = squad.reset(); kills = np.zeros(B); lives = np.zeros(B)
    for t in range(args.ticks):
        a = reflex(obs, squad); obs, rew, done, infos = squad.step(a)
        for i, c in enumerate(ctrls): c.update(squad.last_dn[i], squad.dt)
        kills += [i["kill"] for i in infos]; lives += [i["life_lost"] for i in infos]
    k, l = kills / (args.ticks / 900), lives / (args.ticks / 900)
    results["reflex pilot"] = (k, l); print(f"reflex pilot         kills {k.mean():.2f} ± {k.std():.2f}  lives {l.mean():.2f} ± {l.std():.2f}", flush=True)
    squad.close()
    for ck_path in args.ckpts:
        ck = torch.load(ck_path, weights_only=False)
        ablate = "none" if ck["brain"] == "connectome" else ck["brain"]
        squad = FlySquad(batch=B, seeds=args.seeds, brain_kwargs={"ablate": ablate}, features=ck["features"], episode_ticks=900)
        net = ActorCritic(squad.n_features, ck["n_actions"]).to(dev); net.load_state_dict(ck["net"]); net.eval()
        mean, var = ck["norm_mean"], ck["norm_var"]
        def greedy(o, s):
            x = ((o - mean) / np.sqrt(var + 1e-8)).astype(np.float32)
            with torch.no_grad():
                logits, _ = net(torch.from_numpy(x).to(dev))
            return logits.argmax(-1).cpu().numpy()
        k, l = run(squad, greedy, args.ticks)
        name = f"{ck['brain']} head ({ck['features']})"
        results[name] = (k, l); print(f"{name:20s} kills {k.mean():.2f} ± {k.std():.2f}  lives {l.mean():.2f} ± {l.std():.2f}", flush=True)
        squad.close()
    OUT.mkdir(exist_ok=True)
    (OUT / "eval_readout.json").write_text(json.dumps({n: {"kills": k.tolist(), "lives": l.tolist()} for n, (k, l) in results.items()}, indent=1))
    print("wrote out/eval_readout.json")


if __name__ == "__main__":
    main()
