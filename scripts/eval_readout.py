"""Evaluate trained readout heads (MLP or Recurrent GRU) on fresh audited seeds
against random play and both legacy and active reflex pilots.
Output: out/eval_readout.json (or --out) and a summary table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_readout import (  # noqa: E402
    ActorCritic,
    RecurrentActorCritic,
    build_action_codec,
    make_teacher,
    motor_efference,
)

from noflyzone.controls import CockpitControls  # noqa: E402
from noflyzone.provenance import dump_json  # noqa: E402
from noflyzone.rl_env import FlySquad  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"


def run_squad_policy(squad: FlySquad, step_fn, ticks: int):
    obs = squad.reset()
    kills = np.zeros(squad.B)
    lives = np.zeros(squad.B)
    episodes = []
    done = np.zeros(squad.B, dtype=bool)
    for _ in range(ticks):
        a = step_fn(obs, done, squad)
        obs, rew, done, infos = squad.step(a)
        kills += [i["kill"] for i in infos]
        lives += [i["life_lost"] for i in infos]
        episodes.extend(
            {"fly": i, **inf["episode"]}
            for i, inf in enumerate(infos)
            if "episode" in inf
        )
    per = ticks / 900.0
    return kills / per, lives / per, episodes


def load_policy_runner(ck: dict, squad: FlySquad, dev: torch.device, sample: bool = False):
    arch = ck.get("arch", "mlp")
    hidden = int(ck.get("hidden", 256))
    mean = ck["norm_mean"]
    var = ck["norm_var"]
    var_floor = float(ck.get("norm_var_floor", 0.0))
    clip = ck.get("norm_clip", None)

    def normalize_obs(o: np.ndarray) -> np.ndarray:
        z = (o - mean) / np.sqrt(np.maximum(var, var_floor) + 1e-8)
        if clip is not None:
            z = np.clip(z, -float(clip), float(clip))
        return z.astype(np.float32)

    if arch == "mlp":
        net = ActorCritic(squad.n_features, ck["n_actions"], hidden=hidden).to(dev)
        net.load_state_dict(ck["net"])
        net.eval()

        def step_mlp(obs: np.ndarray, done_prev: np.ndarray, sq: FlySquad) -> np.ndarray:
            x = normalize_obs(obs)
            with torch.no_grad():
                logits, _ = net(torch.from_numpy(x).to(dev))
                if sample:
                    act = torch.distributions.Categorical(logits=logits).sample()
                else:
                    act = logits.argmax(-1)
            return act.cpu().numpy()

        return step_mlp

    net = RecurrentActorCritic(squad.n_features, n_motor=4, hidden=hidden).to(dev)
    net.load_state_dict(ck["net"])
    net.eval()
    _, f_to_a = build_action_codec(squad.action_names)
    B = squad.B
    h = torch.zeros((B, hidden), dtype=torch.float32, device=dev)
    prev_fact = np.tile(np.array([1, 1, 0], dtype=np.int64), (B, 1))
    since_fire = np.full(B, 20, dtype=np.int64)

    def step_gru(obs: np.ndarray, done_prev: np.ndarray, sq: FlySquad) -> np.ndarray:
        nonlocal prev_fact, since_fire
        if np.any(done_prev):
            for i in np.flatnonzero(done_prev):
                h[i].zero_()
                prev_fact[i] = (1, 1, 0)
                since_fire[i] = 20
        x = normalize_obs(obs)
        m = motor_efference(prev_fact, since_fire)
        with torch.no_grad():
            (l_th, l_st, l_fi), _, h_next = net.step(
                torch.from_numpy(x).to(dev),
                torch.from_numpy(m).to(dev),
                h,
            )
            h.copy_(h_next)
            if sample:
                th = torch.distributions.Categorical(logits=l_th).sample().cpu().numpy()
                st = torch.distributions.Categorical(logits=l_st).sample().cpu().numpy()
                fi = torch.distributions.Categorical(logits=l_fi).sample().cpu().numpy()
            else:
                th = l_th.argmax(-1).cpu().numpy()
                st = l_st.argmax(-1).cpu().numpy()
                fi = l_fi.argmax(-1).cpu().numpy()
        f_exec = np.stack([th, st, fi], axis=-1)
        a = f_to_a[th, st, fi]
        since_fire = np.where(fi == 1, 0, since_fire + 1)
        prev_fact = f_exec
        return a

    return step_gru


def squad_kwargs_from_ckpt(ck: dict) -> dict:
    ck_args = ck.get("args", {})
    ablate = "none" if ck["brain"] == "connectome" else ck["brain"]
    legacy_eyes = ck_args.get("legacy_eyes", "arch" not in ck)
    eye_fovy = None if legacy_eyes else ck_args.get("eye_fovy", 54.0)
    converge_eyes = not legacy_eyes
    center_y = ck_args.get("center_y", 0.5 if legacy_eyes else 0.46)
    return {
        "brain_kwargs": {
            "ablate": ablate,
            "ablate_seed": int(ck.get("ablate_seed", ck_args.get("ablate_seed", 0))),
            "weight_scale": float(ck.get("weight_scale", ck_args.get("weight_scale", 1.0))),
        },
        "features": ck["features"],
        "injection": ck_args.get("injection", "v4"),
        "zoom": float(ck_args.get("zoom", 2.0)),
        "eye_offset": float(ck_args.get("eye_offset", 0.15)),
        "retina_center": (0.5, float(center_y)),
        "eye_fovy": eye_fovy,
        "converge_eyes": converge_eyes,
        "suppress_turn_transients": bool(ck_args.get("suppress_turn_transients", False)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104])
    ap.add_argument("--ticks", type=int, default=1800)
    ap.add_argument("--neural-seed", type=int, default=0)
    ap.add_argument("--sample", action="store_true", help="sample stochastic actions instead of greedy argmax")
    ap.add_argument("--ckpts", nargs="+", default=[
        "out/readout_connectome_brain_s0.pt",
        "out/readout_rewired_brain_s0.pt",
        "out/readout_connectome_retina_s0.pt",
    ])
    ap.add_argument("--out", default="out/eval_readout.json")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda")
    results = {}
    episode_records = {}
    protocols = {}
    B = len(args.seeds)

    if not args.skip_baselines:
        squad = FlySquad(batch=B, seeds=args.seeds, neural_seed=args.neural_seed,
                         features="brain", episode_ticks=900)
        rng = np.random.default_rng(0)
        k, l, episodes = run_squad_policy(
            squad, lambda o, d, s: rng.integers(0, s.n_actions, size=s.B), args.ticks
        )
        episode_records["random policy"] = episodes
        protocols["random policy"] = squad.protocol
        results["random policy"] = (k, l)
        print(
            f"{'random policy':36s} kills {k.mean():.2f} ± {k.std():.2f}  "
            f"lives {l.mean():.2f} ± {l.std():.2f}  util {(k - 0.5 * l).mean():+.2f}  per 900 ticks over {B} seeds",
            flush=True,
        )

        ctrls = [
            CockpitControls(squad.action_names, calibration="pool4", saccade_hold=3,
                            startle_refractory=15, steer_mode="li")
            for _ in range(B)
        ]
        obs = squad.reset()
        kills = np.zeros(B)
        lives = np.zeros(B)
        episodes = []
        for _ in range(args.ticks):
            a = np.array([c.to_action() for c in ctrls])
            obs, rew, done, infos = squad.step(a)
            for i, c in enumerate(ctrls):
                if done[i]:
                    ctrls[i] = CockpitControls(
                        squad.action_names, calibration="pool4", saccade_hold=3,
                        startle_refractory=15, steer_mode="li",
                    )
                    episodes.append({"fly": i, **infos[i]["episode"]})
                else:
                    c.update(squad.last_dn[i], squad.dt)
            kills += [i["kill"] for i in infos]
            lives += [i["life_lost"] for i in infos]
        k, l = kills / (args.ticks / 900.0), lives / (args.ticks / 900.0)
        episode_records["reflex pilot (legacy)"] = episodes
        protocols["reflex pilot (legacy)"] = squad.protocol
        results["reflex pilot (legacy)"] = (k, l)
        print(
            f"{'reflex pilot (legacy)':36s} kills {k.mean():.2f} ± {k.std():.2f}  "
            f"lives {l.mean():.2f} ± {l.std():.2f}  util {(k - 0.5 * l).mean():+.2f}",
            flush=True,
        )
        squad.close()

        # Active binocular reflex baseline on the same game & neural seeds
        squad_act = FlySquad(
            batch=B, seeds=args.seeds, neural_seed=args.neural_seed,
            features="brain", episode_ticks=900, injection="v4",
            zoom=1.15, eye_offset=0.11, retina_center=(0.5, 0.46),
            eye_fovy=54.0, converge_eyes=True, suppress_turn_transients=True,
        )
        ctrls_act = [make_teacher(squad_act.action_names, i) for i in range(B)]
        squad_act.reset()
        kills = np.zeros(B)
        lives = np.zeros(B)
        episodes = []
        for _ in range(args.ticks):
            a = np.array([c.to_action() for c in ctrls_act])
            _, _, done, infos = squad_act.step(a)
            for i, c in enumerate(ctrls_act):
                if done[i]:
                    ctrls_act[i] = make_teacher(squad_act.action_names, i)
                    episodes.append({"fly": i, **infos[i]["episode"]})
                else:
                    c.update(squad_act.last_dn[i], squad_act.dt)
            kills += [i["kill"] for i in infos]
            lives += [i["life_lost"] for i in infos]
        k, l = kills / (args.ticks / 900.0), lives / (args.ticks / 900.0)
        episode_records["reflex pilot (active binocular)"] = episodes
        protocols["reflex pilot (active binocular)"] = squad_act.protocol
        results["reflex pilot (active binocular)"] = (k, l)
        print(
            f"{'reflex pilot (active binocular)':36s} kills {k.mean():.2f} ± {k.std():.2f}  "
            f"lives {l.mean():.2f} ± {l.std():.2f}  util {(k - 0.5 * l).mean():+.2f}",
            flush=True,
        )
        squad_act.close()

    for ck_path in args.ckpts:
        ck = torch.load(ck_path, weights_only=False)
        sq_kw = squad_kwargs_from_ckpt(ck)
        squad = FlySquad(
            batch=B, seeds=args.seeds, neural_seed=args.neural_seed,
            episode_ticks=900, **sq_kw,
        )
        runner = load_policy_runner(ck, squad, dev, sample=args.sample)
        k, l, episodes = run_squad_policy(squad, runner, args.ticks)
        name = f"{ck['brain']} ({ck['features']}, {ck.get('arch', 'mlp')}) {Path(ck_path).stem.replace('readout_', '')}"
        episode_records[name] = episodes
        protocols[name] = squad.protocol
        results[name] = (k, l)
        print(
            f"{name:36s} kills {k.mean():.2f} ± {k.std():.2f}  "
            f"lives {l.mean():.2f} ± {l.std():.2f}  util {(k - 0.5 * l).mean():+.2f}",
            flush=True,
        )
        squad.close()

    OUT.mkdir(exist_ok=True)
    dump_json(
        Path(args.out),
        {
            n: {
                "kills": k.tolist(),
                "lives": l.tolist(),
                "utility": (k - 0.5 * l).tolist(),
                "kills_mean": float(k.mean()),
                "lives_mean": float(l.mean()),
                "utility_mean": float((k - 0.5 * l).mean()),
                "episodes": episode_records[n],
                "protocol": protocols[n],
            }
            for n, (k, l) in results.items()
        },
        args=args,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
