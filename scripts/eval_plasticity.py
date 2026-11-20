"""Frozen weights on matched, audited starts, with no external teaching pulses.

Historical checkpoints retain their original training-protocol limitations.
This evaluates those particular weights; it does not create independent
training histories or establish a general learning benefit.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from noflyzone.controls import CockpitControls
from noflyzone.game import BattleZone
from noflyzone.provenance import dump_json
from noflyzone.rl_env import EPISODE_PROTOCOL, FlySquad, episode_seed

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = [
    f"out/plasticity_on_vis_dn_latch_s{i}_weights.npz" for i in range(4)
] + [
    "out/plasticity_chimera_s0_with_s2_loom_weights.npz",
    "out/plasticity_chimera_s2_with_s0_loom_weights.npz",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_starts(seeds, episodes):
    """Audit the exact planned starts and a common action trace before scoring."""
    game = BattleZone(auto_reset=False)
    rows = []
    actions = np.random.default_rng(123).integers(game.n_actions, size=450)
    try:
        for root in seeds:
            for ep in range(episodes):
                seed, warmup = episode_seed(root, ep, 0), episode_seed(root, ep, 1)
                start = game.reset(seed=seed, warmup_seed=warmup)
                assert game.reset(seed=seed, warmup_seed=warmup) == start
                digest = hashlib.sha256()
                for action in actions:
                    frame, _, _, ended = game.step(int(action))
                    digest.update(frame.tobytes())
                    if ended:
                        break
                rows.append({"game_seed_root": root, "episode_index": ep,
                             **start, "action_trace_sha256": digest.hexdigest()})
    finally:
        game.close()
    if len({r["action_trace_sha256"] for r in rows}) != len(rows):
        raise RuntimeError("planned starts do not produce distinct action traces")
    return rows


def controller(squad, calibration):
    return CockpitControls(squad.action_names, calibration=calibration,
                           saccade_hold=3, startle_refractory=15, steer_mode="li")


def run_condition(squad, episodes, calibration, planned):
    squad.reset()
    ctrls = [controller(squad, calibration) for _ in range(squad.B)]
    completed = np.zeros(squad.B, np.int64)
    rows = []
    start_time = time.monotonic()
    while np.any(completed < episodes):
        actions = np.array([c.to_action() for c in ctrls])
        _, _, done, infos = squad.step(actions, active=completed < episodes)
        for i, c in enumerate(ctrls):
            if completed[i] >= episodes:
                continue
            if done[i]:
                ep = infos[i]["episode"]
                start = ep["start"]
                expected = planned[(start["game_seed_root"], start["episode_index"])]
                for key in ("start_frame_sha256", "start_ram_sha256", "warmup_actions"):
                    if start[key] != expected[key]:
                        raise RuntimeError(f"scored start differs from audited start: {key}")
                rows.append({"fly": i, **ep})
                completed[i] += 1
                ctrls[i] = controller(squad, calibration)
                print(f"  {len(rows)}/{squad.B * episodes} episodes: fly {i}, "
                      f"kills {ep['kills']}, lives {ep['lives_lost']}, "
                      f"ticks {ep['ticks']} ({time.monotonic() - start_time:.0f}s)", flush=True)
            else:
                c.update(squad.last_dn[i], squad.dt)
    rows.sort(key=lambda r: (r["fly"], r["start"]["episode_index"]))
    return rows


def summarise(rows, baseline):
    kills = np.array([r["kills"] for r in rows])
    lives = np.array([r["lives_lost"] for r in rows])
    ticks = sum(r["ticks"] for r in rows)
    return {"episodes": len(rows), "kills_mean": float(kills.mean()),
            "lives_lost_mean": float(lives.mean()), "ticks_total": ticks,
            "kills_per_900_ticks": float(kills.sum() * 900 / ticks),
            "lives_lost_per_900_ticks": float(lives.sum() * 900 / ticks),
            "paired_kills_delta": float(np.mean(kills - [r["kills"] for r in baseline])),
            "paired_lives_delta": float(np.mean(lives - [r["lives_lost"] for r in baseline]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", nargs="+", default=DEFAULT_WEIGHTS)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[900, 901, 902, 903])
    ap.add_argument("--neural-seed", type=int, default=250925)
    ap.add_argument("--episodes", type=int, default=4, help="independently reset episodes per seed")
    ap.add_argument("--episode-ticks", type=int, default=900)
    ap.add_argument("--calibration", default="pool4")
    ap.add_argument("--out", type=Path, default=Path("out/eval_plasticity_2026-09-25.json"))
    args = ap.parse_args()
    if args.episodes < 1 or args.episode_ticks < 1 or len(set(args.seeds)) != len(args.seeds):
        ap.error("episode counts must be positive and seed roots must be distinct")
    if args.out.exists():
        ap.error(f"output already exists: {args.out}; choose a new --out")
    checkpoints = []
    for path in ([] if args.baseline_only else args.weights):
        with np.load(path, allow_pickle=False) as z:
            provenance = json.loads(str(z["provenance"])) if "provenance" in z else None
            checkpoints.append({"name": Path(path).stem.removeprefix("plasticity_"),
                                "path": path, "sha256": sha256(path),
                                "training_provenance": provenance,
                                **{k: z[k].copy() for k in ("pos", "pre", "post", "w0", "w")}})
    if len({ck["name"] for ck in checkpoints}) != len(checkpoints):
        ap.error("checkpoint names must be distinct")
    audit = audit_starts(args.seeds, args.episodes)
    print(f"Audited {len(audit)} distinct, reproducible starts and action traces.", flush=True)
    planned = {(r["game_seed_root"], r["episode_index"]): r for r in audit}
    payload = {"protocol": EPISODE_PROTOCOL, "status": "running",
               "external_teaching": "off", "learning": "off; all weights frozen",
               "training_caveat": "Training metadata is retained per checkpoint when available; "
                   "frozen evaluation alone does not validate training histories.",
               "audit_action_seed": 123, "audit_action_ticks": 450,
               "start_audit": audit, "conditions": {},
               "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                   [*sorted((ROOT / "src/noflyzone").glob("*.py")), Path(__file__).resolve()]}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    squad = FlySquad(batch=len(args.seeds), seeds=args.seeds, neural_seed=args.neural_seed,
                     episode_ticks=args.episode_ticks)
    try:
        baseline_edges = []
        for ck in checkpoints:
            pos = squad.brain.csc_positions(ck["pre"], ck["post"])
            if not np.array_equal(pos, ck["pos"]) or (pos < 0).any():
                raise ValueError(f"checkpoint edge positions do not match: {ck['path']}")
            if not np.isfinite(ck["w"]).all():
                raise ValueError(f"non-finite checkpoint weights: {ck['path']}")
            if not np.array_equal(squad.brain.W.data[pos], ck["w0"]):
                raise ValueError(f"checkpoint measured weights do not match: {ck['path']}")
            baseline_edges.append((pos, squad.brain.W.data[pos].copy()))
        dump_json(args.out, payload, args=args, brain=squad.brain)
        baseline = None
        for ck in [None, *checkpoints]:
            for pos, measured in baseline_edges:
                squad.brain.set_edge_weights(pos, measured)
            name = "measured" if ck is None else ck["name"]
            if ck is not None:
                squad.brain.set_edge_weights(ck["pos"], ck["w"])
            before = hashlib.sha256(squad.brain.W.data.tobytes()).hexdigest()
            print(f"Evaluating {name}", flush=True)
            rows = run_condition(squad, args.episodes, args.calibration, planned)
            if hashlib.sha256(squad.brain.W.data.tobytes()).hexdigest() != before:
                raise RuntimeError("weights changed during frozen evaluation")
            if baseline is None:
                baseline = rows
            for row, ref in zip(rows, baseline):
                if row["start"] != ref["start"]:
                    raise RuntimeError("conditions used different game or neural starts")
            summary = summarise(rows, baseline)
            payload["conditions"][name] = {"checkpoint": None if ck is None else
                {k: ck[k] for k in ("path", "sha256", "training_provenance")}, "weights_sha256": before,
                "summary": summary, "episodes": rows}
            dump_json(args.out, payload, args=args, brain=squad.brain)
            print(f"{name}: kills {summary['kills_mean']:.3f}, "
                  f"lives {summary['lives_lost_mean']:.3f} per episode", flush=True)
        payload["status"] = "complete"
        dump_json(args.out, payload, args=args, brain=squad.brain)
    finally:
        squad.close()
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
