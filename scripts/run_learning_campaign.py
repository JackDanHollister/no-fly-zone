"""Run the fixed September 25 learning screen, recording every step.

The JSON plan is frozen and hashed before any training. No resume, tuning,
checkpoint selection or retry: errors and the wall limit mean incomplete.
Training/evaluation logs and checkpoints remain in out/; the final compact
report and audited episode evidence can be copied to docs/research/evidence/.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from eval_plasticity import audit_starts, sha256
from noflyzone.provenance import stamp
from noflyzone.rl_env import EPISODE_PROTOCOL, episode_seed

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "docs/research/evidence/learning-campaign-2026-09-25-plan.json"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def tag(plan, history, condition):
    return f"{plan['campaign']}_h{history}_{condition}"


def checkpoint_name(plan, history, condition):
    return f"{tag(plan, history, condition)}_weights"


def rates(rows):
    ticks = sum(r["ticks"] for r in rows)
    if not rows or ticks <= 0:
        raise ValueError("cannot score an empty condition")
    kills = sum(r["kills"] for r in rows)
    losses = sum(r["lives_lost"] for r in rows)
    return {"kills": kills, "lives_lost": losses, "ticks": ticks,
            "kills_per_900_ticks": kills * 900 / ticks,
            "lives_lost_per_900_ticks": losses * 900 / ticks,
            "utility_per_900_ticks": (kills - 0.5 * losses) * 900 / ticks}


def analyse_panel(plan, payload):
    """Score ALL histories, independently in each panel, from episode rows."""
    if payload["status"] != "complete":
        raise ValueError("evaluation is incomplete")
    conditions = payload["conditions"]
    n = len(plan["training"]["seeds"])
    expected = {"measured"} | {checkpoint_name(plan, h, c)
        for h in range(n) for c in ("on", "shuffled")}
    if set(conditions) != expected:
        raise ValueError("evaluation conditions differ from the plan")
    count = len(plan["evaluation"]["panels"][0]["seeds"]) * plan["evaluation"]["episodes_per_fly"]
    if any(len(c["episodes"]) != count for c in conditions.values()):
        raise ValueError("evaluation episode quota differs from the plan")
    scored = {name: rates(c["episodes"]) for name, c in conditions.items()}
    keys = ("kills_per_900_ticks", "lives_lost_per_900_ticks", "utility_per_900_ticks")
    means = {c: {k: float(np.mean([scored[checkpoint_name(plan, h, c)][k]
                for h in range(n)])) for k in keys} for c in ("on", "shuffled")}
    base = scored["measured"]
    decision = plan["decision"]
    wins = [h for h in range(n)
            if scored[checkpoint_name(plan, h, "on")][keys[2]] > base[keys[2]]
            and scored[checkpoint_name(plan, h, "on")][keys[2]] >
                scored[checkpoint_name(plan, h, "shuffled")][keys[2]]]
    gates = {"history_consistency": len(wins) >= decision["minimum_histories_better_than_both_controls"]}
    for label, control in (("measured", base), ("shuffled", means["shuffled"])):
        gates[f"utility_vs_{label}"] = (means["on"][keys[2]] - control[keys[2]] >=
                                          decision["minimum_utility_gain"])
        gates[f"kills_vs_{label}"] = (means["on"][keys[0]] >=
            decision["minimum_kill_rate_ratio"] * control[keys[0]])
        gates[f"lives_vs_{label}"] = (means["on"][keys[1]] <=
            decision["maximum_life_loss_rate_ratio"] * control[keys[1]])
    return {"pass": all(gates.values()), "gates": gates, "histories_beating_both": wins,
            "measured": base, "history_means": means, "conditions": scored}


def validate_training(plan, history, condition, audit):
    name = tag(plan, history, condition)
    rows = [json.loads(line) for line in (ROOT / f"out/plasticity_{name}.jsonl").read_text().splitlines()]
    training = plan["training"]
    planned = {(r["game_seed_root"], r["episode_index"]): r for r in audit}
    expected = {(training["seeds"][history] * 100 + fly, ep)
                for fly in range(training["batch"]) for ep in range(training["episodes_per_fly"])}
    actual = [(r["start"]["game_seed_root"], r["start"]["episode_index"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f"training quota or starts differ: {name}")
    for row in rows:
        start = row["start"]
        key = (start["game_seed_root"], start["episode_index"])
        wanted = planned[key]
        for field in ("game_seed", "warmup_seed", "warmup_actions", "start_lives",
                      "start_frame_sha256", "start_ram_sha256"):
            if start[field] != wanted[field]:
                raise ValueError(f"training start mismatch: {name}, {key}, {field}")
        neural_root = training["neural_seeds"][history]
        if (start["neural_seed_root"] != neural_root or
                start["neural_seed"] != episode_seed(neural_root, *key, 2) or
                start["neural_row"] != row["fly"] or start["protocol"] != EPISODE_PROTOCOL):
            raise ValueError(f"training neural seed mismatch: {name}")
        if not 0 < row["ticks"] <= training["episode_ticks"]:
            raise ValueError(f"invalid training duration: {name}")
    weights = ROOT / f"out/plasticity_{name}_weights.npz"
    with np.load(weights, allow_pickle=False) as z:
        if not np.isfinite(z["w"]).all():
            raise ValueError(f"non-finite weights: {name}")
        unchanged = bool(np.array_equal(z["w"], z["w0"]))
        if condition == "lr0" and not unchanged:
            raise ValueError(f"rule-off weights changed: {name}")
        info = {"weights": str(weights.relative_to(ROOT)), "sha256": sha256(weights),
                "edges": z["w"].size, "changed_edges": int(np.count_nonzero(z["w"] != z["w0"])),
                "unchanged": unchanged, "episodes": rows, "totals": rates(rows)}
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["protocol"] != EPISODE_PROTOCOL:
        parser.error("plan names a different episode protocol")
    destination = ROOT / "out" / f"campaign_{plan['campaign']}"
    destination.mkdir(exist_ok=False)
    start = time.monotonic()
    deadline = start + plan["budget"]["maximum_wall_seconds"]
    paths = [*sorted((ROOT / "src/noflyzone").glob("*.py")), Path(__file__).resolve(),
             ROOT / "scripts/train_plasticity.py", ROOT / "scripts/eval_plasticity.py"]
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    record = {"status": "preflight", "plan": plan, "plan_sha256": sha256(args.plan),
              "source_sha256": sources, "provenance": stamp(), "training": {}, "evaluation": {}}
    manifest = destination / "manifest.json"
    write_json(destination / "plan.json", plan)
    write_json(manifest, record)

    def check_frozen():
        if sha256(args.plan) != record["plan_sha256"]:
            raise RuntimeError("plan changed during campaign")
        for path, digest in sources.items():
            if sha256(ROOT / path) != digest:
                raise RuntimeError(f"source changed during campaign: {path}")
        if time.monotonic() >= deadline:
            raise TimeoutError("campaign wall budget exhausted")

    def run(name, command):
        check_frozen()
        record["current"] = {"name": name, "command": command,
                             "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        record["elapsed_seconds"] = time.monotonic() - start
        write_json(manifest, record)
        print(f"RUN {name} -> {destination / (name + '.log')}", flush=True)
        with (destination / f"{name}.log").open("x") as log:
            subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True,
                           timeout=max(0.01, deadline - time.monotonic()),
                           env={**os.environ, "MUJOCO_GL": "egl"})
        check_frozen()

    try:
        training, evaluation = plan["training"], plan["evaluation"]
        audits = {}
        for h, seed in enumerate(training["seeds"]):
            print(f"Auditing training history {h}", flush=True)
            audits[f"h{h}"] = audit_starts([seed * 100 + i for i in range(training["batch"])],
                                           training["episodes_per_fly"])
        for panel in evaluation["panels"]:
            print(f"Auditing reserved panel {panel['name']}", flush=True)
            audits[panel["name"]] = audit_starts(panel["seeds"], evaluation["episodes_per_fly"])
        all_starts = [row for rows in audits.values() for row in rows]
        if len({r["action_trace_sha256"] for r in all_starts}) != len(all_starts):
            raise RuntimeError("overlap between training and reserved start traces")
        write_json(destination / "start_audit.json", audits)
        record["start_audit_sha256"] = sha256(destination / "start_audit.json")
        record["distinct_starts"] = len(all_starts)
        record["status"] = "training"
        for h, seed in enumerate(training["seeds"]):
            matched = None
            for condition in training["condition_order"][h]:
                config = training["conditions"][condition]
                name = tag(plan, h, condition)
                command = ["scripts/train_plasticity.py", "--tag", name,
                           "--rule", config["rule"], "--edge-set", config["edge_set"],
                           "--seed", str(seed), "--neural-seed", str(training["neural_seeds"][h]),
                           "--edge-seed", str(seed), "--batch", str(training["batch"]),
                           "--episodes", str(training["episodes_per_fly"]),
                           "--episode-ticks", str(training["episode_ticks"])]
                for key in ("eta", "eligibility", "scale", "elig_cap", "teaching", "dopamine", "readout", "calibration"):
                    command.extend([f"--{key.replace('_', '-')}", str(training[key])])
                if training["latch"]:
                    command.append("--latch")
                run(name, command)
                result = validate_training(plan, h, condition, audits[f"h{h}"])
                starts = sorted([r["start"] for r in result["episodes"]],
                                key=lambda r: (r["game_seed_root"], r["episode_index"]))
                if matched is not None and starts != matched:
                    raise RuntimeError("training controls did not share game and neural starts")
                matched = starts
                record["training"][name] = result
                write_json(manifest, record)
                print(f"DONE {name}: {len(result['episodes'])} episodes, "
                      f"{result['changed_edges']} changed edges", flush=True)
        weights = [record["training"][tag(plan, h, c)]["weights"]
                   for h in range(len(training["seeds"])) for c in ("on", "shuffled")]
        record["status"] = "evaluating"
        for panel in evaluation["panels"]:
            name = f"eval_{panel['name']}"
            output = destination / f"{name}.json"
            run(name, ["scripts/eval_plasticity.py", "--weights", *weights,
                       "--seeds", *map(str, panel["seeds"]),
                       "--neural-seed", str(panel["neural_seed"]),
                       "--episodes", str(evaluation["episodes_per_fly"]),
                       "--episode-ticks", str(evaluation["episode_ticks"]),
                       "--calibration", training["calibration"], "--out", str(output)])
            payload = json.loads(output.read_text())
            if payload["start_audit"] != audits[panel["name"]]:
                raise RuntimeError("evaluation start audit changed since preflight")
            for h in range(len(training["seeds"])):
                for c in ("on", "shuffled"):
                    checkpoint = payload["conditions"][checkpoint_name(plan, h, c)]["checkpoint"]
                    if checkpoint["sha256"] != record["training"][tag(plan, h, c)]["sha256"]:
                        raise RuntimeError("checkpoint changed between training and evaluation")
            record["evaluation"][panel["name"]] = analyse_panel(plan, payload)
            write_json(manifest, record)
        check_frozen()
        record["pass"] = all(p["pass"] for p in record["evaluation"].values())
        record["decision"] = plan["decision"]["if_pass" if record["pass"] else "if_fail"]
        record["status"] = "complete"
        print(record["decision"], flush=True)
    except BaseException as error:
        record["status"] = "incomplete"
        record["error"] = f"{type(error).__name__}: {error}"
        record["decision"] = plan["decision"]["if_incomplete"]
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - start
        write_json(manifest, record)
        print(f"Campaign {record['status']}; record: {manifest}", flush=True)


if __name__ == "__main__":
    main()
