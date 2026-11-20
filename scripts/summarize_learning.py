"""Tables for the learning experiments from the logs in out/.

plasticity_<tag>.jsonl : per-episode rows from train_plasticity.py
readout_<tag>.jsonl    : per-update rows from train_readout.py
eval_*.json            : from eval_readout.py

Usage: python scripts/summarize_learning.py [--glob plasticity_on_vis_dn*]
"""

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "out"


def rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def halves(vals):
    v = np.asarray(vals, float); h = v.size // 2
    return v[:h].mean(), v[h:].mean()


def plasticity(pattern):
    print(f"| run | episodes | kills first / second half | lives lost first / second half | synapses changed | mean drift | PAM / PPL1 Hz |")
    print("|---|---|---|---|---|---|---|")
    for p in sorted(OUT.glob(f"{pattern}.jsonl")):
        r = rows(p)
        if not r or "kills" not in r[0]:
            continue
        r.sort(key=lambda x: x["global_episode"])
        k0, k1 = halves([x["kills"] for x in r]); l0, l1 = halves([x["lives_lost"] for x in r])
        last = r[-1]
        print(f"| {p.stem.replace('plasticity_', '')} | {len(r)} | {k0:.2f} / {k1:.2f} | {l0:.2f} / {l1:.2f} | "
              f"{last['frac_changed'] * 100:.1f}% | {last['drift'] * 100:.1f}% | "
              f"{np.mean([x['pam_hz'] for x in r]):.1f} / {np.mean([x['ppl1_hz'] for x in r]):.1f} |")


def readout(pattern, window=50):
    print(f"| run | updates | ticks | kills per episode first / last {window} updates | lives lost first / last | entropy at end |")
    print("|---|---|---|---|---|---|")
    for p in sorted(OUT.glob(f"{pattern}.jsonl")):
        r = [x for x in rows(p) if x.get("ep_kills") is not None]
        if not r:
            continue
        a, b = r[:window], r[-window:]
        print(f"| {p.stem.replace('readout_', '')} | {r[-1]['update']} | {r[-1]['ticks']} | "
              f"{np.mean([x['ep_kills'] for x in a]):.2f} / {np.mean([x['ep_kills'] for x in b]):.2f} | "
              f"{np.mean([x['ep_lives'] for x in a]):.2f} / {np.mean([x['ep_lives'] for x in b]):.2f} | "
              f"{r[-1]['entropy']:.2f} |")


def evals(pattern):
    for p in sorted(OUT.glob(f"{pattern}.json")):
        d = json.loads(p.read_text())
        if "conditions" in d and "start_audit" in d:
            print(f"\n{p.name} ({d['status']}, teaching {d['external_teaching']})\n"
                  "| frozen brain | episodes | kills per episode | lives lost per episode |\n"
                  "|---|---:|---:|---:|")
            for name, result in d["conditions"].items():
                s = result["summary"]
                print(f"| {name} | {s['episodes']} | {s['kills_mean']:.3f} | "
                      f"{s['lives_lost_mean']:.3f} |")
            continue
        print(f"\n{p.name}\n| policy | kills per 900 ticks | lives lost per 900 ticks |\n|---|---|---|")
        for name, v in d.items():
            if not isinstance(v, dict) or "kills" not in v or "lives" not in v:
                continue
            k, l = np.asarray(v["kills"]), np.asarray(v["lives"])
            print(f"| {name} | {k.mean():.2f} ± {k.std():.2f} | {l.mean():.2f} ± {l.std():.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plasticity", default="plasticity_*")
    ap.add_argument("--readout", default="readout_*")
    ap.add_argument("--evals", default="eval_*")
    args = ap.parse_args()
    print("## Plasticity (reflex pilot plays; weights learn)\n"); plasticity(args.plasticity)
    print("\n## Trained readouts (PPO on the frozen brain)\n"); readout(args.readout)
    print("\n## Greedy evaluations on fresh seeds"); evals(args.evals)


if __name__ == "__main__":
    main()
