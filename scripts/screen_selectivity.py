"""Selectivity screen: what do the descending neurons compute beyond which
eye saw it? A within-hemisphere rewiring keeps left from right by anatomy
(docs/research/wiring-controls-2026-09-18.md), so this screen measures the
things such a shuffle cannot inherit: size tuning (a small object versus a
loom), radial versus wide-field selectivity (a loom versus a full-field
flash), and direction selectivity (a bar moving one way versus the other
within one eye), for every descending type, in the measured brain and in
its controls. Every brain gets the same procedure; nothing is chosen on the
measured brain and then applied to a control.

Brain tokens: measured, rewired[@seed], rewired_ipsi[@seed], optionally
with x<weight_scale>. Output: out/screen_selectivity.json and a table.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_frontend import DT_GAME, stimuli  # noqa: E402
from screen_dns import run  # noqa: E402

from noflyzone.brain_backend import BACKENDS, describe, make_brain  # noqa: E402
from noflyzone.brain_connectome import ConnectomeBrain  # noqa: E402
from noflyzone.brain_lif import annotations  # noqa: E402
from noflyzone.retina import HexRetina  # noqa: E402
from noflyzone.provenance import dump_json

OUT = Path(__file__).resolve().parent.parent / "out"
RESPONSIVE_HZ = 5.0   # a type must exceed its dark rate by this much on some stimulus
SEL = 1.0 / 3.0       # index threshold: the preferred stimulus at least 2x the other


def parse(token):
    m = re.fullmatch(r"(measured|rewired|rewired_ipsi)(?:@(\d+))?(?:x([\d.]+))?", token)
    if not m:
        raise SystemExit(f"bad brain token {token!r}")
    name, seed, scale = m.groups()
    return {"ablate": "none" if name == "measured" else name,
            "ablate_seed": int(seed or 0), "weight_scale": float(scale or 1.0)}


def index(a, b):
    return (a - b) / (a + b + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, default="auto")
    ap.add_argument("--injection", default="v4")
    ap.add_argument("--brains", nargs="+",
                    default=["measured", "rewired_ipsi@0", "rewired_ipsi@1", "rewired_ipsi@2", "rewired@0"])
    ap.add_argument("--aminergic", default="zero")
    ap.add_argument("--sensory", default="external_only")
    ap.add_argument("--dataset", default="783", help="brain_lif.DATASETS key: 783 (FlyWire) or malecns")
    ap.add_argument("--out", default="screen_selectivity.json")
    args = ap.parse_args()

    retina = HexRetina()
    xy = retina.centers_xy.astype(np.float64)
    from scipy.spatial import cKDTree
    spacing = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    st = stimuli(xy, spacing)
    dark = [np.zeros(len(xy), np.float32)] * 12
    # (frames, eye) per condition; each stimulus is shown to each eye in turn
    # a receding disc: the loom played backwards, same edges, opposite
    # direction of expansion; the control that tells expansion from "an edge
    # moved" (nsfm/fly-afterlife withdrew a loom claim on exactly this)
    recede = st["loom"][:2] + st["loom"][2:12][::-1] + st["loom"][12:]
    base = {"dark": dark, "small": st["small_disc_LtoR"] + st["small_disc_RtoL"], "recede": recede,
            "small_LtoR": st["small_disc_LtoR"], "small_RtoL": st["small_disc_RtoL"],
            "tank": st["disc_r1.5"], "disc6": st["disc_r6"], "loom": st["loom"],
            "flash": st["full_flash"], "bar_LtoR": st["bar_LtoR"], "bar_RtoL": st["bar_RtoL"]}
    conds = {f"{k}@{e[0].upper()}": (v, e) for k, v in base.items() for e in ("left", "right")}

    results = {}
    for token in args.brains:
        kw = parse(token)
        core = make_brain(args.backend, version=args.dataset, aminergic=args.aminergic,
                          sensory_input=args.sensory, **kw)
        ann = core.annotations().drop_duplicates("root_id").set_index("root_id").reindex(core.root_ids)
        ct = ann["cell_type"].astype(str).to_numpy()
        side = ann["side"].astype(str).to_numpy()
        dn = ann["super_class"].astype(str).to_numpy() == "descending"
        types = sorted(set(ct[dn]))
        idx = {t: {s: np.flatnonzero((ct == t) & (side == s)) for s in ("left", "right")} for t in types}
        brain = ConnectomeBrain(retina=retina, brain=core, injection=args.injection)
        print(f"\n########## {token}: {describe(core)} | injection {args.injection}", flush=True)
        res = {c: run(brain, f, e, xy) for c, (f, e) in conds.items()}
        hz = {c: r[0] for c, r in res.items()}
        regime = float(np.mean([r[1] for r in res.values()]))
        opp = {"left": "right", "right": "left"}
        rows = []
        for t in types:
            if not (idx[t]["left"].size and idx[t]["right"].size):
                continue
            # ipsi response: the stimulated eye's side, averaged over both eyes
            def ipsi(k):
                return float(np.mean([hz[f"{k}@L"][idx[t]["left"]].mean(), hz[f"{k}@R"][idx[t]["right"]].mean()]))
            def contra(k):
                return float(np.mean([hz[f"{k}@L"][idx[t]["right"]].mean(), hz[f"{k}@R"][idx[t]["left"]].mean()]))
            r = {k: ipsi(k) for k in base}
            rc = {k: contra(k) for k in base}
            d = r["dark"]
            peak = max(r[k] for k in base if k != "dark")
            responsive = peak - d > RESPONSIVE_HZ
            li = index(r["small"], rc["small"])
            # direction selectivity within the stimulated eye; sign must agree across eyes
            ds_l = index(hz["bar_LtoR@L"][idx[t]["left"]].mean(), hz["bar_RtoL@L"][idx[t]["left"]].mean())
            ds_r = index(hz["bar_LtoR@R"][idx[t]["right"]].mean(), hz["bar_RtoL@R"][idx[t]["right"]].mean())
            ds_small_l = index(hz["small_LtoR@L"][idx[t]["left"]].mean(), hz["small_RtoL@L"][idx[t]["left"]].mean())
            ds_small_r = index(hz["small_LtoR@R"][idx[t]["right"]].mean(), hz["small_RtoL@R"][idx[t]["right"]].mean())
            rows.append({
                "type": t, "n": int(idx[t]["left"].size + idx[t]["right"].size),
                "dark": d, "small": r["small"], "small_contra": rc["small"], "tank": r["tank"],
                "disc6": r["disc6"], "loom": r["loom"], "flash": r["flash"], "recede": r["recede"],
                "bar": float(np.mean([r["bar_LtoR"], r["bar_RtoL"]])),
                "responsive": bool(responsive), "LI": li,
                # size tuning: small moving object versus the loom (the largest stimulus)
                "SSI_loom": index(r["small"], r["loom"]),
                # size tuning against the full-field flash
                "SSI_flash": index(r["small"], r["flash"]),
                # radial expansion versus a wide-field flash of the same final extent
                "RSI": index(r["loom"], r["flash"]),
                # expansion versus contraction with the same edges
                "expand_vs_recede": index(r["loom"], r["recede"]),
                # loom versus a wide-field moving bar (self-motion look-alike)
                "loom_vs_bar": index(r["loom"], float(np.mean([r["bar_LtoR"], r["bar_RtoL"]]))),
                "DS_bar": (ds_l, ds_r), "DS_small": (ds_small_l, ds_small_r),
                "DSI_bar": float(np.mean([abs(ds_l), abs(ds_r)])) if np.sign(ds_l) == np.sign(ds_r) != 0 else 0.0,
                "DSI_small": float(np.mean([abs(ds_small_l), abs(ds_small_r)])) if np.sign(ds_small_l) == np.sign(ds_small_r) != 0 else 0.0,
            })
        resp = [x for x in rows if x["responsive"]]
        small_resp = [x for x in resp if x["small"] - x["dark"] > RESPONSIVE_HZ]
        loom_resp = [x for x in resp if x["loom"] - x["dark"] > RESPONSIVE_HZ]
        bar_resp = [x for x in resp if x["bar"] - x["dark"] > RESPONSIVE_HZ]
        summary = {
            "brain": describe(core), "meta": core.meta, "spiking_per_tick": regime,
            "types_tested": len(rows), "responsive": len(resp),
            "small_responsive": len(small_resp), "loom_responsive": len(loom_resp), "bar_responsive": len(bar_resp),
            "lateralised_0.8": sum(abs(x["LI"]) > 0.8 for x in small_resp),
            "small_selective_vs_loom": sum(x["SSI_loom"] > SEL for x in small_resp),
            "small_selective_vs_flash": sum(x["SSI_flash"] > SEL for x in small_resp),
            "loom_selective_vs_flash": sum(x["RSI"] > SEL for x in loom_resp),
            "loom_selective_vs_bar": sum(x["loom_vs_bar"] > SEL for x in loom_resp),
            "expand_selective_vs_recede": sum(x["expand_vs_recede"] > SEL for x in loom_resp),
            "contract_selective_vs_expand": sum(x["expand_vs_recede"] < -SEL for x in loom_resp),
            "best_expand_vs_recede": max((x["expand_vs_recede"] for x in loom_resp), default=float("nan")),
            "direction_selective_bar": sum(x["DSI_bar"] > SEL for x in bar_resp),
            "direction_selective_small": sum(x["DSI_small"] > SEL for x in small_resp),
            "best_small_hz": max((x["small"] for x in rows), default=0.0),
            "best_SSI_loom": max((x["SSI_loom"] for x in small_resp), default=float("nan")),
            "best_RSI": max((x["RSI"] for x in loom_resp), default=float("nan")),
            "best_DSI_bar": max((x["DSI_bar"] for x in bar_resp), default=0.0),
        }
        top = lambda key, pool, n=6: [(x["type"], round(x[key], 2), round(x["small"], 1), round(x["loom"], 1),
                                       round(x["flash"], 1), round(x["bar"], 1))
                                      for x in sorted(pool, key=lambda x: x[key], reverse=True)[:n]]
        print(f"  spiking per tick {regime:.0f}; DN types {len(rows)}, responsive {len(resp)} "
              f"(small {len(small_resp)}, loom {len(loom_resp)}, bar {len(bar_resp)})")
        print(f"  lateralised |LI|>0.8: {summary['lateralised_0.8']}  small-selective vs loom: {summary['small_selective_vs_loom']}  "
              f"vs flash: {summary['small_selective_vs_flash']}  loom-selective vs flash: {summary['loom_selective_vs_flash']}  "
              f"vs bar: {summary['loom_selective_vs_bar']}  direction-selective (bar): {summary['direction_selective_bar']}  "
              f"(small): {summary['direction_selective_small']}")
        print("  columns: type, index, small, loom, flash, bar (Hz, ipsi)")
        print("  top small-vs-loom:", top("SSI_loom", small_resp))
        print("  top loom-vs-flash:", top("RSI", loom_resp))
        print("  top loom-vs-bar:  ", top("loom_vs_bar", loom_resp))
        print("  top expand-vs-recede:", top("expand_vs_recede", loom_resp))
        print("  top recede-vs-expand:", [(x["type"], round(x["expand_vs_recede"], 2), round(x["loom"], 1), round(x["recede"], 1))
                                          for x in sorted(loom_resp, key=lambda x: x["expand_vs_recede"])[:6]])
        print("  top direction (bar):", top("DSI_bar", bar_resp))
        print("  top small responders:", top("small", small_resp))
        results[token] = {"summary": summary, "rows": rows}
        del brain, core
        import torch, gc
        gc.collect(); torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    keys = ["responsive", "lateralised_0.8", "small_selective_vs_loom", "small_selective_vs_flash",
            "loom_selective_vs_flash", "expand_selective_vs_recede", "contract_selective_vs_expand",
            "loom_selective_vs_bar", "direction_selective_bar", "direction_selective_small",
            "best_small_hz", "best_SSI_loom", "best_RSI", "best_expand_vs_recede", "best_DSI_bar", "spiking_per_tick"]
    print(f"{'brain':18s}" + "".join(f"{k[:14]:>15s}" for k in keys))
    for token, r in results.items():
        s = r["summary"]
        print(f"{token:18s}" + "".join(f"{s[k]:15.2f}" if isinstance(s[k], float) else f"{s[k]:15d}" for k in keys))
    OUT.mkdir(exist_ok=True)
    dump_json(OUT / args.out, {"args": vars(args), "results": results}, args=args)
    print(f"wrote {OUT / args.out}")


if __name__ == "__main__":
    main()
