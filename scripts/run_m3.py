"""M3: the REAL connectome flies the tank.

Full closed loop with the 138,639-neuron FlyWire v783 LIF brain: screen ->
real R1-6 photoreceptors -> whole-brain spiking -> real descending neurons ->
levers -> Battlezone. The CPU backend runs ~8x slower than real time; the
video is written at game time, so it plays back at true speed.
"""

import argparse
import time
from pathlib import Path

from noflyzone import dashboard
from noflyzone.brain_backend import BACKENDS, describe, make_brain
from noflyzone.brain_connectome import BrainView, ConnectomeBrain
from noflyzone.loop import PilotLoop

OUT = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=90,
                    help="game steps (15 per second of game time)")
    ap.add_argument("--video", action="store_true", help="write out/m3_demo.mp4")
    ap.add_argument("--backend", choices=BACKENDS, default="auto",
                    help="brain backend: cpu (sparse SciPy) or cuda (Triton)")
    ap.add_argument("--injection", default="v0",
                    help="named injection map from brain_connectome.INJECTION_MAPS")
    ap.add_argument("--readout", default="v0",
                    help="named readout map from brain_connectome.READOUT_MAPS")
    ap.add_argument("--calibration", default=None,
                    help="dn.CALIBRATION_MAPS key (default: same as --readout)")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="retina zoom: lattice over the central 1/zoom of the eye")
    ap.add_argument("--eye-offset", type=float, default=0.0,
                    help="shift each eye's retina window sideways by this image fraction")
    ap.add_argument("--idle-scan", action="store_true",
                    help="scripted forward-right idle scan (off by default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", choices=["none", "rewired", "silenced"],
                    default="none", help="wiring control")
    ap.add_argument("--sensory", choices=["free", "external_only"], default="free",
                    help="external_only: primary sensory neurons fire only from injected input")
    ap.add_argument("--signs", choices=["predicted", "known_first"], default="predicted")
    ap.add_argument("--aminergic", choices=["excitatory", "zero"], default="excitatory",
                    help="zero: dopamine/octopamine/serotonin carry no fast weight")
    ap.add_argument("--saccade-hold", type=int, default=0,
                    help="ticks of visual suppression after each movement")
    ap.add_argument("--startle-refractory", type=int, default=0,
                    help="ticks after a startle during which loom is ignored")
    ap.add_argument("--steer-mode", choices=["diff", "li"], default="diff",
                    help="diff: normalised-drive difference; li: lateralisation index")
    ap.add_argument("--trace", default=None,
                    help="write per-tick DN rates/actions to out/<name>.json")
    args = ap.parse_args()
    calibration = args.calibration or args.readout

    print("building loop + connectome brain...", flush=True)
    t0 = time.time()
    loop = PilotLoop(seed=args.seed, retina_zoom=args.zoom,
                     retina_center=(0.5, 0.5), eye_offset=args.eye_offset)
    brain = ConnectomeBrain(retina=loop.retina,
                            brain=make_brain(args.backend, ablate=args.ablate,
                                             sensory_input=args.sensory,
                                             signs=args.signs, aminergic=args.aminergic),
                            injection=args.injection, readout=args.readout)
    loop.brain = brain
    loop.controls = type(loop.controls)(
        loop.game.action_meanings, calibration=calibration,
        saccade_hold=args.saccade_hold,
        startle_refractory=args.startle_refractory, steer_mode=args.steer_mode)
    print(f"brain backend: {describe(brain.brain)}")
    if args.idle_scan:
        # scripted idle scan: rotation sweeps the horizon across the retina —
        # a turning tank paints motion onto every edge. Labelled on the HUD.
        loop.controls.idle_throttle = 0.35
        loop.controls.idle_steer = 0.5
    mode = (f"injection {args.injection} | readout {args.readout} | "
            f"calibration {calibration} | zoom {args.zoom:g}"
            + (f" eyes +-{args.eye_offset:g}" if args.eye_offset else "") + " | "
            f"{'IDLE SCAN (scripted)' if args.idle_scan else 'CONNECTOME STEERING'}"
            + (f" | saccade hold {args.saccade_hold}" if args.saccade_hold else "")
            + (f" | startle refractory {args.startle_refractory}"
               if args.startle_refractory else "")
            + (" | steer LI" if args.steer_mode == "li" else "")
            + (f" | WIRING {args.ablate.upper()}" if args.ablate != "none" else "")
            + (f" | sensory {args.sensory}" if args.sensory != "free" else "")
            + (f" | signs {args.signs}" if args.signs != "predicted" else "")
            + (f" | aminergic {args.aminergic}" if args.aminergic != "excitatory" else ""))
    print(mode)
    view = BrainView(brain.brain)
    diag_types = ["Mi1", "T4a", "T4b", "T5a", "LC4", "LPLC2", "LC10a"]
    diag = {t: brain.brain.indices_for_type(t) for t in diag_types}
    print(f"ready in {time.time() - t0:.0f} s | {brain.describe()}")

    writer = None
    if args.video:
        import imageio.v2 as imageio
        OUT.mkdir(exist_ok=True)
        writer = imageio.get_writer(OUT / "m3_demo.mp4",
                                    fps=round(1 / loop.dt),
                                    codec="libx264", quality=8)

    t0 = time.time()
    trace = []
    names = loop.game.action_meanings
    for i in range(args.steps):
        state = loop.step()
        spiking = int((brain.window_counts > 0).sum())
        trace.append({"tick": i, "dn": state["dn"], "drive": dict(loop.controls.drive),
                      "suppressed": bool(loop.controls.suppressed),
                      "steer": loop.controls.steer, "throttle": loop.controls.throttle,
                      "fire": bool(loop.controls.fire),
                      "action": names[loop.controls.to_action()],
                      "reward": float(state["reward"]),
                      "life_lost": bool(state["life_lost"]),
                      "episode_end": bool(state["episode_end"]),
                      "spiking": spiking,
                      "retina_lr": float(state["drives"]["right"][:, 0].mean()
                                         - state["drives"]["left"][:, 0].mean())})
        if writer is not None:
            writer.append_data(dashboard.compose(
                loop.cockpit.render("beauty"), loop.game.frame, loop.controls,
                state["dn"], loop.total_reward, i * loop.dt,
                brain=view.render(brain.window_counts),
                brain_label=f"138,639 neurons | {spiking:,} spiking",
                footer=f"NO-FLY-ZONE // FlyWire v783 // {mode}"))
        if (i + 1) % 15 == 0:
            dn_str = "  ".join(f"{k} {v:5.1f}" for k, v in state["dn"].items())
            cascade = "  ".join(
                f"{t} {brain.window_counts[ix].mean() / loop.dt:4.0f}"
                for t, ix in diag.items())
            print(f"t={(i + 1) * loop.dt:5.1f}s  spiking {spiking:6,}  {dn_str}",
                  flush=True)
            print(f"        cascade Hz: {cascade}", flush=True)

    wall = time.time() - t0
    if writer is not None:
        writer.close()
        print(f"video: {OUT / 'm3_demo.mp4'}")
    game_t = args.steps * loop.dt
    speed = game_t / wall
    print(f"\n{args.steps} steps = {game_t:.1f} s game time in {wall:.1f} s wall "
          f"({speed:.1f}x real time)")
    print("latency:")
    print(loop.latency_report())
    assert (brain.window_counts > 0).sum() > 0, "brain silent — injection broken"
    if args.trace:
        import json
        OUT.mkdir(exist_ok=True)
        (OUT / f"{args.trace}.json").write_text(json.dumps(
            {"mode": mode, "steps": args.steps, "trace": trace}))
        from collections import Counter
        acts = Counter(t["action"] for t in trace)
        print(f"actions: {dict(acts.most_common())}")
        print(f"total reward {loop.total_reward:.0f}, lives lost "
              f"{sum(t['life_lost'] for t in trace)}, fire ticks "
              f"{sum(t['fire'] for t in trace)}; trace -> out/{args.trace}.json")

    # who was active in the final tick, by brain region class
    from noflyzone.brain_lif import annotations
    ann = annotations().set_index("root_id")
    ann = ann[~ann.index.duplicated()]
    import pandas as pd
    classes = ann.reindex(loop.brain.brain.root_ids)["super_class"]
    active = pd.Series(brain.window_counts > 0)
    print("\nspiking neurons by super_class (final tick):")
    print(classes[active.to_numpy()].value_counts().head(10).to_string())
    print("PASS: the connectome flew the tank")


if __name__ == "__main__":
    main()
