"""M3: the REAL connectome flies the tank.

Full closed loop with the 138,639-neuron FlyWire v783 LIF brain: screen ->
real R1-6 photoreceptors -> whole-brain spiking -> real descending neurons ->
levers -> Battlezone. The CPU backend runs ~8x slower than real time; the
video is written at game time, so it plays back at true speed.
"""

import argparse
import numpy as np
import time
from pathlib import Path

from noflyzone import dashboard
from noflyzone.brain_backend import BACKENDS, describe, make_brain
from noflyzone.plasticity import EDGE_SETS, MBPlasticity
from noflyzone.brain_lif import ABLATIONS

DATASET_LABELS = {"783": "FlyWire v783 (female)", "malecns": "MaleCNS v1.0 (male)"}
from noflyzone.brain_connectome import BrainView, ConnectomeBrain
from noflyzone.loop import PilotLoop
from noflyzone.provenance import dump_json, git_state, savez

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
    ap.add_argument("--dataset", default="783", help="brain_lif.DATASETS key: 783 (FlyWire) or malecns")
    ap.add_argument("--ablate", choices=list(ABLATIONS), default="none",
                    help="wiring control: rewired (whole-brain target shuffle), "
                         "rewired_ipsi (targets shuffled within each hemisphere), silenced")
    ap.add_argument("--ablate-seed", type=int, default=0, help="RNG seed of the rewiring")
    ap.add_argument("--weight-scale", type=float, default=1.0,
                    help="multiply every synaptic weight (gain-matched controls)")
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
    ap.add_argument("--learned-weights", default=None,
                    help="out/plasticity_<tag>_weights.npz from train_plasticity.py: install "
                         "those synaptic weights (rule off) and label the footer LEARNED WEIGHTS")
    ap.add_argument("--learn", default=None, choices=list(EDGE_SETS),
                    help="run the reward-modulated rule live on this edge set while the fly "
                         "plays (CUDA only): kills pulse PAM, hits pulse PPL1; footer shows drift")
    ap.add_argument("--learn-eta", type=float, default=2e-5)
    ap.add_argument("--learn-seed", type=int, default=0, help="seed of a shuffled edge set")
    ap.add_argument("--video-name", default="m3_demo", help="out/<name>.mp4 with --video")
    ap.add_argument("--live", action="store_true",
                    help="open a window showing the dashboard as the fly plays (tkinter; needs a display); "
                         "close it or press q to stop")
    ap.add_argument("--no-eyes", action="store_true", help="leave the eyes panel off the dashboard")
    ap.add_argument("--pilot", choices=["wired", "embodied"], default="wired",
                    help="wired: descending rates set the levers directly; embodied: they drive the "
                         "forelegs, the physical levers move by contact, the action reads the levers")
    args = ap.parse_args()
    calibration = args.calibration or args.readout

    print("building loop + connectome brain...", flush=True)
    t0 = time.time()
    loop = PilotLoop(seed=args.seed, retina_zoom=args.zoom,
                     retina_center=(0.5, 0.5), eye_offset=args.eye_offset, pilot=args.pilot)
    core = make_brain(args.backend, version=args.dataset,
                      ablate=args.ablate, ablate_seed=args.ablate_seed,
                      sensory_input=args.sensory, signs=args.signs,
                      aminergic=args.aminergic, weight_scale=args.weight_scale)
    extra_sites = None
    if args.learn:
        # the dopamine populations join the Poisson set as teaching sites
        ann = core.annotations().drop_duplicates("root_id").set_index("root_id").reindex(core.root_ids)
        ct = ann["cell_type"].fillna("").astype(str).to_numpy()
        extra_sites = {"pam": np.flatnonzero(np.char.startswith(ct.astype(str), "PAM")),
                       "ppl1": np.flatnonzero(np.char.startswith(ct.astype(str), "PPL1"))}
    brain = ConnectomeBrain(retina=loop.retina, brain=core,
                            injection=args.injection, readout=args.readout,
                            extra_sites=extra_sites)
    loop.brain = brain
    learned = None
    if args.learned_weights:
        z = np.load(args.learned_weights)
        if not np.allclose(core.W.data[z["pos"]], z["w0"]):
            raise SystemExit("learned weights were saved from a different brain "
                             "(dataset, wiring or corrections differ)")
        core.set_edge_weights(z["pos"], z["w"])
        learned = (f"LEARNED WEIGHTS {Path(args.learned_weights).stem.replace('plasticity_', '').replace('_weights', '')}"
                   f" ({int((z['w'] != z['w0']).sum()):,} of {z['w'].size:,} synapses changed)")
        print(learned)
    plast = None
    if args.learn:
        if not hasattr(core, "set_edge_weights"):
            raise SystemExit("--learn needs the CUDA backend")
        plast = MBPlasticity(core, dt=loop.dt, eta=args.learn_eta, edge_set=args.learn,
                             seed=args.learn_seed)
        print(f"plasticity on: {plast.n_edges:,} edges ({plast.edge_set}, {plast.eligibility}, "
              f"{plast.scale}); dopamine from PAM/PPL1 teaching pulses, applied as a global "
              f"modulator (ENGINEERED)")
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
            + (f" | WIRING {args.ablate.upper()}"
               + (f" seed {args.ablate_seed}" if args.ablate_seed else "")
               if args.ablate != "none" else "")
            + (f" | weights x{args.weight_scale:g}" if args.weight_scale != 1.0 else "")
            + (f" | sensory {args.sensory}" if args.sensory != "free" else "")
            + (f" | signs {args.signs}" if args.signs != "predicted" else "")
            + (f" | aminergic {args.aminergic}" if args.aminergic != "excitatory" else "")
            + (f" | {learned}" if learned else "")
            + (" | EMBODIED: forelegs push physical levers (IK targets, claw adhesion); action reads the levers"
               if args.pilot == "embodied" else ""))
    g = git_state()
    if g["short"]:
        mode += f" | {g['short']}{'*' if g['dirty'] else ''}"   # the code that made this frame
    print(mode)
    view = BrainView(brain.brain)
    diag_types = ["Mi1", "T4a", "T4b", "T5a", "LC4", "LPLC2", "LC10a"]
    diag = {t: brain.brain.indices_for_type(t) for t in diag_types}
    print(f"ready in {time.time() - t0:.0f} s | {brain.describe()}")

    live = None
    if args.live:
        from noflyzone.liveview import LiveView
        live = LiveView(title="NO-FLY-ZONE")
    writer = None
    if args.video:
        import imageio.v2 as imageio
        OUT.mkdir(exist_ok=True)
        writer = imageio.get_writer(OUT / f"{args.video_name}.mp4",
                                    fps=round(1 / loop.dt),
                                    codec="libx264", quality=8)

    t0 = time.time()
    trace = []
    names = loop.game.action_meanings
    peak_spiking = 0
    learn_label = ""
    prev_rgb = None
    for i in range(args.steps):
        if plast is not None:
            for name, hz in plast.teaching_rates().items():
                brain.set_extra_rate(name, hz)
        state = loop.step()
        spiking = int((brain.window_counts > 0).sum())
        peak_spiking = max(peak_spiking, spiking)
        if plast is not None:
            d = plast.step(brain.window_counts[None, :], kills=np.array([state["reward"] > 0]),
                           hits=np.array([bool(state["life_lost"])]))
            learn_label = (f" | PLASTICITY ON {plast.edge_set}: {d['frac_changed'] * 100:.0f}% of "
                           f"{plast.n_edges:,} synapses moved, drift {d['drift'] * 100:.1f}% "
                           f"(dopamine ENGINEERED)")
        trace.append({"tick": i, "dn": state["dn"], "drive": dict(loop.controls.drive),
                      "suppressed": bool(loop.controls.suppressed),
                      "steer": loop.controls.steer, "throttle": loop.controls.throttle,
                      "levers": list(loop.controls.measured) if loop.controls.measured else None,
                      "fire": bool(loop.controls.fire),
                      "action": names[loop.controls.to_action()],
                      "reward": float(state["reward"]),
                      "life_lost": bool(state["life_lost"]),
                      "episode_end": bool(state["episode_end"]),
                      "spiking": spiking,
                      "retina_lr": float(state["drives"]["right"][:, 0].mean()
                                         - state["drives"]["left"][:, 0].mean())})
        if writer is not None or live is not None:
            eyes = None if args.no_eyes else dashboard.eyes_panel(
                loop.retina, loop.retina_right, state["rgb"], prev_rgb)
            frame_img = dashboard.compose(
                loop.cockpit.render("beauty"), loop.game.frame, loop.controls,
                state["dn"], loop.total_reward, i * loop.dt,
                brain=view.render(brain.window_counts),
                brain_label=f"{loop.brain.brain.n:,} neurons | {spiking:,} spiking",
                footer=f"NO-FLY-ZONE // {DATASET_LABELS.get(args.dataset, args.dataset)} // {mode}{learn_label}",
                eyes=eyes)
            if writer is not None:
                writer.append_data(frame_img)
            if live is not None and not live.show(frame_img):
                print("live view closed", flush=True)
                break
        prev_rgb = state["rgb"]
        if (i + 1) % 15 == 0:
            dn_str = "  ".join(f"{k} {v:5.1f}" for k, v in state["dn"].items())
            cascade = "  ".join(
                f"{t} {brain.window_counts[ix].mean() / loop.dt:4.0f}"
                for t, ix in diag.items())
            print(f"t={(i + 1) * loop.dt:5.1f}s  spiking {spiking:6,}  {dn_str}",
                  flush=True)
            print(f"        cascade Hz: {cascade}", flush=True)

    wall = time.time() - t0
    if live is not None:
        live.close()
    if writer is not None:
        writer.close()
        print(f"video: {OUT / f'{args.video_name}.mp4'}")
    game_t = args.steps * loop.dt
    speed = game_t / wall
    print(f"\n{args.steps} steps = {game_t:.1f} s game time in {wall:.1f} s wall "
          f"({speed:.1f}x real time)")
    print("latency:")
    print(loop.latency_report())
    # a corrected brain is genuinely silent on a dark game-over screen, so
    # gate on the whole run, not the final tick
    assert peak_spiking > 0, "brain silent for the whole run — injection broken"
    if args.pilot == "embodied":
        lv = np.array([t["levers"] for t in trace if t["levers"] is not None])
        print(f"embodied: levers reached left {lv[:, 0].min():+.2f}..{lv[:, 0].max():+.2f}, "
              f"right {lv[:, 1].min():+.2f}..{lv[:, 1].max():+.2f} of the fly's push range; "
              f"physics {1000 * np.mean(loop.timings['physics']):.0f} ms per tick")
    if plast is not None:
        OUT.mkdir(exist_ok=True)
        wpath = OUT / f"m3_learned_{plast.edge_set}_s{args.seed}_weights.npz"
        savez(wpath, args=args, brain=core, pos=plast.pos, w0=plast.w0, w=plast.w, pre=plast.pre, post=plast.post)
        print(f"plasticity: {learn_label.strip(' |')}; weights -> {wpath}")
    if args.trace:
        import json
        OUT.mkdir(exist_ok=True)
        dump_json(OUT / f"{args.trace}.json", {"mode": mode, "steps": args.steps, "trace": trace},
                  args=args, brain=core)
        from collections import Counter
        acts = Counter(t["action"] for t in trace)
        print(f"actions: {dict(acts.most_common())}")
        print(f"total reward {loop.total_reward:.0f}, lives lost "
              f"{sum(t['life_lost'] for t in trace)}, fire ticks "
              f"{sum(t['fire'] for t in trace)}; trace -> out/{args.trace}.json")

    # who was active in the final tick, by brain region class
    ann = loop.brain.brain.annotations().set_index("root_id")
    ann = ann[~ann.index.duplicated()]
    import pandas as pd
    classes = ann.reindex(loop.brain.brain.root_ids)["super_class"]
    active = pd.Series(brain.window_counts > 0)
    print("\nspiking neurons by super_class (final tick):")
    print(classes[active.to_numpy()].value_counts().head(10).to_string())
    print("PASS: the connectome flew the tank")


if __name__ == "__main__":
    main()
