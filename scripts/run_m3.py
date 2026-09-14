"""M3: the REAL connectome flies the tank.

Full closed loop with the 138,639-neuron FlyWire v783 LIF brain: screen ->
real R1-6 photoreceptors -> whole-brain spiking -> real descending neurons ->
levers -> Battlezone. Runs ~5x slower than real time on Mac CPU; the video is
written at game time, so it plays back at true speed.
"""

import argparse
import time
from pathlib import Path

from noflyzone import dashboard
from noflyzone.brain_connectome import BrainView, ConnectomeBrain
from noflyzone.loop import PilotLoop

OUT = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=90,
                    help="game steps (15 per second of game time)")
    ap.add_argument("--video", action="store_true", help="write out/m3_demo.mp4")
    args = ap.parse_args()

    print("building loop + connectome brain...", flush=True)
    t0 = time.time()
    loop = PilotLoop()
    brain = ConnectomeBrain(retina=loop.retina)
    loop.brain = brain
    # idle scan: rotation sweeps the horizon across the retina — Battlezone's
    # flat ground gives forward motion almost no optic flow, but a turning
    # tank paints motion onto every edge (change is the visual signal)
    loop.controls.idle_throttle = 0.35
    loop.controls.idle_steer = 0.5
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
    for i in range(args.steps):
        state = loop.step()
        spiking = int((brain.window_counts > 0).sum())
        if writer is not None:
            writer.append_data(dashboard.compose(
                loop.cockpit.render("beauty"), loop.game.frame, loop.controls,
                state["dn"], loop.total_reward, i * loop.dt,
                brain=view.render(brain.window_counts),
                brain_label=f"138,639 neurons | {spiking:,} spiking",
                footer="NO-FLY-ZONE  //  pilot: FlyWire connectome (female, v783)"))
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
    print(f"\n{args.steps} steps = {game_t:.1f} s game time in {wall:.0f} s wall "
          f"({wall / game_t:.1f}x slower than real time)")
    print("latency:")
    print(loop.latency_report())
    assert (brain.window_counts > 0).sum() > 0, "brain silent — injection broken"

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
