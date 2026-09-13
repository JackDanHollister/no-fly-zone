"""M2: full closed loop with the reflex stub brain.

Proves the plumbing end to end and measures per-stage latency. Optionally
writes a demo video: beauty cam | what the eye camera sees | fly's-eye mosaic.
"""

import argparse
from pathlib import Path

import numpy as np

from noflyzone.loop import PilotLoop

OUT = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--video", action="store_true", help="write out/m2_demo.mp4")
    args = ap.parse_args()

    loop = PilotLoop()
    writer = None
    if args.video:
        import imageio.v2 as imageio
        OUT.mkdir(exist_ok=True)
        writer = imageio.get_writer(OUT / "m2_demo.mp4",
                                    fps=round(1 / loop.dt),
                                    codec="libx264", quality=8)

    fire_steps = 0
    steer_span = 0.0
    for i in range(args.steps):
        state = loop.step()
        fire_steps += int(loop.controls.fire)
        steer_span = max(steer_span, abs(loop.controls.steer))
        if writer is not None:
            beauty = loop.cockpit.render("beauty")
            mosaic = loop.retina.reconstruct(state["rgb"]["left"])
            panels = [beauty, state["eye_left"], mosaic]
            if loop.cockpit.pov_camera:
                panels.append(loop.cockpit.render(loop.cockpit.pov_camera))
            writer.append_data(np.hstack(panels))
        if (i + 1) % 200 == 0:
            print(f"step {i + 1}: reward so far {loop.total_reward}, "
                  f"steer {loop.controls.steer:+.2f}, "
                  f"throttle {loop.controls.throttle:+.2f}")

    if writer is not None:
        writer.close()
        print(f"video: {OUT / 'm2_demo.mp4'}")

    print(f"\n{args.steps} steps | reward {loop.total_reward} | "
          f"episodes {loop.game.episodes} | fired on {fire_steps} steps")
    print("latency (video capture excluded):")
    print(loop.latency_report())

    assert fire_steps > 0, "fly never fired — song circuit stub broken"
    assert steer_span > 0.05, "steering never moved — retinotopic turn pathway dead"
    print(f"PASS: closed loop ran, fly fired {fire_steps}x, "
          f"peak steer {steer_span:.2f}")
    loop.close()


if __name__ == "__main__":
    main()
