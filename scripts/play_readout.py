"""Play one fly with a trained readout policy (MLP or Recurrent GRU) and record
or view the dashboard, including the retinal mosaics.

The footer says TRAINED READOUT so nobody mistakes it for the reflex pilot:
the connectome computes the features, the policy chooses the action.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_readout import load_policy_runner, squad_kwargs_from_ckpt  # noqa: E402

from noflyzone import dashboard, dn  # noqa: E402
from noflyzone.brain_connectome import BrainView  # noqa: E402
from noflyzone.rl_env import FlySquad  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"


class ActionShim:
    """What the dashboard draws, derived from the chosen action name and DN rates."""

    def __init__(self):
        self.steer = self.throttle = 0.0
        self.fire = False
        self.drive = {k: 0.0 for k in dn.DN_NAMES}

    def set(self, name: str, rates: dict | None = None):
        self.steer = 1.0 if "RIGHT" in name else -1.0 if "LEFT" in name else 0.0
        self.throttle = 1.0 if "UP" in name else -1.0 if "DOWN" in name else 0.0
        self.fire = "FIRE" in name
        if rates:
            self.drive = dn.normalize(rates, dn.CALIBRATION_MAPS["pool4"])

    @property
    def lever_left(self):
        return float(np.clip(self.throttle - self.steer, -1, 1))

    @property
    def lever_right(self):
        return float(np.clip(self.throttle + self.steer, -1, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--neural-seed", type=int, default=0)
    ap.add_argument("--sample", action="store_true", help="sample actions instead of argmax")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-name", default="readout_demo", help="out/<name>.mp4 with --video")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--realtime", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, weights_only=False)
    sq_kw = squad_kwargs_from_ckpt(ck)
    squad = FlySquad(
        batch=1,
        seeds=[args.seed],
        neural_seed=args.neural_seed,
        episode_ticks=args.steps,
        **sq_kw,
    )
    dev = torch.device("cuda")
    runner = load_policy_runner(ck, squad, dev, sample=args.sample)

    obs = squad.reset()
    loop = squad.loops[0]
    view = BrainView(squad.brain) if (args.video or args.live) else None
    shim = ActionShim()
    writer = None
    if args.video:
        import imageio.v2 as imageio
        OUT.mkdir(exist_ok=True)
        writer = imageio.get_writer(OUT / f"{args.video_name}.mp4", fps=15, codec="libx264", quality=8)
    live = None
    if args.live:
        from noflyzone.liveview import LiveView
        live = LiveView(title="NO-FLY-ZONE (TRAINED READOUT)")

    arch_label = ck.get("arch", "mlp").upper()
    mode = (
        f"TRAINED READOUT ({arch_label}) on {ck['brain']} brain | "
        f"features {ck['features']} ({squad.n_features}d) | "
        f"{'sampled' if args.sample else 'greedy'}"
    )
    kills = lives = 0
    done = np.zeros(1, dtype=bool)
    prev_rgb = None
    t0 = time.time()
    next_frame = time.monotonic()

    try:
        for i in range(args.steps):
            a_arr = runner(obs, done, squad)
            a = int(a_arr[0])
            name = squad.action_names[a]
            obs, rew, done, infos = squad.step(a_arr)
            shim.set(name, squad.last_dn[0])
            loop.cockpit.set_controls(shim.lever_left, shim.lever_right, shim.fire)
            kills += int(infos[0]["kill"])
            lives += int(infos[0]["life_lost"])

            if writer is not None or live is not None:
                counts = squad.last_counts[0]
                cur_obs = loop.observe()
                eyes = dashboard.eyes_panel(loop.retina, loop.retina_right, cur_obs["rgb"], prev_rgb)
                prev_rgb = None if done[0] else cur_obs["rgb"]
                frame_img = dashboard.compose(
                    loop.cockpit.render("beauty"),
                    loop.game.frame,
                    shim,
                    squad.last_dn[0],
                    kills * 1000.0,
                    (i + 1) * loop.dt,
                    brain=view.render(counts),
                    brain_label=f"{squad.brain.n:,} neurons | {int((counts > 0).sum()):,} spiking",
                    footer=f"NO-FLY-ZONE // {mode}",
                    eyes=eyes,
                )
                if writer is not None:
                    writer.append_data(frame_img)
                if live is not None and not live.show(frame_img):
                    break

            if args.realtime:
                next_frame = max(next_frame + loop.dt, time.monotonic())
                time.sleep(max(0.0, next_frame - time.monotonic()))

            if (i + 1) % 150 == 0:
                print(f"t={(i + 1) * loop.dt:5.1f}s kills {kills} lives lost {lives} action {name}", flush=True)
    finally:
        if writer is not None:
            writer.close()
            print(f"video: {OUT / f'{args.video_name}.mp4'}")
        if live is not None:
            live.close()
        squad.close()

    print(f"{args.steps} ticks in {time.time() - t0:.1f} s: {kills} kills, {lives} lives lost", flush=True)


if __name__ == "__main__":
    main()
