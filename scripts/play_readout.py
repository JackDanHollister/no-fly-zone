"""Play one fly with a trained readout policy and record the dashboard.

The footer says TRAINED READOUT so nobody mistakes it for the reflex pilot:
the connectome computes the features, the policy chooses the action.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from noflyzone import dashboard
from noflyzone.brain_connectome import BrainView
from noflyzone.rl_env import FlySquad

OUT = Path(__file__).resolve().parent.parent / "out"


class ActionShim:
    """What the dashboard draws, derived from the chosen action name."""

    def __init__(self):
        self.steer = self.throttle = 0.0
        self.fire = False

    def set(self, name: str):
        self.steer = 1.0 if "RIGHT" in name else -1.0 if "LEFT" in name else 0.0
        self.throttle = 1.0 if "UP" in name else -1.0 if "DOWN" in name else 0.0
        self.fire = "FIRE" in name

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
    ap.add_argument("--sample", action="store_true", help="sample actions instead of argmax")
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, weights_only=False)
    from train_readout import ActorCritic  # noqa: E402  (same folder)
    ablate = "none" if ck["brain"] == "connectome" else ck["brain"]
    squad = FlySquad(batch=1, seeds=[args.seed], brain_kwargs={"ablate": ablate},
                     features=ck["features"], episode_ticks=args.steps,
                     injection=ck["args"].get("injection", "v4"),
                     zoom=ck["args"].get("zoom", 2.0), eye_offset=ck["args"].get("eye_offset", 0.15))
    dev = torch.device("cuda")
    net = ActorCritic(squad.n_features, ck["n_actions"]).to(dev)
    net.load_state_dict(ck["net"]); net.eval()
    mean, var = ck["norm_mean"], ck["norm_var"]
    obs = squad.reset()
    loop = squad.loops[0]
    view = BrainView(squad.brain) if args.video else None
    shim = ActionShim()
    writer = None
    if args.video:
        import imageio.v2 as imageio
        OUT.mkdir(exist_ok=True)
        writer = imageio.get_writer(OUT / "readout_demo.mp4", fps=15, codec="libx264", quality=8)
    mode = (f"TRAINED READOUT (PPO) on {ck['brain']} brain | features {ck['features']} | "
            f"{'sampled' if args.sample else 'greedy'}")
    kills = lives = 0
    t0 = time.time()
    for i in range(args.steps):
        x = ((obs - mean) / np.sqrt(var + 1e-8)).astype(np.float32)
        with torch.no_grad():
            logits, _ = net(torch.from_numpy(x).to(dev))
            a = (torch.distributions.Categorical(logits=logits).sample() if args.sample
                 else logits.argmax(-1)).item()
        name = squad.action_names[a]
        shim.set(name)
        obs, rew, done, infos = squad.step(np.array([a]))
        kills += int(infos[0]["kill"]); lives += int(infos[0]["life_lost"])
        if writer is not None:
            counts = squad.last_counts[0]
            writer.append_data(dashboard.compose(
                loop.cockpit.render("beauty"), loop.game.frame, shim, squad.last_dn[0],
                loop.total_reward, i * loop.dt, brain=view.render(counts),
                brain_label=f"138,639 neurons | {int((counts > 0).sum()):,} spiking",
                footer=f"NO-FLY-ZONE // {mode}"))
        if (i + 1) % 150 == 0:
            print(f"t={(i + 1) * loop.dt:5.1f}s kills {kills} lives lost {lives} action {name}", flush=True)
    if writer is not None:
        writer.close(); print(f"video: {OUT / 'readout_demo.mp4'}")
    print(f"{args.steps} ticks in {time.time() - t0:.0f} s: {kills} kills, {lives} lives lost")
    squad.close()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
