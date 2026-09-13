"""The pilot loop: game -> screen -> eyes -> retina -> brain -> controls -> game.

Single-threaded by design (macOS GL constraint). Any brain exposing the
contract in brain_stub's docstring — step(drives, dt, signals) -> DN rates
(Hz, keys per dn.DN_NAMES) plus reset() — can be dropped in: the reflex stub
now, the LIF connectome later.

Timing note: each step applies the action derived from the PREVIOUS step's
percept — a deliberate one-step (~67 ms) sensorimotor latency, standing in
for reaction time.
"""

import time
from collections import defaultdict

import numpy as np

from .controls import CockpitControls
from .cockpit import Cockpit
from .game import BattleZone
from .retina import HexRetina

LAMP_FLASH_SECONDS = 0.5


class PilotLoop:
    def __init__(self, brain=None, seed: int | None = 0):
        self.game = BattleZone(seed=seed)
        self.dt = self.game.dt
        self.cockpit = Cockpit()
        self.retina = HexRetina(image_shape=(self.cockpit.render_height,
                                             self.cockpit.render_width))
        if brain is None:
            from .brain_stub import ReflexBrain
            brain = ReflexBrain(retina_x=self.retina.centers_x)
        self.brain = brain
        self.controls = CockpitControls(self.game.action_meanings)
        self.timings: dict[str, list[float]] = defaultdict(list)
        self.total_reward = 0.0
        self._lamp_steps = 0
        self._lamp_flash_steps = max(1, round(LAMP_FLASH_SECONDS / self.dt))

    def _timed(self, stage: str, fn, *args):
        t0 = time.perf_counter()
        out = fn(*args)
        self.timings[stage].append(time.perf_counter() - t0)
        return out

    def step(self) -> dict:
        action = self.controls.to_action()
        frame, reward, life_lost, episode_end = self._timed(
            "game", self.game.step, action)
        self.total_reward += reward

        if life_lost:
            self._lamp_steps = self._lamp_flash_steps
        self.cockpit.set_lamp(self._lamp_steps > 0)
        self._lamp_steps = max(0, self._lamp_steps - 1)

        self._timed("texture", self.cockpit.set_screen, frame)
        cam_l, cam_r = self.cockpit.eye_cameras
        eye_l = self._timed("render", self.cockpit.render, cam_l)
        eye_r = self._timed("render", self.cockpit.render, cam_r)
        rgb = {
            "left": self._timed("retina", self.retina.sample, eye_l),
            "right": self._timed("retina", self.retina.sample, eye_r),
        }
        drives = {side: self.retina.photoreceptors(s) for side, s in rgb.items()}

        if episode_end:
            # the world jumped: temporal state must not bridge two episodes
            self.brain.reset()
        signals = {"damage": life_lost, "reward": reward,
                   "episode_end": episode_end}
        dn = self._timed("brain", self.brain.step, drives, self.dt, signals)
        self.controls.update(dn, self.dt)
        self.cockpit.set_controls(self.controls.lever_left,
                                  self.controls.lever_right,
                                  self.controls.fire)
        return {"eye_left": eye_l, "rgb": rgb, "drives": drives, "dn": dn,
                "reward": reward, "life_lost": life_lost,
                "episode_end": episode_end}

    def latency_report(self) -> str:
        lines = []
        step_total = 0.0
        for stage, samples in self.timings.items():
            ms = 1000.0 * float(np.mean(samples))
            per_step = ms * (2 if stage in ("render", "retina") else 1)
            step_total += per_step
            lines.append(f"  {stage:<8} {ms:7.2f} ms" +
                         (" (x2/step)" if per_step != ms else ""))
        lines.append(f"  {'total':<8} {step_total:7.2f} ms/step "
                     f"(~{1000.0 / step_total:.0f} Hz)")
        return "\n".join(lines)

    def close(self) -> None:
        self.cockpit.close()
        self.game.close()
