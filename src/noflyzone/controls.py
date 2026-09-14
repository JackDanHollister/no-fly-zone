"""Motor mapping: normalized DN drives -> cockpit control state -> ALE action.

Battlezone's cabinet had dual tread sticks; we keep that fiction: two lever
positions derived from throttle +/- steer, plus a fire button. All thresholds
here are in normalized [0, 1] drive units (see dn.py) — brain-agnostic.
"""

import numpy as np

from . import dn


class CockpitControls:
    def __init__(self, action_meanings: list[str], idle_throttle: float = 0.0):
        """idle_throttle: baseline forward drive (central-pattern stand-in) so
        a quiet brain still generates optic flow for its own visual system."""
        self._action_id = {name: i for i, name in enumerate(action_meanings)}
        self.idle_throttle = idle_throttle
        self.idle_steer = 0.0  # scanning turn: rotation makes optic flow
        self.steer = 0.0     # -1 (left) .. +1 (right)
        self.throttle = 0.0  # -1 (reverse) .. +1 (forward)
        self.fire = False

    def update(self, rates: dict[str, float], dt: float) -> None:
        drive = dn.normalize(rates)
        target_steer = np.clip(
            2.0 * (drive["DNa02_R"] - drive["DNa02_L"]) + self.idle_steer,
            -1.0, 1.0)
        if drive["DNp01"] > 0.5:  # giant-fiber startle: back off
            target_throttle = -1.0
        else:
            target_throttle = max(drive["DNg100"], self.idle_throttle)
        # low-pass so levers move like levers, not switches
        alpha = 1.0 - np.exp(-dt / 0.12)
        self.steer += alpha * (target_steer - self.steer)
        self.throttle += alpha * (target_throttle - self.throttle)
        self.fire = drive["pIP10"] > 0.5

    @property
    def lever_left(self) -> float:
        return float(np.clip(self.throttle - self.steer, -1.0, 1.0))

    @property
    def lever_right(self) -> float:
        return float(np.clip(self.throttle + self.steer, -1.0, 1.0))

    def to_action(self) -> int:
        parts = []
        if self.throttle > 0.25:
            parts.append("UP")
        elif self.throttle < -0.25:
            parts.append("DOWN")
        if self.steer > 0.3:
            parts.append("RIGHT")
        elif self.steer < -0.3:
            parts.append("LEFT")
        name = "".join(parts) or "NOOP"
        if self.fire:
            name = "FIRE" if name == "NOOP" else name + "FIRE"
        return self._action_id.get(name, self._action_id["NOOP"])
