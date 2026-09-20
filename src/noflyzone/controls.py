"""Motor mapping: normalized DN drives -> cockpit control state -> ALE action.

Battlezone's cabinet had dual tread sticks; we keep that fiction: two lever
positions derived from throttle +/- steer, plus a fire button. All thresholds
here are in normalized [0, 1] drive units (see dn.py) — brain-agnostic.
"""

import numpy as np

from . import dn


class CockpitControls:
    def __init__(self, action_meanings: list[str], idle_throttle: float = 0.0,
                 calibration: str = "v0", saccade_hold: int = 0,
                 startle_ticks: int = 8, startle_refractory: int = 0,
                 steer_mode: str = "diff", steer_min_hz: float = 8.0):
        """idle_throttle: baseline forward drive (central-pattern stand-in) so
        a quiet brain still generates optic flow for its own visual system.
        Off by default: with idle_throttle and idle_steer at 0, a silent
        brain yields NOOP. calibration: a dn.CALIBRATION_MAPS key.

        saccade_hold > 0 enables saccadic gating, the fly's own trick for
        telling the world's motion from its own: after any tick that moved the
        tank, the visual DN drives are ignored for that many ticks (efference
        copy), so the pilot moves in short bursts and looks between them.
        startle_ticks: how long a giant-fibre startle reverses; then loom is
        ignored for startle_refractory ticks so self-motion cannot re-trigger
        it. Both are motor programs, not brain properties; they are named on
        the HUD.

        steer_mode "diff": steer from the difference of the normalised
        left/right drives (the M2 mapping; saturates when both pools are
        busy). "li": steer from the lateralisation index (R - L) / (R + L) of
        the raw pool rates, gated on R + L >= steer_min_hz — contrast
        normalisation, so an object on one side gives a full turn while a
        wide-field event that lights both pools gives none."""
        self._action_id = {name: i for i, name in enumerate(action_meanings)}
        self.idle_throttle = idle_throttle
        self.idle_steer = 0.0  # scanning turn: rotation makes optic flow
        self.calibration = calibration
        # "adaptive": each pool's span is its own running peak rate (decay
        # 30 s, floor 4 Hz), so thresholds are relative to the brain's own
        # dynamic range and a gain change of the descending neurons does not
        # break them. The spans start at the floor: the brain's first events
        # set the scale, nothing is assumed from the measured brain. pool4's
        # spans are kept only to scale the steer gate.
        self.adaptive = calibration == "adaptive"
        base = "pool4" if self.adaptive else calibration
        self._cal = {k: dict(v) for k, v in dn.CALIBRATION_MAPS[base].items()}
        self._span0 = {k: v["span"] for k, v in self._cal.items()}
        self.peak_tau_s = 30.0
        self.peak_floor_hz = 4.0
        self.peak = {k: (self.peak_floor_hz if self.adaptive else v["span"]) for k, v in self._cal.items()}
        if self.adaptive:
            for k in self._cal:
                self._cal[k]["span"] = self.peak[k]
        self.drive: dict[str, float] = {}
        self.saccade_hold = saccade_hold
        self.startle_ticks = startle_ticks
        self.startle_refractory = startle_refractory
        self.steer_mode = steer_mode
        self.steer_min_hz = steer_min_hz
        self._hold = 0        # fixation ticks left (visual drives suppressed)
        self._startle = 0     # startle-reverse ticks left
        self._refractory = 0  # ticks left during which loom is ignored
        self.suppressed = False
        self.steer = 0.0     # -1 (left) .. +1 (right)
        self.throttle = 0.0  # -1 (reverse) .. +1 (forward)
        self.fire = False
        # embodied mode: the levers' measured positions (left, right) in
        # [-1, 1]; when set, the action and the treads come from these, not
        # from the intended steer/throttle above
        self.measured: tuple[float, float] | None = None
        # physical levers do not return all the way: after a push they rest
        # 0.15-0.26 of the push range from zero (measured 2026-09-20), so the
        # measured values pass through a dead zone, as a joystick's do
        self.deadzone = 0.35

    def update(self, rates: dict[str, float], dt: float) -> None:
        if self.adaptive:
            decay = float(np.exp(-dt / self.peak_tau_s))
            for k in self._cal:
                self.peak[k] = max(self.peak[k] * decay, float(rates[k]), self.peak_floor_hz)
                self._cal[k]["span"] = self.peak[k]
        drive = dn.normalize(rates, self._cal)
        self.drive = drive
        # efference copy: while we move (a saccade's aftermath or a startle
        # reverse) the view is our own motion, so the visual drives are ignored
        self.suppressed = self.saccade_hold > 0 and (self._hold > 0 or self._startle > 0)
        if self.suppressed:
            drive = {k: 0.0 for k in drive}
            if self._hold > 0:
                self._hold -= 1
        if self._refractory > 0:
            self._refractory -= 1
        elif drive["DNp01"] > 0.5:  # giant-fiber startle: back off
            self._startle = self.startle_ticks
            self._refractory = self.startle_refractory
        if self.steer_mode == "li":
            left = 0.0 if self.suppressed else float(rates["DNa02_L"])
            right = 0.0 if self.suppressed else float(rates["DNa02_R"])
            total = left + right
            gate = self.steer_min_hz
            if self.adaptive:   # the gate scales with the steer pools' own range
                gate *= (self.peak["DNa02_L"] + self.peak["DNa02_R"]) / (
                    self._span0["DNa02_L"] + self._span0["DNa02_R"])
            li = (right - left) / total if total >= gate else 0.0
            target_steer = float(np.clip(2.0 * li + self.idle_steer, -1.0, 1.0))
        else:
            target_steer = float(np.clip(
                2.0 * (drive["DNa02_R"] - drive["DNa02_L"]) + self.idle_steer,
                -1.0, 1.0))
        if self._startle > 0:
            target_throttle = -1.0
            self._startle -= 1
            if self._startle == 0 and self.saccade_hold > 0:
                self._hold = self.saccade_hold  # settle before looking again
        else:
            target_throttle = max(drive["DNg100"], self.idle_throttle)
        # low-pass so levers move like levers, not switches
        alpha = 1.0 - np.exp(-dt / 0.12)
        self.steer += alpha * (target_steer - self.steer)
        self.throttle += alpha * (target_throttle - self.throttle)
        self.fire = drive["pIP10"] > 0.5

    @property
    def lever_left(self) -> float:
        if self.measured is not None:
            return self.measured[0]
        return float(np.clip(self.throttle - self.steer, -1.0, 1.0))

    @property
    def lever_right(self) -> float:
        if self.measured is not None:
            return self.measured[1]
        return float(np.clip(self.throttle + self.steer, -1.0, 1.0))

    @property
    def effective(self) -> tuple[float, float]:
        """(steer, throttle) that the action is taken from: the intent in
        wired mode, the levers' measured positions in embodied mode."""
        if self.measured is None:
            return self.steer, self.throttle
        left, right = (self._dead(v) for v in self.measured)
        return (right - left) / 2.0, (left + right) / 2.0

    def _dead(self, v: float) -> float:
        dz = self.deadzone
        return float(np.sign(v) * max(0.0, abs(v) - dz) / (1.0 - dz))

    def to_action(self) -> int:
        steer, throttle = self.effective
        parts = []
        if throttle > 0.25:
            parts.append("UP")
        elif throttle < -0.25:
            parts.append("DOWN")
        if steer > 0.3:
            parts.append("RIGHT")
        elif steer < -0.3:
            parts.append("LEFT")
        name = "".join(parts) or "NOOP"
        if parts and self.saccade_hold > 0:
            self._hold = self.saccade_hold  # we are about to move: look away
        if self.fire:
            name = "FIRE" if name == "NOOP" else name + "FIRE"
        return self._action_id.get(name, self._action_id["NOOP"])
