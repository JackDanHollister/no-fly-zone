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
                 startle_thresh: float = 0.5,
                 steer_mode: str = "diff", steer_min_hz: float = 8.0,
                 sharp_steer_hz: float = 0.0,
                 search_steer: float = 0.0, search_delay: int = 2,
                 patrol_sweep: int = 0, patrol_stride: int = 0,
                 fovea_li: float = 0.0, pip_lock_drive: float = 0.75,
                 off_axis_fire: bool = True,
                 direct_steer: bool = False):
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
        self.startle_thresh = float(startle_thresh)
        self.steer_mode = steer_mode
        self.steer_min_hz = steer_min_hz
        self.sharp_steer_hz = float(sharp_steer_hz)
        self.search_steer = float(search_steer)
        self.search_delay = int(search_delay)
        self.patrol_sweep = int(patrol_sweep)
        self.patrol_stride = int(patrol_stride)
        self.fovea_li = float(fovea_li)
        self.pip_lock_drive = float(pip_lock_drive)
        self.off_axis_fire = bool(off_axis_fire)
        self.direct_steer = bool(direct_steer)
        self._search_sign = 1.0
        self._escape_sign = -1.0
        self._quiet_ticks = 0
        self._patrol_saccades = 0
        self._patrol_sweeps = 0
        self._stride_ticks = 0
        self._override: tuple[float, float, bool] | None = None
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
        self.fire_measured: bool | None = None   # embodied: the button's own state
        # physical levers do not return all the way: after a push they rest
        # 0.15-0.26 of the push range from zero (measured 2026-09-20), so the
        # measured values pass through a dead zone, as a joystick's do
        self.deadzone = 0.35

    def set_command(self, steer: float, throttle: float, fire: bool,
                    rates: dict[str, float] | None = None) -> None:
        """Override motor intent directly (e.g. from a trained readout policy),
        preserving normalized DN drive display and embodied physical lever IK."""
        if rates is not None:
            self.drive = dn.normalize(rates, self._cal)
        self.suppressed = False
        self.steer = float(np.clip(steer, -1.0, 1.0))
        self.throttle = float(np.clip(throttle, -1.0, 1.0))
        self.fire = bool(fire)
        self._override = (self.steer, self.throttle, self.fire)

    def update(self, rates: dict[str, float], dt: float) -> None:
        self._override = None
        if self.adaptive:
            decay = float(np.exp(-dt / self.peak_tau_s))
            for k in self._cal:
                self.peak[k] = max(self.peak[k] * decay, float(rates[k]), self.peak_floor_hz)
                self._cal[k]["span"] = self.peak[k]
        raw_drive = dn.normalize(rates, self._cal)
        self.drive = raw_drive
        drive = raw_drive
        # efference copy: while we move (a saccade's aftermath or a startle
        # reverse) the view is our own motion, so the visual drives are ignored
        self.suppressed = self.saccade_hold > 0 and (self._hold > 0 or self._startle > 0)
        if self.suppressed:
            drive = {k: 0.0 for k in drive}
            if self._hold > 0:
                self._hold -= 1
        if self._refractory > 0:
            self._refractory -= 1
        elif drive["DNp01"] > self.startle_thresh:  # giant-fiber startle: back off
            self._startle = self.startle_ticks
            self._refractory = self.startle_refractory
            self._escape_sign = -1.0 if float(rates["DNa02_R"]) >= float(rates["DNa02_L"]) else 1.0
        total = 0.0
        li = 0.0
        locked = False
        gated = False
        striding = False
        gate = self.steer_min_hz
        if self.steer_mode == "li":
            left = 0.0 if self.suppressed else float(rates["DNa02_L"])
            right = 0.0 if self.suppressed else float(rates["DNa02_R"])
            total = left + right
            if self.adaptive:   # the gate scales with the steer pools' own range
                gate *= (self.peak["DNa02_L"] + self.peak["DNa02_R"]) / (
                    self._span0["DNa02_L"] + self._span0["DNa02_R"])
            if total > 0.0:
                li = (right - left) / total
            gated = (total >= gate) or (
                self.sharp_steer_hz > 0.0 and total >= self.sharp_steer_hz and abs(li) >= 0.60
            )
            if (not self.suppressed and self.fovea_li > 0.0
                    and total >= gate and abs(li) <= self.fovea_li
                    and (drive["pIP10"] >= self.pip_lock_drive or total >= 1.75 * gate)):
                locked = True
                self._quiet_ticks = 0
                self._stride_ticks = 0
                if self.patrol_sweep == 0:
                    self._patrol_saccades = 0
                target_steer = 0.0
            elif gated:
                self._quiet_ticks = 0
                self._stride_ticks = 0
                if self.patrol_sweep == 0:
                    self._patrol_saccades = 0
                if abs(li) > self.fovea_li:
                    turn_sign = 1.0 if li >= 0.0 else -1.0
                    if self.patrol_sweep == 0:
                        self._search_sign = turn_sign
                    target_steer = float(np.clip(2.0 * li + self.idle_steer, -1.0, 1.0))
                    if self.direct_steer:
                        target_steer = turn_sign
                else:
                    target_steer = 0.0
            else:
                if not self.suppressed and self.search_steer > 0.0:
                    self._quiet_ticks += 1
                    if self._stride_ticks > 0:
                        self._stride_ticks -= 1
                        striding = True
                        target_steer = 0.0
                    elif self._quiet_ticks >= self.search_delay:
                        target_steer = float(np.clip(self._search_sign * self.search_steer, -1.0, 1.0))
                        if self.patrol_sweep > 0:
                            self._patrol_saccades += 1
                            if self._patrol_saccades >= self.patrol_sweep:
                                self._patrol_saccades = 0
                                if self._patrol_sweeps % 2 == 0:
                                    self._search_sign *= -1.0
                                self._patrol_sweeps += 1
                                self._stride_ticks = self.patrol_stride
                    else:
                        target_steer = float(np.clip(self.idle_steer, -1.0, 1.0))
                else:
                    target_steer = 0.0 if self.suppressed else float(np.clip(self.idle_steer, -1.0, 1.0))
        else:
            target_steer = float(np.clip(
                2.0 * (drive["DNa02_R"] - drive["DNa02_L"]) + self.idle_steer,
                -1.0, 1.0))
        in_startle = self._startle > 0
        if in_startle:
            rem = self._startle
            target_throttle = -1.0
            if self.search_steer > 0.0:
                target_steer = self._escape_sign if rem >= max(1, self.startle_ticks - 2) else 0.0
            self._startle -= 1
            if self._startle == 0 and self.saccade_hold > 0:
                self._hold = self.saccade_hold  # settle before looking again
        elif locked:
            target_throttle = 1.0 if drive["DNg100"] > 0.20 else self.idle_throttle
        elif striding or (self.suppressed and self._stride_ticks > 0):
            target_throttle = 1.0
        else:
            if self.fovea_li > 0.0 and gated:
                # close diagonally when nearly aligned; pivot in place toward far off-axis targets
                target_throttle = (1.0 if (abs(li) < 0.45 and drive["DNg100"] > 0.25)
                                   else self.idle_throttle)
            elif self.fovea_li > 0.0 and self.patrol_sweep > 0:
                target_throttle = self.idle_throttle
            else:
                target_throttle = max(drive["DNg100"], self.idle_throttle)
        # low-pass so levers move like levers, not switches (unless direct_steer is requested)
        alpha = 1.0 - np.exp(-dt / 0.12)
        if self.direct_steer:
            self.steer = target_steer
            self.throttle = target_throttle
        else:
            self.steer += alpha * (target_steer - self.steer)
            self.throttle += alpha * (target_throttle - self.throttle)
        if in_startle and self.search_steer > 0.0:
            self.fire = False
        elif self.fovea_li > 0.0 and not self.suppressed:
            self.fire = locked or (self.off_axis_fire and total >= 2.2 * gate and abs(li) < 0.32)
        else:
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

    def to_action(self, *, commit: bool = True) -> int:
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
        moved_for_hold = ("LEFT" in name or "RIGHT" in name) if self.direct_steer else bool(parts)
        if commit and moved_for_hold and self.saccade_hold > 0:
            self._hold = self.saccade_hold  # we are about to move: look away
        fire = self.fire if self.fire_measured is None else self.fire_measured
        if fire:
            name = "FIRE" if name == "NOOP" else name + "FIRE"
        return self._action_id.get(name, self._action_id["NOOP"])
