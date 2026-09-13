"""Stub pilot brain: hand-written reflexes behind the same interface the real
LIF connectome brain will expose at M3:

    step(drives, dt, signals=None) -> raw DN rates in Hz (keys: dn.DN_NAMES)
    reset()                           clear temporal state (world jumped)

drives: {"left": (n, 2), "right": (n, 2)} photoreceptor drives in [0, 1]
(channel 0 = R1-6 broadband, channel 1 = R8 blue; see retina.photoreceptors).
signals: {"damage": bool, "reward": float, "episode_end": bool} — the hook
M7's PPL101 dopamine-on-damage will use.
"""

import numpy as np

# gains are in normalized-drive units (photoreceptor values 0-1)
TURN_GAIN = 15000.0
FALLBACK_MOTION_GAIN = 3000.0
JUMP_THRESHOLD = 0.07


class ReflexBrain:
    def __init__(self, retina_x: np.ndarray | None = None):
        """retina_x: per-ommatidium horizontal position in [-1, 1]; enables
        retinotopic steering (turn toward motion) instead of eye-vs-eye."""
        self.retina_x = retina_x
        self.reset()

    def reset(self) -> None:
        self._prev_lum: dict[str, np.ndarray] = {}
        self._prev_total: float | None = None
        self._song_t = 0.0
        self._startle = 0.0
        self._turn_baseline = 0.0

    def step(self, drives: dict[str, np.ndarray], dt: float,
             signals: dict | None = None) -> dict[str, float]:
        motion = {}
        brightness = {}
        turn_drive = 0.0
        for side, chans in drives.items():
            lum = chans[:, 0]
            prev = self._prev_lum.get(side)
            if prev is not None:
                diff = np.abs(lum - prev)
                motion[side] = float(diff.mean())
                if self.retina_x is not None:
                    turn_drive += float((diff * self.retina_x).mean())
            else:
                motion[side] = 0.0
            brightness[side] = float(lum.mean())
            self._prev_lum[side] = lum

        if self.retina_x is not None:
            # habituate to sustained motion asymmetry (self-motion during a
            # turn), otherwise turning right sweeps the scene into a signal
            # that commands more right turn, forever
            alpha = 1.0 - np.exp(-dt / 0.8)
            self._turn_baseline += alpha * (turn_drive - self._turn_baseline)
            turn = np.clip((turn_drive - self._turn_baseline) * TURN_GAIN,
                           -100.0, 100.0)
        else:
            turn = np.clip((motion["right"] - motion["left"])
                           * FALLBACK_MOTION_GAIN, -100.0, 100.0)

        # startle on a global luminance jump (explosion / muzzle flash nearby)
        total = brightness["left"] + brightness["right"]
        jump = 0.0 if self._prev_total is None else abs(total - self._prev_total)
        self._prev_total = total
        if jump > JUMP_THRESHOLD:
            self._startle = 0.6
        self._startle = max(0.0, self._startle - dt)
        startled = self._startle > 0.0

        # courtship-song oscillator: ~0.5 s bouts every ~2.5 s, suppressed by startle
        self._song_t = (self._song_t + dt) % 2.5
        singing = (self._song_t < 0.5) and not startled

        return {
            "DNa02_L": max(0.0, 60.0 - turn),
            "DNa02_R": max(0.0, 60.0 + turn),
            "DNg100": 0.0 if startled else 80.0,
            "pIP10": 150.0 if singing else 0.0,
            "DNp01": 15.0 if startled else 0.0,
        }
