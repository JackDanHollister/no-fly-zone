"""The descending-neuron contract — single owner of DN names, units, and
calibration. Any brain (stub or connectome) emits raw rates in Hz; controls
consume only the normalized [0, 1] drives produced here. Swapping brains at M3
means editing the calibration entries below, never the motor mapping.

Windowing convention: a "rate" is the brain's estimate over the dt it was
stepped with (spike count / dt for a spiking brain). DNp01 (giant fiber) is an
event neuron — a single spike in a window is a full startle, so its
calibration saturates on one spike at the 15 Hz control tick (1 spike /
(1/15 s) = 15 Hz -> drive 1.0).
"""

CALIBRATION_MAPS: dict[str, dict[str, dict[str, float]]] = {
    # name: baseline Hz subtracted, rate span Hz mapping to drive 1.0
    # "v0": tuned for the reflex stub and single-cell readouts
    "v0": {
        "DNa02_L": {"baseline": 0.0, "span": 160.0},  # steer left drive
        "DNa02_R": {"baseline": 0.0, "span": 160.0},  # steer right drive
        "DNg100": {"baseline": 0.0, "span": 100.0},   # forward drive
        "pIP10": {"baseline": 0.0, "span": 150.0},    # courtship song -> FIRE
        "DNp01": {"baseline": 0.0, "span": 15.0},     # giant fiber: 1 spike/tick saturates
    },
    # "pool4": provisional, from the 2026-09-14 DN screen with injection v4
    # (docs/research/cascade-probe-2026-09-14.md): a small object on one eye
    # drives that side's pool at 12-15 Hz mean and the other at ~0; DNg100
    # runs 50-80 Hz for any visual event; the loom pool reaches 15-25 Hz.
    # To be replaced by measured closed-loop distributions.
    "pool4": {
        "DNa02_L": {"baseline": 0.0, "span": 24.0},
        "DNa02_R": {"baseline": 0.0, "span": 24.0},
        "DNg100": {"baseline": 0.0, "span": 80.0},
        "pIP10": {"baseline": 0.0, "span": 12.0},
        "DNp01": {"baseline": 0.0, "span": 20.0},
    },
}
# the rewired_ipsi control's own pools use pool4's spans so that only the
# pool membership differs between the measured brain and its control
CALIBRATION_MAPS["pool_ipsi0"] = CALIBRATION_MAPS["pool4"]
CALIBRATION_MAPS["pool_male"] = CALIBRATION_MAPS["pool4"]
CALIBRATION_MAPS["yaw"] = CALIBRATION_MAPS["pool4"]
DN_CALIBRATION = CALIBRATION_MAPS["v0"]

DN_NAMES = tuple(DN_CALIBRATION)


def normalize(rates: dict[str, float],
              calibration: dict[str, dict[str, float]] | None = None
              ) -> dict[str, float]:
    """Raw DN rates (Hz) -> normalized drives in [0, 1]. Validates keys."""
    cal_map = calibration if calibration is not None else DN_CALIBRATION
    missing = [n for n in DN_NAMES if n not in rates]
    if missing:
        raise KeyError(f"brain output missing DN rates: {missing}; "
                       f"contract requires {list(DN_NAMES)}")
    out = {}
    for name in DN_NAMES:
        cal = cal_map[name]
        drive = (float(rates[name]) - cal["baseline"]) / cal["span"]
        out[name] = min(max(drive, 0.0), 1.0)
    return out
