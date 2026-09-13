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

DN_CALIBRATION: dict[str, dict[str, float]] = {
    # name: baseline Hz subtracted, rate span Hz mapping to drive 1.0
    "DNa02_L": {"baseline": 0.0, "span": 160.0},  # steer left drive
    "DNa02_R": {"baseline": 0.0, "span": 160.0},  # steer right drive
    "DNg100": {"baseline": 0.0, "span": 100.0},   # forward drive
    "pIP10": {"baseline": 0.0, "span": 150.0},    # courtship song -> FIRE
    "DNp01": {"baseline": 0.0, "span": 15.0},     # giant fiber: 1 spike/tick saturates
}

DN_NAMES = tuple(DN_CALIBRATION)


def normalize(rates: dict[str, float]) -> dict[str, float]:
    """Raw DN rates (Hz) -> normalized drives in [0, 1]. Validates keys."""
    missing = [n for n in DN_NAMES if n not in rates]
    if missing:
        raise KeyError(f"brain output missing DN rates: {missing}; "
                       f"contract requires {list(DN_NAMES)}")
    out = {}
    for name in DN_NAMES:
        cal = DN_CALIBRATION[name]
        drive = (float(rates[name]) - cal["baseline"]) / cal["span"]
        out[name] = min(max(drive, 0.0), 1.0)
    return out
