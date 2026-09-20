"""The adaptive calibration makes the controller invariant to a gain change
of the descending neurons, which a fixed calibration is not."""

import numpy as np

from noflyzone.controls import CockpitControls

ACTIONS = ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT",
           "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE",
           "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"]
DT = 1 / 15


def play(calibration, gain, ticks=120):
    """A loom (DNp01 burst) after 60 quiet ticks, all rates scaled by gain."""
    c = CockpitControls(ACTIONS, calibration=calibration, saccade_hold=3,
                        startle_refractory=15, steer_mode="li")
    startled = False
    for t in range(ticks):
        loom = 60 <= t < 66
        rates = {"DNa02_L": 3.0, "DNa02_R": 3.0, "DNg100": 40.0 if loom else 20.0,
                 "pIP10": 0.0, "DNp01": 14.0 if loom else 0.0}
        c.update({k: v * gain for k, v in rates.items()}, DT)
        startled |= c.throttle < -0.5
    return c, startled


def test_fixed_calibration_breaks_under_gain_change():
    _, startled_full = play("pool4", 1.0)
    _, startled_half = play("pool4", 0.5)
    assert startled_full and not startled_half     # 7 Hz never crosses the fixed 10 Hz threshold


def test_adaptive_calibration_keeps_the_startle():
    c1, startled_full = play("adaptive", 1.0)
    c2, startled_half = play("adaptive", 0.5)
    assert startled_full and startled_half
    # the spans re-measured themselves to the brain's own range
    assert abs(c2.peak["DNg100"] / c1.peak["DNg100"] - 0.5) < 0.05
    assert c1.peak["DNp01"] >= 2.0 and all(v >= 2.0 for v in c1.peak.values())


def test_adaptive_starts_at_the_floor_and_decays_back_to_it():
    c = CockpitControls(ACTIONS, calibration="adaptive")
    assert c.peak["DNg100"] == c.peak_floor_hz
    burst = {"DNa02_L": 0.0, "DNa02_R": 0.0, "DNg100": 60.0, "pIP10": 0.0, "DNp01": 0.0}
    quiet = {k: 0.0 for k in burst}
    c.update(burst, DT)
    assert c.peak["DNg100"] == 60.0 and c.drive["DNg100"] == 1.0
    for _ in range(int(300 / DT)):
        c.update(quiet, DT)
    assert c.peak["DNg100"] == c.peak_floor_hz
    # a fixed calibration is untouched by all this
    f = CockpitControls(ACTIONS, calibration="pool4")
    f.update(burst, DT)
    assert f.peak["DNg100"] == 80.0 and f.drive["DNg100"] == 0.75
