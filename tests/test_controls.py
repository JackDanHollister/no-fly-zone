"""Motor programs, no data needed."""

import numpy as np

from noflyzone.controls import CockpitControls

NAMES = ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT",
         "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE",
         "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"]
DT = 1 / 15


def rates(**kw):
    r = {"DNa02_L": 0.0, "DNa02_R": 0.0, "DNg100": 0.0, "pIP10": 0.0, "DNp01": 0.0}
    r.update(kw)
    return r


def test_silence_yields_noop():
    c = CockpitControls(NAMES, calibration="pool4", steer_mode="li")
    for _ in range(100):
        c.update(rates(), DT)
        assert NAMES[c.to_action()] == "NOOP"


def test_one_sided_object_turns_toward_it_then_fixates():
    c = CockpitControls(NAMES, calibration="pool4", saccade_hold=3, steer_mode="li")
    seq = []
    for _ in range(8):
        c.update(rates(DNa02_L=12.0), DT)
        seq.append(NAMES[c.to_action()])
    assert seq[0] == "LEFT" and seq[1:4] == ["NOOP"] * 3 and seq[4] == "LEFT"


def test_wide_field_event_gives_no_turn():
    c = CockpitControls(NAMES, calibration="pool4", steer_mode="li")
    for _ in range(5):
        c.update(rates(DNa02_L=60.0, DNa02_R=60.0), DT)
        assert "LEFT" not in NAMES[c.to_action()] and "RIGHT" not in NAMES[c.to_action()]


def test_startle_reverses_then_refractory():
    c = CockpitControls(NAMES, calibration="pool4", saccade_hold=3, startle_ticks=4,
                        startle_refractory=6, steer_mode="li")
    acts = []
    for k in range(12):
        c.update(rates(DNp01=40.0), DT)   # loom every tick
        acts.append(NAMES[c.to_action()])
    assert acts[0].startswith("DOWN")
    assert any(a == "NOOP" for a in acts[4:10]), "refractory should hold the reverse off"
