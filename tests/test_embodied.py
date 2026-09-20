"""M6: the levers are read from the controller as measured positions, and the
seated fly moves the physical levers with its forelegs (needs the flybody)."""

import numpy as np
import pytest

from noflyzone.cockpit import FLY_XML, Cockpit
from noflyzone.controls import CockpitControls

ACTIONS = ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT",
           "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE",
           "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"]


def test_measured_levers_drive_the_action_not_the_intent():
    c = CockpitControls(ACTIONS)
    c.throttle, c.steer = 1.0, 0.0                 # the intent says forward
    assert ACTIONS[c.to_action()] == "UP"
    c.measured = (-0.8, -0.8)                      # the levers say the legs pushed back
    s, t = c.effective
    assert s == 0.0 and -0.8 < t < -0.5              # through the dead zone
    assert ACTIONS[c.to_action()] == "DOWN"
    assert (c.lever_left, c.lever_right) == (-0.8, -0.8)
    c.measured = (-0.8, 0.8)                       # right forward, left back: a right turn
    s, t = c.effective
    assert s > 0.5 and t == 0.0
    assert ACTIONS[c.to_action()] == "RIGHT"
    c.measured = (-0.25, -0.2)                     # the resting hysteresis is not a command
    assert c.effective == (0.0, 0.0) and ACTIONS[c.to_action()] == "NOOP"
    c.fire = True                                  # the intent to fire is not a shot
    c.fire_measured = False                        # until the button is pressed
    assert ACTIONS[c.to_action()] == "NOOP"
    c.fire_measured = True
    assert ACTIONS[c.to_action()] == "FIRE"


@pytest.mark.skipif(not FLY_XML.exists(), reason="flybody assets not fetched")
def test_seated_fly_moves_the_physical_levers():
    c = Cockpit(physics=True)
    assert c.physics and abs(c.lever_values()[0]) < 0.1     # zeroed at rest
    assert c.lever_contacts()["left"] > 0 and c.lever_contacts()["right"] > 0
    c.set_lever_targets(1.0, 1.0); c.step_physics(0.4)
    fwd = c.lever_values()
    assert fwd[0] > 0.5 and fwd[1] > 0.5
    c.set_lever_targets(-1.0, 1.0); c.step_physics(0.4)
    turn = c.lever_values()
    assert turn[0] < -0.4 and turn[1] > 0.4
    assert np.abs(c.data.qvel).max() < 50                   # no chatter
    assert c.lever_contacts()["left"] > 0 and c.lever_contacts()["right"] > 0
    # the proboscis presses the button and lets it go
    assert c.fire_pressed() < 0.2
    c.set_fire_intent(True)
    for _ in range(6):
        c.step_physics(1 / 15)
    assert c.fire_pressed() >= 0.35
    c.set_fire_intent(False)
    for _ in range(6):
        c.step_physics(1 / 15)
    assert c.fire_pressed() < 0.2
    c.close()
