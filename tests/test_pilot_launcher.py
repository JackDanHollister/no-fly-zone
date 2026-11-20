"""Usable launch settings and honest records when a viewer closes early."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from noflyzone.controls import CockpitControls
from noflyzone.dn import DN_NAMES

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot, m3 = load("pilot"), load("run_m3")


def test_launch_preset_and_explicit_overrides():
    args = m3.parse_args(["--preset", "pilot"])
    assert {k: getattr(args, k) for k in m3.PILOT_PRESET} == m3.PILOT_PRESET
    assert m3.parse_args([]).injection == "v0"
    assert m3.parse_args(["--zoom", "1.5", "--preset", "pilot"]).zoom == 1.5


def test_headless_default_and_live_pacing(monkeypatch):
    monkeypatch.setattr(pilot, "desktop_available", lambda: False)
    args = pilot.parse_args([])
    assert args.mode == "record"
    assert "--live" not in pilot.command(args, "test")
    with pytest.raises(SystemExit):
        pilot.parse_args(["watch"])
    monkeypatch.setattr(pilot, "desktop_available", lambda: True)
    args = pilot.parse_args(["--seconds", "2", "--embodied"])
    cmd = pilot.command(args, "test")
    assert "--live" in cmd and "--realtime" in cmd
    assert cmd[cmd.index("--steps") + 1] == "30"
    assert cmd[cmd.index("--pilot") + 1] == "embodied"


def test_invalid_duration_and_existing_output_rejected(monkeypatch, tmp_path):
    for duration in ("0", "-1", "nan", "inf"):
        with pytest.raises(SystemExit):
            pilot.parse_args(["record", "--seconds", duration])
    monkeypatch.setattr(m3, "OUT", tmp_path)
    (tmp_path / "existing.mp4").write_bytes(b"keep me")
    with pytest.raises(SystemExit):
        m3.parse_args(["--video", "--video-name", "existing"])
    assert (tmp_path / "existing.mp4").read_bytes() == b"keep me"


def test_loop_closes_when_brain_initialization_fails(monkeypatch):
    closed = []
    loop = SimpleNamespace(retina=None, close=lambda: closed.append(True))
    monkeypatch.setattr(m3, "PilotLoop", lambda **kwargs: loop)

    def broken(*args, **kwargs):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(m3, "make_brain", broken)
    with pytest.raises(RuntimeError, match="backend unavailable"):
        m3.main(["--steps", "1"])
    assert closed == [True]


def test_closing_live_view_records_actual_duration_and_closes_resources(monkeypatch, tmp_path):
    closed = []
    frame = np.zeros((16, 16, 3), np.uint8)
    core = SimpleNamespace(n=1, indices_for_type=lambda t: np.array([], dtype=int))
    brain = SimpleNamespace(brain=core, window_counts=np.array([1]), describe=lambda: "fixture")
    controls = CockpitControls(["NOOP", "FIRE"])
    loop = SimpleNamespace(dt=1 / 15, retina=None, retina_right=None, controls=controls,
        game=SimpleNamespace(action_meanings=["NOOP", "FIRE"], frame=frame), total_reward=0,
        cockpit=SimpleNamespace(render=lambda view: frame), close=lambda: closed.append("loop"),
        latency_report=lambda: "fixture")
    loop.step = lambda: {"reward": 0, "life_lost": False, "episode_end": False,
        "dn": {name: 0.0 for name in DN_NAMES}, "rgb": {},
        "drives": {side: np.zeros((1, 2)) for side in ("left", "right")}}
    monkeypatch.setattr(m3, "OUT", tmp_path)
    monkeypatch.setattr(m3, "PilotLoop", lambda **kwargs: loop)
    seen = []
    monkeypatch.setattr(m3, "make_brain", lambda *args, **kw: seen.append(kw["seed"]) or core)
    monkeypatch.setattr(m3, "ConnectomeBrain", lambda **kwargs: brain)
    monkeypatch.setattr(m3, "BrainView", lambda core: SimpleNamespace(render=lambda counts: frame))
    monkeypatch.setattr(m3, "git_state", lambda: {"short": None})
    monkeypatch.setattr(m3.dashboard, "compose", lambda *args, **kwargs: frame)
    monkeypatch.setattr(m3, "dump_json", lambda path, payload, **kw: path.write_text(json.dumps(payload)))
    from noflyzone import liveview
    monkeypatch.setattr(liveview, "LiveView", lambda **kw: SimpleNamespace(
        show=lambda frame: False, close=lambda: closed.append("window")))
    m3.main(["--preset", "pilot", "--steps", "10", "--seed", "7", "--live",
             "--quiet", "--no-eyes", "--trace", "closed"])
    saved = json.loads((tmp_path / "closed.json").read_text())
    assert seen == [7]
    assert saved["steps"] == len(saved["trace"]) == 1
    assert saved["requested_steps"] == 10 and saved["stopped_early"]
    assert closed == ["window", "loop"]
