"""Tests for binocular crosshair coverage, active foraging controls, bilateral
feature pooling, and the factored recurrent GRU readout."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from noflyzone.cockpit import Cockpit
from noflyzone.controls import CockpitControls
from noflyzone.retina import HexRetina
from noflyzone import rl_env

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train_readout = _load_script("train_readout")
pilot = _load_script("pilot")
m3 = _load_script("run_m3")

ACTIONS = [
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT",
    "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
    "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE",
]


def test_converged_eyes_eliminate_crosshair_blind_spot_and_bezel_bleed():
    # Legacy setting (fovy=45 parallel, zoom=2.0, eye_offset=0.15) has a blind spot at x=80
    legacy = Cockpit(with_fly=False)
    try:
        leg_l = HexRetina(image_shape=(240, 320), zoom=2.0, center=(0.35, 0.5))
        leg_r = HexRetina(image_shape=(240, 320), zoom=2.0, center=(0.65, 0.5))
        crosshair_bar = np.zeros((210, 160, 3), dtype=np.uint8)
        crosshair_bar[85:115, 78:83] = 255
        legacy.set_screen(crosshair_bar)
        assert (leg_l.sample(legacy.render("eye_left"))[:, 0] > 10).sum() == 0
        assert (leg_r.sample(legacy.render("eye_right"))[:, 0] > 10).sum() == 0
    finally:
        legacy.close()

    # Converged fovy=54 setting keeps 0 bezel bleed, sees radar, and overlaps on x=80
    conv = Cockpit(with_fly=False, eye_fovy=54.0, converge_eyes=True)
    try:
        ret_l = HexRetina(image_shape=(240, 320), zoom=1.15, center=(0.5 - 0.11, 0.46))
        ret_r = HexRetina(image_shape=(240, 320), zoom=1.15, center=(0.5 + 0.11, 0.46))

        conv.set_screen(np.zeros((210, 160, 3), dtype=np.uint8))
        assert conv.render("eye_left").max() == 0
        assert conv.render("eye_right").max() == 0

        conv.set_screen(crosshair_bar)
        sl = ret_l.sample(conv.render("eye_left"))[:, 0]
        sr = ret_r.sample(conv.render("eye_right"))[:, 0]
        assert (sl > 10).sum() >= 15 and (sr > 10).sum() >= 15

        radar_box = np.zeros((210, 160, 3), dtype=np.uint8)
        radar_box[6:34, 65:95] = 255
        conv.set_screen(radar_box)
        assert (ret_l.sample(conv.render("eye_left"))[:, 0] > 10).sum() >= 5
        assert (ret_r.sample(conv.render("eye_right"))[:, 0] > 10).sum() >= 5
    finally:
        conv.close()


def test_sparse_retina_sample_matches_reference_bincount():
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    ret = HexRetina(image_shape=(240, 320), zoom=1.35, center=(0.42, 0.51))
    got = ret.sample(img)
    flat = img.reshape(-1, 3).astype(np.float32)
    ids = ret._omm_of_pixel[ret._valid]
    expected = np.stack(
        [np.bincount(ids, weights=flat[ret._valid, c], minlength=ret.n_ommatidia) for c in range(3)],
        axis=1,
    ) / ret._counts[:, None]
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_active_foraging_controls_search_lock_and_commit_free_peek():
    c = CockpitControls(
        ACTIONS,
        calibration="pool4",
        saccade_hold=1,
        steer_mode="li",
        steer_min_hz=6.0,
        search_steer=1.0,
        search_delay=2,
        fovea_li=0.20,
        direct_steer=True,
    )
    quiet = {"DNa02_L": 0.0, "DNa02_R": 0.0, "DNg100": 0.0, "pIP10": 0.0, "DNp01": 0.0}
    c.update(quiet, 1 / 15)
    assert ACTIONS[c.to_action()] == "NOOP"
    c.update(quiet, 1 / 15)
    # Peek without commit must not arm saccade suppression early
    assert ACTIONS[c.to_action(commit=False)] == "RIGHT"
    assert c._hold == 0
    assert ACTIONS[c.to_action(commit=True)] == "RIGHT"
    assert c._hold == 1

    # Next tick is the post-saccade fixation tick -> NOOP
    c.update(quiet, 1 / 15)
    assert c.suppressed and ACTIONS[c.to_action()] == "NOOP"

    # Target centered in binocular fovea -> hold aim (no turn), advance and FIRE
    centered = {"DNa02_L": 15.0, "DNa02_R": 16.0, "DNg100": 50.0, "pIP10": 15.0, "DNp01": 0.0}
    c.update(centered, 1 / 15)
    assert ACTIONS[c.to_action()] == "UPFIRE"
    # Forward-only movement under direct_steer must not blind the next tick
    assert c._hold == 0


def test_bilateral_feature_pooling_distinguishes_left_and_right(monkeypatch):
    fake_ann = pd.DataFrame({
        "root_id": [10, 11, 12],
        "cell_type": ["DNa02", "DNa02", "DNg99"],
        "super_class": ["descending", "descending", "descending"],
        "side": ["left", "right", "center"],
    })
    monkeypatch.setattr(rl_env, "annotations", lambda version="783": fake_ann)

    dummy_self = SimpleNamespace(
        brain=SimpleNamespace(root_ids=np.array([10, 11, 12]), n=3, version="783"),
        loops=[SimpleNamespace(retina=SimpleNamespace(centers_xy=np.zeros((4, 2), np.float32)))],
        dt=1 / 15,
    )
    counts_left = np.array([[10, 0, 0]], dtype=np.int32)
    counts_right = np.array([[0, 10, 0]], dtype=np.int32)

    dummy_self.features_mode = "brain"
    rl_env.FlySquad._build_feature_pooling(dummy_self, (2, 2))
    sym_l = rl_env.FlySquad._brain_features(dummy_self, counts_left)
    sym_r = rl_env.FlySquad._brain_features(dummy_self, counts_right)
    np.testing.assert_allclose(sym_l, sym_r)  # legacy mode collapses hemispheres

    dummy_self.features_mode = "brain_bilateral"
    rl_env.FlySquad._build_feature_pooling(dummy_self, (2, 2))
    bil_l = rl_env.FlySquad._brain_features(dummy_self, counts_left)
    bil_r = rl_env.FlySquad._brain_features(dummy_self, counts_right)
    assert dummy_self.n_features == 3
    assert not np.allclose(bil_l, bil_r)


def test_action_codec_bijection_and_gru_episode_reset():
    a_to_f, f_to_a = train_readout.build_action_codec(ACTIONS)
    assert a_to_f.shape == (18, 3) and f_to_a.shape == (3, 3, 2)
    reconstructed = [int(f_to_a[th, st, fi]) for th, st, fi in a_to_f]
    assert reconstructed == list(range(18))

    torch.manual_seed(0)
    net = train_readout.RecurrentActorCritic(n_in=6, n_motor=4, hidden=16)
    net.eval()
    x_seq = torch.randn(4, 2, 6)
    m_seq = torch.zeros(4, 2, 4)
    done_prev = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    h0 = torch.randn(2, 16)
    fact = torch.zeros(4, 2, 3, dtype=torch.long)

    logp_a, _, val_a, _ = net.evaluate_chunk(x_seq, m_seq, done_prev, h0, fact)
    # Changing initial h0 for fly 0 must NOT change post-reset step t=2 or t=3 for fly 0
    h0_alt = h0.clone()
    h0_alt[0] = torch.randn(16) * 5.0
    logp_b, _, val_b, _ = net.evaluate_chunk(x_seq, m_seq, done_prev, h0_alt, fact)
    assert not torch.allclose(val_a[:2, 0], val_b[:2, 0])
    torch.testing.assert_close(val_a[2:, 0], val_b[2:, 0])
    torch.testing.assert_close(logp_a[2:, 0], logp_b[2:, 0])


def test_active_preset_and_controller_selection(monkeypatch, tmp_path):
    args_act = m3.parse_args(["--preset", "active"])
    assert args_act.converge_eyes and args_act.eye_fovy == 54.0 and args_act.direct_steer
    assert args_act.injection == "v11" and args_act.suppress_turn_transients
    assert args_act.zoom == 1.75 and args_act.eye_offset == 0.21 and args_act.patrol_sweep == 6
    assert args_act.no_off_axis_fire and args_act.sharp_steer_hz == 6.0
    fake_ckpt = tmp_path / "champ.pt"
    fake_ckpt.write_bytes(b"pt")
    monkeypatch.setattr(pilot, "resolve_champion_ckpt", lambda: fake_ckpt)

    # Default launcher uses pure frozen-connectome active reflex (zero RL) even if a GRU checkpoint exists
    p_def = pilot.parse_args(["record"])
    cmd_def = pilot.command(p_def, "t_def")
    assert cmd_def[cmd_def.index("--preset") + 1] == "active"
    assert "--readout-ckpt" not in cmd_def

    # Explicit --controller learned opts into the trained GRU readout
    p_lrn = pilot.parse_args(["record", "--controller", "learned"])
    cmd_lrn = pilot.command(p_lrn, "t_lrn")
    assert "--readout-ckpt" in cmd_lrn and str(fake_ckpt) in cmd_lrn

    p_leg = pilot.parse_args(["record", "--controller", "legacy"])
    cmd_leg = pilot.command(p_leg, "t_leg")
    assert cmd_leg[cmd_leg.index("--preset") + 1] == "pilot"
    p_act = pilot.parse_args(["record", "--controller", "active", "--seed", "3"])
    cmd_act = pilot.command(p_act, "t_act")
    assert cmd_act[cmd_act.index("--preset") + 1] == "active"
    assert "--warmup" in cmd_act


def test_symmetric_patrol_and_directional_loom_escape():
    c = CockpitControls(
        ACTIONS,
        calibration="pool4",
        saccade_hold=1,
        startle_ticks=5,
        startle_refractory=18,
        startle_thresh=0.60,
        steer_mode="li",
        steer_min_hz=4.0,
        search_steer=1.0,
        search_delay=2,
        patrol_sweep=2,
        patrol_stride=2,
        fovea_li=0.22,
        direct_steer=True,
    )
    quiet = {"DNa02_L": 0.0, "DNa02_R": 0.0, "DNg100": 0.0, "pIP10": 0.0, "DNp01": 0.0}
    seen = []
    for _ in range(18):
        c.update(quiet, 1 / 15)
        seen.append(ACTIONS[c.to_action()])
    # Sweeps both RIGHT and LEFT and takes forward optic-flow strides in between
    assert "RIGHT" in seen and "LEFT" in seen and "UP" in seen

    # Close loom louder on right side -> jinks diagonally DOWNLEFT without firing
    loom_right = {"DNa02_L": 4.0, "DNa02_R": 16.0, "DNg100": 40.0, "pIP10": 12.0, "DNp01": 14.0}
    c.update(loom_right, 1 / 15)
    act = ACTIONS[c.to_action()]
    assert act == "DOWNLEFT" and not c.fire


def test_signed_chroma_and_tight_crosshair_fire_discipline():
    from noflyzone.brain_connectome import compact_chroma, directional_surround_matrices

    ret = HexRetina(image_shape=(240, 320), zoom=1.75, center=(0.5, 0.48))
    H_w, V_w = directional_surround_matrices(ret.centers_xy)
    xy = ret.centers_xy

    # Horizontal sky-over-ground step: uniform along x -> 0 compact chromatic response
    lum = np.where(xy[:, 1] > 0.0, 0.43, 0.40).astype(np.float32)
    blue = np.where(xy[:, 1] > 0.0, 0.64, 0.15).astype(np.float32)
    assert compact_chroma(lum, blue, H_w, V_w).max() == 0.0

    # Compact blue object on green ground -> strong positive response at object centre
    spot = (xy[:, 0] ** 2 + (xy[:, 1] + 0.15) ** 2) < 0.12 ** 2
    lum_obj = np.where(spot, 0.28, lum)
    blue_obj = np.where(spot, 0.55, blue)
    cc = compact_chroma(lum_obj, blue_obj, H_w, V_w)
    assert cc[spot].max() > 0.20 and cc[~spot].max() == 0.0

    # Tight trigger discipline: off-axis target turns without firing; sharp unilateral 7.5 Hz turns LEFT; centred target locks & fires
    c = CockpitControls(
        ACTIONS, calibration="pool4", saccade_hold=1, startle_ticks=5,
        startle_refractory=18, startle_thresh=0.80, steer_mode="li",
        steer_min_hz=8.0, sharp_steer_hz=6.0, search_steer=1.0,
        patrol_sweep=6, patrol_stride=4, fovea_li=0.18, pip_lock_drive=0.50,
        off_axis_fire=False, direct_steer=True,
    )
    # Sharp unilateral left target (7.5 Hz < 8.0 Hz gate, |LI|=1.0 >= 0.60) -> turns LEFT, no off-axis shot
    c.update({"DNa02_L": 7.5, "DNa02_R": 0.0, "DNg100": 40.0, "pIP10": 0.0, "DNp01": 0.0}, 1 / 15)
    assert ACTIONS[c.to_action(commit=False)] == "LEFT" and not c.fire
    # Centred binocular target -> UPFIRE
    c.update({"DNa02_L": 7.5, "DNa02_R": 7.5, "DNg100": 60.0, "pIP10": 7.5, "DNp01": 0.0}, 1 / 15)
    assert ACTIONS[c.to_action()] == "UPFIRE" and c.fire


def test_running_norm_variance_floor_tames_quiet_channels():
    norm = train_readout.RunningNorm(2, clip=10.0, var_floor=1.0)
    # Channel 0 is nearly silent; channel 1 has high variance (var ~ 4)
    samples = np.zeros((200, 2), dtype=np.float32)
    samples[0, 0] = 0.5
    samples[:, 1] = np.where(np.arange(200) % 2 == 0, 0.0, 4.0)
    norm.update(samples)
    assert norm.var[0] < 0.01 and norm.var[1] > 3.5
    # A single stray spike of 2.0 on channel 0 must NOT be amplified >10x
    out = norm(np.array([[2.0, 4.0]], dtype=np.float32))
    assert 1.8 < out[0, 0] < 2.1
    assert 0.9 < out[0, 1] < 1.1
