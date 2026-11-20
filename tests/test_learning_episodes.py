"""Episode independence: actual ALE starts and the batched reset contract."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from noflyzone.game import BattleZone
from noflyzone import rl_env


def test_actual_game_starts_are_distinct_and_reproducible():
    game = BattleZone(auto_reset=False)
    try:
        starts = []
        frames = []
        trajectories = []
        actions = np.random.default_rng(123).integers(game.n_actions, size=450)
        for seed in (700, 701, 702, 703, 800, 801, 802, 803):
            start = game.reset(seed=rl_env.episode_seed(seed, 0, 0),
                               warmup_seed=rl_env.episode_seed(seed, 0, 1))
            starts.append(start)
            assert game.episode_reward == 0
            frames.append(game.step(0)[0].copy())
            digest = hashlib.sha256()
            for action in actions:
                frame, _, _, ended = game.step(int(action))
                digest.update(frame.tobytes())
                if ended:
                    break
            trajectories.append(digest.hexdigest())
        assert len({s["start_frame_sha256"] for s in starts}) == 8
        assert len({s["start_ram_sha256"] for s in starts}) == 8
        assert len(set(trajectories)) == 8
        for j, seed in enumerate((700, 701, 702, 703, 800, 801, 802, 803)):
            repeated = game.reset(seed=rl_env.episode_seed(seed, 0, 0),
                                  warmup_seed=rl_env.episode_seed(seed, 0, 1))
            assert repeated == starts[j]
            np.testing.assert_array_equal(game.step(0)[0], frames[j])
    finally:
        game.close()


def test_game_terminal_frame_is_not_replaced_when_reset_is_explicit():
    terminal = np.full((2, 2, 3), 17, np.uint8)
    game = BattleZone.__new__(BattleZone)
    game.auto_reset = False
    game._lives = 1
    game.episode_reward = 0
    game.episodes = 0
    game.env = SimpleNamespace(step=lambda action: (terminal, 1000, True, False, {"lives": 0}))
    frame, reward, lost, ended = game.step(0)
    assert frame is terminal and reward == 1000 and lost and ended
    assert game.episode_reward == 1000 and game.episodes == 1


@pytest.fixture
def squad(monkeypatch):
    class Brain:
        def __init__(self, backend, batch, seed, **kwargs):
            self.n, self.B, self.seed, self.meta = 2, batch, seed, {}
            self.state = np.zeros((batch, 2), np.int32)
            self.row_seeds = [None] * batch

        def reset(self):
            self.state[:] = 0

        def reset_rows(self, rows, *, seeds):
            self.state[rows] = 0
            for i, seed in zip(rows, seeds):
                self.row_seeds[i] = seed

        def update_rates(self, rates):
            pass

        def run_counts(self, steps):
            self.state += 1
            return self.state.copy()

    class Loop:
        def __init__(self, seed, **kwargs):
            assert kwargs["auto_reset"] is False
            self.game = SimpleNamespace(action_meanings=["NOOP", "FIRE"])
            self.dt, self.retina, self.steps = 1 / 15, None, 0
            self.ends_at = 2 if seed == 701 else 100

        def reset(self, seed, *, warmup_seed):
            self.steps = 0
            return {"game_seed": seed, "warmup_seed": warmup_seed,
                    "start_frame_sha256": str(warmup_seed)}

        def sense(self, action):
            self.steps += 1
            return {"drives": {s: np.ones((1, 2)) for s in ("left", "right")},
                    "reward": 1000, "life_lost": self.steps == self.ends_at,
                    "episode_end": self.steps == self.ends_at}

        def close(self):
            pass

    class Connectome:
        def __init__(self, **kwargs):
            self._pr_all = np.array([0])
            self.n_visual, self._pops = 0, []
            self._readout, self._derived = {"DNp01": np.array([0])}, {}

        def _register(self):
            pass

    monkeypatch.setattr(rl_env, "make_brain", Brain)
    monkeypatch.setattr(rl_env, "PilotLoop", Loop)
    monkeypatch.setattr(rl_env, "ConnectomeBrain", Connectome)
    monkeypatch.setattr(rl_env.FlySquad, "_build_feature_pooling",
                        lambda self, grid: setattr(self, "n_features", 1))
    monkeypatch.setattr(rl_env.FlySquad, "_brain_features",
                        lambda self, counts: counts.sum(axis=1, keepdims=True).astype(np.float32))
    return rl_env.FlySquad(batch=2, seeds=[700, 701], neural_seed=73, episode_ticks=3)


def test_reset_replays_start_schedule_without_a_hidden_tick(squad):
    obs = squad.reset()
    original = [s.copy() for s in squad.starts]
    assert squad.brain.seed == 73
    assert len(set(squad.brain.row_seeds)) == 2
    assert not obs.any() and not squad.tick.any()
    assert [loop.steps for loop in squad.loops] == [0, 0]
    squad.step([0, 0])
    squad.extra_rates[:] = 100
    squad.reset()
    assert squad.starts == original
    assert not squad.extra_rates.any() and not squad.last_counts.any()
    assert not squad.tick.any() and not squad.ep_kills.any()


def test_game_over_and_tick_limit_reset_individual_worlds(squad):
    squad.reset()
    original = [s.copy() for s in squad.starts]
    squad.step([0, 0])
    obs, _, done, infos = squad.step([0, 0])
    assert done.tolist() == [False, True]
    assert squad.tick.tolist() == [2, 0]
    assert squad.loops[0].steps == 2 and squad.loops[1].steps == 0
    assert infos[1]["episode"]["ticks"] == 2
    assert infos[1]["episode"]["kills"] == 2
    assert infos[1]["episode"]["end_reason"] == "game_over"
    assert infos[1]["episode"]["start"] == original[1]
    assert obs[1, 0] == 0 and infos[1]["terminal_observation"][0] > 0
    assert squad.last_counts[1].sum() > 0  # available for the terminal learning update
    assert squad.starts[1]["neural_seed"] != original[1]["neural_seed"]
    squad.extra_rates[:] = 100
    obs, _, done, infos = squad.step([0, 0])
    assert done.tolist() == [True, False]
    assert infos[0]["episode"]["ticks"] == 3
    assert infos[0]["episode"]["kills"] == 3
    assert infos[0]["episode"]["end_reason"] == "tick_limit"
    assert obs[0, 0] == 0 and squad.extra_rates[0, 0] == 0
    assert squad._prev[0] == {"left": None, "right": None}
    assert squad.loops[0].steps == 0 and squad.loops[1].steps == 1


def test_inactive_flies_do_not_accumulate_extra_episodes(squad):
    squad.reset()
    for _ in range(4):
        _, reward, done, infos = squad.step([0, 0], active=[True, False])
        assert reward[1] == 0 and not done[1] and infos[1] == {}
    assert squad.loops[1].steps == 0 and squad.episode_index[1] == 0
    assert squad.last_counts[1].sum() == 0
