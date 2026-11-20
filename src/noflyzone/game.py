"""Battlezone game layer: ALE via gymnasium, frames out, joystick actions in.

Deliberate choices: sticky actions OFF (repeat_action_probability=0) so
behavior is a deterministic function of motor output — with stickiness on,
ALE stochastically repeats the previous action, which would confound M3's
closed-loop circuit validation. Frameskip stays at the v5 default (4 frames
at 60 Hz -> dt = 1/15 s per step); `dt` is exported here as the single
source of truth for the whole loop.

The env auto-resets on episode end: the fly keeps living across episodes
(stated decision). The boundary is reported via episode_end so temporal
state that must not span two worlds can be cleared.
"""

import hashlib

import ale_py
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)

ENV_ID = "ALE/BattleZone-v5"
# 15-60 ticks can stay in the input-insensitive opening and produce the
# same gameplay for different action sequences. Five to ten seconds reaches
# controllable play; start and trajectory diversity are checked separately.
WARMUP_MIN, WARMUP_MAX = 75, 150


class BattleZone:
    def __init__(self, seed: int | None = None, *, auto_reset: bool = True):
        self.env = gym.make(ENV_ID, repeat_action_probability=0.0)
        frameskip = self.env.unwrapped._frameskip
        assert isinstance(frameskip, int), (
            f"stochastic frameskip {frameskip!r} unsupported: dt must be fixed")
        self.dt = frameskip / 60.0
        self.n_actions = self.env.action_space.n
        self.action_meanings = self.env.unwrapped.get_action_meanings()
        self.auto_reset = auto_reset
        self.episodes = 0
        self.reset(seed=seed)

    def reset(self, seed: int | None = None, *, warmup_seed: int | None = None) -> dict:
        """Start a game, optionally diversifying ALE's otherwise identical starts.

        Warm-up actions are outside the scored episode. Record the actual
        starting image and RAM, not just the requested seed label.
        """
        self.frame, info = self.env.reset(seed=seed)
        actions = []
        warmup_ends = 0
        if warmup_seed is not None:
            rng = np.random.default_rng(warmup_seed)
            actions = rng.integers(self.n_actions, size=int(
                rng.integers(WARMUP_MIN, WARMUP_MAX + 1))).tolist()
            for action in actions:
                self.frame, _, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    warmup_ends += 1
                    self.frame, info = self.env.reset()
        self._lives = info.get("lives", 0)
        self.episode_reward = 0.0
        return {"game_seed": seed, "warmup_seed": warmup_seed,
                "warmup_actions": actions, "warmup_ticks": len(actions),
                "warmup_game_overs": warmup_ends, "start_lives": int(self._lives),
                "start_frame_sha256": hashlib.sha256(self.frame.tobytes()).hexdigest(),
                "start_ram_sha256": hashlib.sha256(
                    self.env.unwrapped.ale.getRAM().tobytes()).hexdigest()}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        """Advance one env step.

        Returns (frame, reward, life_lost, episode_end): life_lost = the tank
        took fatal damage (a lives drop or terminal death); episode_end = the
        game ended (death or ALE's truncation). With auto_reset=True the
        returned frame is from the new world; otherwise it is terminal.
        """
        self.frame, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += float(reward)
        lives = info.get("lives", 0)
        life_lost = lives < self._lives or terminated
        self._lives = lives
        episode_end = terminated or truncated
        if episode_end:
            self.episodes += 1
            if self.auto_reset:
                self.reset()
        return self.frame, float(reward), life_lost, episode_end

    def close(self) -> None:
        self.env.close()
