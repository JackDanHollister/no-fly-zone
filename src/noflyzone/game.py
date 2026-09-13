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

import ale_py
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)

ENV_ID = "ALE/BattleZone-v5"


class BattleZone:
    def __init__(self, seed: int | None = None):
        self.env = gym.make(ENV_ID, repeat_action_probability=0.0)
        frameskip = self.env.unwrapped._frameskip
        assert isinstance(frameskip, int), (
            f"stochastic frameskip {frameskip!r} unsupported: dt must be fixed")
        self.dt = frameskip / 60.0
        self.n_actions = self.env.action_space.n
        self.action_meanings = self.env.unwrapped.get_action_meanings()
        self.frame, info = self.env.reset(seed=seed)
        self._lives = info.get("lives", 0)
        self.episode_reward = 0.0
        self.episodes = 0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        """Advance one env step.

        Returns (frame, reward, life_lost, episode_end): life_lost = the tank
        took fatal damage (a lives drop or terminal death); episode_end = the
        world is about to be a fresh episode (death or ALE's 108k-frame
        truncation) and the returned frame is already from the new world.
        """
        self.frame, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += float(reward)
        lives = info.get("lives", 0)
        life_lost = lives < self._lives or terminated
        self._lives = lives
        episode_end = terminated or truncated
        if episode_end:
            self.episodes += 1
            self.episode_reward = 0.0
            self.frame, info = self.env.reset()
            self._lives = info.get("lives", 0)
        return self.frame, float(reward), life_lost, episode_end

    def close(self) -> None:
        self.env.close()
