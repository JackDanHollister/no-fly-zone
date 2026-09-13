"""Battlezone game layer: ALE via gymnasium, frames out, joystick actions in."""

import ale_py
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)

ENV_ID = "ALE/BattleZone-v5"


class BattleZone:
    """Thin wrapper: reset-on-death loop yielding RGB frames (210x160x3)."""

    def __init__(self, seed: int | None = None):
        self.env = gym.make(ENV_ID)
        self.n_actions = self.env.action_space.n
        self.frame, _ = self.env.reset(seed=seed)
        self.episode_reward = 0.0
        self.episodes = 0

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Advance one env step. Returns (frame, reward, died_this_step)."""
        self.frame, reward, terminated, truncated, _ = self.env.step(action)
        self.episode_reward += float(reward)
        done = terminated or truncated
        if done:
            self.episodes += 1
            self.episode_reward = 0.0
            self.frame, _ = self.env.reset()
        return self.frame, float(reward), done

    def close(self) -> None:
        self.env.close()
