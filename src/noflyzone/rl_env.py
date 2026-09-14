"""A squad of flies for training a readout on the frozen connectome.

B pilot loops (game, cockpit, retinas) feed one batched CUDA brain. Each
tick the policy picks an action per fly, the loops sense, the injection map
turns each fly's retina into Poisson rates, the brain runs one game tick for
all flies at once, and the observation is the brain's spike counts pooled by
cell type over the descending, visual-projection and visual-centrifugal
neurons. The connectome is never trained; only the policy on top of it is.

Conditions: brain_kwargs chooses the wiring (measured, rewired, silenced);
features "retina" bypasses the brain entirely and pools the photoreceptor
drives, the no-brain ceiling. Reward: kills (+1 per 1000 points) minus a
life-loss penalty. An episode is a fixed number of ticks or the game ending.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .brain_backend import make_brain
from .brain_connectome import TRANSIENT_CAP, ConnectomeBrain
from .brain_lif import DT, annotations
from .loop import PilotLoop

FEATURE_CLASSES = ("descending", "visual_projection", "visual_centrifugal")


class FlySquad:
    def __init__(self, batch: int = 8, seeds: list[int] | None = None,
                 injection: str = "v4", readout: str = "pool4",
                 zoom: float = 2.0, eye_offset: float = 0.15,
                 brain_kwargs: dict | None = None, episode_ticks: int = 900,
                 life_penalty: float = 0.5, features: str = "brain",
                 retina_grid: tuple[int, int] = (12, 9),
                 extra_sites: dict[str, np.ndarray] | None = None):
        self.B = batch
        kw = {"aminergic": "zero", "sensory_input": "external_only"}
        kw.update(brain_kwargs or {})
        self.brain = make_brain("cuda", batch=batch, **kw)
        self.loops = [PilotLoop(seed=(seeds[i] if seeds else i), retina_zoom=zoom,
                                eye_offset=eye_offset) for i in range(batch)]
        self.dt = self.loops[0].dt
        self.n_steps = round(self.dt / DT)
        # the injection map and readouts, registered once on the batched brain
        self.cb = ConnectomeBrain(retina=self.loops[0].retina, brain=self.brain,
                                  injection=injection, readout=readout,
                                  extra_sites=extra_sites)
        self.n_poi = len(self.cb._pr_all)
        self.extra_rates = np.zeros((batch, self.n_poi - self.cb.n_visual), np.float32)
        self._prev = [{"left": None, "right": None} for _ in range(batch)]
        self.episode_ticks = episode_ticks
        self.life_penalty = life_penalty
        self.features_mode = features
        self.action_names = list(self.loops[0].game.action_meanings)
        self.n_actions = len(self.action_names)
        self._build_feature_pooling(retina_grid)
        self.tick = np.zeros(batch, np.int64)
        self.ep_reward = np.zeros(batch)
        self.ep_kills = np.zeros(batch, np.int64)
        self.ep_lives = np.zeros(batch, np.int64)
        self.last_counts = np.zeros((batch, self.brain.n), np.int32)
        self.last_dn = [dict() for _ in range(batch)]

    # -- features ----------------------------------------------------------

    def _build_feature_pooling(self, grid):
        ann = annotations().drop_duplicates("root_id").set_index("root_id")
        ann = ann.reindex(self.brain.root_ids)
        ct = ann["cell_type"].fillna("").astype(str).to_numpy()
        sc = ann["super_class"].fillna("").astype(str).to_numpy()
        keep = np.isin(sc, FEATURE_CLASSES)
        types = sorted(set(ct[keep]))
        rows, cols, vals = [], [], []
        for r, t in enumerate(types):
            idx = np.flatnonzero(keep & (ct == t))
            rows += [r] * idx.size
            cols += idx.tolist()
            vals += [1.0 / idx.size] * idx.size
        self.pool = sparse.csr_matrix((vals, (rows, cols)),
                                      shape=(len(types), self.brain.n), dtype=np.float32)
        self.feature_types = types
        # retina pooling: coarse grid over the hex lattice, per eye, per channel
        xy = self.loops[0].retina.centers_xy
        gx = np.clip(((xy[:, 0] + 1) / 2 * grid[0]).astype(int), 0, grid[0] - 1)
        gy = np.clip(((xy[:, 1] + 1) / 2 * grid[1]).astype(int), 0, grid[1] - 1)
        cell = gy * grid[0] + gx
        n_cells = grid[0] * grid[1]
        counts = np.bincount(cell, minlength=n_cells).astype(np.float32)
        self.retina_pool = sparse.csr_matrix(
            (1.0 / counts[cell], (cell, np.arange(len(cell)))),
            shape=(n_cells, len(cell)), dtype=np.float32)
        self.n_features = (len(types) if self.features_mode == "brain"
                           else 4 * n_cells * 2)  # lum + delta, 2 channels, 2 eyes

    def _brain_features(self, counts: np.ndarray) -> np.ndarray:
        hz = (self.pool @ counts.T.astype(np.float32)).T / self.dt  # (B, F)
        return np.log1p(hz).astype(np.float32)

    def _retina_features(self, drives: dict, delta: dict) -> np.ndarray:
        parts = []
        for side in ("left", "right"):
            for arr in (drives[side], delta[side]):
                parts.append(self.retina_pool @ arr)  # (cells, 2)
        return np.concatenate([p.ravel() for p in parts]).astype(np.float32)

    def set_teaching(self, name: str, hz, flies=None) -> None:
        """Poisson rate for an extra site (e.g. 'pam'), for all flies or the
        given ones. Extra slices are in whole-activation-set coordinates."""
        sl = self.cb._extra[name]
        cols = slice(sl.start - self.cb.n_visual, sl.stop - self.cb.n_visual)
        if flies is None:
            self.extra_rates[:, cols] = hz
        else:
            self.extra_rates[np.asarray(flies), cols] = hz

    # -- injection, per fly --------------------------------------------------

    def _rates_and_delta(self, i: int, drives: dict):
        lum = {s: drives[s][:, 0] for s in ("left", "right")}
        delta = {}
        for s in lum:
            prev = self._prev[i][s]
            delta[s] = (lum[s] - prev) if prev is not None else np.zeros_like(lum[s])
            self._prev[i][s] = lum[s]
        rates = np.zeros(self.n_poi, np.float32)
        rates[self.cb.n_visual:] = self.extra_rates[i]
        for mode, gain, sl, omm, side in self.cb._pops:
            if mode == "lum":
                sig = lum[side][omm]
            elif mode == "on":
                sig = np.clip(delta[side][omm], 0.0, None)
            elif mode == "off":
                sig = np.clip(-delta[side][omm], 0.0, None)
            else:
                sig = np.abs(delta[side][omm])
            rates[sl] = np.minimum(sig * gain, TRANSIENT_CAP)
        full_delta = {s: np.stack([delta[s], delta[s]], 1) for s in delta}
        return rates, full_delta

    # -- the tick ------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.brain.reset()
        self.cb._register()
        self._prev = [{"left": None, "right": None} for _ in range(self.B)]
        self.tick[:] = 0
        self.ep_reward[:] = 0
        self.ep_kills[:] = 0
        self.ep_lives[:] = 0
        obs, *_ = self.step(np.zeros(self.B, np.int64))  # one NOOP tick
        return obs

    def step(self, actions):
        B = self.B
        rates = np.zeros((B, self.n_poi), np.float32)
        rewards = np.zeros(B, np.float32)
        dones = np.zeros(B, bool)
        infos = []
        retina_feats = []
        for i, loop in enumerate(self.loops):
            sensed = loop.sense(int(actions[i]))
            r, delta = self._rates_and_delta(i, sensed["drives"])
            rates[i] = r
            kill = sensed["reward"] > 0
            rewards[i] = sensed["reward"] / 1000.0 - self.life_penalty * float(sensed["life_lost"])
            self.ep_reward[i] += rewards[i]
            self.ep_kills[i] += int(kill)
            self.ep_lives[i] += int(sensed["life_lost"])
            self.tick[i] += 1
            game_over = bool(sensed["episode_end"])
            dones[i] = game_over or self.tick[i] >= self.episode_ticks
            infos.append({"game_over": game_over, "kill": bool(kill),
                          "life_lost": bool(sensed["life_lost"])})
            if self.features_mode == "retina":
                retina_feats.append(self._retina_features(sensed["drives"], delta))
        self.brain.update_rates(rates)
        counts = self.brain.run_counts(self.n_steps)
        counts = counts.reshape(B, -1)
        self.last_counts = counts
        for i in range(B):
            dn = {name: float(counts[i, idx].mean() / self.dt)
                  for name, idx in self.cb._readout.items()}
            for name, (kind, (a, b)) in self.cb._derived.items():
                dn[name] = min(dn[a], dn[b])
            self.last_dn[i] = dn
        obs = (self._brain_features(counts) if self.features_mode == "brain"
               else np.stack(retina_feats))
        # episode bookkeeping and per-fly resets
        for i in np.flatnonzero(dones):
            infos[i]["episode"] = {"reward": float(self.ep_reward[i]),
                                   "kills": int(self.ep_kills[i]),
                                   "lives_lost": int(self.ep_lives[i]),
                                   "ticks": int(self.tick[i])}
            self.ep_reward[i] = 0
            self.ep_kills[i] = 0
            self.ep_lives[i] = 0
            self.tick[i] = 0
            if infos[i]["game_over"]:      # the world jumped: no temporal bridge
                self.brain.reset_rows([i])
                self._prev[i] = {"left": None, "right": None}
        return obs, rewards, dones, infos

    def close(self) -> None:
        for loop in self.loops:
            loop.close()
