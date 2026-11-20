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
life-loss penalty. An episode is a fixed number of ticks or the game ending;
both boundaries reset the world and neural row. Reset observations describe
a resting brain (zero spikes), with no hidden NOOP tick in the score budget.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .brain_backend import make_brain
from .brain_connectome import TRANSIENT_CAP, ConnectomeBrain
from .brain_lif import DT, annotations
from .game import WARMUP_MIN, WARMUP_MAX
from .loop import PilotLoop

FEATURE_CLASSES = ("descending", "visual_projection", "visual_centrifugal")
EPISODE_PROTOCOL = "independent-starts-v1"


def episode_seed(*parts: int) -> int:
    """A reproducible uint32 seed, separate from any global NumPy RNG."""
    return int(np.random.SeedSequence(parts).generate_state(1)[0])


class FlySquad:
    def __init__(self, batch: int = 8, seeds: list[int] | None = None,
                 injection: str = "v4", readout: str = "pool4",
                 zoom: float = 2.0, eye_offset: float = 0.15,
                 brain_kwargs: dict | None = None, episode_ticks: int = 900,
                 life_penalty: float = 0.5, features: str = "brain",
                 retina_grid: tuple[int, int] = (12, 9),
                 extra_sites: dict[str, np.ndarray] | None = None,
                 neural_seed: int = 0,
                 retina_center: tuple[float, float] = (0.5, 0.5),
                 eye_fovy: float | None = None,
                 converge_eyes: bool = False,
                 suppress_turn_transients: bool = False):
        if batch < 1 or episode_ticks < 1:
            raise ValueError("batch and episode_ticks must be positive")
        if features not in ("brain", "brain_bilateral", "retina"):
            raise ValueError("features must be 'brain', 'brain_bilateral' or 'retina'")
        self.seeds = list(range(batch)) if seeds is None else list(seeds)
        if len(self.seeds) != batch or any(s < 0 for s in self.seeds):
            raise ValueError("provide one nonnegative game seed per fly")
        self.B = batch
        kw = {"aminergic": "zero", "sensory_input": "external_only"}
        kw.update(brain_kwargs or {})
        self.neural_seed = int(kw.pop("seed", neural_seed))
        if self.neural_seed < 0:
            raise ValueError("neural_seed must be nonnegative")
        self.brain = make_brain("cuda", batch=batch, seed=self.neural_seed, **kw)
        self.protocol = {"name": EPISODE_PROTOCOL, "game_seeds": self.seeds,
                         "neural_seed": self.neural_seed, "episode_ticks": episode_ticks,
                         "warmup_ticks": [WARMUP_MIN, WARMUP_MAX],
                         "reset_observation": "resting brain; no scored tick",
                         "rng": "Philox: episode seed, fly index, episode step, neuron"}
        self.brain.meta["learning_protocol"] = self.protocol
        self.suppress_turn_transients = bool(suppress_turn_transients)
        loop_extra = {}
        if retina_center != (0.5, 0.5):
            loop_extra["retina_center"] = retina_center
        if eye_fovy is not None:
            loop_extra["eye_fovy"] = eye_fovy
        if converge_eyes:
            loop_extra["converge_eyes"] = converge_eyes
        if suppress_turn_transients:
            loop_extra["suppress_turn_transients"] = suppress_turn_transients
        self.loops = [PilotLoop(seed=self.seeds[i], retina_zoom=zoom,
                                eye_offset=eye_offset, auto_reset=False,
                                **loop_extra) for i in range(batch)]
        self.dt = self.loops[0].dt
        self.n_steps = round(self.dt / DT)
        # the injection map and readouts, registered once on the batched brain
        self.cb = ConnectomeBrain(retina=self.loops[0].retina, brain=self.brain,
                                  injection=injection, readout=readout,
                                  extra_sites=extra_sites)
        self.n_poi = len(self.cb._pr_all)
        self.extra_rates = np.zeros((batch, self.n_poi - self.cb.n_visual), np.float32)
        self._prev = [{"left": None, "right": None} for _ in range(batch)]
        self._lp = [{"left": None, "right": None} for _ in range(batch)]
        self._prev_onoff = [{"left": None, "right": None} for _ in range(batch)]
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
        self.episode_index = np.zeros(batch, np.int64)
        self.starts = [dict() for _ in range(batch)]

    # -- features ----------------------------------------------------------

    def _build_feature_pooling(self, grid):
        version = getattr(self.brain, "version", "783")
        ann = annotations(version).drop_duplicates("root_id").set_index("root_id")
        ann = ann.reindex(self.brain.root_ids)
        ct = ann["cell_type"].fillna("").astype(str).to_numpy()
        sc = ann["super_class"].fillna("").astype(str).to_numpy()
        side = ann["side"].fillna("").astype(str).to_numpy()
        keep = np.isin(sc, FEATURE_CLASSES)
        rows, cols, vals = [], [], []
        if self.features_mode == "brain_bilateral":
            # Separate left, right, and unpaired populations so lateralised
            # visual/descending channels are preserved rather than averaged out.
            keys = sorted({(str(ct[k]), str(side[k])) for k in np.flatnonzero(keep)})
            types = [f"{t}/{s}" if s else t for t, s in keys]
            for r, (t, s) in enumerate(keys):
                idx = np.flatnonzero(keep & (ct == t) & (side == s))
                rows += [r] * idx.size
                cols += idx.tolist()
                vals += [1.0 / idx.size] * idx.size
        else:
            types = sorted(set(ct[keep]))
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
        self.n_features = (len(types) if self.features_mode in ("brain", "brain_bilateral")
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

    def _rates_and_delta(self, i: int, drives: dict, *, self_turn: bool = False):
        rates = np.zeros(self.n_poi, np.float32)
        rates[self.cb.n_visual:] = self.extra_rates[i]
        if hasattr(self.cb, "encode_stream"):
            state = {"prev_lum": self._prev[i], "lp": self._lp[i], "prev_onoff": self._prev_onoff[i]}
            _, delta = self.cb.encode_stream(
                drives, self.dt, state,
                self_turn=self_turn and self.suppress_turn_transients,
                out_rates=rates[:self.cb.n_visual],
            )
        else:
            lum = {s: drives[s][:, 0] for s in ("left", "right")}
            delta = {}
            for s in lum:
                prev = self._prev[i][s]
                delta[s] = (lum[s] - prev) if prev is not None else np.zeros_like(lum[s])
                self._prev[i][s] = lum[s]
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

    def _reset_rows(self, rows) -> np.ndarray:
        """Reset selected worlds and neural streams; preserve terminal counts
        in last_counts/last_dn for the caller's learning update.
        """
        obs = np.zeros((len(rows), self.n_features), np.float32)
        neural_seeds = []
        for j, i in enumerate(rows):
            base, ep = self.seeds[i], int(self.episode_index[i])
            neural = episode_seed(self.neural_seed, base, ep, 2)
            neural_seeds.append(neural)
            start = self.loops[i].reset(seed=episode_seed(base, ep, 0),
                                        warmup_seed=episode_seed(base, ep, 1))
            self.starts[i] = {**start, "protocol": EPISODE_PROTOCOL,
                              "game_seed_root": base, "episode_index": ep,
                              "neural_seed_root": self.neural_seed,
                              "neural_seed": neural, "neural_row": int(i)}
            self._prev[i] = {"left": None, "right": None}
            self._lp[i] = {"left": None, "right": None}
            self._prev_onoff[i] = {"left": None, "right": None}
            self.extra_rates[i] = 0
            self.tick[i] = 0
            self.ep_reward[i] = 0
            self.ep_kills[i] = 0
            self.ep_lives[i] = 0
            if self.features_mode == "retina":
                drives = self.loops[i].observe()["drives"]
                _, delta = self._rates_and_delta(i, drives)
                obs[j] = self._retina_features(drives, delta)
        self.brain.reset_rows(rows, seeds=neural_seeds)
        return obs

    def reset(self) -> np.ndarray:
        """Replay the configured start schedule, including actual game resets."""
        self.brain.reset()
        self.cb._register()
        self.episode_index[:] = 0
        self.last_counts[:] = 0
        self.last_dn = [dict() for _ in range(self.B)]
        return self._reset_rows(np.arange(self.B))

    def step(self, actions, *, active=None):
        """Step active flies, returning reset observations at either boundary.

        last_counts/last_dn retain this tick's terminal activity. Controllers
        must reset on done, rather than feeding that activity into a new game.
        Inactive rows are excluded from game time, rewards and observations.
        """
        B = self.B
        active = np.ones(B, bool) if active is None else np.asarray(active, bool)
        if active.shape != (B,):
            raise ValueError("active must have one entry per fly")
        rates = np.zeros((B, self.n_poi), np.float32)
        rewards = np.zeros(B, np.float32)
        dones = np.zeros(B, bool)
        infos = []
        retina_feats = np.zeros((B, self.n_features), np.float32) \
            if self.features_mode == "retina" else None
        for i, loop in enumerate(self.loops):
            if not active[i]:
                infos.append({})
                continue
            sensed = loop.sense(int(actions[i]))
            self_turn = bool(getattr(loop, "_last_rotated", False))
            r, delta = self._rates_and_delta(i, sensed["drives"], self_turn=self_turn)
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
                retina_feats[i] = self._retina_features(sensed["drives"], delta)
        self.brain.update_rates(rates)
        counts = self.brain.run_counts(self.n_steps)
        counts = counts.reshape(B, -1).copy()
        counts[~active] = 0
        self.last_counts = counts
        for i in range(B):
            dn = {name: float(counts[i, idx].mean() / self.dt)
                  for name, idx in self.cb._readout.items()}
            for name, (kind, (a, b)) in self.cb._derived.items():
                dn[name] = min(dn[a], dn[b])
            self.last_dn[i] = dn
        obs = (self._brain_features(counts) if self.features_mode in ("brain", "brain_bilateral")
               else retina_feats)
        # episode bookkeeping and per-fly resets
        for i in np.flatnonzero(dones):
            infos[i]["episode"] = {"reward": float(self.ep_reward[i]),
                                   "kills": int(self.ep_kills[i]),
                                   "lives_lost": int(self.ep_lives[i]),
                                   "ticks": int(self.tick[i]),
                                   "start": self.starts[i],
                                   "end_reason": "game_over" if infos[i]["game_over"]
                                       else "tick_limit"}
            infos[i]["terminal_observation"] = obs[i].copy()
            self.episode_index[i] += 1
        rows = np.flatnonzero(dones)
        if rows.size:
            obs[rows] = self._reset_rows(rows)
        return obs, rewards, dones, infos

    def close(self) -> None:
        for loop in self.loops:
            loop.close()
