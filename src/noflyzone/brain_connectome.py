"""The real pilot: FlyWire v783 connectome behind the loop's brain contract.

step(drives, dt, signals) -> DN rates + reset(), same as the reflex stub:
- drives (per-eye ommatidium luminance in [0,1]) become Poisson rates on
  REAL neurons, assigned retinotopically: each cell's 3D position is
  PCA-projected to 2D per eye, axis-aligned to the FAFB frame (u ~ lateral x,
  v ~ vertical y), and matched to the nearest retina ommatidium. Orientation
  is a heuristic — good enough for luminance and motion energy; proper
  ommatidial ray mapping is future work.
- Injection sites: R1-6 get luminance (context/gain control). The retina and
  lamina both signal through sign-inverting graded synapses (histaminergic
  R1-6, glutamatergic L1) that a pure LIF network cannot propagate — cells
  receiving only inhibition never spike. So the retina+lamina stage is
  modeled explicitly at its output: the medulla ON channel (Mi1) gets
  luminance-increment transients, the OFF channels (Tm1, Tm2) get decrement
  transients, per ommatidium per game tick. From the medulla down the
  pathway is cholinergic/excitatory (Mi1/Tm -> T4/T5 -> LC4/LPLC2 -> GF) and
  runs on the untouched connectome.
- the whole 138,639-neuron LIF brain then runs dt of biological time
  (667 x 0.1 ms steps per 15 Hz game tick),
- DN rates are read out from the REAL descending neurons and emitted through
  the dn.py contract. Female-brain casting notes: pIP10 (male courtship song)
  does not exist in this brain — the FIRE trigger is the oviposition
  descending neurons (oviDN*): she shoots when broody.

On the CPU backend the full brain runs ~8x slower than real time (Mac and
workstation alike); the CUDA backend (brain_cuda) replays the tick as one
graph.
"""

import numpy as np
from scipy.spatial import cKDTree

from . import dn
from .brain_lif import DT, ShiuBrain, annotations

MAX_PR_RATE = 150.0  # Hz at full-white, the reference model's default drive

# Named injection maps: (cell type, drive mode, gain in Hz at full-scale
# signal). Modes: "lum" = luminance, "on" = rectified luminance increment,
# "off" = rectified decrement, "onoff" = |change|. Which map a run used is
# part of its provenance (describe()); see docs/research/cascade-probe-*.md.
INJECTION_MAPS = {
    # the M3/M4 map: retina context plus ON/OFF transients at three types
    "v0": [
        ("R1-6", "lum", MAX_PR_RATE),
        ("Mi1", "on", 3000.0),   # medulla ON channel: luminance increments
        ("Tm1", "off", 3000.0),  # medulla OFF channels: luminance decrements
        ("Tm2", "off", 2000.0),
    ],
    # the full columnar inputs of T4 (Mi1, Tm3) and T5 (Tm1, Tm2, Tm4, Tm9)
    "v1": [
        ("R1-6", "lum", MAX_PR_RATE),
        ("Mi1", "on", 3000.0), ("Tm3", "on", 3000.0),
        ("Tm1", "off", 3000.0), ("Tm2", "off", 2000.0),
        ("Tm4", "off", 2000.0), ("Tm9", "off", 2000.0),
    ],
    # v1 plus the small-object relays that feed LC11 (T2, T3, T2a) and the
    # loom relay input TmY3 (LC4), all as rectified transients
    "v2": [
        ("R1-6", "lum", MAX_PR_RATE),
        ("Mi1", "on", 3000.0), ("Tm3", "on", 3000.0),
        ("Tm1", "off", 3000.0), ("Tm2", "off", 2000.0),
        ("Tm4", "off", 2000.0), ("Tm9", "off", 2000.0),
        ("T2", "onoff", 3000.0), ("T3", "onoff", 3000.0),
        ("T2a", "onoff", 3000.0), ("TmY3", "onoff", 2000.0),
    ],
    # v1 plus the LC11 small-object inputs that do not also feed the LC4 loom
    # detector (T3, T2a); T2 and TmY3 are left out so a small object does not
    # fire the giant-fibre escape
    "v3": [
        ("R1-6", "lum", MAX_PR_RATE),
        ("Mi1", "on", 3000.0), ("Tm3", "on", 3000.0),
        ("Tm1", "off", 3000.0), ("Tm2", "off", 2000.0),
        ("Tm4", "off", 2000.0), ("Tm9", "off", 2000.0),
        ("T3", "onoff", 3000.0), ("T2a", "onoff", 3000.0),
    ],
    # v1 plus T3 only: T2a also drives DNp01 (escape) via LC15/LC17
    "v4": [
        ("R1-6", "lum", MAX_PR_RATE),
        ("Mi1", "on", 3000.0), ("Tm3", "on", 3000.0),
        ("Tm1", "off", 3000.0), ("Tm2", "off", 2000.0),
        ("Tm4", "off", 2000.0), ("Tm9", "off", 2000.0),
        ("T3", "onoff", 3000.0),
    ],
}
INJECTION_SITES = INJECTION_MAPS["v0"]
TRANSIENT_CAP = 400.0  # Hz ceiling on injected transient rates

# Named readout maps: dn.py contract name -> list of selectors
# (kind, value, sides); a pool's rate is the mean over all its cells.
# Special kind "min_of" takes the minimum of two other contract rates.
STEER_POOL = ["DNp07", "DNb05", "DNg52", "DNb06", "DNa02"]
# DNp12 is the loom-selective type in the screen (24 Hz loom, 0.6 Hz small
# object); DNbe001 was dropped because it also fires for a tank sitting ahead
LOOM_POOL = ["DNp12", "DNp01"]
READOUT_MAPS = {
    # the M3/M4 map: single named cells
    "v0": {
        "DNa02_L": [("cell_type", "DNa02", ("left",))],
        "DNa02_R": [("cell_type", "DNa02", ("right",))],
        "DNg100": [("cell_type", "DNg100", None)],
        "pIP10": [("cell_type_prefix", "oviDN", None)],  # female stand-in
        "DNp01": [("cell_type", "DNp01", None)],
    },
    # from the 2026-09-14 DN screen with injection v4: the descending types
    # that fire on the same side as a small object (steer pools, one per
    # side), DNg100 for drive, the loom-selective types for the startle, and
    # FIRE when both steer pools fire at once (object straight ahead)
    "pool4": {
        "DNa02_L": [("cell_type", t, ("left",)) for t in STEER_POOL],
        "DNa02_R": [("cell_type", t, ("right",)) for t in STEER_POOL],
        "DNg100": [("cell_type", "DNg100", None)],
        "pIP10": [("min_of", ("DNa02_L", "DNa02_R"), None)],
        "DNp01": [("cell_type", t, None) for t in LOOM_POOL],
    },
}
READOUT = READOUT_MAPS["v0"]


def _pca2d(pos: np.ndarray) -> np.ndarray:
    """3D points -> 2D principal plane, normalized to [-1, 1], axes signed to
    follow FAFB x (lateral) and y (vertical)."""
    centered = pos - pos.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    uv = centered @ vt[:2].T
    if np.corrcoef(uv[:, 0], pos[:, 0])[0, 1] < 0:
        uv[:, 0] *= -1
    if np.corrcoef(uv[:, 1], pos[:, 1])[0, 1] < 0:
        uv[:, 1] *= -1
    return uv / np.abs(uv).max(axis=0)


class ConnectomeBrain:
    def __init__(self, retina, brain: ShiuBrain | None = None, seed: int = 0,
                 injection: str = "v0", readout: str = "v0"):
        self.brain = brain if brain is not None else ShiuBrain(seed=seed)
        self.injection = injection
        self.sites = INJECTION_MAPS[injection]
        self.readout = readout
        ann = annotations()

        # retinotopic injection map: per (population, eye)
        self._pops: list[tuple[str, float, slice, np.ndarray, str]] = []
        tree = cKDTree(retina.centers_xy)
        chunks = []
        start = 0
        for cell_type, mode, gain in self.sites:
            for side in ("left", "right"):
                rows = ann[(ann["cell_type"] == cell_type)
                           & (ann["side"] == side)]
                ids = rows["root_id"].to_numpy(np.int64)
                pos = rows[["pos_x", "pos_y", "pos_z"]].to_numpy(float)
                keep = np.array([int(r) in self.brain.index_of for r in ids])
                ids, pos = ids[keep], pos[keep]
                bidx = np.array([self.brain.index_of[int(r)] for r in ids],
                                dtype=np.int64)
                omm = tree.query(_pca2d(pos))[1]
                self._pops.append((mode, gain,
                                   slice(start, start + len(bidx)), omm, side))
                chunks.append(bidx)
                start += len(bidx)
        self._pr_all = np.concatenate(chunks)
        self._rates = np.zeros(len(self._pr_all), np.float32)
        self._prev_lum: dict[str, np.ndarray | None] = {"left": None,
                                                        "right": None}

        self._readout: dict[str, np.ndarray] = {}
        self._derived: dict[str, tuple[str, tuple[str, str]]] = {}
        for name, specs in READOUT_MAPS[readout].items():
            parts = []
            for kind, value, sides in specs:
                if kind == "min_of":
                    self._derived[name] = ("min_of", value)
                    continue
                rows = ann[ann["cell_type"].astype(str).str.startswith(value)] \
                    if kind == "cell_type_prefix" else ann[ann["cell_type"] == value]
                if sides is not None:
                    rows = rows[rows["side"].isin(sides)]
                idx, _ = self.brain.indices_of(rows["root_id"].to_numpy())
                parts.append(idx)
            if name in self._derived:
                continue
            idx = np.concatenate(parts) if parts else np.empty(0, np.int64)
            if len(idx) == 0:
                raise RuntimeError(f"no neurons found for DN readout {name}")
            self._readout[name] = idx
        assert set(self._readout) | set(self._derived) == set(dn.DN_NAMES)

        self.window_counts = np.zeros(self.brain.n, np.int32)
        self._register()

    def describe(self) -> str:
        pops = {}
        for (mode, _, sl, _, side), (ct, _, _) in zip(
                self._pops, [s for s in self.sites for _ in (0, 1)]):
            pops[f"{ct}/{side}"] = sl.stop - sl.start
        ro = {k: len(v) for k, v in self._readout.items()}
        ro.update({k: f"{v[0]}{v[1]}" for k, v in self._derived.items()})
        return (f"injection map {self.injection} {pops}, "
                f"readout map {self.readout} {ro}")

    def _register(self) -> None:
        self.brain.set_poisson({int(i): 0.0 for i in self._pr_all})

    def reset(self) -> None:
        self.brain.reset()
        self._register()
        self._prev_lum = {"left": None, "right": None}

    def step(self, drives: dict[str, np.ndarray], dt: float,
             signals: dict | None = None) -> dict[str, float]:
        lum = {side: drives[side][:, 0] for side in ("left", "right")}
        delta = {}
        for side in lum:
            prev = self._prev_lum[side]
            delta[side] = (lum[side] - prev) if prev is not None \
                else np.zeros_like(lum[side])
            self._prev_lum[side] = lum[side]
        for mode, gain, sl, omm, side in self._pops:
            if mode == "lum":
                sig = lum[side][omm]
            elif mode == "on":
                sig = np.clip(delta[side][omm], 0.0, None)
            elif mode == "off":
                sig = np.clip(-delta[side][omm], 0.0, None)
            else:  # onoff: any change
                sig = np.abs(delta[side][omm])
            self._rates[sl] = np.minimum(sig * gain, TRANSIENT_CAP)
        self.brain.update_rates(self._rates)

        counts = self.window_counts
        n_steps = round(dt / DT)
        run_counts = getattr(self.brain, "run_counts", None)
        if run_counts is not None:          # GPU backend: one graph replay
            counts[:] = run_counts(n_steps)
        else:
            counts[:] = 0
            for _ in range(n_steps):
                spk = self.brain.step()
                if spk.size:
                    counts[spk] += 1

        rates = {name: float(counts[idx].mean() / dt)
                 for name, idx in self._readout.items()}
        for name, (kind, (a, b)) in self._derived.items():
            rates[name] = min(rates[a], rates[b])
        return rates


class BrainView:
    """Front-view soma point cloud of the whole brain, spikes glowing through
    it — the dashboard's third panel."""

    def __init__(self, brain: ShiuBrain, width: int = 480, height: int = 360):
        self.w, self.h = width, height
        ann = annotations().set_index("root_id")
        ann = ann[~ann.index.duplicated()]
        pos = ann.reindex(brain.root_ids)[["pos_x", "pos_y"]].to_numpy(float)
        self._valid = np.flatnonzero(~np.isnan(pos).any(axis=1))
        p = pos[self._valid]
        lo, hi = p.min(axis=0), p.max(axis=0)
        scale = min((width - 20) / (hi[0] - lo[0]),
                    (height - 20) / (hi[1] - lo[1]))
        xy = (p - lo) * scale
        self._px = (xy[:, 0] + (width - (hi[0] - lo[0]) * scale) / 2
                    ).astype(np.int32)
        self._py = (xy[:, 1] + (height - (hi[1] - lo[1]) * scale) / 2
                    ).astype(np.int32)
        density = np.zeros((height, width), np.float32)
        np.add.at(density, (self._py, self._px), 1.0)
        base = np.log1p(density)
        base /= base.max()
        self._base = base
        self._trail = np.zeros((height, width), np.float32)

    def render(self, window_counts: np.ndarray) -> np.ndarray:
        act = window_counts[self._valid] > 0
        frame = np.zeros((self.h, self.w), np.float32)
        if act.any():
            np.maximum.at(frame, (self._py[act], self._px[act]), 1.0)
        self._trail = np.maximum(self._trail * 0.78, frame)
        img = np.zeros((self.h, self.w, 3), np.float32)
        img[..., 0] = self._base * 42
        img[..., 1] = self._base * 48
        img[..., 2] = self._base * 66
        img[..., 0] += self._trail * 235
        img[..., 1] += self._trail * 130
        img[..., 2] += self._trail * 40
        return np.clip(img, 0, 255).astype(np.uint8)
