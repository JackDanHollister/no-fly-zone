"""Sparse NumPy reimplementation of the Shiu et al. 2024 whole-brain LIF model
(Nature 634:210; github.com/philshiu/Drosophila_brain_model, MIT).

Faithful to their Brian2 code, not just the paper text:
- dv/dt = (v0 - v + g)/t_mbr, dg/dt = -g/tau, exact-linear update at dt=0.1 ms
- strict threshold v > v_th; on spike v -> v_rst AND g -> 0
- 2.2 ms refractory during which v and g are frozen (no decay) but arriving
  synaptic kicks still accumulate into g
- w = (±1 NT sign) x (synapse count) x 0.275 mV, homogeneous 1.8 ms delay
- activated neurons: no refractory period, Poisson events add 68.75 mV
  straight into v (effectively forced spiking at the Poisson rate)
- silencing zeroes OUTGOING weights only

Data: Completeness_783.csv fixes the neuron index order (138,639 neurons);
Connectivity_783.parquet holds 15.1M signed connections (no synapse-count
threshold). FlyWire v783; annotations from flyconnectome/flywire_annotations.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

DATA = Path(__file__).resolve().parents[2] / "data"
SHIU = DATA / "shiu_model"
ANNOTATIONS_TSV = (DATA / "flywire_annotations" / "supplemental_files"
                   / "Supplemental_file1_neuron_annotations.tsv")

DT = 1e-4          # s (Brian2 default clock)
V_REST = -52.0     # mV
V_RESET = -52.0
V_TH = -45.0
T_MBR = 20e-3      # s
TAU_SYN = 5e-3
T_RFC = 2.2e-3
T_DLY = 1.8e-3
W_SYN = 0.275      # mV per anatomical synapse
POISSON_KICK = W_SYN * 250  # 68.75 mV, forces a spike

REFR_STEPS = round(T_RFC / DT)   # 22
DELAY_STEPS = round(T_DLY / DT)  # 18

_annotations_cache: pd.DataFrame | None = None


def annotations() -> pd.DataFrame:
    global _annotations_cache
    if _annotations_cache is None:
        _annotations_cache = pd.read_csv(ANNOTATIONS_TSV, sep="\t",
                                         low_memory=False)
    return _annotations_cache


class ShiuBrain:
    def __init__(self, version: str = "783", seed: int = 0):
        comp = pd.read_csv(SHIU / f"Completeness_{version}.csv")
        self.root_ids = comp.iloc[:, 0].to_numpy(dtype=np.int64)
        self.n = len(self.root_ids)
        self.index_of = {int(r): i for i, r in enumerate(self.root_ids)}
        con = pd.read_parquet(
            SHIU / f"Connectivity_{version}.parquet",
            columns=["Presynaptic_Index", "Postsynaptic_Index",
                     "Excitatory x Connectivity"])
        w = (con["Excitatory x Connectivity"].to_numpy(np.float32) * W_SYN)
        self.W = sparse.csc_matrix(
            (w, (con["Postsynaptic_Index"].to_numpy(np.int64),
                 con["Presynaptic_Index"].to_numpy(np.int64))),
            shape=(self.n, self.n))
        self._decay_m = np.float32(np.exp(-DT / T_MBR))
        self._decay_s = np.float32(np.exp(-DT / TAU_SYN))
        self._b_coef = np.float32(TAU_SYN / (TAU_SYN - T_MBR))
        self._seed = seed
        self.reset()

    # -- neuron lookup -----------------------------------------------------

    def indices_of(self, root_ids) -> tuple[np.ndarray, list[int]]:
        """(present indices, missing root ids)."""
        present, missing = [], []
        for r in root_ids:
            i = self.index_of.get(int(r))
            (present.append(i) if i is not None else missing.append(int(r)))
        return np.array(present, dtype=np.int64), missing

    def indices_for_type(self, cell_type: str, side: str | None = None
                         ) -> np.ndarray:
        df = annotations()
        rows = df[df["cell_type"] == cell_type]
        if side is not None:
            rows = rows[rows["side"] == side]
        idx, _ = self.indices_of(rows["root_id"].to_numpy())
        return idx

    # -- state -------------------------------------------------------------

    def reset(self) -> None:
        self.v = np.full(self.n, V_REST, np.float32)
        self.g = np.zeros(self.n, np.float32)
        self.refr = np.zeros(self.n, np.int32)
        self._buffer = [np.empty(0, np.int64) for _ in range(DELAY_STEPS)]
        self._buf_pos = 0
        self._activated = np.zeros(self.n, bool)
        self._poi_idx = np.empty(0, np.int64)
        self._poi_p = np.empty(0, np.float32)
        self._rng = np.random.default_rng(self._seed)

    def set_poisson(self, rates: dict[int, float]) -> None:
        """Activate neurons (by index) with Poisson drive in Hz. Replaces the
        previous activation set. Activated neurons lose their refractory
        period, per the reference model."""
        self._poi_idx = np.fromiter(rates.keys(), np.int64, len(rates))
        self._poi_p = np.array([r * DT for r in rates.values()], np.float32)
        self._activated[:] = False
        self._activated[self._poi_idx] = True

    def silence(self, idx: np.ndarray) -> None:
        """Zero all OUTGOING weights of the given neurons (reference-model
        semantics: inputs still arrive, spikes have no effect)."""
        indptr, data = self.W.indptr, self.W.data
        for c in np.asarray(idx, dtype=np.int64):
            data[indptr[c]:indptr[c + 1]] = 0.0

    # -- dynamics ----------------------------------------------------------

    def _deliver(self, cols: np.ndarray) -> None:
        indptr, indices, data = self.W.indptr, self.W.indices, self.W.data
        rows = np.concatenate([indices[indptr[c]:indptr[c + 1]] for c in cols])
        vals = np.concatenate([data[indptr[c]:indptr[c + 1]] for c in cols])
        if rows.size:
            self.g += np.bincount(rows, weights=vals,
                                  minlength=self.n).astype(np.float32)

    def step(self) -> np.ndarray:
        """Advance one 0.1 ms step; returns indices of neurons that spiked."""
        frozen = self.refr > 0
        live = ~frozen
        b = self._b_coef * self.g[live]
        self.v[live] = (V_REST + (self.v[live] - V_REST - b) * self._decay_m
                        + b * self._decay_s)
        self.g[live] *= self._decay_s

        spk = np.flatnonzero(self.v > V_TH)
        if spk.size:
            self.v[spk] = V_RESET
            self.g[spk] = 0.0
            self.refr[spk] = np.where(self._activated[spk], 0, REFR_STEPS)

        arriving = self._buffer[self._buf_pos]
        if arriving.size:
            self._deliver(arriving)
        self._buffer[self._buf_pos] = spk
        self._buf_pos = (self._buf_pos + 1) % DELAY_STEPS

        if self._poi_idx.size:
            hit = self._rng.random(self._poi_idx.size) < self._poi_p
            self.v[self._poi_idx[hit]] += POISSON_KICK

        self.refr[frozen] -= 1
        return spk

    def run(self, duration_s: float, progress_every: int = 0
            ) -> np.ndarray:
        """Run for a duration; returns per-neuron spike counts."""
        counts = np.zeros(self.n, np.int64)
        steps = round(duration_s / DT)
        for k in range(steps):
            spk = self.step()
            if spk.size:
                counts[spk] += 1
            if progress_every and (k + 1) % progress_every == 0:
                print(f"  step {k + 1}/{steps}, total spikes {counts.sum()}",
                      flush=True)
        return counts
