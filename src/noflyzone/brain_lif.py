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


ABLATIONS = ("none", "rewired", "silenced")
SENSORY_MODES = ("free", "external_only")
SENSORY_CLASSES = ("sensory", "sensory_ascending")
SIGN_MODES = ("predicted", "known_first")
AMINERGIC_MODES = ("excitatory", "zero")
NT_INHIBITORY = ("gaba", "glutamate", "histamine")
NT_EXCITATORY = ("acetylcholine",)
NT_AMINERGIC = ("dopamine", "octopamine", "serotonin")


def transmitter_per_cell(root_ids: np.ndarray, prefer_known: bool) -> np.ndarray:
    """One transmitter label per cell: the literature-known one (first token
    of `known_nt`) when prefer_known and it is one of the seven classes,
    else the FlyWire prediction `top_nt`."""
    ann = annotations().drop_duplicates("root_id").set_index("root_id").reindex(root_ids)
    top = ann["top_nt"].astype(str).str.lower().to_numpy()
    if not prefer_known:
        return top
    known = (ann["known_nt"].astype(str).str.lower()
             .str.split(r"[;,]").str[0].str.strip().to_numpy())
    ok = np.isin(known, NT_INHIBITORY + NT_EXCITATORY + NT_AMINERGIC)
    return np.where(ok, known, top)


class ShiuBrain:
    def __init__(self, version: str = "783", seed: int = 0,
                 ablate: str = "none", ablate_seed: int = 0,
                 sensory_input: str = "free", signs: str = "predicted",
                 aminergic: str = "excitatory"):
        """ablate: "none" is the measured wiring; "rewired" permutes every
        edge's postsynaptic target with a seeded RNG (each presynaptic cell
        keeps its out-degree, synapse counts and transmitter sign, the
        topology is destroyed); "silenced" zeroes all weights. Controls for
        any claim that the wiring matters.

        sensory_input: "free" is the reference model, where primary sensory
        neurons (photoreceptors, olfactory and other receptor neurons; the
        sensory super-classes) are ordinary LIF units that central feedback
        can drive. "external_only" zeroes every synapse ONTO those neurons,
        so they fire only when their sense organ is driven (Poisson input).
        Motivation, 2026-09-14: a strong visual event tips the reference model
        into a self-sustaining olfactory state (receptor neurons, antennal
        lobe local neurons, Kenyon cells, APL and DPM at 400+ Hz) that
        persists indefinitely in darkness.

        signs: "predicted" keeps the file's signs (FlyWire top_nt: GABA and
        glutamate -1, everything else +1). "known_first" re-signs every
        presynaptic cell whose literature transmitter is known: GABA,
        glutamate, histamine -1; acetylcholine +1; amines per `aminergic`.
        aminergic: "excitatory" is the reference (dopamine, octopamine,
        serotonin as fast +1 synapses); "zero" removes their fast weight —
        they are slow modulators, and as +1 synapses the ~300 PAM dopamine
        cells close a Kenyon-cell -> PAM -> Kenyon-cell loop that sustains
        the persistent state above. Every option is recorded in `meta`."""
        if ablate not in ABLATIONS:
            raise ValueError(f"ablate must be one of {ABLATIONS}")
        if sensory_input not in SENSORY_MODES:
            raise ValueError(f"sensory_input must be one of {SENSORY_MODES}")
        if signs not in SIGN_MODES or aminergic not in AMINERGIC_MODES:
            raise ValueError("bad signs/aminergic option")
        self.ablate = ablate
        self.ablate_seed = ablate_seed
        self.sensory_input = sensory_input
        self.signs = signs
        self.aminergic = aminergic
        self.meta = {"version": version, "ablate": ablate, "ablate_seed": ablate_seed,
                     "sensory_input": sensory_input, "signs": signs,
                     "aminergic": aminergic}
        comp = pd.read_csv(SHIU / f"Completeness_{version}.csv")
        self.root_ids = comp.iloc[:, 0].to_numpy(dtype=np.int64)
        self.n = len(self.root_ids)
        self.index_of = {int(r): i for i, r in enumerate(self.root_ids)}
        con = pd.read_parquet(
            SHIU / f"Connectivity_{version}.parquet",
            columns=["Presynaptic_Index", "Postsynaptic_Index",
                     "Excitatory x Connectivity"])
        w = (con["Excitatory x Connectivity"].to_numpy(np.float32) * W_SYN)
        post = con["Postsynaptic_Index"].to_numpy(np.int64)
        pre = con["Presynaptic_Index"].to_numpy(np.int64)
        if signs == "known_first" or aminergic == "zero":
            # literature transmitter first: the FlyWire prediction calls every
            # Kenyon cell dopaminergic, and they are cholinergic
            nt = transmitter_per_cell(self.root_ids, prefer_known=True)
            counts = np.abs(w)
            if signs == "known_first":
                sign = np.where(np.isin(nt, NT_INHIBITORY), -1.0,
                                np.where(np.isin(nt, NT_EXCITATORY), 1.0, np.nan))
                cell_sign = sign[pre]
                w = np.where(np.isnan(cell_sign), w, counts * cell_sign).astype(np.float32)
                self.meta["resigned_edges"] = int((~np.isnan(cell_sign)).sum())
            if aminergic == "zero":
                am = np.isin(nt, NT_AMINERGIC)[pre]
                w = np.where(am, 0.0, w).astype(np.float32)
                self.meta["aminergic_edges_zeroed"] = int(am.sum())
        if ablate == "rewired":
            post = np.random.default_rng(ablate_seed).permutation(post)
        elif ablate == "silenced":
            w = np.zeros_like(w)
        if sensory_input == "external_only":
            ann = annotations().drop_duplicates("root_id").set_index("root_id")
            sclass = ann["super_class"].reindex(self.root_ids).astype(str).to_numpy()
            sensory = np.isin(sclass, SENSORY_CLASSES)
            self.n_sensory_deafferented = int(sensory.sum())
            w = np.where(sensory[post], 0.0, w).astype(np.float32)
        self.W = sparse.csc_matrix((w, (post, pre)), shape=(self.n, self.n))
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

    def update_rates(self, rates_hz: np.ndarray) -> None:
        """Fast per-frame rate update for the CURRENT activation set, in the
        order that set_poisson registered it."""
        self._poi_p = (np.asarray(rates_hz, np.float32) * DT)

    def csc_positions(self, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        """Positions in W.data of the edges (pre -> post), or -1 where no
        such edge exists. W is CSC with columns = presynaptic cells."""
        pre = np.asarray(pre, np.int64); post = np.asarray(post, np.int64)
        indptr, indices = self.W.indptr, self.W.indices
        out = np.full(pre.size, -1, np.int64)
        for k, (c, r) in enumerate(zip(pre, post)):
            lo, hi = indptr[c], indptr[c + 1]
            rows = indices[lo:hi]
            j = np.searchsorted(rows, r) if self.W.has_sorted_indices else None
            if j is not None and j < rows.size and rows[j] == r:
                out[k] = lo + j
            elif j is None:
                hit = np.flatnonzero(rows == r)
                if hit.size:
                    out[k] = lo + hit[0]
        return out

    def silence(self, idx: np.ndarray) -> None:
        """Zero all OUTGOING weights of the given neurons (reference-model
        semantics: inputs still arrive, spikes have no effect)."""
        indptr, data = self.W.indptr, self.W.data
        for c in np.asarray(idx, dtype=np.int64):
            data[indptr[c]:indptr[c + 1]] = 0.0

    # -- dynamics ----------------------------------------------------------

    def _deliver(self, cols: np.ndarray) -> None:
        # sparse column-gather via C-level matvec: cost scales with the
        # spiking columns' edges, not with total nnz
        vec = sparse.csc_matrix(
            (np.ones(cols.size, np.float32), cols,
             np.array([0, cols.size])), shape=(self.n, 1))
        out = (self.W @ vec).tocoo()
        self.g[out.row] += out.data.astype(np.float32)

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
