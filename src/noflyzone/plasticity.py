"""Dopamine-gated plasticity on the Kenyon-cell to MBON synapses (M7).

The mushroom body is where flies learn: Kenyon cells (KCs) carry a sparse
sensory code, MBONs read it out toward the descending neurons, and the
dopamine neurons PAM (reward) and PPL1 (punishment) gate changes of the KC to
MBON synapses. Here the game supplies the reinforcers: a kill pulses the PAM
population and a hit pulses the PPL1 population (Poisson teaching input for a
few ticks), the real dopamine neurons' spiking is the teaching signal, and a
three-factor rule changes the 62,261 measured KC to MBON edges in place in
the shared connectivity, between graph replays.

Rule (a simplification of the fly's, stated as such): eligibility is a
presynaptic Kenyon-cell trace, as in the fly, where dopamine acts on the KC
axon terminals and the MBON need not fire (an option keeps a postsynaptic
factor); the dopamine term is PAM rate minus PPL1 rate, centred on its
running mean so a hit-heavy game cannot only ever depress; dw = eta *
eligibility * dopamine; weights are bounded to [0.1, 2.0] times their
measured value. Measured 2026-09-14: in this model a full-field flash
activates about 5 of 5,177 Kenyon cells and no MBON, so the mushroom body
is only weakly in the visual loop; the experiment measures whether that is
enough. In the fly,
dopamine paired with KC activity mostly depresses; the sign convention here
is "reward strengthens what was active, punishment weakens it", which is the
behaviourally useful reading and is recorded as the rule name.

Controls, pre-registered: rule="lr0" computes everything and applies nothing;
edge_set="shuffled" applies the same rule to an equal number of random
non-mushroom-body edges. Every fly in a batch shares the weights, so a batch
is one fly living several lives at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .brain_lif import SHIU, annotations

RULES = ("on", "lr0")
EDGE_SETS = ("kc_mbon", "shuffled")


class MBPlasticity:
    def __init__(self, brain, dt: float, eta: float = 2e-3, tau_trace_s: float = 0.5,
                 bounds: tuple[float, float] = (0.1, 2.0), teaching_hz: float = 100.0,
                 pulse_ticks: int = 6, da_tau_s: float = 20.0, rule: str = "on",
                 edge_set: str = "kc_mbon", seed: int = 0,
                 eligibility: str = "presynaptic"):
        assert rule in RULES and edge_set in EDGE_SETS
        assert eligibility in ("presynaptic", "hebbian")
        self.brain, self.dt, self.eta, self.rule, self.edge_set = brain, dt, eta, rule, edge_set
        self.eligibility = eligibility
        self.lo, self.hi = bounds
        self.teaching_hz, self.pulse_ticks = teaching_hz, pulse_ticks
        self._trace_decay = float(np.exp(-dt / tau_trace_s))
        self._da_decay = float(np.exp(-dt / da_tau_s))
        ann = annotations().drop_duplicates("root_id").set_index("root_id").reindex(brain.root_ids)
        ct = ann["cell_type"].fillna("").astype(str).to_numpy()
        starts = lambda s: np.flatnonzero(np.char.startswith(ct.astype(str), s))
        self.kc, self.mbon, self.pam, self.ppl1 = starts("KC"), starts("MBON"), starts("PAM"), starts("PPL1")
        con = pd.read_parquet(SHIU / "Connectivity_783.parquet",
                              columns=["Presynaptic_Index", "Postsynaptic_Index"])
        pre = con["Presynaptic_Index"].to_numpy(np.int64); post = con["Postsynaptic_Index"].to_numpy(np.int64)
        m = np.isin(pre, self.kc) & np.isin(post, self.mbon)
        if edge_set == "shuffled":
            rng = np.random.default_rng(seed)
            pool = np.flatnonzero(~m & ~np.isin(pre, self.kc) & ~np.isin(post, self.mbon))
            m = np.zeros_like(m); m[rng.choice(pool, size=int(np.isin(pre, self.kc).sum() and (np.isin(pre, self.kc) & np.isin(post, self.mbon)).sum()), replace=False)] = True
        pairs = pd.DataFrame({"pre": pre[m], "post": post[m]}).drop_duplicates()
        self.pre, self.post = pairs["pre"].to_numpy(), pairs["post"].to_numpy()
        self.pos = brain.csc_positions(self.pre, self.post)
        ok = self.pos >= 0
        self.pre, self.post, self.pos = self.pre[ok], self.post[ok], self.pos[ok]
        self.w0 = brain.W.data[self.pos].astype(np.float32).copy()
        self.w = self.w0.copy()
        self.sign = np.sign(self.w0); self.mag = np.abs(self.w0)
        n = brain.n
        self.trace_pre = np.zeros(n, np.float32)
        self.trace_post = np.zeros(n, np.float32)
        self.da_mean = 0.0
        self.pulse = {"pam": 0, "ppl1": 0}
        self.n_edges = int(self.pos.size)

    # teaching input: which extra rates to inject next tick (shared by all flies)
    def teaching_rates(self) -> dict[str, float]:
        return {k: (self.teaching_hz if v > 0 else 0.0) for k, v in self.pulse.items()}

    def step(self, counts: np.ndarray, kills: np.ndarray, hits: np.ndarray) -> dict:
        """counts: (B, n) spike counts of the tick; kills/hits: (B,) events.
        Returns diagnostics. Applies the weight change unless rule == 'lr0'."""
        counts = counts.reshape(-1, self.brain.n)
        mean = counts.mean(axis=0).astype(np.float32)          # shared weights: pooled experience
        hz = mean / self.dt
        pam_hz, ppl1_hz = float(hz[self.pam].mean()), float(hz[self.ppl1].mean())
        da = pam_hz - ppl1_hz
        self.da_mean = self._da_decay * self.da_mean + (1 - self._da_decay) * da
        da_c = da - self.da_mean
        self.trace_pre *= self._trace_decay; self.trace_pre += mean
        self.trace_post *= self._trace_decay; self.trace_post += mean
        elig = self.trace_pre[self.pre]
        if self.eligibility == "hebbian":
            elig = elig * self.trace_post[self.post]
        dw = self.eta * elig * da_c
        if self.rule == "on" and da_c != 0.0:
            new_mag = np.clip(self.mag + dw, self.lo * np.abs(self.w0), self.hi * np.abs(self.w0))
            changed = new_mag != self.mag
            if changed.any():
                self.mag = new_mag
                self.w = (self.sign * self.mag).astype(np.float32)
                self.brain.set_edge_weights(self.pos[changed], self.w[changed])
        # schedule teaching pulses from the game's reinforcers
        for k, ev in (("pam", kills), ("ppl1", hits)):
            self.pulse[k] = max(self.pulse[k] - 1, 0)
            if np.any(ev):
                self.pulse[k] = self.pulse_ticks
        drift = float(np.mean(np.abs(self.mag - np.abs(self.w0)) / np.maximum(np.abs(self.w0), 1e-6)))
        return {"pam_hz": pam_hz, "ppl1_hz": ppl1_hz, "da": da, "da_centred": da_c,
                "elig_mean": float(elig.mean()), "drift": drift,
                "frac_changed": float(np.mean(self.mag != np.abs(self.w0)))}

    def reset_traces(self) -> None:
        self.trace_pre[:] = 0; self.trace_post[:] = 0
        self.pulse = {"pam": 0, "ppl1": 0}
