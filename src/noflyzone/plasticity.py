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

Added 2026-09-19, after the mushroom body was measured to be out of the
visual loop: the same three-factor rule on the synapses the game does
reach. edge_set="dn_in" is every measured edge onto a descending neuron
(332,213), "vis_dn" the subset whose presynaptic cell is a visual projection
or visual centrifugal neuron (24,110); each has a "<set>_shuffled" control of
equal size drawn from edges that touch neither end. For these sets the
eligibility is Hebbian (presynaptic trace times postsynaptic trace, so an
input is credited only when its descending neuron fired) and the change is
relative to the measured weight (scale="relative"), which is
reward-modulated Hebbian learning, the spiking form of a policy gradient
(Fremaux & Gerstner 2016). The dopamine term is still the fly's own PAM
minus PPL1 population rate, driven by the game's teaching pulses; those
neurons do not contact the descending neurons' inputs in the wiring, so the
signal is applied as a global modulator and labelled ENGINEERED wherever it
is shown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .brain_lif import SHIU, annotations

RULES = ("on", "lr0")
BASE_SETS = ("kc_mbon", "dn_in", "vis_dn")
EDGE_SETS = ("kc_mbon", "shuffled", "dn_in", "vis_dn",
             "kc_mbon_shuffled", "dn_in_shuffled", "vis_dn_shuffled")
VISUAL_CLASSES = ("visual_projection", "visual_centrifugal")


class MBPlasticity:
    def __init__(self, brain, dt: float, eta: float = 2e-3, tau_trace_s: float = 0.5,
                 bounds: tuple[float, float] = (0.1, 2.0), teaching_hz: float = 100.0,
                 pulse_ticks: int = 6, da_tau_s: float = 20.0, rule: str = "on",
                 edge_set: str = "kc_mbon", seed: int = 0,
                 eligibility: str | None = None, scale: str | None = None,
                 elig_cap: float | None = None, latch: bool = False):
        assert rule in RULES and edge_set in EDGE_SETS
        if edge_set == "shuffled":
            edge_set = "kc_mbon_shuffled"
        base = edge_set.replace("_shuffled", "")
        shuffled = edge_set.endswith("_shuffled")
        # defaults per set: the mushroom-body rule as run on 2026-09-14,
        # the descending-input rule as Hebbian and relative
        if eligibility is None:
            eligibility = "presynaptic" if base == "kc_mbon" else "hebbian"
        if scale is None:
            scale = "absolute" if base == "kc_mbon" else "relative"
        assert eligibility in ("presynaptic", "hebbian") and scale in ("absolute", "relative")
        self.brain, self.dt, self.eta, self.rule, self.edge_set = brain, dt, eta, rule, edge_set
        self.base_set, self.shuffled = base, shuffled
        self.eligibility, self.scale = eligibility, scale
        # the Hebbian eligibility is heavy-tailed (measured 2026-09-19 on
        # vis_dn: mean 6, 90th percentile 18, maximum above 10,000), so it is
        # bounded for the relative rule; a standard saturating eligibility
        if elig_cap is None:
            elig_cap = np.inf if scale == "absolute" else 50.0
        self.elig_cap = float(elig_cap)
        # latch (2026-09-20): a reinforcer's pulse lasts several ticks, during
        # which the brain is responding to the event's own consequence (the
        # screen flash of a hit, the explosion of a kill); with latch the
        # dopamine acts on the eligibility as it stood the tick before the
        # event, so what is credited is what preceded the outcome, not the
        # response to it
        self.latch = bool(latch)
        self._elig_prev = None
        self._elig_latched = None
        self.lo, self.hi = bounds
        self.teaching_hz, self.pulse_ticks = teaching_hz, pulse_ticks
        self._trace_decay = float(np.exp(-dt / tau_trace_s))
        self._da_decay = float(np.exp(-dt / da_tau_s))
        ann = annotations().drop_duplicates("root_id").set_index("root_id").reindex(brain.root_ids)
        ct = ann["cell_type"].fillna("").astype(str).to_numpy()
        starts = lambda s: np.flatnonzero(np.char.startswith(ct.astype(str), s))
        self.kc, self.mbon, self.pam, self.ppl1 = starts("KC"), starts("MBON"), starts("PAM"), starts("PPL1")
        sc = ann["super_class"].fillna("").astype(str).to_numpy()
        con = pd.read_parquet(SHIU / "Connectivity_783.parquet",
                              columns=["Presynaptic_Index", "Postsynaptic_Index"])
        pairs = pd.DataFrame({"pre": con["Presynaptic_Index"].to_numpy(np.int64),
                              "post": con["Postsynaptic_Index"].to_numpy(np.int64)}).drop_duplicates()
        pre = pairs["pre"].to_numpy(); post = pairs["post"].to_numpy()
        if base == "kc_mbon":
            pre_ok, post_ok = np.isin(pre, self.kc), np.isin(post, self.mbon)
        elif base == "dn_in":
            pre_ok, post_ok = np.ones(pre.size, bool), sc[post] == "descending"
        else:  # vis_dn
            pre_ok, post_ok = np.isin(sc[pre], VISUAL_CLASSES), sc[post] == "descending"
        m = pre_ok & post_ok
        if shuffled:
            # equal number of edges touching neither the set's sources nor its
            # targets; when the sources are every cell (dn_in) only the targets
            # are excluded, so the control is "the rule on synapses that do not
            # end on a descending neuron"
            rng = np.random.default_rng(seed)
            excl = post_ok if pre_ok.all() else (pre_ok | post_ok)
            pool = np.flatnonzero(~excl)
            n_set = int(m.sum())
            m = np.zeros_like(m); m[rng.choice(pool, size=n_set, replace=False)] = True
        pairs = pairs[m]
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
        # edges whose measured weight is zero (aminergic, under aminergic="zero")
        # sit in the set but cannot move: the bounds are 0.1 x 0 to 2 x 0
        self.n_live = int(np.count_nonzero(self.w0))

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
        if np.isfinite(self.elig_cap):
            elig = np.minimum(elig, self.elig_cap)
        if self.latch:
            event = bool(np.any(kills)) or bool(np.any(hits))
            if event and self._elig_prev is not None:
                self._elig_latched = self._elig_prev.copy()
            if self._elig_latched is not None and (self.pulse["pam"] > 0 or self.pulse["ppl1"] > 0 or event):
                elig_used = self._elig_latched
            else:
                elig_used = elig
                self._elig_latched = None
            self._elig_prev = elig
        else:
            elig_used = elig
        dw = self.eta * elig_used * da_c
        if self.scale == "relative":
            dw = dw * np.abs(self.w0)
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
                "elig_mean": float(elig_used.mean()), "drift": drift,
                "frac_changed": float(np.mean(self.mag != np.abs(self.w0)))}

    def load_weights(self, path) -> None:
        """Install weights saved by train_plasticity.py (same edge set) and
        treat them as the starting point; with rule='lr0' this is a frozen
        evaluation of a learned brain."""
        z = np.load(path)
        if z["pos"].size != self.pos.size or not np.array_equal(z["pos"], self.pos):
            raise ValueError("saved weights are for a different edge set")
        self.w = z["w"].astype(np.float32); self.mag = np.abs(self.w)
        self.brain.set_edge_weights(self.pos, self.w)

    def reset_traces(self) -> None:
        self.trace_pre[:] = 0; self.trace_post[:] = 0
        self.pulse = {"pam": 0, "ppl1": 0}
        self._elig_prev = None; self._elig_latched = None
