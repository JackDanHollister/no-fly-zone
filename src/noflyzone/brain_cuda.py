"""CUDA backend for the Shiu whole-brain LIF (brain_lif.ShiuBrain semantics,
state on the GPU, Triton kernels, one CUDA graph per game tick).

Per 0.1 ms step, four launches on the current stream:
  K0  zero the ring-buffer slot this step will write into
  K1  per-neuron: exact-linear decay (live only), strict threshold, reset of
      v and g, refractory bookkeeping, Poisson kick, spike record (per-neuron
      count + append to the delay ring via an atomic counter)
  K2  delivery: for every spike recorded DELAY_STEPS ago, atomic-add its
      outgoing CSC column into g (cost scales with spiking columns' edges,
      as on the CPU)
  K3  step counter += 1

Everything is static-shape, so a whole tick (667 steps) is captured once as a
CUDA graph and replayed; only the per-neuron spike counts come back to the
host. Each ring row holds n entries, so it cannot overflow.

Batching: `batch=B` runs B independent flies over ONE connectivity
allocation. State is (B, n), the delay ring (R, B, n), Poisson rates (B, n)
so each fly sees its own screen; the kernels take the fly index as the
second grid axis and the RNG offset folds it in so fly 0 of a batch is
bit-identical to an unbatched run. `reset_rows()` restarts individual flies
(episode ends) without touching the others. `set_edge_weights()` writes into
the shared weights in place, between graph replays, for plasticity. Randomness is Philox (seed, step * n + neuron) — stateless, so reset()
restores the same stream as the reference, though not the same numbers as
NumPy's generator.

Ordering matches brain_lif.ShiuBrain.step: K1 does decay -> threshold/reset
-> Poisson kick (v only) -> refractory decrement for neurons frozen at the
start of the step; K2 then delivers into the already-decayed, just-zeroed g.
The CPU kicks after delivering, but the kick touches only v and delivery only
g, so the two orders are equivalent. The per-neuron kernel is launched with
FP fusion off so the decay arithmetic rounds like NumPy's separate ops; the
remaining ~1e-6 mV drift is float32 atomic summation order in delivery.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import triton
import triton.language as tl

from .brain_lif import (DELAY_STEPS, DT, POISSON_KICK, REFR_STEPS, ShiuBrain,
                        V_RESET, V_REST, V_TH)

RING = DELAY_STEPS + 1            # read slot (k - D) and write slot k never collide
NEURON_BLOCK = 1024
EDGE_BLOCK = 128
DELIVER_PROGRAMS = 4096           # K2 programs; each strides over the spike list
MAX_GRAPH_STEPS = 2500            # larger runs are chunked into several replays


@triton.jit
def _k_zero_slot(ring_cnt_ptr, ctr_ptr, R: tl.constexpr, B: tl.constexpr):
    fly = tl.program_id(0)
    c = tl.load(ctr_ptr)
    tl.store(ring_cnt_ptr + (c % R) * B + fly, 0)


@triton.jit
def _k_neuron(v_ptr, g_ptr, refr_ptr, p_ptr, act_ptr, counts_ptr,
              ring_list_ptr, ring_cnt_ptr, ctr_ptr,
              n, seed, decay_m, decay_s, b_coef,
              V_REST_C: tl.constexpr, V_RESET_C: tl.constexpr,
              V_TH_C: tl.constexpr, REFR: tl.constexpr, KICK: tl.constexpr,
              R: tl.constexpr, B: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    fly = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    row = fly.to(tl.int64) * n
    c = tl.load(ctr_ptr)
    slot = c % R

    v = tl.load(v_ptr + row + offs, mask=mask, other=V_REST_C)
    g = tl.load(g_ptr + row + offs, mask=mask, other=0.0)
    refr = tl.load(refr_ptr + row + offs, mask=mask, other=0)
    act = tl.load(act_ptr + row + offs, mask=mask, other=0)
    p = tl.load(p_ptr + row + offs, mask=mask, other=0.0)

    frozen = refr > 0
    b = b_coef * g
    v_new = V_REST_C + (v - V_REST_C - b) * decay_m + b * decay_s
    g_new = g * decay_s
    v = tl.where(frozen, v, v_new)
    g = tl.where(frozen, g, g_new)

    spk = (v > V_TH_C) & mask
    v = tl.where(spk, V_RESET_C, v)
    g = tl.where(spk, 0.0, g)
    refr = tl.where(spk, tl.where(act != 0, 0, REFR), refr)

    cnt = tl.load(counts_ptr + row + offs, mask=mask, other=0)
    tl.store(counts_ptr + row + offs, cnt + spk.to(tl.int32), mask=mask)
    pos = tl.atomic_add(ring_cnt_ptr + slot * B + fly + tl.zeros([BLOCK], tl.int32),
                        1, mask=spk)
    tl.store(ring_list_ptr + (slot * B + fly).to(tl.int64) * n + pos, offs,
             mask=spk)  # pos < n always

    # fly 0 of any batch draws the same stream as an unbatched run
    u = tl.rand(seed, fly.to(tl.int64) * (1 << 50) + c.to(tl.int64) * n
                + offs.to(tl.int64))
    hit = (p > 0.0) & (u < p)
    v = tl.where(hit, v + KICK, v)

    refr = tl.where(frozen, refr - 1, refr)
    tl.store(v_ptr + row + offs, v, mask=mask)
    tl.store(g_ptr + row + offs, g, mask=mask)
    tl.store(refr_ptr + row + offs, refr, mask=mask)


@triton.jit
def _k_deliver(g_ptr, indptr_ptr, indices_ptr, data_ptr,
               ring_list_ptr, ring_cnt_ptr, ctr_ptr, n,
               R: tl.constexpr, D: tl.constexpr, B: tl.constexpr,
               PROGRAMS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    fly = tl.program_id(1)
    c = tl.load(ctr_ptr)
    slot = (c + (R - D)) % R
    cnt = tl.load(ring_cnt_ptr + slot * B + fly)
    list_base = (slot * B + fly).to(tl.int64) * n
    g_base = fly.to(tl.int64) * n
    for e in range(pid, cnt, PROGRAMS):      # grid-stride over the spike list
        col = tl.load(ring_list_ptr + list_base + e)
        start = tl.load(indptr_ptr + col)
        end = tl.load(indptr_ptr + col + 1)
        for off in range(start, end, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            m = idx < end
            rows = tl.load(indices_ptr + idx, mask=m, other=0)
            vals = tl.load(data_ptr + idx, mask=m, other=0.0)
            tl.atomic_add(g_ptr + g_base + rows, vals, mask=m)


class ShiuBrainCUDA(ShiuBrain):
    """Drop-in for ShiuBrain. Fast path: run_counts(n_steps) -> per-neuron
    spike counts (numpy). step() and run() keep the reference API."""

    def __init__(self, version: str = "783", seed: int = 0,
                 device: str = "cuda", use_graph: bool = True,
                 ablate: str = "none", ablate_seed: int = 0,
                 sensory_input: str = "free", signs: str = "predicted",
                 aminergic: str = "excitatory", weight_scale: float = 1.0,
                 batch: int = 1):
        if not torch.cuda.is_available():
            raise RuntimeError("ShiuBrainCUDA: no CUDA device visible to torch")
        if batch < 1:
            raise ValueError("batch must be >= 1")
        self.B = batch
        self._device = torch.device(device)
        self._use_graph = use_graph
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._ready = False
        super().__init__(version=version, seed=seed, ablate=ablate,
                         ablate_seed=ablate_seed, sensory_input=sensory_input,
                         signs=signs, aminergic=aminergic,
                         weight_scale=weight_scale)  # loads W, reset()
        dev = self._device
        W = self.W
        self._indptr = torch.from_numpy(W.indptr.astype(np.int32)).to(dev)
        self._indices = torch.from_numpy(W.indices.astype(np.int32)).to(dev)
        self._data = torch.from_numpy(W.data.astype(np.float32)).to(dev)
        n, B = self.n, self.B
        self._v = torch.empty((B, n), dtype=torch.float32, device=dev)
        self._g = torch.empty((B, n), dtype=torch.float32, device=dev)
        self._refr = torch.empty((B, n), dtype=torch.int32, device=dev)
        self._p = torch.zeros((B, n), dtype=torch.float32, device=dev)
        self._act = torch.zeros((B, n), dtype=torch.int32, device=dev)
        self._counts = torch.zeros((B, n), dtype=torch.int32, device=dev)
        self._ring_list = torch.zeros((RING, B, n), dtype=torch.int32, device=dev)
        self._ring_cnt = torch.zeros((RING, B), dtype=torch.int32, device=dev)
        self._ctr = torch.zeros(1, dtype=torch.int64, device=dev)  # no wrap
        self._poi_idx_dev = torch.empty(0, dtype=torch.int64, device=dev)
        self._ready = True
        self.reset()

    @property
    def backend(self) -> str:
        return f"cuda:{torch.cuda.get_device_name(self._device)}" + \
            (" graph" if self._use_graph else " eager") + \
            (f" x{self.B}" if self.B > 1 else "")

    # -- state -------------------------------------------------------------

    def reset(self) -> None:
        """Full reset of every fly, the Poisson registration and the step
        counter (the reference semantics)."""
        super().reset()
        if not self._ready:
            return
        self._v.fill_(V_REST)
        self._g.zero_()
        self._refr.zero_()
        self._p.zero_()
        self._act.zero_()
        self._counts.zero_()
        self._ring_cnt.zero_()
        self._ctr.zero_()
        self._poi_idx_dev = torch.empty(0, dtype=torch.int64,
                                        device=self._device)

    def reset_rows(self, rows) -> None:
        """Restart the given flies (membrane, synapses, refractory, delay
        ring, counts) and leave the others and the Poisson registration
        untouched; for per-fly episode ends."""
        rows = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self._device)
        self._v[rows] = V_REST
        self._g[rows] = 0.0
        self._refr[rows] = 0
        self._counts[rows] = 0
        self._ring_cnt[:, rows] = 0

    def set_poisson(self, rates: dict[int, float]) -> None:
        """Same activation set and rates for every fly."""
        super().set_poisson(rates)
        self._p.zero_()
        self._act.zero_()
        self._poi_idx_dev = torch.from_numpy(self._poi_idx).to(self._device)
        if self._poi_idx.size:
            self._p[:, self._poi_idx_dev] = torch.from_numpy(self._poi_p).to(
                self._device)
            self._act[:, self._poi_idx_dev] = 1

    def update_rates(self, rates_hz: np.ndarray) -> None:
        """Per-frame rates for the current activation set: shape (n_poi,)
        applies to every fly, (B, n_poi) gives each fly its own."""
        rates = np.asarray(rates_hz, np.float32)
        super().update_rates(rates[0] if rates.ndim == 2 else rates)
        if self._poi_idx.size:
            p = torch.from_numpy(np.ascontiguousarray(rates * DT)).to(self._device)
            self._p[:, self._poi_idx_dev] = p  # broadcasts (n,) or (B, n)

    def set_edge_weights(self, positions, values) -> None:
        """Write weights (mV, signed) into the shared connectivity in place,
        by CSC data position (see ShiuBrain.csc_positions). Safe between
        graph replays; affects every fly."""
        pos = torch.as_tensor(np.asarray(positions, dtype=np.int64), device=self._device)
        val = torch.as_tensor(np.asarray(values, dtype=np.float32), device=self._device)
        self._data[pos] = val
        self.W.data[np.asarray(positions, dtype=np.int64)] = np.asarray(values, np.float32)

    def silence(self, idx: np.ndarray) -> None:
        super().silence(idx)
        self._data.copy_(torch.from_numpy(self.W.data.astype(np.float32)))

    # -- dynamics ----------------------------------------------------------

    def _launch_step(self) -> None:
        n, B = self.n, self.B
        _k_zero_slot[(B,)](self._ring_cnt, self._ctr, R=RING, B=B)
        _k_neuron[(triton.cdiv(n, NEURON_BLOCK), B)](
            self._v, self._g, self._refr, self._p, self._act, self._counts,
            self._ring_list, self._ring_cnt, self._ctr,
            n, int(self._seed), float(self._decay_m), float(self._decay_s),
            float(self._b_coef),
            V_REST_C=V_REST, V_RESET_C=V_RESET, V_TH_C=V_TH,
            REFR=REFR_STEPS, KICK=POISSON_KICK, R=RING, B=B, BLOCK=NEURON_BLOCK,
            enable_fp_fusion=False)
        _k_deliver[(DELIVER_PROGRAMS, B)](
            self._g, self._indptr, self._indices, self._data,
            self._ring_list, self._ring_cnt, self._ctr, n,
            R=RING, D=DELAY_STEPS, B=B, PROGRAMS=DELIVER_PROGRAMS,
            BLOCK=EDGE_BLOCK)
        self._ctr.add_(1)

    def _snapshot(self) -> list[torch.Tensor]:
        return [t.clone() for t in (self._v, self._g, self._refr, self._counts,
                                    self._ring_list, self._ring_cnt,
                                    self._ctr)]

    def _restore(self, snap: list[torch.Tensor]) -> None:
        for dst, src in zip((self._v, self._g, self._refr, self._counts,
                             self._ring_list, self._ring_cnt, self._ctr),
                            snap):
            dst.copy_(src)

    def _capture(self, n_steps: int) -> torch.cuda.CUDAGraph:
        """Warm up (compiles the kernels, advances state), capture, then
        restore the pre-warm-up state whether or not capture succeeded."""
        snap = self._snapshot()
        try:
            side = torch.cuda.Stream(device=self._device)
            side.wait_stream(torch.cuda.current_stream(self._device))
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._launch_step()
            torch.cuda.current_stream(self._device).wait_stream(side)
            torch.cuda.synchronize(self._device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(n_steps):
                    self._launch_step()
        finally:
            self._restore(snap)
            torch.cuda.synchronize(self._device)
        self._graphs[(n_steps, self._seed)] = graph
        return graph

    def _run_steps(self, n_steps: int) -> None:
        if self._use_graph:
            graph = self._graphs.get((n_steps, self._seed))
            if graph is None:
                try:
                    graph = self._capture(n_steps)
                except RuntimeError as exc:  # pragma: no cover - driver bound
                    warnings.warn(f"CUDA graph capture failed ({exc!r}); "
                                  "falling back to eager launches")
                    # a failed torch.cuda.graph __enter__ can leave its
                    # capture stream current; go back to the default stream
                    torch.cuda.set_stream(torch.cuda.default_stream(self._device))
                    self._use_graph = False
                    self._run_steps(n_steps)
                    return
            graph.replay()
        else:
            for _ in range(n_steps):
                self._launch_step()

    def run_counts(self, n_steps: int) -> np.ndarray:
        """Advance n_steps x 0.1 ms on the GPU for every fly; return per-neuron
        spike counts for that window as int32, shape (n,) for a single fly or
        (B, n) for a batch."""
        with torch.cuda.device(self._device):
            self._counts.zero_()
            done = 0
            while done < n_steps:
                k = min(MAX_GRAPH_STEPS, n_steps - done)
                self._run_steps(k)
                done += k
            counts = self._counts.cpu().numpy()
        return counts[0] if self.B == 1 else counts

    def step(self) -> np.ndarray:
        """Reference API: one step, returns sorted spiking indices (slow
        path: host syncs per call). Prefer run_counts(). Single fly only."""
        if self.B != 1:
            raise RuntimeError("step() is for batch=1; use run_counts()")
        with torch.cuda.device(self._device):
            slot = int(self._ctr.item()) % RING
            self._launch_step()
            cnt = int(self._ring_cnt[slot, 0].item())
            spk = self._ring_list[slot, 0, :cnt].cpu().numpy().astype(np.int64)
        return np.sort(spk)

    def run(self, duration_s: float, progress_every: int = 0) -> np.ndarray:
        steps = round(duration_s / DT)
        counts = np.zeros(self.n if self.B == 1 else (self.B, self.n), np.int64)
        chunk = progress_every if progress_every else steps
        done = 0
        while done < steps:
            k = min(chunk, steps - done)
            counts += self.run_counts(k)
            done += k
            if progress_every:
                print(f"  step {done}/{steps}, total spikes {counts.sum()}",
                      flush=True)
        return counts
