"""Equivalence and speed check: CUDA backend vs the reference sparse-SciPy
LIF core.

A. Deterministic drive (Poisson p = 1 -> a kick every step) on the sugar
   GRNs. A1: the per-step spike sets must match the CPU port exactly for the
   first 500 steps (50 ms) — any semantic slip (delay slot, refractory,
   ordering) shows up here within a few steps of the first delivery at step
   19. A2: over 0.3 s the aggregate counts must agree within 1%; individual
   neurons may differ because float32 atomic summation order (~1e-6 mV) is
   amplified by the recurrent network — measured 2026-09-14 with FP fusion
   off: 0 of 1500 traced steps differ on 3 GRNs, totals 314061 vs 313881 on
   20 GRNs.
B. The validate_brain gates on both backends: silence, sugar -> MN9,
   LC4 -> giant fibre. Rates should agree within Poisson noise.
C. Wall time for one game tick (667 steps) and for 1 s of sim, both backends.
"""

import argparse
import time

import numpy as np

from noflyzone.brain_backend import describe, make_brain
from noflyzone.brain_lif import DT

SUGAR_GRN_IDS = [
    720575940611875570, 720575940612670570, 720575940616885538,
    720575940617000768, 720575940617937543, 720575940620900446,
    720575940621502051, 720575940621754367, 720575940624963786,
    720575940628853239, 720575940629176663, 720575940630233916,
    720575940630797113, 720575940632425919, 720575940632889389,
    720575940633143833, 720575940637568838, 720575940638202345,
    720575940639198653, 720575940639332736, 720575940640649691,
]
MN9_ID = 720575940660219265
TICK_STEPS = 667


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-graph", action="store_true",
                    help="CUDA backend with eager launches (no CUDA graph)")
    args = ap.parse_args()

    cpu = make_brain("cpu")
    gpu = make_brain("cuda", use_graph=not args.no_graph)
    print(f"cpu: {describe(cpu)}\ngpu: {describe(gpu)}")
    sugar, _ = cpu.indices_of(SUGAR_GRN_IDS)
    mn9 = cpu.indices_of([MN9_ID])[0][0]
    lc4 = cpu.indices_for_type("LC4")
    gf = np.concatenate([cpu.indices_for_type("DNp01", side=s)
                         for s in ("left", "right")])

    # A1 — deterministic, step-exact over a short horizon
    print("\n[A1] kick-every-step drive on 3 sugar GRNs: per-step spike sets, "
          "500 steps")
    for b in (cpu, gpu):
        b.reset()
        b.set_poisson({int(i): 1.0 / DT for i in sugar[:3]})  # p = 1
    first_mismatch = None
    total = 0
    for k in range(500):
        sc = set(cpu.step().tolist())
        sg = set(gpu.step().tolist())
        total += len(sc)
        if sc != sg and first_mismatch is None:
            first_mismatch = (k, sorted(sc ^ sg))
    print(f"    {total} CPU spikes over 500 steps; first differing step: "
          f"{first_mismatch}")
    assert total > 0, "no spikes under kick-every-step drive"
    assert first_mismatch is None, \
        f"GPU spike set diverged at step {first_mismatch[0]} — semantic bug"

    # A2 — deterministic, aggregate over a chaotic horizon
    print("\n[A2] kick-every-step drive on 20 sugar GRNs, 0.3 s aggregate")
    res = {}
    for name, b in (("cpu", cpu), ("gpu", gpu)):
        b.reset()
        b.set_poisson({int(i): 1.0 / DT for i in sugar})  # p = 1
        counts, wall = timed(b.run, 0.3)
        res[name] = counts
        print(f"    {name}: {counts.sum()} spikes, {(counts > 0).sum()} active, "
              f"MN9 {counts[mn9] / 0.3:.0f} Hz, {wall:.1f} s wall")
    diff = res["gpu"].astype(np.int64) - res["cpu"].astype(np.int64)
    print(f"    neurons with different counts: {int((diff != 0).sum())} "
          f"(max |diff| {np.abs(diff).max()}, sum diff {diff.sum()})")
    rel = abs(int(res["gpu"].sum()) - int(res["cpu"].sum())) / res["cpu"].sum()
    rel_active = abs(int((res["gpu"] > 0).sum()) - int((res["cpu"] > 0).sum())) \
        / (res["cpu"] > 0).sum()
    assert rel < 0.01 and rel_active < 0.01, \
        "aggregate spike statistics differ by more than 1% — not rounding noise"

    # B — statistical gates
    print("\n[B] validate_brain gates on both backends")
    for name, b in (("cpu", cpu), ("gpu", gpu)):
        b.reset()
        silent = b.run(0.2).sum()
        b.reset()
        b.set_poisson({int(i): 150.0 for i in sugar})
        c, w_sugar = timed(b.run, 1.0)
        mn9_hz = c[mn9]
        b.reset()
        b.set_poisson({int(i): 100.0 for i in lc4})
        c, w_lc4 = timed(b.run, 0.5)
        gf_hz = [int(x) for x in c[gf] * 2]
        print(f"    {name}: silence {silent} spikes | sugar->MN9 {mn9_hz} Hz "
              f"({w_sugar:.1f} s) | LC4->GF {gf_hz} Hz ({w_lc4:.1f} s)")
        assert silent == 0 and mn9_hz > 5 and max(gf_hz) > 0

    # C — tick timing
    print("\n[C] one game tick (667 steps) under sugar drive, after warm-up")
    for name, b in (("cpu", cpu), ("gpu", gpu)):
        b.reset()
        b.set_poisson({int(i): 150.0 for i in sugar})
        if hasattr(b, "run_counts"):
            b.run_counts(TICK_STEPS)  # warm-up / graph capture
            t0 = time.perf_counter()
            for _ in range(10):
                b.run_counts(TICK_STEPS)
            per_tick = (time.perf_counter() - t0) / 10
        else:
            t0 = time.perf_counter()
            for _ in range(TICK_STEPS):
                b.step()
            per_tick = time.perf_counter() - t0
        print(f"    {name}: {per_tick * 1000:.1f} ms per tick "
              f"({(TICK_STEPS * DT) / per_tick:.2f}x real time)")
    print("\nPASS: CUDA backend matches the reference")


if __name__ == "__main__":
    main()
