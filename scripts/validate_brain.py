"""M3 validation gate: does our sparse-NumPy port of the Shiu whole-brain LIF
reproduce the published behavior?

1. Silence control: no input -> the network should be essentially silent.
2. Sugar -> MN9: activating the 21 labellar sugar GRNs (Shiu et al.'s own
   IDs) must drive the proboscis motor neuron MN9 — their headline result.
3. Looming -> Giant Fiber: activating LC4 visual projection neurons must
   drive DNp01 (the escape circuit our pilot uses).
"""

import argparse
import time

import numpy as np

from noflyzone.brain_backend import BACKENDS, describe, make_brain

SUGAR_GRN_IDS = [  # Shiu et al. example.ipynb, FlyWire v630/783
    720575940611875570, 720575940612670570, 720575940616885538,
    720575940617000768, 720575940617937543, 720575940620900446,
    720575940621502051, 720575940621754367, 720575940624963786,
    720575940628853239, 720575940629176663, 720575940630233916,
    720575940630797113, 720575940632425919, 720575940632889389,
    720575940633143833, 720575940637568838, 720575940638202345,
    720575940639198653, 720575940639332736, 720575940640649691,
]
MN9_ID = 720575940660219265


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, default="auto")
    args = ap.parse_args()
    t0 = time.time()
    print("loading connectome (138k neurons, 15.1M edges)...", flush=True)
    brain = make_brain(args.backend)
    print(f"loaded in {time.time() - t0:.0f} s | backend {describe(brain)}")

    # 1 — silence control
    brain.reset()
    counts = brain.run(0.2)
    print(f"\n[1] no input, 200 ms: {counts.sum()} spikes network-wide")
    assert counts.sum() == 0, "network is spontaneously active — dynamics wrong"

    # 2 — sugar -> MN9
    sugar_idx, missing = brain.indices_of(SUGAR_GRN_IDS)
    print(f"\n[2] sugar GRNs present in v783: {len(sugar_idx)}/21 "
          f"(missing {len(missing)})")
    mn9_idx, mn9_missing = brain.indices_of([MN9_ID])
    assert len(mn9_idx) == 1, f"MN9 {MN9_ID} not in v783 completeness"
    brain.reset()
    brain.set_poisson({int(i): 150.0 for i in sugar_idx})
    t0 = time.time()
    counts = brain.run(1.0, progress_every=2500)
    mn9_rate = counts[mn9_idx[0]]
    print(f"    1 s sim in {time.time() - t0:.1f} s wall; "
          f"network spikes {counts.sum()}, neurons active "
          f"{(counts > 0).sum()}")
    print(f"    MN9 rate: {mn9_rate} Hz "
          f"(published: robust firing under sugar activation)")
    assert mn9_rate > 5, "MN9 barely fires — feeding circuit not reproduced"

    # 3 — looming -> Giant Fiber (our escape pathway)
    lc4_idx = brain.indices_for_type("LC4")
    gf_idx = np.concatenate([brain.indices_for_type("DNp01", side=s)
                             for s in ("left", "right")])
    print(f"\n[3] LC4 neurons: {len(lc4_idx)}, DNp01 (GF): {len(gf_idx)}")
    brain.reset()
    brain.set_poisson({int(i): 100.0 for i in lc4_idx})
    t0 = time.time()
    counts = brain.run(0.5, progress_every=2500)
    gf_rates = counts[gf_idx] * 2  # 0.5 s -> Hz
    print(f"    0.5 s sim in {time.time() - t0:.1f} s wall")
    print(f"    GF rates under LC4 looming drive: {[int(r) for r in gf_rates]} Hz")
    assert gf_rates.max() > 0, "Giant Fiber silent under LC4 drive"

    print("\nPASS: LIF core reproduces silence, feeding, and escape behavior")


if __name__ == "__main__":
    main()
