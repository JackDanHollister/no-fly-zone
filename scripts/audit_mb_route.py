"""Go/no-go audit for mushroom-body plasticity (M7): does the visual drive we
inject reach the Kenyon cells, do the MBONs reach the descending neurons we
read out, and how big are the plastic and teaching edge sets?

Output: out/audit_mb_route.json and a table on stdout.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from noflyzone.brain_connectome import LOOM_POOL, STEER_POOL
from noflyzone.brain_lif import SHIU, ShiuBrain, annotations

OUT = Path(__file__).resolve().parent.parent / "out"
INJECT = ["Mi1", "Tm3", "Tm1", "Tm2", "Tm4", "Tm9", "T3", "R1-6"]
READOUT = STEER_POOL + ["DNg100"] + LOOM_POOL


def main() -> None:
    brain = ShiuBrain()
    n = brain.n
    ann = annotations().drop_duplicates("root_id").set_index("root_id").reindex(brain.root_ids)
    ct = ann["cell_type"].astype(str).to_numpy()
    con = pd.read_parquet(SHIU / "Connectivity_783.parquet",
                          columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    pre = con["Presynaptic_Index"].to_numpy(np.int64); post = con["Postsynaptic_Index"].to_numpy(np.int64)
    w = con["Excitatory x Connectivity"].to_numpy(np.float32); cnt = np.abs(w)
    starts = lambda s: np.flatnonzero(np.char.startswith(ct.astype(str), s))
    kc, mbon, pam, ppl = starts("KC"), starts("MBON"), starts("PAM"), starts("PPL1")
    kc_types = pd.Series(ct[kc]).value_counts()
    print(f"Kenyon cells {kc.size} in {len(kc_types)} types: {kc_types.head(12).to_dict()}")
    print(f"MBONs {mbon.size}, PAM {pam.size}, PPL1 {ppl.size}")

    # plastic set and teaching edges
    m_kc_mbon = np.isin(pre, kc) & np.isin(post, mbon)
    m_pam_kc = np.isin(pre, pam) & np.isin(post, kc)
    m_ppl_kc = np.isin(pre, ppl) & np.isin(post, kc)
    print(f"\nKC -> MBON edges: {m_kc_mbon.sum():,} ({cnt[m_kc_mbon].sum():,.0f} synapses); "
          f"PAM -> KC edges: {m_pam_kc.sum():,}; PPL1 -> KC edges: {m_ppl_kc.sum():,}")
    by_mbon = pd.Series(ct[post[m_kc_mbon]]).value_counts()
    print("KC -> MBON edges by MBON type (top 12):", by_mbon.head(12).to_dict())

    # BFS hops over excitatory edges with >= 3 synapses (a spike can propagate)
    keep = (cnt >= 3) & (w > 0)
    adj = sparse.csr_matrix((np.ones(keep.sum(), np.int8), (pre[keep], post[keep])), shape=(n, n))
    def hops_from(src, targets, max_hops=6):
        reached = np.zeros(n, bool); reached[src] = True; frontier = reached.copy()
        first = {t: (0 if reached[i].any() else None) for t, i in targets.items()}
        for hop in range(1, max_hops + 1):
            nxt = (adj.T @ frontier.astype(np.int32)) > 0
            frontier = nxt & ~reached; reached |= nxt
            for t, i in targets.items():
                if first[t] is None and reached[i].any():
                    first[t] = hop
            if not frontier.any():
                break
        return first, reached
    kc_targets = {t: np.flatnonzero(ct == t) for t in kc_types.index[:12]}
    inj = np.flatnonzero(np.isin(ct, INJECT))
    first, reached = hops_from(inj, kc_targets)
    print("\nhops from the injected visual types to each KC type (excitatory edges >= 3 syn):", first)
    kc_reached = int(reached[kc].sum())
    print(f"Kenyon cells reachable from the visual injection within 6 hops: {kc_reached}/{kc.size}")
    # which KCs get visual input in 2-3 hops (the visual KC subtypes)
    ro_targets = {t: np.flatnonzero(ct == t) for t in READOUT}
    first_ro, _ = hops_from(mbon, ro_targets)
    print("hops from MBONs to the readout DN types:", first_ro)
    # direct MBON -> readout edges
    ro_idx = np.flatnonzero(np.isin(ct, READOUT))
    m_mbon_ro = np.isin(pre, mbon) & np.isin(post, ro_idx)
    print(f"direct MBON -> readout-DN edges: {m_mbon_ro.sum()} ({cnt[m_mbon_ro].sum():.0f} synapses); "
          f"by DN type: {pd.Series(ct[post[m_mbon_ro]]).value_counts().to_dict()}")
    OUT.mkdir(exist_ok=True)
    (OUT / "audit_mb_route.json").write_text(json.dumps({
        "kc_types": kc_types.to_dict(), "kc_mbon_edges": int(m_kc_mbon.sum()),
        "pam_kc_edges": int(m_pam_kc.sum()), "ppl_kc_edges": int(m_ppl_kc.sum()),
        "hops_visual_to_kc": first, "kc_reached": kc_reached, "hops_mbon_to_readout": first_ro,
        "mbon_readout_edges": int(m_mbon_ro.sum())}, indent=1, default=str))


if __name__ == "__main__":
    main()
