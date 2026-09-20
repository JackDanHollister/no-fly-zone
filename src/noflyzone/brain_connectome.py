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
    # v4 plus the slow inhibitory inputs of T4: Mi4 (GABA, sustained ON,
    # low-passed luminance) and Mi9 (glutamate, sustained in the dark,
    # low-passed darkness). The fly's T4 direction selectivity comes from
    # these arriving from neighbouring columns with slower kinetics than
    # Mi1/Tm3; the columns are in the wiring, the kinetics have to be
    # injected. Gains in Hz at full-scale signal; tau in LOWPASS_TAU.
    "v5": [
        ("R1-6", "lum", MAX_PR_RATE),
        ("Mi1", "on", 3000.0), ("Tm3", "on", 3000.0),
        ("Tm1", "off", 3000.0), ("Tm2", "off", 2000.0),
        ("Tm4", "off", 2000.0), ("Tm9", "off", 2000.0),
        ("T3", "onoff", 3000.0),
        ("Mi4", "lum_lp", 80.0), ("Mi9", "dark_lp", 80.0),
    ],
}
LOWPASS_TAU = 0.15  # s, the slow channels' time constant
SURROUND_RADIUS = 3.0   # lattice spacings; wider than the screen's small disc (radius 2) so its centre keeps 0.6 of its drive
SURROUND_K = 1.0
WIDEFIELD_K, WIDEFIELD_FRAC = 1.0, 0.10   # v8: opponency against the eye's mean surround energy; at 0.10 a small disc keeps 0.82, a loom edge past radius 5 nothing

# v6: direction computed OUTSIDE the wiring. A Hassenstein-Reichardt
# correlator over neighbouring ommatidia (this tick's ON or OFF transient at
# the neighbour times last tick's at the cell, minus the reverse) gives four
# direction channels per eye; they are injected at the four T4 subtypes (ON)
# and four T5 subtypes (OFF). The probe of 2026-09-18 showed the LIF cannot
# make T4 direction-selective from columnar inputs (point neurons, one fixed
# delay), so this replaces exactly that stage with an engineered one and
# leaves everything downstream of T4/T5 on the connectome. The letter-to-
# direction assignment (a: +x, b: -x, c: +y, d: -y on the retina's
# normalised axes) is a convention and is recorded with the map.
# v7: v4 with the medulla transient channels surround-antagonised at the
# retina (centre minus the mean of the ommatidia within SURROUND_RADIUS,
# rectified), the stage a point LIF over columnar inputs cannot compute and
# the fly does in the lamina and medulla (L2/C3 and Mi4/Mi9/CT1 surrounds).
# Labelled ENGINEERED like v6; the question is whether the wiring downstream
# builds size-tuned descending neurons from it, as it built HS/VS from v6.
INJECTION_MAPS["v7"] = [
    ("R1-6", "lum", MAX_PR_RATE),
    ("Mi1", "on_cs", 3000.0), ("Tm3", "on_cs", 3000.0),
    ("Tm1", "off_cs", 3000.0), ("Tm2", "off_cs", 2000.0),
    ("Tm4", "off_cs", 2000.0), ("Tm9", "off_cs", 2000.0),
    ("T3", "onoff_cs", 3000.0),
]
# v8: v7 plus wide-field opponency on the same channels, the second stage
# a small-object detector needs (2026-09-20, docs/research/size-tuning-2026-09-20.md):
# a loom's edge passes a surround everywhere at once and suppresses itself
# against the eye's mean; a lone small disc does not. ENGINEERED.
INJECTION_MAPS["v8"] = [
    ("R1-6", "lum", MAX_PR_RATE),
    ("Mi1", "on_wf", 3000.0), ("Tm3", "on_wf", 3000.0),
    ("Tm1", "off_wf", 3000.0), ("Tm2", "off_wf", 2000.0),
    ("Tm4", "off_wf", 2000.0), ("Tm9", "off_wf", 2000.0),
    ("T3", "onoff_wf", 3000.0),
]
# v9: v4 with the small-object channel (v8's surround plus wide-field
# opponency) only on T2, T2a and T3, the columnar cells that feed the
# lobula's small-object detectors (LC11, LC18); the medulla ON/OFF channels
# that feed the loom detectors keep their raw transients. Does the wiring
# route the engineered small-object signal to specific descending neurons
# while keeping its loom response? ENGINEERED.
INJECTION_MAPS["v9"] = [
    ("R1-6", "lum", MAX_PR_RATE),
    ("Mi1", "on", 3000.0), ("Tm3", "on", 3000.0),
    ("Tm1", "off", 3000.0), ("Tm2", "off", 2000.0),
    ("Tm4", "off", 2000.0), ("Tm9", "off", 2000.0),
    ("T2", "onoff_wf", 3000.0), ("T2a", "onoff_wf", 3000.0), ("T3", "onoff_wf", 3000.0),
]
INJECTION_MAPS["v6"] = INJECTION_MAPS["v4"] + [
    ("T4a", "on_dir_px", 20000.0), ("T4b", "on_dir_mx", 20000.0),
    ("T4c", "on_dir_py", 20000.0), ("T4d", "on_dir_my", 20000.0),
    ("T5a", "off_dir_px", 20000.0), ("T5b", "off_dir_mx", 20000.0),
    ("T5c", "off_dir_py", 20000.0), ("T5d", "off_dir_my", 20000.0),
]
DIR_VECTORS = {"px": (1.0, 0.0), "mx": (-1.0, 0.0), "py": (0.0, 1.0), "my": (0.0, -1.0)}
# v6m: the same correlator with the right eye's horizontal channels mirrored
# (T4a/T5a get -x, T4b/T5b get +x on the right), so "a" means the same
# body-relative direction in both eyes as it does in the fly, where the two
# eyes' front-to-back axes are mirror images. With v6 a full-screen sweep
# drives the same subtype on both sides; with v6m it drives opposite ones,
# which is what yaw optic flow does to a real fly. Which of the two screen
# directions is "front-to-back" is a convention, recorded here as: on the
# left eye +x (screen left to right) is channel a.
INJECTION_MAPS["v6m"] = INJECTION_MAPS["v4"] + [
    ("T4a", "on_dir_px|mx", 20000.0), ("T4b", "on_dir_mx|px", 20000.0),
    ("T4c", "on_dir_py", 20000.0), ("T4d", "on_dir_my", 20000.0),
    ("T5a", "off_dir_px|mx", 20000.0), ("T5b", "off_dir_mx|px", 20000.0),
    ("T5c", "off_dir_py", 20000.0), ("T5d", "off_dir_my", 20000.0),
]


def surround_matrix(xy: np.ndarray, radius: float = 2.0):
    """Row-normalised sparse averaging matrix over the ommatidia within
    `radius` lattice spacings of each one (itself excluded): the surround
    of a centre-surround receptive field on the hex lattice."""
    from scipy import sparse
    xy = np.asarray(xy, np.float64)
    tree = cKDTree(xy)
    spacing = float(np.median(tree.query(xy, k=2)[0][:, 1]))
    rows, cols = [], []
    for i, nb in enumerate(tree.query_ball_point(xy, r=radius * spacing + 1e-6)):
        for j in nb:
            if j != i:
                rows.append(i); cols.append(j)
    rows, cols = np.array(rows), np.array(cols)
    counts = np.bincount(rows, minlength=len(xy)).astype(np.float64)
    vals = 1.0 / counts[rows]
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(xy), len(xy)))


def centre_surround(sig: np.ndarray, S, k: float = 1.0) -> np.ndarray:
    """Centre minus k times the mean of the surround, rectified: a small
    object lights a centre without its surround, a large one lights both."""
    return np.clip(sig - k * (S @ sig), 0.0, None)


def wide_field_opponent(cs: np.ndarray, k: float = 1.0, frac: float = 0.10) -> np.ndarray:
    """A small-object channel: the surround-antagonised signal minus k times
    its whole-eye mean, rectified, with the mean scaled by the fraction of
    the eye a small object covers. A lone small disc lights a few ommatidia
    and the mean stays near zero; a loom's edge lights a ring everywhere
    and suppresses itself (the lobula's wide-field inhibition, LPi-like,
    computed rather than wired)."""
    return np.clip(cs - k * cs.mean() / frac, 0.0, None)


def lattice_neighbours(xy: np.ndarray) -> dict[str, np.ndarray]:
    """For each ommatidium, the index of its neighbour one lattice spacing
    away in each of DIR_VECTORS (-1 at the edge)."""
    xy = np.asarray(xy, np.float64)
    tree = cKDTree(xy)
    spacing = float(np.median(tree.query(xy, k=2)[0][:, 1]))
    out = {}
    for key, (dx, dy) in DIR_VECTORS.items():
        dist, j = tree.query(xy + spacing * np.array([dx, dy]))
        out[key] = np.where(dist < 0.6 * spacing, j, -1)
    return out


def reichardt(cur: np.ndarray, prev: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """Hassenstein-Reichardt output per ommatidium for motion toward the
    neighbour: a transient that was here last tick and is at the neighbour
    now, minus the reverse, rectified."""
    ok = nb >= 0
    fwd = np.zeros_like(cur); back = np.zeros_like(cur)
    fwd[ok] = cur[nb[ok]] * prev[ok]
    back[ok] = cur[ok] * prev[nb[ok]]
    return np.clip(fwd - back, 0.0, None)
INJECTION_SITES = INJECTION_MAPS["v0"]
TRANSIENT_CAP = 400.0  # Hz ceiling on injected transient rates

# Named readout maps: dn.py contract name -> list of selectors
# (kind, value, sides); a pool's rate is the mean over all its cells.
# Special kind "min_of" takes the minimum of two other contract rates.
STEER_POOL = ["DNp07", "DNb05", "DNg52", "DNb06", "DNa02"]
# DNp12 is the loom-selective type in the screen (24 Hz loom, 0.6 Hz small
# object); DNbe001 was dropped because it also fires for a tank sitting ahead
LOOM_POOL = ["DNp12", "DNp01"]
# the within-hemisphere rewiring control's OWN screened pools (screen_dns.py
# --ablate rewired_ipsi --ablate-seed 0, injection v4, 2026-09-18): the five
# descending types that lateralise best with a small object in THAT graph,
# and its two loom-selective types. A control brain gets the same readout
# procedure as the measured one, not the measured one's pools.
STEER_POOL_IPSI0 = ["DNp103", "DNge041", "DNge104", "DNa10", "DNae009"]
LOOM_POOL_IPSI0 = ["DNp102", "DNg84"]
# MaleCNS v1.0 at weights x0.6, its own screen (screen_dns.py --dataset
# malecns --weight-scale 0.6, injection v4, 2026-09-18): descending types
# lateralised (LI > 0.5) on both a small object and a bar, ranked by
# response, and its loudest loom-selective types plus the giant fibre.
# DNb05 and DNp07 are in the female pool too; DNp04, DNp02 and DNp20 are not.
STEER_POOL_MALE = ["DNp04", "DNb05", "DNp02", "DNp20", "DNp07"]
LOOM_POOL_MALE = ["DNg33", "DNp12", "DNp01"]
STEER_POOL_MALE_HEX = ["DNp04", "DNp02", "DNp15", "DNp07", "DNb05"]
LOOM_POOL_MALE_HEX = ["DNg33", "DNg36_a", "DNp12", "DNge141"]
# "yaw": the descending types whose left-minus-right rate flips sign with
# the direction of yaw optic flow under injection v6m (probe_motion.py
# --yaw, 2026-09-18): DNp15, DNg41, DNae002, DNa02, DNa15, all with
# |direction index| > 0.85 on both sides. Under the v6m convention the LEFT
# cells fire for screen-rightward motion, so to steer TOWARD the motion
# (pursuit; also the stabilising sense of the optomotor response) the
# contract's left key reads the right-side cells and vice versa. Drive and
# startle as pool4. An engineered front end feeds it; labelled as such.
YAW_POOL = ["DNp15", "DNg41", "DNae002", "DNa02", "DNa15"]
READOUT_MAPS = {
    # the M3/M4 map: single named cells
    "v0": {
        "DNa02_L": [("cell_type", "DNa02", ("left",))],
        "DNa02_R": [("cell_type", "DNa02", ("right",))],
        "DNg100": [("cell_type", "DNg100", None)],
        # pIP10 itself where the dataset has it (MaleCNS); the female
        # FlyWire brain has none, so the oviposition DNs stand in
        "pIP10": [("cell_type", "pIP10", None), ("cell_type_prefix", "oviDN", None)],
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
    "yaw": {
        "DNa02_L": [("cell_type", t, ("right",)) for t in YAW_POOL],
        "DNa02_R": [("cell_type", t, ("left",)) for t in YAW_POOL],
        "DNg100": [("cell_type", "DNg100", None)],
        "pIP10": [("min_of", ("DNa02_L", "DNa02_R"), None)],
        "DNp01": [("cell_type", t, None) for t in LOOM_POOL],
    },
    # pool4's recipe applied to the male brain's own screen
    "pool_male": {
        "DNa02_L": [("cell_type", t, ("left",)) for t in STEER_POOL_MALE],
        "DNa02_R": [("cell_type", t, ("right",)) for t in STEER_POOL_MALE],
        "DNg100": [("cell_type", "DNg100", None)],
        "pIP10": [("min_of", ("DNa02_L", "DNa02_R"), None)],
        "DNp01": [("cell_type", t, None) for t in LOOM_POOL_MALE],
    },
    # pool4's recipe applied to the male's screen under the hex-column
    # placement of 2026-09-20 (docs/research/male-retinotopy-2026-09-20.md):
    # steer = the five lateralised (LI 1.0, consistent) types with the
    # highest small-object rate; loom = the loom-selective, non-lateralised
    # types above 300 Hz, plus DNp12
    "pool_male_hex": {
        "DNa02_L": [("cell_type", t, ("left",)) for t in STEER_POOL_MALE_HEX],
        "DNa02_R": [("cell_type", t, ("right",)) for t in STEER_POOL_MALE_HEX],
        "DNg100": [("cell_type", "DNg100", None)],
        "pIP10": [("min_of", ("DNa02_L", "DNa02_R"), None)],
        "DNp01": [("cell_type", t, None) for t in LOOM_POOL_MALE_HEX],
    },
    # pool4's recipe applied to the rewired_ipsi seed-0 brain's own screen
    "pool_ipsi0": {
        "DNa02_L": [("cell_type", t, ("left",)) for t in STEER_POOL_IPSI0],
        "DNa02_R": [("cell_type", t, ("right",)) for t in STEER_POOL_IPSI0],
        "DNg100": [("cell_type", "DNg100", None)],
        "pIP10": [("min_of", ("DNa02_L", "DNa02_R"), None)],
        "DNp01": [("cell_type", t, None) for t in LOOM_POOL_IPSI0],
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


def hex_to_plane(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Axial hex-column coordinates -> points on a hex lattice in the plane.
    Any axial basis of the lattice gives the same point set up to an
    isometry, which place_cells() fits afterwards, so the annotation's
    convention need not be known."""
    h1 = np.asarray(h1, np.float64); h2 = np.asarray(h2, np.float64)
    return np.stack([np.sqrt(3.0) * (h1 + h2 / 2.0), 1.5 * h2], axis=1)


def place_cells(pos: np.ndarray, hex1=None, hex2=None, retinotopy: str = "auto") -> tuple[np.ndarray, int]:
    """2D retinotopic coordinates in [-1, 1]^2 for one (type, side) population.
    Cells with a hex column (MaleCNS assignedOlHex1/2) are placed by column
    identity, on a hex lattice aligned to the type's soma-position PCA frame
    by an orthogonal Procrustes fit (rotation or reflection), so the axes
    keep the PCA convention (x lateral, y vertical) and cells without a
    column, placed by PCA, share the frame. Returns (uv, n_placed_by_hex).
    retinotopy="pca" ignores the columns (the 2026-09-18 behaviour)."""
    pos = np.asarray(pos, float)
    has_pos = np.isfinite(pos).all(axis=1)
    uv = np.full((len(pos), 2), np.nan)
    if has_pos.sum() >= 3:
        uv[has_pos] = _pca2d(pos[has_pos])
    if retinotopy == "pca" or hex1 is None:
        return uv, 0
    hexed = np.isfinite(np.asarray(hex1, float)) & np.isfinite(np.asarray(hex2, float))
    n_hex = int(hexed.sum())
    if n_hex < 3:
        return uv, 0
    xy = hex_to_plane(np.asarray(hex1, float)[hexed], np.asarray(hex2, float)[hexed])
    a = xy - xy.mean(axis=0)
    both = hexed & has_pos
    if both.sum() >= 3:
        # orient the lattice to the soma frame with an orthogonal fit
        b = uv[both] - uv[both].mean(axis=0)
        ab = xy[both[hexed]] - xy[both[hexed]].mean(axis=0)
        u, _, vt = np.linalg.svd(ab.T @ b)
        a = a @ (u @ vt)
    aligned = a / np.abs(a).max(axis=0)
    out = uv.copy()
    out[hexed] = aligned
    return out, n_hex


def inherit_hex(W, hex1: np.ndarray, hex2: np.ndarray, side: np.ndarray,
                cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Give cells without a hex column the synapse-weighted majority column
    of their same-side partners (inputs and outputs) that have one. In the
    fly a column is defined by its connectivity, so a photoreceptor whose
    soma lies outside the volume (MaleCNS: 13 of 3,377 R1-6 have a
    position) is placed at the lamina column it drives. W is CSC with
    columns = presynaptic cells. Returns (hex1, hex2, n_inherited)."""
    hex1 = np.array(hex1, float); hex2 = np.array(hex2, float)
    has = np.isfinite(hex1) & np.isfinite(hex2)
    csc = W.tocsc(); csr = W.tocsr()
    n_done = 0
    for i in cells:
        if has[i]:
            continue
        votes: dict[tuple[float, float], float] = {}
        for mat, sl in ((csc, slice(csc.indptr[i], csc.indptr[i + 1])),     # outputs
                        (csr, slice(csr.indptr[i], csr.indptr[i + 1]))):    # inputs
            partners, w = mat.indices[sl], np.abs(mat.data[sl])
            for j, wj in zip(partners, w):
                if has[j] and side[j] == side[i] and wj > 0:
                    k = (hex1[j], hex2[j]); votes[k] = votes.get(k, 0.0) + float(wj)
        if votes:
            hex1[i], hex2[i] = max(votes.items(), key=lambda kv: kv[1])[0]
            n_done += 1
    return hex1, hex2, n_done


class ConnectomeBrain:
    def __init__(self, retina, brain: ShiuBrain | None = None, seed: int = 0,
                 injection: str = "v0", readout: str = "v0",
                 extra_sites: dict[str, np.ndarray] | None = None,
                 retinotopy: str = "auto"):
        """extra_sites: name -> brain indices to add to the Poisson activation
        set with their own rate slots (e.g. dopamine neurons as teaching
        signals); set them per tick with set_extra_rate(). retinotopy:
        "auto" places cells by hex column where the annotation has one
        (MaleCNS), else by soma-position PCA; "pca" always PCA."""
        self.brain = brain if brain is not None else ShiuBrain(seed=seed)
        self.injection = injection
        self.sites = INJECTION_MAPS[injection]
        self.readout = readout
        self.retinotopy = retinotopy
        self.retinotopy_counts = {"hex": 0, "pca": 0, "inherited": 0}
        ann = self.brain.annotations()
        has_hex = "hex1" in ann.columns and "hex2" in ann.columns and retinotopy != "pca"
        if has_hex:
            # columns for the injected cells that lack one, from their partners
            ann = ann.drop_duplicates("root_id").set_index("root_id").reindex(self.brain.root_ids)
            h1 = ann["hex1"].to_numpy(float); h2 = ann["hex2"].to_numpy(float)
            sides = ann["side"].fillna("").astype(str).to_numpy()
            types = ann["cell_type"].fillna("").astype(str).to_numpy()
            wanted = np.flatnonzero(np.isin(types, [t for t, _, _ in self.sites]))
            h1, h2, n_inh = inherit_hex(self.brain.W, h1, h2, sides, wanted)
            self.retinotopy_counts["inherited"] = n_inh
            hex_of = dict(zip(self.brain.root_ids.tolist(), zip(h1.tolist(), h2.tolist())))
            ann = self.brain.annotations()

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
                in_brain = np.array([int(r) in self.brain.index_of for r in ids])
                if has_hex:
                    hx = np.array([hex_of.get(int(r), (np.nan, np.nan)) for r in ids], float).reshape(-1, 2)
                    # a cell is placeable with a position or a (possibly inherited) column
                    keep = in_brain & (np.isfinite(pos).all(axis=1) | np.isfinite(hx).all(axis=1))
                    hx = hx[keep]
                else:
                    # cells without a position cannot be placed on the retina
                    keep = in_brain & np.isfinite(pos).all(axis=1)
                ids, pos = ids[keep], pos[keep]
                bidx = np.array([self.brain.index_of[int(r)] for r in ids],
                                dtype=np.int64)
                if has_hex:
                    uv, n_hex = place_cells(pos, hx[:, 0], hx[:, 1], retinotopy)
                else:
                    uv, n_hex = _pca2d(pos), 0
                self.retinotopy_counts["hex"] += n_hex
                self.retinotopy_counts["pca"] += len(bidx) - n_hex
                omm = tree.query(uv)[1]
                self._pops.append((mode, gain,
                                   slice(start, start + len(bidx)), omm, side))
                chunks.append(bidx)
                start += len(bidx)
        self._extra: dict[str, slice] = {}
        for name, idx in (extra_sites or {}).items():
            idx = np.asarray(idx, dtype=np.int64)
            self._extra[name] = slice(start, start + len(idx))
            chunks.append(idx)
            start += len(idx)
        self._pr_all = np.concatenate(chunks)
        self._rates = np.zeros(len(self._pr_all), np.float32)
        self._S = surround_matrix(retina.centers_xy, SURROUND_RADIUS) \
            if any(mode.endswith(("_cs", "_wf")) for mode, _, _, _, _ in self._pops) else None
        self.n_visual = int(sum(sl.stop - sl.start for _, _, sl, _, _ in self._pops))
        self._prev_lum: dict[str, np.ndarray | None] = {"left": None,
                                                        "right": None}
        self._lp: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._prev_onoff: dict[str, tuple[np.ndarray, np.ndarray] | None] = {"left": None, "right": None}
        # neighbour index in each lattice direction (-1 at the edge), for v6
        self._nb = lattice_neighbours(retina.centers_xy)

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
        rc = self.retinotopy_counts
        placed = (f", retinotopy hex-column {rc['hex']} cells ({rc['inherited']} by partners) / PCA {rc['pca']}"
                  if rc["hex"] else "")
        return (f"injection map {self.injection} {pops}, "
                f"readout map {self.readout} {ro}{placed}")

    def _register(self) -> None:
        self.brain.set_poisson({int(i): 0.0 for i in self._pr_all})

    def set_extra_rate(self, name: str, hz: float) -> None:
        self._rates[self._extra[name]] = hz

    def reset(self) -> None:
        self.brain.reset()
        self._register()
        self._prev_lum = {"left": None, "right": None}
        self._lp = {"left": None, "right": None}
        self._prev_onoff = {"left": None, "right": None}

    def _direction(self, cur: np.ndarray, prev: np.ndarray, key: str) -> np.ndarray:
        return reichardt(cur, prev, self._nb[key])

    def step(self, drives: dict[str, np.ndarray], dt: float,
             signals: dict | None = None) -> dict[str, float]:
        lum = {side: drives[side][:, 0] for side in ("left", "right")}
        delta = {}
        alpha = 1.0 - np.exp(-dt / LOWPASS_TAU)
        for side in lum:
            prev = self._prev_lum[side]
            delta[side] = (lum[side] - prev) if prev is not None \
                else np.zeros_like(lum[side])
            self._prev_lum[side] = lum[side]
            lp = self._lp[side]
            self._lp[side] = lum[side].copy() if lp is None \
                else lp + alpha * (lum[side] - lp)
        onoff = {side: (np.clip(delta[side], 0.0, None), np.clip(-delta[side], 0.0, None))
                 for side in lum}
        cs = {}
        if self._S is not None:
            for side in lum:
                cs[(side, "on_cs")] = centre_surround(onoff[side][0], self._S, SURROUND_K)
                cs[(side, "off_cs")] = centre_surround(onoff[side][1], self._S, SURROUND_K)
                cs[(side, "onoff_cs")] = cs[(side, "on_cs")] + cs[(side, "off_cs")]
                for ch in ("on", "off"):
                    cs[(side, f"{ch}_wf")] = wide_field_opponent(cs[(side, f"{ch}_cs")], WIDEFIELD_K, WIDEFIELD_FRAC)
                cs[(side, "onoff_wf")] = cs[(side, "on_wf")] + cs[(side, "off_wf")]
        dirs: dict[tuple[str, str], np.ndarray] = {}
        for side in lum:
            prev = self._prev_onoff[side]
            if prev is not None:
                for key in DIR_VECTORS:
                    dirs[(side, "on_dir_" + key)] = self._direction(onoff[side][0], prev[0], key)
                    dirs[(side, "off_dir_" + key)] = self._direction(onoff[side][1], prev[1], key)
            self._prev_onoff[side] = onoff[side]
        for mode, gain, sl, omm, side in self._pops:
            if mode == "lum":
                sig = lum[side][omm]
            elif mode == "on":
                sig = np.clip(delta[side][omm], 0.0, None)
            elif mode == "off":
                sig = np.clip(-delta[side][omm], 0.0, None)
            elif mode == "lum_lp":     # slow sustained ON
                sig = self._lp[side][omm]
            elif mode == "dark_lp":    # slow sustained in the dark
                sig = 1.0 - self._lp[side][omm]
            elif mode.endswith(("_cs", "_wf")):   # v7 surround, v8 surround + wide-field opponency
                sig = cs[(side, mode)][omm]
            elif mode.startswith(("on_dir_", "off_dir_")):  # v6 correlator channels
                key = mode
                if "|" in mode:  # "left|right" channel names (v6m mirrors the right eye)
                    base, alt = mode.split("|")
                    key = base if side == "left" else base.rsplit("_", 1)[0] + "_" + alt
                d = dirs.get((side, key))
                sig = d[omm] if d is not None else np.zeros(len(omm), np.float32)
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
        ann = brain.annotations().set_index("root_id")
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
