"""Hex-column placement keeps column neighbours next to each other on the
retina, in the frame the soma positions define, and PCA-only is unchanged."""

import numpy as np

from scipy import sparse

from noflyzone.brain_connectome import _pca2d, hex_to_plane, inherit_hex, place_cells


def synthetic(n_side=12, seed=0):
    rng = np.random.default_rng(seed)
    h1, h2 = np.meshgrid(np.arange(n_side), np.arange(n_side))
    h1, h2 = h1.ravel().astype(float), h2.ravel().astype(float)
    xy = hex_to_plane(h1, h2)
    # somas: the lattice rotated and reflected in 3D, plus noise as big as a column
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    pos = np.c_[xy[:, 0], -xy[:, 1], 0.1 * xy[:, 0]] @ rot.T + rng.normal(0, 1.0, (len(h1), 3))
    return h1, h2, xy, pos


def test_hex_placement_preserves_neighbours_and_frame():
    h1, h2, xy, pos = synthetic()
    uv, n_hex = place_cells(pos, h1, h2)
    assert n_hex == len(h1)
    assert np.abs(uv).max() <= 1.0 + 1e-9
    # hex neighbours (h1+1) are the nearest cells on the plane
    d = np.linalg.norm(uv[:, None] - uv[None], axis=-1); np.fill_diagonal(d, np.inf)
    nn = d.argmin(axis=1)
    is_neighbour = (np.abs(h1[nn] - h1) + np.abs(h2[nn] - h2) <= 2) & (np.abs(h1[nn] - h1) <= 1) & (np.abs(h2[nn] - h2) <= 1)
    assert is_neighbour.mean() > 0.95
    # the frame follows the soma PCA (same orientation, up to noise)
    pca = _pca2d(pos)
    assert np.corrcoef(uv[:, 0], pca[:, 0])[0, 1] > 0.9
    assert np.corrcoef(uv[:, 1], pca[:, 1])[0, 1] > 0.9


def test_missing_columns_fall_back_to_pca_in_the_same_frame():
    h1, h2, xy, pos = synthetic()
    h1m, h2m = h1.copy(), h2.copy()
    h1m[::3] = np.nan; h2m[::3] = np.nan
    uv, n_hex = place_cells(pos, h1m, h2m)
    assert n_hex == len(h1) - len(h1[::3])
    pca = _pca2d(pos)
    assert np.allclose(uv[::3], pca[::3])          # the PCA cells are exactly PCA
    assert np.corrcoef(uv[:, 0], pca[:, 0])[0, 1] > 0.9
    uv_pca, n = place_cells(pos, h1, h2, retinotopy="pca")
    assert n == 0 and np.allclose(uv_pca, pca)


def test_inherit_hex_takes_the_weighted_majority_same_side_partner_column():
    # cells 0,1,2 have columns; 3 is a photoreceptor driving 1 (weight 5) and 2 (weight 2);
    # 4 has only a partner on the other side; 5 has none
    hex1 = np.array([1, 2, 3, np.nan, np.nan, np.nan]); hex2 = np.array([1, 2, 3, np.nan, np.nan, np.nan])
    side = np.array(["left"] * 4 + ["left", "right"])
    W = sparse.lil_matrix((6, 6))
    W[1, 3] = 5.0; W[2, 3] = -2.0          # column 3 = presynaptic cell 3
    W[0, 4] = 1.0; side[0] = "right"       # cell 4's only partner is on the other side
    h1, h2, n = inherit_hex(W.tocsc(), hex1, hex2, side, np.array([3, 4, 5]))
    assert n == 1 and (h1[3], h2[3]) == (2, 2)
    assert np.isnan(h1[4]) and np.isnan(h1[5])


def test_hex_only_cells_are_placed_without_positions():
    h1, h2, xy, pos = synthetic()
    pos2 = pos.copy(); pos2[::2] = np.nan            # half the cells have no soma
    uv, n_hex = place_cells(pos2, h1, h2)
    assert n_hex == len(h1) and np.isfinite(uv).all()
    d = np.linalg.norm(uv[:, None] - uv[None], axis=-1); np.fill_diagonal(d, np.inf)
    nn = d.argmin(axis=1)
    assert ((np.abs(h1[nn] - h1) <= 1) & (np.abs(h2[nn] - h2) <= 1)).mean() > 0.95
    pca = _pca2d(pos[1::2])
    assert abs(np.corrcoef(uv[1::2, 0], pca[:, 0])[0, 1]) > 0.9
