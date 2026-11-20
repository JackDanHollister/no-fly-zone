"""Compound-eye retina: bins a camera image onto a hexagonal ommatidia lattice.

15 hex rings -> 721 ommatidia, matching flygym's per-eye count. Each pixel is
assigned to its nearest ommatidium centre (precomputed); sampling is then a
per-channel mean over each ommatidium's pixels.
"""

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

# measured luminance of a full-white CRT at the eye cameras; the calibration
# reference for photoreceptor drives
WHITE_REF = 255.0


class HexRetina:
    def __init__(self, n_rings: int = 15, image_shape: tuple[int, int] = (240, 320),
                 zoom: float = 1.0, center: tuple[float, float] = (0.5, 0.5)):
        """zoom > 1 lays the lattice over the central 1/zoom of the image
        (a foveal view: the fly leans in), centred at `center` as image
        fractions (x, y). zoom 1 is the whole image."""
        self.image_shape = image_shape
        self.zoom = zoom
        self.center = center
        centers = []
        for q in range(-n_rings, n_rings + 1):
            r_lo = max(-n_rings, -q - n_rings)
            r_hi = min(n_rings, -q + n_rings)
            for r in range(r_lo, r_hi + 1):
                centers.append((np.sqrt(3.0) * (q + r / 2.0), 1.5 * r))
        centers = np.asarray(centers, dtype=np.float64)
        self.n_ommatidia = len(centers)
        # normalized ommatidium positions in [-1, 1]^2 (x: left..right,
        # y: down..up) — the retinotopic frame used for motor readouts and
        # for mapping onto real photoreceptor neurons
        scale = np.abs(centers).max()
        self.centers_xy = (centers / scale).astype(np.float32)
        self.centers_x = self.centers_xy[:, 0]

        # scale lattice to the largest disc that fits the image, centred
        h, w = image_shape
        radius = min(h, w) / 2.0 / zoom
        lattice_radius = np.abs(centers).max() * 1.05
        centers = centers / lattice_radius * radius
        centers += np.array([w * center[0], h * center[1]])

        ys, xs = np.mgrid[0:h, 0:w]
        pixels = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
        dist, idx = cKDTree(centers).query(pixels)
        spacing = np.sqrt(3.0) / lattice_radius * radius
        omm = np.where(dist <= spacing * 1.5, idx, -1).astype(np.int64)

        self._omm_of_pixel = omm
        self._valid = omm >= 0
        self._counts = np.bincount(omm[self._valid], minlength=self.n_ommatidia)
        self._counts = np.maximum(self._counts, 1).astype(np.float32)
        valid_px = np.flatnonzero(self._valid)
        omm_ids = omm[valid_px]
        self._sample_mat = sparse.csr_matrix(
            ((1.0 / self._counts[omm_ids]).astype(np.float32), (omm_ids, valid_px)),
            shape=(self.n_ommatidia, h * w),
            dtype=np.float32,
        )

    def sample(self, img: np.ndarray) -> np.ndarray:
        """RGB image (H, W, 3) -> per-ommatidium mean RGB (n_ommatidia, 3)."""
        assert img.shape[:2] == self.image_shape, (
            f"retina built for {self.image_shape}, got {img.shape[:2]} — "
            "cockpit render size and retina image_shape must match")
        flat = img.reshape(-1, 3).astype(np.float32)
        return np.asarray(self._sample_mat @ flat, dtype=np.float32)

    def photoreceptors(self, sample: np.ndarray) -> np.ndarray:
        """Ommatidium mean RGB (0-255) -> photoreceptor drives (n, 2) in [0, 1].

        Channel 0: R1-6 broadband (green-weighted luminance); channel 1: R8
        blue. WHITE_REF is the calibration constant — re-measure it after any
        art pass that changes screen/scene brightness (M5), so cosmetic
        changes become a one-number recalibration, not a silent brain bias.
        """
        lum = (0.3 * sample[:, 0] + 0.5 * sample[:, 1] + 0.2 * sample[:, 2])
        drives = np.stack([lum, sample[:, 2]], axis=1) / WHITE_REF
        return np.clip(drives, 0.0, 1.0).astype(np.float32)

    def reconstruct(self, sample: np.ndarray) -> np.ndarray:
        """Ommatidium values -> fly's-eye mosaic image (for visualisation)."""
        h, w = self.image_shape
        flat = np.zeros((h * w, 3), dtype=np.float32)
        flat[self._valid] = sample[self._omm_of_pixel[self._valid]]
        return flat.reshape(h, w, 3).astype(np.uint8)
