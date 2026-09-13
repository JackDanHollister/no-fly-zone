"""Compound-eye retina: bins a camera image onto a hexagonal ommatidia lattice.

15 hex rings -> 721 ommatidia, matching flygym's per-eye count. Each pixel is
assigned to its nearest ommatidium centre (precomputed); sampling is then a
per-channel mean over each ommatidium's pixels.
"""

import numpy as np
from scipy.spatial import cKDTree

# measured luminance of a full-white CRT at the eye cameras; the calibration
# reference for photoreceptor drives
WHITE_REF = 255.0


class HexRetina:
    def __init__(self, n_rings: int = 15, image_shape: tuple[int, int] = (240, 320)):
        self.image_shape = image_shape
        centers = []
        for q in range(-n_rings, n_rings + 1):
            r_lo = max(-n_rings, -q - n_rings)
            r_hi = min(n_rings, -q + n_rings)
            for r in range(r_lo, r_hi + 1):
                centers.append((np.sqrt(3.0) * (q + r / 2.0), 1.5 * r))
        centers = np.asarray(centers, dtype=np.float64)
        self.n_ommatidia = len(centers)
        # normalized horizontal position of each ommatidium, -1 (left) .. +1
        # (right) — the retinotopic axis motor readouts weight against
        self.centers_x = (centers[:, 0] / np.abs(centers).max()).astype(np.float32)

        # scale lattice to the largest disc that fits the image, centred
        h, w = image_shape
        radius = min(h, w) / 2.0
        lattice_radius = np.abs(centers).max() * 1.05
        centers = centers / lattice_radius * radius
        centers += np.array([w / 2.0, h / 2.0])

        ys, xs = np.mgrid[0:h, 0:w]
        pixels = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
        dist, idx = cKDTree(centers).query(pixels)
        spacing = np.sqrt(3.0) / lattice_radius * radius
        omm = np.where(dist <= spacing * 1.5, idx, -1).astype(np.int64)

        self._omm_of_pixel = omm
        self._valid = omm >= 0
        self._counts = np.bincount(omm[self._valid], minlength=self.n_ommatidia)
        self._counts = np.maximum(self._counts, 1).astype(np.float32)

    def sample(self, img: np.ndarray) -> np.ndarray:
        """RGB image (H, W, 3) -> per-ommatidium mean RGB (n_ommatidia, 3)."""
        assert img.shape[:2] == self.image_shape, (
            f"retina built for {self.image_shape}, got {img.shape[:2]} — "
            "cockpit render size and retina image_shape must match")
        flat = img.reshape(-1, 3).astype(np.float32)
        out = np.empty((self.n_ommatidia, 3), dtype=np.float32)
        ids = self._omm_of_pixel[self._valid]
        for c in range(3):
            out[:, c] = np.bincount(ids, weights=flat[self._valid, c],
                                    minlength=self.n_ommatidia)
        out /= self._counts[:, None]
        return out

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
