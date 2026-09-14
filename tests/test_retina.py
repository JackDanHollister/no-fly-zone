import numpy as np

from noflyzone.retina import HexRetina


def test_lattice_size_and_zoom():
    r1 = HexRetina(image_shape=(240, 320))
    r2 = HexRetina(image_shape=(240, 320), zoom=2.0, center=(0.35, 0.5))
    assert r1.n_ommatidia == r2.n_ommatidia == 721
    img = np.zeros((240, 320, 3), np.uint8)
    img[:, :160] = 255  # bright left half
    s1, s2 = r1.sample(img), r2.sample(img)
    assert s1.shape == (721, 3)
    # the zoomed left-eye window sits over the bright half, so it sees more light
    assert s2[:, 0].mean() > s1[:, 0].mean()


def test_photoreceptor_drives_are_bounded():
    r = HexRetina()
    d = r.photoreceptors(np.full((721, 3), 255.0, np.float32))
    assert d.shape == (721, 2) and d.max() <= 1.0 and d.min() >= 0.0
