"""The crux smoke test: a live Battlezone frame, textured onto the CRT inside
the MuJoCo cockpit, seen by the fly's eye camera. If this works, the whole
project is assembly."""

from pathlib import Path

import numpy as np
from PIL import Image

from noflyzone.cockpit import Cockpit
from noflyzone.game import BattleZone

OUT = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    game = BattleZone(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(120):
        game.step(int(rng.integers(game.n_actions)))

    cockpit = Cockpit()
    cockpit.set_screen(game.frame)

    OUT.mkdir(exist_ok=True)
    for cam in ("eye_left", "eye_right", "beauty"):
        px = cockpit.render(camera=cam)
        Image.fromarray(px).save(OUT / f"cockpit_{cam}.png")
        print(f"{cam}: {px.shape}, brightness mean {px.mean():.1f}")

    eye = cockpit.render(camera="eye_left")
    assert eye.std() > 5.0, "eye camera sees nothing — screen not visible"

    # orientation check: bright-top / dark-bottom pattern must arrive
    # right-way-up at the eye (guards the texture-upload path against flips)
    pattern = np.zeros((210, 160, 3), dtype=np.uint8)
    pattern[:105] = 255
    cockpit.set_screen(pattern)
    view = cockpit.render(camera="eye_left").mean(axis=2)
    top, bottom = view[: view.shape[0] // 3], view[-view.shape[0] // 3:]
    assert top.mean() > bottom.mean() + 30, (
        f"screen upside down: top {top.mean():.0f} vs bottom {bottom.mean():.0f}")
    print("PASS: eye camera sees the game screen, right way up")
    cockpit.close()
    game.close()


if __name__ == "__main__":
    main()
