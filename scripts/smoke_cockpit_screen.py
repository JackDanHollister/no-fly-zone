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
    print("PASS: eye camera sees the game screen")
    cockpit.close()
    game.close()


if __name__ == "__main__":
    main()
