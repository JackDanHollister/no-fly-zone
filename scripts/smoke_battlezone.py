"""Smoke test: Battlezone boots, steps, and yields frames."""

from pathlib import Path

import numpy as np
from PIL import Image

from noflyzone.game import BattleZone

OUT = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    game = BattleZone(seed=0)
    print(f"env: {game.env.spec.id}, actions: {game.n_actions}")
    rng = np.random.default_rng(0)
    total = 0.0
    for _ in range(500):
        _, reward, _, _ = game.step(int(rng.integers(game.n_actions)))
        total += reward
    print(f"500 random steps, reward: {total}, frame: {game.frame.shape}")
    OUT.mkdir(exist_ok=True)
    Image.fromarray(game.frame).save(OUT / "battlezone_frame.png")
    print(f"saved {OUT / 'battlezone_frame.png'}")
    game.close()


if __name__ == "__main__":
    main()
