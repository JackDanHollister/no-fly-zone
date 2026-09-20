"""M6 gate: the seated fly moves the physical levers with its forelegs.

Builds the cockpit in embodied mode, holds the stance, then commands
forward, back, left and right through the legs and reads the levers.
PASS if each command moves the levers the right way by at least
--min-value of the normalised range, the claws stay on the knobs, and the
fly does not chatter. Reports the physics cost per game tick.
"""

import argparse
import time

import numpy as np

from noflyzone.cockpit import FLY_XML, LEVER_PUSH, Cockpit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", type=float, default=0.4, help="seconds of physics per command")
    ap.add_argument("--min-value", type=float, default=0.5)
    args = ap.parse_args()
    if not FLY_XML.exists():
        raise SystemExit("flybody assets missing: run scripts/fetch_assets.py")
    c = Cockpit(physics=True)
    c.step_physics(0.2)
    print(f"stance held: levers {np.round(c.lever_values(), 3)}, contacts {c.lever_contacts()}, "
          f"max|qvel| {np.abs(c.data.qvel).max():.2f}")
    t0 = time.perf_counter(); c.step_physics(1 / 15); tick = time.perf_counter() - t0
    print(f"physics cost: {tick * 1000:.0f} ms per 66.7 ms game tick ({0.0667 / tick:.2f}x real time)")
    ok = True
    for name, (l, r), want in (("forward", (1, 1), (+1, +1)), ("back", (-1, -1), (-1, -1)),
                               ("left", (1, -1), (+1, -1)), ("right", (-1, 1), (-1, +1)),
                               ("release", (0, 0), (0, 0))):
        c.set_lever_targets(l, r)
        c.step_physics(args.settle)
        v = c.lever_values(); con = c.lever_contacts(); q = float(np.abs(c.data.qvel).max())
        good = all((abs(x) < 0.35) if w == 0 else (np.sign(x) == w and abs(x) >= args.min_value) for x, w in zip(v, want))
        good &= con["left"] > 0 and con["right"] > 0 and q < 50
        ok &= good
        print(f"  {name:8s} levers target ({l:+d},{r:+d}) -> ({v[0]:+.2f},{v[1]:+.2f}) "
              f"[{np.degrees(-v[0] * LEVER_PUSH):+.1f} deg, {np.degrees(-v[1] * LEVER_PUSH):+.1f} deg] "
              f"contacts {con['left']}/{con['right']} max|qvel| {q:.1f} {'ok' if good else 'FAIL'}")
    print("PASS: the fly works the levers" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
