"""Focused moving-target tracking benchmark through the cockpit CRT, compound
eyes, whole-brain LIF connectome and motor controller.

Addresses Milestone 1 in README.md: compare measured FlyWire wiring, whole-brain
shuffled controls (`rewired`), within-hemisphere shuffled controls
(`rewired_ipsi`, both with the measured pools and its own screened pools), and
a silenced brain on reserved target trajectories using identical visual inputs
and motor rules.

Output: out/bench_tracking.json (with provenance) and a summary table.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np

from noflyzone.brain_backend import BACKENDS, describe, make_brain
from noflyzone.brain_connectome import ConnectomeBrain
from noflyzone.cockpit import Cockpit
from noflyzone.controls import CockpitControls
from noflyzone.provenance import dump_json
from noflyzone.retina import HexRetina

OUT = Path(__file__).resolve().parent.parent / "out"
DT = 1.0 / 15.0
FRAME_H, FRAME_W = 210, 160
PX_PER_DEG = 2.2       # ~72.7 deg horizontal CRT viewport across 160 pixels
TURN_DEG_PER_TICK = 7.5
LOCK_DEG = 8.0         # crosshair firing window (+/- 8 deg ~ +/- 17.6 px)
ACTIONS = ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT",
           "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE",
           "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"]


def wrap180(deg: float) -> float:
    return float((deg + 180.0) % 360.0 - 180.0)


def render_target_scene(err_deg: float, tick: int, rng: np.random.Generator) -> np.ndarray:
    """Render a vector-Battlezone style CRT scene with a compact high-contrast target at `err_deg`.

    Background is horizontally uniform so zero lateral cue leaks from sky or horizon;
    target size (~22x26 px, ~2 ommatidia radius) matches `probe_frontend.py`'s small
    object stimulus projected onto the 160x210 cockpit CRT.
    """
    img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    img[36:96] = (6, 14, 10)
    img[96:182] = (10, 24, 14)
    img[95:97] = (28, 70, 34)  # faint uniform horizon line
    cx = round(FRAME_W / 2.0 + err_deg * PX_PER_DEG)
    jitter = int(rng.integers(-1, 2))
    # Subtle tread/turret vector flicker so stationary-in-fovea targets remain visible
    hi = 250 if (tick % 2 == 0) else 195
    x0, x1 = cx - 11 + jitter, cx + 11 + jitter
    if x1 > 0 and x0 < FRAME_W:
        xa, xb = max(0, x0), min(FRAME_W, x1)
        img[92:110, xa:xb] = (hi, 255, hi)
        txa, txb = max(0, cx - 6 + jitter), min(FRAME_W, cx + 6 + jitter)
        if txb > txa:
            img[84:92, txa:txb] = (220, 255, 255)
    return img


def reserved_trajectories(steps: int) -> dict[str, np.ndarray]:
    """Eight balanced left/right reserved target trajectories (degrees vs world north)."""
    ts = np.arange(steps, dtype=np.float64) * DT
    ramp_up = -30.0 + 60.0 * np.abs(((ts / 4.0) % 2.0) - 1.0)
    return {
        "step_right_24": np.where(ts < 0.3, 0.0, 24.0 + 4.0 * np.sin(2 * np.pi * 0.9 * ts)),
        "step_left_24": np.where(ts < 0.3, 0.0, -24.0 - 4.0 * np.sin(2 * np.pi * 0.9 * ts)),
        "drift_right": np.minimum(36.0, 8.0 + 4.5 * ts) + 3.0 * np.sin(2 * np.pi * 0.8 * ts),
        "drift_left": -np.minimum(36.0, 8.0 + 4.5 * ts) - 3.0 * np.sin(2 * np.pi * 0.8 * ts),
        "sine_pos": 14.0 + 18.0 * np.sin(2 * np.pi * 0.22 * ts),
        "sine_neg": -14.0 - 18.0 * np.sin(2 * np.pi * 0.22 * ts),
        "triangle_pos": ramp_up,
        "triangle_neg": -ramp_up,
    }


def parse_condition(token: str, default_readout: str) -> tuple[str, str, int, float, str]:
    m = re.fullmatch(r"(\w+?)(?:@(\d+))?(?:x([\d.]+))?(?::(\w+))?", token)
    if not m:
        raise SystemExit(f"bad condition token {token!r}")
    name, aseed, scale, readout = m.groups()
    ablate = "none" if name in ("measured", "connectome") else name
    return token, ablate, int(aseed or 0), float(scale or 1.0), (readout or default_readout)


def evaluate_brain_on_panel(
    core,
    cockpit: Cockpit,
    ret_l: HexRetina,
    ret_r: HexRetina,
    panel: dict[str, np.ndarray],
    injection: str,
    readout: str,
    neural_seed: int,
    steer_min_hz: float = 4.0,
    fovea_li: float = 0.22,
) -> list[dict]:
    cb = ConnectomeBrain(retina=ret_l, brain=core, injection=injection, readout=readout)
    rows = []
    for idx, (traj_name, target_deg) in enumerate(panel.items()):
        core._seed = neural_seed + idx * 17
        cb.reset()
        ctrl = CockpitControls(
            ACTIONS,
            calibration=readout,
            saccade_hold=1,
            startle_refractory=18,
            steer_mode="li",
            steer_min_hz=steer_min_hz,
            search_steer=0.0,  # pure tracking test: no blind search bias
            fovea_li=fovea_li,
            direct_steer=True,
        )
        rng = np.random.default_rng(7000 + idx)
        heading_deg = 0.0
        last_rot = False
        errs = []
        turn_events = []  # (sign(err), sign(turn)) when |err| > LOCK_DEG and turn != 0
        spiking = []

        for t, tgt in enumerate(target_deg):
            err = wrap180(float(tgt) - heading_deg)
            errs.append(err)
            frm = render_target_scene(err, t, rng)
            cockpit.set_screen(frm)
            el = cockpit.render("eye_left")
            er = cockpit.render("eye_right")
            drives = {
                "left": ret_l.photoreceptors(ret_l.sample(el)),
                "right": ret_r.photoreceptors(ret_r.sample(er)),
            }
            dn_rates = cb.step(drives, DT, signals={"self_turn": last_rot})
            spiking.append(int((cb.window_counts > 0).sum()))
            ctrl.update(dn_rates, DT)
            act_name = ACTIONS[ctrl.to_action()]
            turn = 1 if "RIGHT" in act_name else (-1 if "LEFT" in act_name else 0)
            if abs(err) > LOCK_DEG and turn != 0:
                turn_events.append((int(np.sign(err)), turn))
            heading_deg = wrap180(heading_deg + turn * TURN_DEG_PER_TICK)
            last_rot = (turn != 0)

        errs_arr = np.asarray(errs, dtype=np.float64)
        rms = float(np.sqrt(np.mean(errs_arr ** 2)))
        mae = float(np.mean(np.abs(errs_arr)))
        lock = float(np.mean(np.abs(errs_arr) <= LOCK_DEG))
        acc_l = [tr == -1 for s, tr in turn_events if s == -1]
        acc_r = [tr == 1 for s, tr in turn_events if s == 1]
        steer_acc = float(np.mean([r for r in (
            float(np.mean(acc_l)) if acc_l else np.nan,
            float(np.mean(acc_r)) if acc_r else np.nan,
        ) if np.isfinite(r)])) if (acc_l or acc_r) else float("nan")
        rows.append({
            "trajectory": traj_name,
            "steps": len(target_deg),
            "rms_error_deg": rms,
            "mae_deg": mae,
            "lock_frac": lock,
            "n_turns": len(turn_events),
            "steer_acc": steer_acc,
            "mean_spiking": float(np.mean(spiking)),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=BACKENDS, default="auto")
    ap.add_argument("--dataset", default="783")
    ap.add_argument("--steps", type=int, default=150, help="ticks per trajectory (10 s at 15 Hz)")
    ap.add_argument("--injection", default="v4")
    ap.add_argument("--readout", default="pool4")
    ap.add_argument("--zoom", type=float, default=1.75)
    ap.add_argument("--eye-offset", type=float, default=0.21)
    ap.add_argument("--center-y", type=float, default=0.48)
    ap.add_argument("--steer-min-hz", type=float, default=6.0)
    ap.add_argument("--fovea-li", type=float, default=0.22)
    ap.add_argument("--neural-seed", type=int, default=42)
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "measured",
            "rewired@0x0.7",
            "rewired@1x0.7",
            "rewired@2x0.7",
            "rewired_ipsi@0x0.7",
            "rewired_ipsi@1x0.7",
            "rewired_ipsi@0x0.7:pool_ipsi0",
            "silenced",
        ],
    )
    ap.add_argument("--out", default="bench_tracking.json")
    args = ap.parse_args()

    cockpit = Cockpit(eye_fovy=54.0, converge_eyes=True)
    ret_l = HexRetina(image_shape=(240, 320), zoom=args.zoom, center=(0.5 - args.eye_offset, args.center_y))
    ret_r = HexRetina(image_shape=(240, 320), zoom=args.zoom, center=(0.5 + args.eye_offset, args.center_y))
    panel = reserved_trajectories(args.steps)

    all_results = {}
    try:
        for tok in args.conditions:
            t0 = time.time()
            label, ablate, aseed, scale, readout = parse_condition(tok, args.readout)
            core = make_brain(
                args.backend,
                version=args.dataset,
                seed=args.neural_seed,
                ablate=ablate,
                ablate_seed=aseed,
                weight_scale=scale,
                aminergic="zero",
                sensory_input="external_only",
            )
            rows = evaluate_brain_on_panel(
                core, cockpit, ret_l, ret_r, panel, args.injection, readout, args.neural_seed,
                steer_min_hz=args.steer_min_hz, fovea_li=args.fovea_li,
            )
            rms_mean = float(np.mean([r["rms_error_deg"] for r in rows]))
            mae_mean = float(np.mean([r["mae_deg"] for r in rows]))
            lock_mean = float(np.mean([r["lock_frac"] for r in rows]))
            turns_tot = int(sum(r["n_turns"] for r in rows))
            valid_acc = [r["steer_acc"] for r in rows if np.isfinite(r["steer_acc"])]
            acc_mean = float(np.mean(valid_acc)) if valid_acc else float("nan")
            spk_mean = float(np.mean([r["mean_spiking"] for r in rows]))
            summary = {
                "brain": describe(core),
                "readout": readout,
                "rms_error_deg": rms_mean,
                "mae_deg": mae_mean,
                "lock_frac": lock_mean,
                "total_turns": turns_tot,
                "steer_acc": acc_mean,
                "mean_spiking": spk_mean,
                "wall_s": time.time() - t0,
            }
            all_results[label] = {"summary": summary, "trajectories": rows}
            print(
                f"{label:28s} | RMS {rms_mean:5.2f}° | MAE {mae_mean:5.2f}° | "
                f"lock {lock_mean:5.2f} | turns {turns_tot:3d} | acc {acc_mean:4.2f} | "
                f"spk/tick {spk_mean:5.0f} ({summary['wall_s']:.1f}s)",
                flush=True,
            )
            del core
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        cockpit.close()

    OUT.mkdir(exist_ok=True)
    out_path = OUT / args.out
    dump_json(out_path, {"args": vars(args), "panel_trajectories": list(panel), "results": all_results}, args=args)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
