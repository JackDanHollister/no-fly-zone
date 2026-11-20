"""One-command local launcher for the whole-brain Battlezone pilot."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHAMPION_CKPTS = (
    "out/readout_champion_connectome_gru_best.pt",
    "out/readout_champion_connectome_gru.pt",
)
ASSETS = (
    "assets/flybody/fruitfly.xml",
    "data/shiu_model/Completeness_783.csv",
    "data/shiu_model/Connectivity_783.parquet",
    "data/flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
)
MODULES = ("noflyzone", "numpy", "scipy", "pandas", "pyarrow", "mujoco",
           "gymnasium", "ale_py", "PIL", "imageio", "imageio_ffmpeg")


def desktop_available():
    return sys.platform == "darwin" or bool(os.environ.get("DISPLAY"))


def resolve_champion_ckpt() -> Path | None:
    for rel in DEFAULT_CHAMPION_CKPTS:
        p = ROOT / rel
        if p.is_file():
            return p
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Default: watch on a desktop, record without a display. Every run saves an MP4 "
        "and a trace in out/. Press Q or Escape, or close the window, to finish early."))
    parser.add_argument("mode", nargs="?", choices=("watch", "record", "doctor", "status"))
    parser.add_argument("--seconds", type=float, default=60, help="game seconds (default: 60)")
    parser.add_argument("--embodied", action="store_true", help="use the fly's physical lever controls")
    parser.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=0, help="game and neural seed (default: 0)")
    parser.add_argument("--controller", choices=("active", "auto", "learned", "legacy"), default="active",
                        help="active (default): pure frozen-connectome descending-neuron hunter (zero RL); "
                             "learned: optional trained GRU readout; legacy: Sept 25 pilot preset")
    parser.add_argument("--ckpt", default=None, help="explicit trained readout checkpoint (.pt)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds < 1 / 15:
        parser.error("--seconds must be finite and at least 1/15 second")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.ckpt and not Path(args.ckpt).is_file():
        parser.error(f"checkpoint not found: {args.ckpt}")
    if args.controller == "learned" and not args.ckpt and resolve_champion_ckpt() is None:
        parser.error("no trained readout checkpoint found; run train_readout.py or pass --ckpt")
    if args.mode is None:
        args.mode = "watch" if desktop_available() else "record"
    if args.mode == "watch" and not desktop_available():
        parser.error("no desktop display is available; use './no-fly-zone record'")
    return args


def doctor(backend="auto", live=False):
    """Check prerequisites, the actual renderer and game before brain loading."""
    missing = [module for module in MODULES if importlib.util.find_spec(module) is None]
    absent = [path for path in ASSETS if not (ROOT / path).is_file()]
    if missing:
        print("Missing Python packages: " + ", ".join(missing), file=sys.stderr)
        print('Install: .venv/bin/python -m pip install -e ".[dev]" -c constraints.txt', file=sys.stderr)
    if absent:
        print("Missing assets: " + ", ".join(absent), file=sys.stderr)
        print("Fetch: .venv/bin/python scripts/fetch_assets.py", file=sys.stderr)
    if missing or absent:
        return False
    if live:
        try:
            import tkinter
            from PIL import ImageTk
        except ImportError:
            print("Live viewing needs tkinter for this Python; use 'record' to save a video.", file=sys.stderr)
            return False
    from noflyzone.brain_backend import resolve_backend
    selected = resolve_backend(backend)
    if selected == "cuda":
        try:
            import torch
            import triton
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device unavailable")
            device = torch.cuda.get_device_name(0)
        except (ImportError, RuntimeError) as error:
            print(f"CUDA is unavailable: {error}. Use --backend cpu or install the CUDA extras.", file=sys.stderr)
            return False
        print(f"Brain: CUDA on {device}", flush=True)
    else:
        print("Brain: CPU (slower than game speed; saved video still plays at normal speed)", flush=True)
    try:
        import imageio_ffmpeg
        import mujoco
        from noflyzone.game import BattleZone
        if not shutil.which(imageio_ffmpeg.get_ffmpeg_exe()):
            raise RuntimeError("FFmpeg encoder is unavailable")
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
        with mujoco.Renderer(model, height=32, width=32) as renderer:
            renderer.update_scene(mujoco.MjData(model))
            renderer.render()
        game = BattleZone()
        game.close()
    except Exception as error:
        print(f"Renderer/game check failed: {error}", file=sys.stderr)
        return False
    print("Ready: assets, renderer, Battlezone and video encoder.", flush=True)
    return True


def select_preset_and_ckpt(args) -> tuple[str, str | None]:
    controller = getattr(args, "controller", "active")
    explicit_ckpt = getattr(args, "ckpt", None)
    if explicit_ckpt:
        return "active", explicit_ckpt
    if controller == "legacy":
        return "pilot", None
    if controller == "learned":
        champ = resolve_champion_ckpt()
        if champ is not None:
            return "active", str(champ)
    return "active", None


def command(args, name):
    preset, ckpt = select_preset_and_ckpt(args)
    result = [sys.executable, str(ROOT / "scripts/run_m3.py"), "--preset", preset,
              "--backend", args.backend, "--seed", str(args.seed),
              "--steps", str(round(args.seconds * 15)), "--video", "--video-name", name,
              "--trace", name, "--quiet", "--pilot", "embodied" if args.embodied else "wired"]
    if ckpt is not None:
        result.extend(["--readout-ckpt", ckpt])
    if args.seed > 0:
        result.append("--warmup")
    if args.mode == "watch":
        result.extend(["--live", "--realtime"])
    return result


def campaign_status():
    manifests = list((ROOT / "out").glob("campaign_*/manifest.json"))
    if manifests:
        path = max(manifests, key=lambda p: p.stat().st_mtime)
        record = json.loads(path.read_text())
        training = sum(len(run["episodes"]) for run in record["training"].values())
        plan = record["plan"]
        per_panel = ((1 + 2 * len(plan["training"]["seeds"])) *
                     len(plan["evaluation"]["panels"][0]["seeds"]) *
                     plan["evaluation"]["episodes_per_fly"])
        evaluated = len(record["evaluation"]) * per_panel
        current = record.get("current", {}).get("name", "start audit")
        if record["status"] == "training" and current not in record["training"]:
            log = ROOT / "out" / f"plasticity_{current}.jsonl"
            if log.exists():
                training += log.read_text().count("\n")
        if record["status"] == "evaluating" and current.removeprefix("eval_") not in record["evaluation"]:
            output = path.parent / f"{current}.json"
            if output.exists():
                try:
                    payload = json.loads(output.read_text())
                    evaluated += sum(len(c["episodes"]) for c in payload["conditions"].values())
                    log = path.parent / f"{current}.log"
                    if log.exists():
                        block = log.read_text().rsplit("Evaluating ", 1)[-1]
                        condition = block.splitlines()[0] if block else ""
                        progress = re.findall(r"^\s+(\d+)/(\d+) episodes:", block, re.MULTILINE)
                        if condition not in payload["conditions"] and progress:
                            evaluated += int(progress[-1][0])
                except json.JSONDecodeError:
                    pass
        print(f"Learning campaign (plasticity): {record['status']}")
        print(f"Training: {training}/{plan['budget']['training_episodes']} episodes; "
              f"evaluation: {evaluated}/{plan['budget']['evaluation_episodes']} episodes.")
        if record["status"] not in ("complete", "incomplete"):
            print(f"Current: {current.removeprefix(plan['campaign'] + '_')}")
        if "decision" in record:
            print(record["decision"])
        if "error" in record:
            print(record["error"])
        print(f"Full record: {path}")
    else:
        print("No plasticity campaign has run here yet.")

    champ_log = ROOT / "out" / "readout_champion_connectome_gru.jsonl"
    if champ_log.is_file():
        lines = [ln for ln in champ_log.read_text().splitlines() if ln.strip()]
        if lines:
            last = json.loads(lines[-1])
            champ_ckpt = resolve_champion_ckpt()
            print(
                f"\nTrained readout (GRU): update {last['update']} ({last.get('phase', 'ppo')}), "
                f"{last['ticks']:,} fly-ticks, {last['episodes']} episodes; "
                f"rolling kills {last.get('ep_kills')}, lives {last.get('ep_lives')}."
            )
            if champ_ckpt is not None:
                print(f"Optional RL readout checkpoint (--controller learned): {champ_ckpt}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "status":
        return campaign_status()
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    if not doctor(args.backend, live=args.mode == "watch"):
        return 1
    if args.mode == "doctor":
        return 0
    preset, ckpt = select_preset_and_ckpt(args)
    label = (f"Whole-brain trained readout ({Path(ckpt).stem})"
             if ckpt else f"Whole-brain {preset} reflex pilot")
    name = "pilot_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    print(f"{label} | {'physical' if args.embodied else 'direct'} lever controls | "
          f"{args.seconds:g} game seconds", flush=True)
    print(f"Recording and trace: {ROOT / 'out' / name}.mp4 / .json", flush=True)
    if args.mode == "watch":
        print("Press Q or Escape, or close the window, to finish and keep the recording.", flush=True)
    os.chdir(ROOT)
    os.execv(sys.executable, command(args, name))


if __name__ == "__main__":
    raise SystemExit(main())
