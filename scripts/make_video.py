"""Cut shareable videos from dashboard recordings and their traces.

  highlights   the windows around kills and turns from one run, with title
               cards and event flashes
  sidebyside   the measured brain over the rewired control from the same
               seed, first-contact window, labelled

Every card states seed, speed, brain and model options; nothing is replay.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "out"
FPS = 15


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def card(size, lines, sub=None, seconds=3.0):
    w, h = size
    img = Image.new("RGB", (w, h), (8, 8, 12))
    d = ImageDraw.Draw(img)
    y = h // 2 - 22 * len(lines)
    for i, (text, px, col) in enumerate(lines):
        f = font(px)
        box = d.textbbox((0, 0), text, font=f)
        d.text(((w - box[2] + box[0]) / 2, y), text, font=f, fill=col)
        y += px + 14
    if sub:
        f = font(16)
        box = d.textbbox((0, 0), sub, font=f)
        d.text(((w - box[2] + box[0]) / 2, h - 44), sub, font=f, fill=(150, 150, 160))
    frame = np.asarray(img)
    return [frame] * int(seconds * FPS)


def overlay(frame, text, pos=(16, 16), px=22, col=(255, 200, 80)):
    img = Image.fromarray(frame.copy())
    d = ImageDraw.Draw(img, "RGBA")
    f = font(px)
    box = d.textbbox(pos, text, font=f)
    d.rectangle([box[0] - 8, box[1] - 6, box[2] + 8, box[3] + 6], fill=(0, 0, 0, 170))
    d.text(pos, text, font=f, fill=col)
    return np.asarray(img)


def highlights(video, trace, out_path):
    tr = json.load(open(trace))["trace"]
    kills = {t["tick"] for t in tr if t["reward"] > 0}
    deaths = {t["tick"] for t in tr if t["life_lost"]}
    turns = {t["tick"] for t in tr if "LEFT" in t["action"] or "RIGHT" in t["action"]}
    windows = []
    for k in sorted(kills):
        windows.append((max(0, k - 55), min(len(tr), k + 40), "kill"))
    for t in sorted(turns):
        if not any(a <= t < b for a, b, _ in windows):
            windows.append((max(0, t - 20), min(len(tr), t + 25), "turn"))
    windows.sort()
    merged = []
    for a, b, tag in windows:
        if merged and a <= merged[-1][1] + 15:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b), merged[-1][2] + "+" + tag)
        else:
            merged.append((a, b, tag))
    first = iio.imread(video, index=0)
    size = (first.shape[1], first.shape[0])
    frames = card(size, [("NO-FLY-ZONE", 54, (255, 120, 40)),
                         ("a real fruit-fly connectome pilots Atari Battlezone", 26, (230, 230, 230)),
                         ("FlyWire v783, 138,639 neurons, spiking on a GPU, through its own eyes", 20, (180, 180, 190))],
                  sub="closed loop, real time, seed 0, measured wiring, aminergic fast weights off, sensory input-only", seconds=3.5)
    for a, b, tag in merged:
        label = {"kill": "contact", "turn": "turning toward a target"}.get(tag.split("+")[0], "contact")
        for i in range(a, b):
            fr = iio.imread(video, index=i)
            if i - a < 22:
                fr = overlay(fr, f"{label}  ·  t = {i / FPS:.0f} s")
            for k in range(0, 15):
                if i - k in kills:
                    fr = overlay(fr, "KILL  +1000", pos=(16, size[1] - 200), px=30, col=(120, 255, 140))
                if i - k in deaths:
                    fr = overlay(fr, "HIT", pos=(16, size[1] - 200), px=30, col=(255, 90, 90))
            if i in turns or (i - 1) in turns:
                fr = overlay(fr, "turn", pos=(size[0] // 3 - 60, 16), px=22, col=(120, 200, 255))
            frames.append(fr)
    n_kills, n_deaths = len(kills), len(deaths)
    frames += card(size, [(f"{n_kills} kills, {n_deaths} lives lost in 100 seconds", 30, (230, 230, 230)),
                          ("a randomly rewired copy of the same brain scores the same", 24, (255, 200, 80)),
                          ("the loop is real; the wiring is not yet shown to matter", 24, (180, 180, 190))],
                   sub="github.com/JackDanHollister/no-fly-zone", seconds=4.0)
    w = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    for fr in frames:
        w.append_data(fr)
    w.close()
    return len(frames) / FPS, merged


PRESETS = {
    # the 2026-09-14 cut: measured wiring over its whole-brain rewiring
    "rewired": {
        "label_a": "MEASURED WIRING", "label_b": "REWIRED CONTROL",
        "title": [("same brain, same seed, same start", 40, (230, 230, 230)),
                  ("top: the measured FlyWire wiring", 26, (120, 255, 140)),
                  ("bottom: every synapse's target randomly permuted (degrees, counts and signs kept)", 22, (255, 200, 80))],
        "title_sub": "closed loop, real time, seed 0, the first 10 seconds after the first enemy appears",
        "closing": [("both fire at the first tank; both score", 32, (230, 230, 230)),
                    ("over ten seeds: measured 10 kills, rewired 10, silenced 0", 26, (180, 180, 190)),
                    ("the wiring is not yet shown to matter", 26, (255, 200, 80))],
        "start": 150, "end": 300,
    },
    # the 2026-09-18 cut: two individuals, two reconstructions, same game
    "fight": {
        "label_a": "FLYWIRE v783 (female)", "label_b": "MaleCNS v1.0 (male, weights x0.6)",
        "title": [("two flies, two connectomes, same game", 40, (230, 230, 230)),
                  ("top: FlyWire v783, female, 138,639 neurons, its own screened pools", 24, (120, 255, 140)),
                  ("bottom: MaleCNS v1.0, male, 166,700 neurons, its own screened pools", 24, (120, 200, 255))],
        "title_sub": "closed loop, real time, seed 0, 100 seconds; weights x0.6 puts the male in the female's firing regime",
        "closing": [("same cells carry the same signals in both flies", 30, (230, 230, 230)),
                    ("DNg100 drive, DNp12 loom, DNp07 side: found in both", 24, (180, 180, 190)),
                    ("neither a random graph nor a hemisphere-respecting one does this", 24, (255, 200, 80))],
        "start": 0, "end": 1500,
    },
}


def sidebyside(video_a, video_b, trace_a, trace_b, out_path, preset="rewired"):
    P = PRESETS[preset]
    start, end = P["start"], P["end"]
    ta = json.load(open(trace_a))["trace"]; tb = json.load(open(trace_b))["trace"]
    ka = {t["tick"] for t in ta if t["reward"] > 0}; kb = {t["tick"] for t in tb if t["reward"] > 0}
    da = {t["tick"] for t in ta if t["life_lost"]}; db = {t["tick"] for t in tb if t["life_lost"]}
    fa = iio.imread(video_a, index=0)
    w, h = fa.shape[1], fa.shape[0]
    size = (w, 2 * h)
    frames = card(size, P["title"], sub=P["title_sub"], seconds=3.5)
    col_a, col_b = (120, 255, 140), (120, 200, 255) if preset == "fight" else (255, 200, 80)
    # stream both videos once (random access re-decodes from the start each time)
    ra, rb = imageio.get_reader(video_a), imageio.get_reader(video_b)
    for i, (fa_i, fb_i) in enumerate(zip(ra, rb)):
        if i < start:
            continue
        if i >= end:
            break
        a = overlay(np.asarray(fa_i), P["label_a"], px=24, col=col_a)
        b = overlay(np.asarray(fb_i), P["label_b"], px=24, col=col_b)
        for k in range(0, 15):
            if i - k in da:
                a = overlay(a, "HIT", pos=(w - 120, h - 200), px=30, col=(255, 90, 90))
            if i - k in db:
                b = overlay(b, "HIT", pos=(w - 120, h - 200), px=30, col=(255, 90, 90))
        for k in range(0, 15):
            if i - k in ka:
                a = overlay(a, "KILL", pos=(16, h - 200), px=30, col=col_a)
            if i - k in kb:
                b = overlay(b, "KILL", pos=(16, h - 200), px=30, col=col_b)
        frames.append(np.concatenate([a, b], axis=0))
    frames += card(size, P["closing"], sub="github.com/JackDanHollister/no-fly-zone", seconds=4.0)
    wr = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    for fr in frames:
        wr.append_data(fr)
    wr.close()
    return len(frames) / FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["highlights", "sidebyside"])
    ap.add_argument("--video", default="out/m3_connectome_seed0.mp4")
    ap.add_argument("--trace", default="out/m3_connectome_seed0.json")
    ap.add_argument("--video-b", default="out/m3_rewired_seed0.mp4")
    ap.add_argument("--trace-b", default="out/m3_rewired_seed0.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preset", choices=list(PRESETS), default="rewired")
    args = ap.parse_args()
    if args.mode == "highlights":
        out = args.out or OUT / "no-fly-zone-highlights.mp4"
        secs, wins = highlights(args.video, args.trace, out)
        print(f"highlights: {secs:.1f} s, windows {wins} -> {out}")
    else:
        out = args.out or OUT / "no-fly-zone-vs-rewired.mp4"
        secs = sidebyside(args.video, args.video_b, args.trace, args.trace_b, out, preset=args.preset)
        print(f"side by side: {secs:.1f} s -> {out}")


if __name__ == "__main__":
    main()
