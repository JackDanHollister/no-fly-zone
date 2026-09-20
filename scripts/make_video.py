"""Cut shareable videos from dashboard recordings and their traces.

  highlights   the windows around kills and turns from one run, with title
               cards and event flashes
  sidebyside   the measured brain over the rewired control from the same
               seed, first-contact window, labelled
  showcase     the 2026-09-20 cut: the female's contacts, the embodied pilot
               working physical levers, the two flies side by side, and the
               measured findings on cards; from the seed-0 recordings in out/

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


def _letterbox(frame, size, caption=None, sub=None):
    """A dashboard frame centred on the showcase canvas, caption above,
    a smaller line below."""
    W, H = size
    canvas = Image.new("RGB", (W, H), (8, 8, 12))
    img = Image.fromarray(frame)
    if img.width != W:
        img = img.resize((W, int(img.height * W / img.width)))
    y0 = (H - img.height) // 2
    canvas.paste(img, (0, y0))
    d = ImageDraw.Draw(canvas)
    if caption:
        f = font(30); box = d.textbbox((0, 0), caption, font=f)
        d.text(((W - box[2] + box[0]) / 2, y0 // 2 - 18), caption, font=f, fill=(255, 200, 80))
    if sub:
        f = font(20); box = d.textbbox((0, 0), sub, font=f)
        d.text(((W - box[2] + box[0]) / 2, y0 + img.height + (H - y0 - img.height) // 2 - 10), sub, font=f, fill=(170, 170, 180))
    return np.asarray(canvas)


def _windows(tr, before=55, after=40, max_windows=3):
    kills = sorted({t["tick"] for t in tr if t["reward"] > 0})
    wins = [(max(0, k - before), min(len(tr), k + after)) for k in kills][:max_windows]
    if not wins:   # no kill: the first startle
        d = [t["tick"] for t in tr if t["life_lost"]]
        wins = [(max(0, d[0] - 60), min(len(tr), d[0] + 30))] if d else [(150, 300)]
    return wins


def _events(tr):
    return ({t["tick"] for t in tr if t["reward"] > 0}, {t["tick"] for t in tr if t["life_lost"]})


def _clip(video, tr, wins, size, caption, sub, fire_key=None):
    kills, deaths = _events(tr)
    reader = imageio.get_reader(video)
    lo, hi = min(a for a, _ in wins), max(b for _, b in wins)
    out = []
    for i, fr in enumerate(reader):
        if i < lo:
            continue
        if i >= hi:
            break
        if not any(a <= i < b for a, b in wins):
            continue
        fr = np.asarray(fr)
        for k in range(0, 15):
            if i - k in kills:
                fr = overlay(fr, "KILL  +1000", pos=(16, fr.shape[0] - 200), px=30, col=(120, 255, 140))
            if i - k in deaths:
                fr = overlay(fr, "HIT", pos=(16, fr.shape[0] - 200), px=30, col=(255, 90, 90))
        if fire_key and tr[i].get(fire_key):
            fr = overlay(fr, "proboscis on the button", pos=(fr.shape[1] // 3 - 60, 16), px=22, col=(255, 120, 40))
        out.append(_letterbox(fr, size, caption, sub))
    reader.close()
    return out


def showcase(out_path):
    W, H = 1440, 1024
    size = (W, H)
    female_v, female_t = OUT / "fight_female_seed0.mp4", OUT / "fight_female_seed0.json"
    embodied_v, embodied_t = OUT / "embodied_seed0.mp4", OUT / "embodied_seed0.json"
    male_v, male_t = OUT / "fight_male_seed0.mp4", OUT / "fight_male_seed0.json"
    tf = json.load(open(female_t))["trace"]; te = json.load(open(embodied_t))["trace"]; tm = json.load(open(male_t))["trace"]
    frames = card(size, [("NO-FLY-ZONE", 64, (255, 120, 40)),
                         ("a real fruit-fly connectome pilots Atari Battlezone", 30, (230, 230, 230)),
                         ("FlyWire v783: 138,639 neurons, 15.1 million synapses, spiking on a GPU", 22, (180, 180, 190)),
                         ("through its own eyes, out of its own descending neurons", 22, (180, 180, 190))],
                  sub="everything here is closed loop and real time; nothing is replay; every number was measured", seconds=4.5)
    frames += card(size, [("1. the pilot", 40, (255, 200, 80)),
                          ("the game on a CRT, two compound eyes watching it, the whole brain in between", 24, (230, 230, 230)),
                          ("bottom right: what each eye sampled this tick", 22, (180, 180, 190))],
                   sub="seed 0, the measured wiring, aminergic fast weights off, sensory input-only", seconds=3.5)
    frames += _clip(str(female_v), tf, _windows(tf), size, "the measured wiring, first contacts",
                    "DNa02 pools steer, DNg100 drives, DNp01 backs off from a loom; pIP10 (oviDN) fires")
    frames += card(size, [("2. embodied", 40, (255, 200, 80)),
                          ("the same brain, but now the fly's own legs push physical levers", 24, (230, 230, 230)),
                          ("claws adhere, levers spring back, the proboscis presses the fire button", 22, (180, 180, 190)),
                          ("the game reads the levers, not the neurons", 22, (180, 180, 190))],
                   sub="MuJoCo at 0.1 ms, the flybody model, real time on the workstation", seconds=4.0)
    frames += _clip(str(embodied_v), te, _windows(te, max_windows=2), size, "embodied: legs on the levers",
                    "treads show the physical levers; a lever that lags is pushed harder", fire_key="fire")
    frames += card(size, [("3. two flies", 40, (255, 200, 80)),
                          ("top: FlyWire v783, female. bottom: MaleCNS v1.0, male, 166,700 neurons", 24, (230, 230, 230)),
                          ("each placed on the retina by its own optic-lobe columns", 22, (180, 180, 190))],
                   sub="seed 0, the same game; weights x0.6 puts the male in the female's firing regime", seconds=3.5)
    kf = sorted({t["tick"] for t in tf if t["reward"] > 0}); k0 = kf[0] if kf else 200
    a0, a1 = max(0, k0 - 60), min(1500, k0 + 45)
    ra, rb = imageio.get_reader(str(female_v)), imageio.get_reader(str(male_v))
    kfa, dfa = _events(tf); kfb, dfb = _events(tm)
    for i, (fa_i, fb_i) in enumerate(zip(ra, rb)):
        if i < a0:
            continue
        if i >= a1:
            break
        a = overlay(np.asarray(fa_i), "FLYWIRE v783 (female)", px=24, col=(120, 255, 140))
        b = overlay(np.asarray(fb_i), "MaleCNS v1.0 (male)", px=24, col=(120, 200, 255))
        for k in range(0, 15):
            if i - k in kfa: a = overlay(a, "KILL", pos=(16, a.shape[0] - 200), px=30, col=(120, 255, 140))
            if i - k in kfb: b = overlay(b, "KILL", pos=(16, b.shape[0] - 200), px=30, col=(120, 200, 255))
            if i - k in dfa: a = overlay(a, "HIT", pos=(a.shape[1] - 120, a.shape[0] - 200), px=30, col=(255, 90, 90))
            if i - k in dfb: b = overlay(b, "HIT", pos=(b.shape[1] - 120, b.shape[0] - 200), px=30, col=(255, 90, 90))
        frames.append(np.concatenate([a, b], axis=0))
    ra.close(); rb.close()
    frames += card(size, [("what the wiring does, measured against rewired copies of itself", 30, (230, 230, 230)),
                          ("keeps left from right: steers toward the target on 62 of 63 looks; rewirings at chance", 22, (120, 255, 140)),
                          ("routes vision to 58 descending types, not 355; 14 prefer expansion to contraction", 22, (120, 255, 140)),
                          ("given direction at T4/T5, builds HS, VS and ten direction-selective descending neurons", 22, (120, 255, 140)),
                          ("given a small-object channel at T2/T3, builds 14 small-object-selective ones; rewirings none", 22, (120, 255, 140))],
                   sub="docs/research/ in the repository has every control", seconds=6.0)
    frames += card(size, [("what it does not do", 30, (230, 230, 230)),
                          ("a random rewiring kills as often; the game's reinforcers pay for none of the above", 22, (255, 200, 80)),
                          ("a trained readout learns to survive, not to kill, on any wiring", 22, (255, 200, 80)),
                          ("a plasticity rule reaches the game and makes the fly worse: the shell that hits it is never seen", 22, (255, 200, 80)),
                          ("nothing here learns yet; nothing here is conscious; no token, no coin, ever", 22, (180, 180, 190))],
                   sub="github.com/JackDanHollister/no-fly-zone", seconds=6.0)
    wr = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    for fr in frames:
        wr.append_data(fr)
    wr.close()
    return len(frames) / FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["highlights", "sidebyside", "showcase"])
    ap.add_argument("--video", default="out/m3_connectome_seed0.mp4")
    ap.add_argument("--trace", default="out/m3_connectome_seed0.json")
    ap.add_argument("--video-b", default="out/m3_rewired_seed0.mp4")
    ap.add_argument("--trace-b", default="out/m3_rewired_seed0.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preset", choices=list(PRESETS), default="rewired")
    args = ap.parse_args()
    if args.mode == "showcase":
        out = args.out or OUT / "no-fly-zone-showcase.mp4"
        print(f"showcase: {showcase(out):.1f} s -> {out}")
        return
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
