"""Presentation layout: fly-in-cockpit | game screen, controls strip below.

compose() builds one video frame:
  +----------------------+----------------------+----------------------+
  |  fly in the cockpit  |     the raw game     |  connectome (spikes) |
  +----------------------+----------------------+----------------------+
  |  treads | DN rates | FIRE / startle / score |  eyes: L / R mosaics |
  +---------------------------------------------+----------------------+
The eyes panel (optional) is what the retinas actually sampled this tick,
reconstructed as the two hex mosaics, with each eye's transient energy.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import dn

PANEL_W, PANEL_H = 480, 360
STRIP_H = 150
W, H = PANEL_W * 2, PANEL_H + STRIP_H

BG = (16, 18, 24)
PANEL_BG = (8, 9, 12)
FG = (200, 205, 215)
DIM = (110, 115, 130)
ACCENT = (235, 90, 40)
GOOD = (90, 200, 120)

try:
    FONT = ImageFont.load_default(size=13)
    FONT_BIG = ImageFont.load_default(size=20)
except TypeError:  # older Pillow
    FONT = FONT_BIG = ImageFont.load_default()


def _fit(img: np.ndarray, w: int, h: int, pixelated: bool) -> Image.Image:
    mode = Image.NEAREST if pixelated else Image.BILINEAR
    return Image.fromarray(img).resize((w, h), mode)


def _bar_h(draw, x, y, w, h, frac, color, label, text):
    draw.rectangle([x, y, x + w, y + h], outline=DIM)
    fw = int(w * max(0.0, min(1.0, frac)))
    if fw:
        draw.rectangle([x, y, x + fw, y + h], fill=color)
    draw.text((x + w + 8, y - 2), f"{label} {text}", font=FONT, fill=FG)


def eyes_panel(retina, retina_right, rgb: dict, prev_rgb: dict | None,
               w: int = PANEL_W, h: int = STRIP_H) -> np.ndarray:
    """The two retinal mosaics of this tick side by side, each with its
    summed |delta luminance| against the previous tick (the transient the
    brain is injected with) as a bar."""
    panel = Image.new("RGB", (w, h), PANEL_BG)
    draw = ImageDraw.Draw(panel)
    half = w // 2
    mh = h - 40
    for i, (side, ret) in enumerate((("left", retina), ("right", retina_right))):
        mosaic = ret.reconstruct(rgb[side])
        ys, xs = np.nonzero(mosaic.any(axis=2))      # crop to the lattice's footprint
        if ys.size:
            mosaic = mosaic[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        mw = int(mh * mosaic.shape[1] / mosaic.shape[0])
        x0 = i * half + (half - mw) // 2
        panel.paste(Image.fromarray(mosaic).resize((mw, mh), Image.NEAREST), (x0, 22))
        lum = rgb[side][:, 0].astype(np.float32)
        energy = float(np.abs(lum - prev_rgb[side][:, 0]).mean() / 255.0) if prev_rgb is not None else 0.0
        draw.text((i * half + 10, 6), f"EYE {side[0].upper()}   {ret.n_ommatidia} ommatidia", font=FONT, fill=DIM)
        _bar_h(draw, i * half + 10, h - 14, half - 130, 8, min(1.0, energy * 8), ACCENT, "transient", f"{energy:.3f}")
    return np.asarray(panel)


def compose(beauty: np.ndarray, game: np.ndarray, controls, rates: dict,
            score: float, t: float, brain: np.ndarray | None = None,
            brain_label: str = "",
            footer: str = "NO-FLY-ZONE", eyes: np.ndarray | None = None) -> np.ndarray:
    n_panels = 3 if brain is not None else 2
    width = PANEL_W * n_panels
    canvas = Image.new("RGB", (width, PANEL_H + STRIP_H), BG)
    canvas.paste(_fit(beauty, PANEL_W, PANEL_H, False), (0, 0))
    canvas.paste(_fit(game, PANEL_W, PANEL_H, True), (PANEL_W, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "PILOT", font=FONT, fill=DIM)
    draw.text((PANEL_W + 10, 8), "BATTLEZONE", font=FONT, fill=DIM)
    if brain is not None:
        canvas.paste(_fit(brain, PANEL_W, PANEL_H, False), (2 * PANEL_W, 0))
        draw.text((2 * PANEL_W + 10, 8), "CONNECTOME", font=FONT, fill=DIM)
        if brain_label:
            draw.text((2 * PANEL_W + 10, PANEL_H - 22), brain_label,
                      font=FONT, fill=FG)

    top = PANEL_H
    draw.rectangle([0, top, width, PANEL_H + STRIP_H], fill=PANEL_BG)

    # treads: two vertical levers, centre line = neutral
    draw.text((20, top + 10), "TREADS", font=FONT, fill=DIM)
    for i, val in enumerate((controls.lever_left, controls.lever_right)):
        x = 30 + i * 34
        y0, y1 = top + 32, H - 18
        mid = (y0 + y1) // 2
        draw.rectangle([x, y0, x + 16, y1], outline=DIM)
        draw.line([x - 3, mid, x + 19, mid], fill=DIM)
        tip = int(mid - val * (y1 - y0) / 2)
        draw.rectangle([x + 2, min(mid, tip), x + 14, max(mid, tip)],
                       fill=ACCENT)

    # descending-neuron rates, normalized against the dn.py calibration
    draw.text((130, top + 10), "DESCENDING NEURONS", font=FONT, fill=DIM)
    drives = dn.normalize(rates)
    for i, name in enumerate(dn.DN_NAMES):
        y = top + 32 + i * 21
        _bar_h(draw, 130, y, 130, 12, drives[name], ACCENT,
               f"{name:<8}", f"{rates[name]:5.0f} Hz")

    # status cluster
    x0 = 420
    firing = controls.fire
    draw.ellipse([x0, top + 26, x0 + 44, top + 70],
                 fill=(220, 40, 30) if firing else (60, 22, 20),
                 outline=DIM)
    draw.text((x0 + 6, top + 76), "FIRE", font=FONT,
              fill=FG if firing else DIM)
    draw.text((x0 + 90, top + 20), f"SCORE {score:6.0f}", font=FONT_BIG, fill=GOOD)
    draw.text((x0 + 90, top + 52),
              f"steer    {controls.steer:+.2f}", font=FONT, fill=FG)
    draw.text((x0 + 90, top + 70),
              f"throttle {controls.throttle:+.2f}", font=FONT, fill=FG)
    draw.text((x0 + 90, top + 88), f"t = {t:6.1f} s", font=FONT, fill=DIM)
    # the footer wraps to two lines within the strip left of the eyes panel
    max_w = (2 * PANEL_W if eyes is not None and n_panels == 3 else width) - (x0 + 90) - 10
    for j, line in enumerate(_wrap(draw, footer, max_w, 2)):
        draw.text((x0 + 90, top + 110 + j * 16), line, font=FONT, fill=DIM)
    if eyes is not None and n_panels == 3:
        canvas.paste(Image.fromarray(eyes).resize((PANEL_W, STRIP_H)), (2 * PANEL_W, top))

    return np.asarray(canvas)


def _wrap(draw, text: str, max_w: int, max_lines: int) -> list[str]:
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=FONT) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur); cur = w
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and draw.textlength(lines[-1] + " ...", font=FONT) > max_w:
            lines[-1] = lines[-1][:-1]
        lines[-1] = lines[-1].rstrip() + " ..."
    return lines
