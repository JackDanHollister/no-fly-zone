"""Presentation layout: fly-in-cockpit | game screen, controls strip below.

compose() builds one video frame:
  +----------------------+----------------------+
  |  fly in the cockpit  |     the raw game     |
  +----------------------+----------------------+
  |  treads | DN rates | FIRE / startle / score |
  +---------------------------------------------+
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


def compose(beauty: np.ndarray, game: np.ndarray, controls, rates: dict,
            score: float, t: float, brain: np.ndarray | None = None,
            brain_label: str = "") -> np.ndarray:
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
    draw.text((x0 + 90, top + 112),
              "NO-FLY-ZONE  //  MaleCNS pilot: reflex stub", font=FONT, fill=DIM)

    return np.asarray(canvas)
