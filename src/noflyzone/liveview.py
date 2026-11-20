"""A live window for the dashboard, on the standard library only.

LiveView.show(frame) paints an RGB array into a tkinter window and pumps
its events; it returns False once the window has been closed or q pressed.
The caller controls pacing. Every submitted frame is painted; this class
does not drop frames or slow a fast simulation to game speed.
"""

from __future__ import annotations

import numpy as np


class LiveView:
    def __init__(self, title: str = "NO-FLY-ZONE", scale: float = 1.0):
        import tkinter as tk
        from PIL import Image, ImageTk
        self._tk, self._Image, self._ImageTk = tk, Image, ImageTk
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg="#101218")
        self.label = tk.Label(self.root, bd=0, bg="#101218")
        self.label.pack()
        self.scale = scale
        self.open = True
        self._photo = None
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Key-q>", lambda e: self._on_close())
        self.root.bind("<Escape>", lambda e: self._on_close())

    def _on_close(self) -> None:
        self.open = False

    def show(self, frame: np.ndarray) -> bool:
        if not self.open:
            return False
        img = self._Image.fromarray(np.ascontiguousarray(frame))
        if self.scale != 1.0:
            img = img.resize((int(img.width * self.scale), int(img.height * self.scale)))
        self._photo = self._ImageTk.PhotoImage(img)
        self.label.configure(image=self._photo)
        try:
            self.root.update_idletasks()
            self.root.update()
        except self._tk.TclError:   # window destroyed under us
            self.open = False
        return self.open

    def close(self) -> None:
        if self.open:
            self.open = False
        try:
            self.root.destroy()
        except self._tk.TclError:
            pass
