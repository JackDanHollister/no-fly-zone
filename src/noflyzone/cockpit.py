"""Cockpit layer: MuJoCo scene with a CRT screen the fly's eye cameras watch.

The game frame is written into the model's texture data; any in-scene camera
then sees the live game. macOS requires GL on the main thread, so all
rendering here is designed for a single-threaded step/upload/render loop.
"""

import mujoco
import numpy as np
from PIL import Image

TEX_SIZE = 256

_XML = f"""
<mujoco model="no-fly-zone-cockpit">
  <visual>
    <headlight ambient="0.08 0.08 0.1" diffuse="0.2 0.2 0.25"/>
  </visual>
  <asset>
    <texture name="crt" type="2d" builtin="flat" rgb1="0 0 0"
             width="{TEX_SIZE}" height="{TEX_SIZE}"/>
    <material name="crt_mat" texture="crt" emission="1" specular="0" shininess="0"/>
    <material name="hull" rgba="0.13 0.14 0.18 1"/>
    <material name="trim" rgba="0.9 0.3 0.1 1" emission="0.6"/>
  </asset>
  <worldbody>
    <light pos="0 -0.2 0.5" dir="0 0.4 -1" diffuse="0.35 0.35 0.45"/>
    <geom name="screen" type="box" pos="0 0.25 0.12" size="0.16 0.004 0.12"
          material="crt_mat"/>
    <geom name="bezel" type="box" pos="0 0.258 0.12" size="0.175 0.003 0.135"
          material="trim"/>
    <geom name="dash" type="box" pos="0 0.22 -0.04" size="0.22 0.06 0.035"
          material="hull"/>
    <camera name="eye_left" pos="-0.03 -0.05 0.10" xyaxes="1 0 0 0 0 1"/>
    <camera name="eye_right" pos="0.03 -0.05 0.10" xyaxes="1 0 0 0 0 1"/>
    <camera name="beauty" pos="-0.25 -0.28 0.22" xyaxes="0.8 -0.6 0 0.18 0.24 0.95"/>
  </worldbody>
</mujoco>
"""


class Cockpit:
    def __init__(self, render_width: int = 320, render_height: int = 240):
        self.model = mujoco.MjModel.from_xml_string(_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self._tex_id = self.model.texture("crt").id
        adr = self.model.tex_adr[self._tex_id]
        self._tex_slice = slice(adr, adr + TEX_SIZE * TEX_SIZE * 3)
        self._renderer = mujoco.Renderer(self.model, height=render_height,
                                         width=render_width)

    def set_screen(self, frame: np.ndarray) -> None:
        """Write an RGB game frame (any size) into the CRT texture."""
        img = Image.fromarray(frame).resize((TEX_SIZE, TEX_SIZE))
        # GL textures are bottom-up; flip so the game reads right way up
        pixels = np.flipud(np.asarray(img, dtype=np.uint8))
        self.model.tex_data[self._tex_slice] = pixels.ravel()
        mujoco.mjr_uploadTexture(self.model, self._renderer._mjr_context,
                                 self._tex_id)

    def render(self, camera: str = "eye_left") -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()
