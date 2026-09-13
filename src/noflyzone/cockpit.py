"""Cockpit layer: MuJoCo scene with a CRT screen the fly's eye cameras watch,
tread levers, a fire button, and a damage lamp.

The game frame is written into the model's texture data; any in-scene camera
then sees the live game. macOS requires GL on the main thread, so all
rendering here is designed for a single-threaded step/upload/render loop.
Wired mode puppets the controls kinematically (qpos set directly).
"""

import mujoco
import numpy as np
from PIL import Image

TEX_SIZE = 256
LEVER_RANGE = 0.55  # rad of hinge travel for |value| = 1

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
    <material name="metal" rgba="0.55 0.58 0.62 1" specular="0.8" shininess="0.6"/>
    <material name="knob" rgba="0.6 0.08 0.08 1"/>
  </asset>
  <worldbody>
    <light pos="0 -0.2 0.5" dir="0 0.4 -1" diffuse="0.35 0.35 0.45"/>
    <geom name="screen" type="box" pos="0 0.25 0.12" size="0.16 0.004 0.12"
          material="crt_mat"/>
    <geom name="bezel" type="box" pos="0 0.258 0.12" size="0.175 0.003 0.135"
          material="trim"/>
    <geom name="dash" type="box" pos="0 0.22 -0.04" size="0.22 0.06 0.035"
          material="hull"/>
    <!-- damage lamp sits above the screen, inside the eye cameras' view, so
         a flash is a real visual stimulus for the fly, not decoration -->
    <geom name="lamp" type="sphere" pos="0 0.25 0.27" size="0.016"
          rgba="0.25 0.05 0.05 1"/>
    <body name="lever_left" pos="-0.09 0.15 -0.005">
      <joint name="lever_left" type="hinge" axis="1 0 0"
             range="-{LEVER_RANGE} {LEVER_RANGE}" limited="true"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.085" size="0.007" material="metal"/>
      <geom type="sphere" pos="0 0 0.09" size="0.013" material="knob"/>
    </body>
    <body name="lever_right" pos="0.09 0.15 -0.005">
      <joint name="lever_right" type="hinge" axis="1 0 0"
             range="-{LEVER_RANGE} {LEVER_RANGE}" limited="true"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.085" size="0.007" material="metal"/>
      <geom type="sphere" pos="0 0 0.09" size="0.013" material="knob"/>
    </body>
    <body name="fire_button" pos="0 0.15 -0.002">
      <joint name="fire_button" type="slide" axis="0 0 1" range="-0.008 0" limited="true"/>
      <geom type="cylinder" size="0.011 0.006" rgba="0.85 0.1 0.1 1"/>
    </body>
    <camera name="eye_left" pos="-0.03 -0.06 0.16" xyaxes="1 0 0 0 0.128 0.992"/>
    <camera name="eye_right" pos="0.03 -0.06 0.16" xyaxes="1 0 0 0 0.128 0.992"/>
    <camera name="beauty" pos="-0.25 -0.28 0.22" xyaxes="0.8 -0.6 0 0.18 0.24 0.95"/>
  </worldbody>
</mujoco>
"""

_LAMP_OFF = np.array([0.25, 0.05, 0.05, 1.0])
_LAMP_ON = np.array([1.0, 0.12, 0.08, 1.0])


class Cockpit:
    def __init__(self, render_width: int = 320, render_height: int = 240):
        self.render_width = render_width
        self.render_height = render_height
        self.model = mujoco.MjModel.from_xml_string(_XML)
        self.data = mujoco.MjData(self.model)
        self._tex_id = self.model.texture("crt").id
        adr = self.model.tex_adr[self._tex_id]
        self._tex_slice = slice(adr, adr + TEX_SIZE * TEX_SIZE * 3)
        self._lamp_id = self.model.geom("lamp").id
        self._jnt = {name: self.model.joint(name).qposadr[0]
                     for name in ("lever_left", "lever_right", "fire_button")}
        mujoco.mj_forward(self.model, self.data)
        self._renderer = mujoco.Renderer(self.model, height=render_height,
                                         width=render_width)

    def set_screen(self, frame: np.ndarray) -> None:
        """Write an RGB game frame (any size) into the CRT texture."""
        img = Image.fromarray(frame).resize((TEX_SIZE, TEX_SIZE))
        # tex_data row 0 renders at the top — upload as-is, no flip
        pixels = np.asarray(img, dtype=np.uint8)
        self.model.tex_data[self._tex_slice] = pixels.ravel()
        mujoco.mjr_uploadTexture(self.model, self._renderer._mjr_context,
                                 self._tex_id)

    def set_controls(self, lever_left: float, lever_right: float,
                     fire: bool) -> None:
        """Puppet the controls (wired mode). Lever values in [-1, 1]."""
        self.data.qpos[self._jnt["lever_left"]] = lever_left * LEVER_RANGE
        self.data.qpos[self._jnt["lever_right"]] = lever_right * LEVER_RANGE
        self.data.qpos[self._jnt["fire_button"]] = -0.008 if fire else 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_lamp(self, on: bool) -> None:
        self.model.geom_rgba[self._lamp_id] = _LAMP_ON if on else _LAMP_OFF

    def render(self, camera: str = "eye_left") -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()
