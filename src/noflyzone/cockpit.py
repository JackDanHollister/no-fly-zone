"""Cockpit layer: MuJoCo scene with a CRT screen, tread levers, a fire button,
a damage lamp — and the pilot: the DeepMind/Janelia `flybody` fruit fly model
seated at the controls (assets/flybody, fetched by scripts/fetch_assets.py).

The cockpit is built around the fly's measured anatomy (stance keyframe):
front claws at (±0.087, +0.090, -0.121) relative to the thorax, stance depth
0.121-0.127, head at +0.057. The seat frame places the thorax so the levers
sit exactly under the front tarsi and the screen faces the head. The fly's
own `eye_left`/`eye_right` cameras (140 deg fisheyes, laterally mounted like
real fly eyes, screen in the medial binocular overlap) feed the retina.

Without the assets the scene falls back to an empty cockpit with fixed
head-position cameras, so tests run before fetching 137 MB of meshes.

All rendering is single-threaded (macOS GL constraint); the fly is puppeted
kinematically (qpos set directly, mj_forward, no physics stepping).
"""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

TEX_SIZE = 256
LEVER_RANGE = 0.55  # rad of hinge travel for |value| = 1

FLY_XML = Path(__file__).resolve().parents[2] / "assets" / "flybody" / "fruitfly.xml"

# seat: fly thorax pose in the cockpit (fly model forward = +x, rotated to +y)
SEAT_POS = (0.0, -0.04, 0.067)
SEAT_QUAT = (0.7071068, 0.0, 0.0, 0.7071068)
# measured front-claw contact points after seating (world frame)
CLAW_X, CLAW_Y, CLAW_Z = 0.087, 0.050, -0.054

_XML = f"""
<mujoco model="no-fly-zone-cockpit">
  <visual>
    <headlight ambient="0.12 0.12 0.14" diffuse="0.3 0.3 0.35"/>
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
    <light pos="0 -0.2 0.5" dir="0 0.4 -1" diffuse="0.4 0.4 0.45"/>
    <light pos="-0.3 0.1 0.3" dir="1 -0.2 -0.8" diffuse="0.25 0.22 0.2"/>
    <geom name="screen" type="box" pos="0 0.25 0.10" size="0.16 0.004 0.12"
          material="crt_mat"/>
    <geom name="bezel" type="box" pos="0 0.258 0.10" size="0.175 0.003 0.135"
          material="trim"/>
    <geom name="console" type="box" pos="0 0.19 -0.10" size="0.20 0.05 0.05"
          material="hull"/>
    <geom name="seat" type="box" pos="0 -0.10 -0.08" size="0.15 0.17 0.02"
          material="hull"/>
    <!-- damage lamp above the screen, inside the eyes' view: a flash is a
         real visual stimulus for the fly, not decoration -->
    <geom name="lamp" type="sphere" pos="0 0.25 0.25" size="0.016"
          rgba="0.25 0.05 0.05 1"/>
    <body name="lever_left" pos="-{CLAW_X} {CLAW_Y} {CLAW_Z - 0.09}">
      <joint name="lever_left" type="hinge" axis="1 0 0"
             range="-{LEVER_RANGE} {LEVER_RANGE}" limited="true"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.085" size="0.007" material="metal"/>
      <geom type="sphere" pos="0 0 0.09" size="0.013" material="knob"/>
    </body>
    <body name="lever_right" pos="{CLAW_X} {CLAW_Y} {CLAW_Z - 0.09}">
      <joint name="lever_right" type="hinge" axis="1 0 0"
             range="-{LEVER_RANGE} {LEVER_RANGE}" limited="true"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.085" size="0.007" material="metal"/>
      <geom type="sphere" pos="0 0 0.09" size="0.013" material="knob"/>
    </body>
    <body name="fire_button" pos="0 {CLAW_Y} {CLAW_Z - 0.02}">
      <joint name="fire_button" type="slide" axis="0 0 1" range="-0.008 0" limited="true"/>
      <geom type="cylinder" size="0.011 0.006" rgba="0.85 0.1 0.1 1"/>
    </body>
    <body name="focus" pos="0 0.06 0.03"/>
    <camera name="beauty" pos="-0.34 -0.32 0.24" mode="targetbody" target="focus"/>
    <camera name="shoulder" pos="0.16 -0.34 0.20" mode="targetbody" target="focus"/>
    <!-- retina cameras ride just ahead of the antennae on the fly's eye
         line (contact-lens cams): behind this point the fly's own head
         occludes the screen -->
    <camera name="eye_left" pos="-0.022 0.075 0.062" xyaxes="1 0 0 0 -0.212 0.977"/>
    <camera name="eye_right" pos="0.022 0.075 0.062" xyaxes="1 0 0 0 -0.212 0.977"/>
  </worldbody>
</mujoco>
"""

_LAMP_OFF = np.array([0.25, 0.05, 0.05, 1.0])
_LAMP_ON = np.array([1.0, 0.12, 0.08, 1.0])


def _load_fly_stance() -> dict[str, float]:
    """Joint angles of the flybody stance keyframe, by joint name."""
    solo = mujoco.MjModel.from_xml_path(str(FLY_XML))
    key_qpos = solo.key_qpos[0]
    stance = {}
    for j in range(solo.njnt):
        if solo.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        stance[solo.joint(j).name] = float(key_qpos[solo.jnt_qposadr[j]])
    return stance


def _build_model(with_fly: bool) -> mujoco.MjModel:
    if not with_fly:
        return mujoco.MjModel.from_xml_string(_XML)
    spec = mujoco.MjSpec.from_string(_XML)
    child = mujoco.MjSpec.from_file(str(FLY_XML))
    for k in list(child.keys):
        child.delete(k)
    root = child.worldbody.bodies[0]
    for j in list(root.joints):  # puppeted: the seat frame replaces the freejoint
        child.delete(j)
    frame = spec.worldbody.add_frame(pos=list(SEAT_POS), quat=list(SEAT_QUAT))
    frame.attach_body(child.worldbody, "fly-", "")
    return spec.compile()


class Cockpit:
    def __init__(self, render_width: int = 320, render_height: int = 240,
                 with_fly: bool | None = None):
        if with_fly is None:
            with_fly = FLY_XML.exists()
        self.with_fly = with_fly
        self.render_width = render_width
        self.render_height = render_height
        self.model = _build_model(with_fly)
        self.data = mujoco.MjData(self.model)
        if with_fly:
            for name, val in _load_fly_stance().items():
                self.data.qpos[self.model.joint("fly-" + name).qposadr[0]] = val
        # retina feed: fixed cameras at the fly's measured head position,
        # aimed at the screen. The fly's anatomical 140-degree fisheye cams
        # (fly-eye_*) put the screen at the image edge — kept as pov_camera
        # for visualisation until M3 maps ommatidia by real ray directions.
        self.eye_cameras = ("eye_left", "eye_right")
        self.pov_camera = "fly-eye_left" if with_fly else None
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

    def render(self, camera: str = "beauty") -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()
