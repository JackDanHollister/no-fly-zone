"""Cockpit layer: MuJoCo scene with a CRT screen, tread levers, a fire button,
a damage lamp — and the pilot: the DeepMind/Janelia `flybody` fruit fly model
seated at the controls (assets/flybody, fetched by scripts/fetch_assets.py).

The cockpit is built around the fly's measured anatomy (stance keyframe):
front claws at (±0.087, +0.090, -0.121) relative to the thorax, stance depth
0.121-0.127, head at +0.057. The seat frame places the thorax so the levers
sit exactly under the front tarsi and the screen faces the head. Retina
cameras ride just ahead of the antennae on the eye line (behind that the
fly's own head occludes the screen); the anatomical 140-degree fisheye cams
are kept as pov_camera.

Set dressing is a cutaway: deck, starboard wall, overhead ribs and screen
pylons — the port/aft quadrant stays open for the beauty and shoulder
cameras, and nothing enters the eye-camera corridor to the screen.

Without the assets the scene falls back to the bare cockpit so tests run
before fetching 137 MB of meshes. All rendering is single-threaded (macOS GL
constraint); the fly is puppeted kinematically (qpos + mj_forward, no
physics stepping).
"""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TEX_SIZE = 256
PLATE_W, PLATE_H = 256, 64
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
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.10 0.10 0.13" diffuse="0.22 0.22 0.28"/>
  </visual>
  <asset>
    <texture name="crt" type="2d" builtin="flat" rgb1="0 0 0"
             width="{TEX_SIZE}" height="{TEX_SIZE}"/>
    <material name="crt_mat" texture="crt" emission="1" specular="0" shininess="0"/>
    <texture name="plate" type="2d" builtin="flat" rgb1="0.05 0.05 0.06"
             width="{PLATE_W}" height="{PLATE_H}"/>
    <material name="plate_mat" texture="plate" emission="0.9" specular="0"/>
    <texture name="deck" type="2d" builtin="checker" rgb1="0.10 0.11 0.14"
             rgb2="0.13 0.14 0.17" width="300" height="300"/>
    <material name="deck_mat" texture="deck" texrepeat="10 10"
              specular="0.3" shininess="0.3"/>
    <material name="hull" rgba="0.13 0.14 0.18 1"/>
    <material name="panel" rgba="0.17 0.18 0.23 1" specular="0.4" shininess="0.4"/>
    <material name="trim" rgba="0.9 0.3 0.1 1" emission="0.6"/>
    <material name="neon_orange" rgba="1.0 0.45 0.15 1" emission="1"/>
    <material name="neon_cyan" rgba="0.30 0.90 1.0 1" emission="1"/>
    <material name="metal" rgba="0.55 0.58 0.62 1" specular="0.8" shininess="0.6"/>
    <material name="knob" rgba="0.6 0.08 0.08 1"/>
    <material name="rubber" rgba="0.09 0.09 0.11 1" specular="0.05"/>
    <material name="glass_green" rgba="0.25 0.75 0.45 1" emission="0.7"/>
    <material name="glass_amber" rgba="0.9 0.65 0.2 1" emission="0.7"/>
  </asset>
  <worldbody>
    <!-- directional key + rim (no spotlight cone edges on the CRT — the
         retina reads that screen); the warm CRT-glow spot only hits the fly -->
    <light directional="true" pos="0 -0.2 0.5" dir="0 0.4 -1"
           diffuse="0.30 0.30 0.36"/>
    <light pos="0 0.22 0.10" dir="0 -0.99 -0.12" diffuse="0.40 0.28 0.16"
           specular="0.05 0.05 0.05"/>
    <light directional="true" pos="-0.35 -0.35 0.30" dir="0.65 0.65 -0.40"
           diffuse="0.08 0.11 0.18"/>

    <!-- screen assembly -->
    <geom name="screen" type="box" pos="0 0.25 0.10" size="0.16 0.004 0.12"
          material="crt_mat"/>
    <geom name="bezel" type="box" pos="0 0.258 0.10" size="0.175 0.003 0.135"
          material="trim"/>
    <geom name="screen_hood" type="box" pos="0 0.24 0.245" size="0.185 0.02 0.008"
          material="hull"/>
    <!-- damage lamp above the screen, inside the eyes' view: a flash is a
         real visual stimulus for the fly, not decoration -->
    <geom name="lamp" type="sphere" pos="0 0.25 0.25" size="0.016"
          rgba="0.25 0.05 0.05 1"/>
    <geom name="pylon_l" type="box" pos="-0.215 0.25 0.10" size="0.018 0.018 0.155"
          material="panel"/>
    <geom name="pylon_r" type="box" pos="0.215 0.25 0.10" size="0.018 0.018 0.155"
          material="panel"/>
    <geom name="pylon_l_neon" type="box" pos="-0.196 0.244 0.10"
          size="0.004 0.006 0.145" material="neon_cyan"/>
    <geom name="pylon_r_neon" type="box" pos="0.196 0.244 0.10"
          size="0.004 0.006 0.145" material="neon_cyan"/>

    <!-- console: toggles, blinkenlights, gauges -->
    <geom name="console" type="box" pos="0 0.19 -0.10" size="0.20 0.05 0.05"
          material="hull"/>
    <geom name="console_top" type="box" pos="0 0.175 -0.049" size="0.19 0.035 0.004"
          material="panel"/>
    <geom name="toggle1" type="cylinder" pos="-0.14 0.17 -0.038" size="0.004 0.008"
          euler="15 0 0" material="metal"/>
    <geom name="toggle2" type="cylinder" pos="-0.115 0.17 -0.038" size="0.004 0.008"
          euler="-15 0 0" material="metal"/>
    <geom name="toggle3" type="cylinder" pos="-0.09 0.17 -0.038" size="0.004 0.008"
          euler="15 0 0" material="metal"/>
    <geom name="blink1" type="sphere" pos="0.09 0.17 -0.045" size="0.005"
          material="glass_green"/>
    <geom name="blink2" type="sphere" pos="0.11 0.17 -0.045" size="0.005"
          material="glass_amber"/>
    <geom name="blink3" type="sphere" pos="0.13 0.17 -0.045" size="0.005"
          material="neon_orange"/>
    <geom name="gauge1" type="cylinder" pos="-0.04 0.165 -0.044" size="0.014 0.002"
          material="glass_green"/>
    <geom name="gauge2" type="cylinder" pos="0.04 0.165 -0.044" size="0.014 0.002"
          material="glass_amber"/>

    <!-- deck and seat pedestal -->
    <geom name="deck" type="box" pos="0 0.02 -0.16" size="0.42 0.44 0.015"
          material="deck_mat"/>
    <geom name="deck_neon_l" type="box" pos="-0.32 0.02 -0.143"
          size="0.006 0.40 0.003" material="neon_orange"/>
    <geom name="deck_neon_r" type="box" pos="0.32 0.02 -0.143"
          size="0.006 0.40 0.003" material="neon_orange"/>
    <geom name="pedestal" type="box" pos="0 -0.10 -0.125" size="0.10 0.13 0.022"
          material="panel"/>
    <geom name="seat" type="box" pos="0 -0.10 -0.08" size="0.15 0.17 0.02"
          material="hull"/>

    <!-- starboard wall (port/aft stay open for the cameras) -->
    <geom name="wall_r" type="box" pos="0.38 0.02 0.09" size="0.012 0.36 0.25"
          material="hull"/>
    <geom name="wall_r_neon" type="box" pos="0.367 0.02 0.30"
          size="0.003 0.34 0.005" material="neon_orange"/>
    <geom name="aux_frame1" type="box" pos="0.368 0.18 0.10" size="0.004 0.042 0.032"
          material="rubber"/>
    <geom name="aux_screen1" type="box" pos="0.366 0.18 0.10" size="0.004 0.034 0.024"
          rgba="0.12 0.35 0.22 1"/>
    <geom name="aux_frame2" type="box" pos="0.368 0.05 0.10" size="0.004 0.042 0.032"
          material="rubber"/>
    <geom name="aux_screen2" type="box" pos="0.366 0.05 0.10" size="0.004 0.034 0.024"
          rgba="0.40 0.30 0.10 1"/>
    <geom name="nameplate" type="box" pos="0.366 0.115 0.20" size="0.004 0.095 0.024"
          material="plate_mat"/>
    <geom name="vent1" type="box" pos="0.366 -0.16 0.02" size="0.004 0.05 0.004"
          material="rubber"/>
    <geom name="vent2" type="box" pos="0.366 -0.16 0.045" size="0.004 0.05 0.004"
          material="rubber"/>
    <geom name="vent3" type="box" pos="0.366 -0.16 0.07" size="0.004 0.05 0.004"
          material="rubber"/>
    <geom name="conduit_wall" type="capsule" fromto="0.36 -0.30 -0.12 0.36 0.30 -0.12"
          size="0.008" material="rubber"/>
    <geom name="conduit_neon" type="capsule" fromto="0.363 -0.30 -0.10 0.363 0.30 -0.10"
          size="0.004" material="neon_cyan"/>

    <!-- background gantry behind the screen (frames the shot, never blocks
         a camera: aft of the screen plane) -->
    <geom name="gantry_l" type="box" pos="-0.30 0.40 0.10" size="0.020 0.020 0.26"
          material="panel"/>
    <geom name="gantry_r" type="box" pos="0.30 0.40 0.10" size="0.020 0.020 0.26"
          material="panel"/>
    <geom name="gantry_beam" type="box" pos="0 0.40 0.382" size="0.32 0.020 0.022"
          material="panel"/>
    <geom name="gantry_neon" type="box" pos="0 0.393 0.358" size="0.30 0.004 0.004"
          material="neon_orange"/>

    <!-- cabling from console to deck -->
    <geom name="cable1" type="capsule" fromto="0.14 0.19 -0.145 0.30 0.06 -0.148"
          size="0.006" material="rubber"/>
    <geom name="cable2" type="capsule" fromto="-0.14 0.19 -0.145 -0.24 0.00 -0.148"
          size="0.006" material="rubber"/>
    <geom name="cable3" type="capsule" fromto="0.10 0.19 -0.145 0.34 -0.10 -0.146"
          size="0.004" material="rubber"/>

    <!-- controls, placed under the fly's front claws -->
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


def _paint_nameplate(model: mujoco.MjModel) -> None:
    """Draw the mission nameplate into its texture (before renderer creation,
    so it uploads with the GL context)."""
    img = Image.new("RGB", (PLATE_W, PLATE_H), (12, 12, 15))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=34)
    except TypeError:
        font = ImageFont.load_default()
    text = "NO-FLY-ZONE"
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((PLATE_W - box[2] + box[0]) / 2, (PLATE_H - box[3] + box[1]) / 2),
              text, font=font, fill=(255, 120, 40))
    draw.rectangle([2, 2, PLATE_W - 3, PLATE_H - 3], outline=(90, 40, 15), width=2)
    # the plate's visible face (-x, on the starboard wall) mirrors the texture
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    tex_id = model.texture("plate").id
    adr = model.tex_adr[tex_id]
    model.tex_data[adr: adr + PLATE_W * PLATE_H * 3] = np.asarray(
        img, dtype=np.uint8).ravel()


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
        _paint_nameplate(self.model)
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
