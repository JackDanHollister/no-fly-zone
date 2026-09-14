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
    <material name="olive" rgba="0.23 0.25 0.17 1" specular="0.15" shininess="0.2"/>
    <material name="gunmetal" rgba="0.15 0.16 0.15 1" specular="0.5" shininess="0.5"/>
    <material name="brass" rgba="0.62 0.48 0.22 1" specular="0.85" shininess="0.7"/>
    <texture name="tally" type="2d" builtin="flat" rgb1="0.1 0.1 0.09"
             width="128" height="96"/>
    <material name="tally_mat" texture="tally" emission="0.5" specular="0"/>
    <texture name="poster1" type="2d" builtin="flat" rgb1="0.1 0.05 0.03"
             width="128" height="176"/>
    <material name="poster1_mat" texture="poster1" emission="0.45" specular="0"/>
    <texture name="poster2" type="2d" builtin="flat" rgb1="0.05 0.07 0.12"
             width="128" height="160"/>
    <material name="poster2_mat" texture="poster2" emission="0.45" specular="0"/>
    <texture name="crate" type="2d" builtin="flat" rgb1="0.2 0.22 0.15"
             width="128" height="64"/>
    <material name="crate_mat" texture="crate" emission="0.35" specular="0"/>
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

    <!-- tank interior: sloped armor glacis plates flanking the screen -->
    <geom name="armor_l" type="box" pos="-0.30 0.33 0.06" size="0.11 0.014 0.22"
          euler="0 18 -20" material="olive"/>
    <geom name="armor_r" type="box" pos="0.30 0.33 0.06" size="0.11 0.014 0.22"
          euler="0 -18 20" material="olive"/>
    <geom name="periscope" type="box" pos="0 0.235 0.275" size="0.045 0.028 0.016"
          material="gunmetal"/>
    <geom name="periscope_slit" type="box" pos="0 0.206 0.275" size="0.032 0.002 0.004"
          rgba="0.05 0.35 0.2 1"/>
    <replicate count="7" offset="0 0.09 0">
      <geom type="sphere" size="0.0045" pos="0.372 -0.30 0.325" material="gunmetal"/>
    </replicate>
    <replicate count="7" offset="0 0.09 0">
      <geom type="sphere" size="0.0045" pos="0.372 -0.30 -0.10" material="gunmetal"/>
    </replicate>
    <replicate count="5" offset="0 0 0.06">
      <geom type="sphere" size="0.004" pos="-0.236 0.245 -0.02" material="gunmetal"/>
    </replicate>
    <replicate count="5" offset="0 0 0.06">
      <geom type="sphere" size="0.004" pos="0.236 0.245 -0.02" material="gunmetal"/>
    </replicate>

    <!-- war-movie paraphernalia -->
    <geom name="ammo_crate" type="box" pos="0.24 -0.02 -0.105" size="0.055 0.075 0.040"
          material="olive"/>
    <geom name="crate_stencil" type="box" pos="0.24 -0.097 -0.10" size="0.045 0.002 0.026"
          material="crate_mat"/>
    <geom name="crate_rope" type="capsule" fromto="0.20 -0.06 -0.065 0.28 -0.06 -0.065"
          size="0.004" material="rubber"/>
    <geom name="helmet" type="ellipsoid" pos="0.24 -0.02 -0.052" size="0.038 0.046 0.026"
          material="olive"/>
    <geom name="helmet_rim" type="cylinder" pos="0.24 -0.02 -0.062" size="0.047 0.004"
          material="olive"/>
    <geom name="shell1" type="cylinder" pos="-0.20 -0.16 -0.140" size="0.006 0.020"
          euler="90 0 15" material="brass"/>
    <geom name="shell2" type="cylinder" pos="-0.24 -0.12 -0.140" size="0.006 0.020"
          euler="90 0 70" material="brass"/>
    <geom name="shell3" type="cylinder" pos="-0.17 -0.10 -0.140" size="0.006 0.020"
          euler="90 0 40" material="brass"/>
    <geom name="dogtag_chain" type="capsule" fromto="-0.17 0.152 -0.052 -0.165 0.145 -0.082"
          size="0.0015" material="metal"/>
    <geom name="dogtag1" type="box" pos="-0.165 0.145 -0.090" size="0.007 0.002 0.010"
          material="metal"/>
    <geom name="dogtag2" type="box" pos="-0.162 0.148 -0.093" size="0.007 0.002 0.010"
          euler="0 0 12" material="metal"/>

    <!-- wall art: kill tally + war posters -->
    <geom name="tally_plate" type="box" pos="0.366 0.115 0.135" size="0.004 0.055 0.033"
          material="tally_mat"/>
    <geom name="poster_apoc" type="box" pos="0.366 -0.04 0.24" size="0.004 0.052 0.072"
          material="poster1_mat"/>
    <geom name="poster_wings" type="box" pos="0.366 -0.215 0.21" size="0.004 0.047 0.060"
          material="poster2_mat"/>

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


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _center_text(draw, img_w, y, text, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((img_w - box[2] + box[0]) / 2, y), text, font=font, fill=fill)


def _paint_plate(img, draw, w, h):
    _center_text(draw, w, (h - 40) / 2, "NO-FLY-ZONE", _font(34), (255, 120, 40))
    draw.rectangle([2, 2, w - 3, h - 3], outline=(90, 40, 15), width=2)


def _paint_tally(img, draw, w, h):
    _center_text(draw, w, 6, "CONFIRMED SWATS", _font(12), (210, 210, 200))
    y0, y1 = 34, 78
    x = 14
    for group in range(2):  # two gates of five
        for i in range(4):
            draw.line([x + i * 9, y0, x + i * 9, y1], fill=(225, 225, 215), width=3)
        draw.line([x - 5, y1 - 4, x + 32, y0 + 4], fill=(225, 225, 215), width=3)
        x += 52
    draw.rectangle([1, 1, w - 2, h - 2], outline=(60, 60, 55), width=2)


def _paint_poster_apoc(img, draw, w, h):
    for i in range(90):  # sunset gradient
        t = i / 90
        draw.line([0, i, w, i],
                  fill=(int(230 - 140 * t), int(110 - 80 * t), int(20 * (1 - t))))
    draw.ellipse([w / 2 - 30, 22, w / 2 + 30, 82], fill=(250, 160, 40))
    draw.rectangle([0, 90, w, h], fill=(15, 10, 8))
    for cx, cy in ((30, 40), (85, 26)):  # gunship dragonflies
        draw.ellipse([cx - 9, cy - 2, cx + 9, cy + 2], fill=(20, 12, 8))
        draw.line([cx - 12, cy - 5, cx + 12, cy - 5], fill=(20, 12, 8), width=2)
    _center_text(draw, w, 102, "APOCALYPSE", _font(19), (245, 200, 120))
    _center_text(draw, w, 124, "FLY NOW", _font(19), (245, 200, 120))
    _center_text(draw, w, 152, "the horror has six legs", _font(10), (170, 120, 70))


def _paint_poster_wings(img, draw, w, h):
    draw.rectangle([4, 4, w - 5, h - 5], outline=(225, 215, 180), width=3)
    _center_text(draw, w, 14, "LOOSE WINGS", _font(17), (235, 225, 190))
    # a wing sinking into the waves
    draw.ellipse([w / 2 - 26, 52, w / 2 + 26, 78], outline=(235, 225, 190), width=3)
    draw.line([w / 2 - 26, 66, w / 2 + 26, 62], fill=(235, 225, 190), width=2)
    for y in (86, 94):
        for x in range(10, w - 10, 16):
            draw.arc([x, y - 4, x + 16, y + 6], 200, 340, fill=(120, 140, 170), width=2)
    _center_text(draw, w, 106, "SINK SHIPS", _font(17), (235, 225, 190))
    _center_text(draw, w, 134, "MINISTRY OF WING SAFETY", _font(9), (150, 150, 140))


def _paint_crate(img, draw, w, h):
    stencil = (25, 22, 15)
    _center_text(draw, w, 8, "SUGAR RATIONS", _font(15), stencil)
    _center_text(draw, w, 32, "GRADE A - 7.62 mmol", _font(11), stencil)
    draw.rectangle([2, 2, w - 3, h - 3], outline=stencil, width=2)


_DECALS = {
    # tex name: (width, height, background, painter, mirror)
    # wall decals face -x, which mirrors the texture -> pre-flip
    "plate": (PLATE_W, PLATE_H, (12, 12, 15), _paint_plate, True),
    "tally": (128, 96, (26, 26, 23), _paint_tally, True),
    "poster1": (128, 176, (26, 13, 8), _paint_poster_apoc, True),
    "poster2": (128, 160, (13, 18, 31), _paint_poster_wings, True),
    "crate": (128, 64, (51, 56, 38), _paint_crate, False),  # faces -y
}


def _paint_decals(model: mujoco.MjModel) -> None:
    """Draw all decal textures (before renderer creation, so they upload
    with the GL context)."""
    for name, (w, h, bg, painter, mirror) in _DECALS.items():
        img = Image.new("RGB", (w, h), bg)
        painter(img, ImageDraw.Draw(img), w, h)
        if mirror:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        tex_id = model.texture(name).id
        adr = model.tex_adr[tex_id]
        model.tex_data[adr: adr + w * h * 3] = np.asarray(
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
        _paint_decals(self.model)
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
