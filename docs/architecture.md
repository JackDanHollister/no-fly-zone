# NO-FLY-ZONE — Design Notes

## Concept

The complete fly connectome, run as a spiking network, is the pilot of a mech
(well, a tank). It never sees the game directly: the game exists as a glowing
CRT inside a 3D cockpit, the fly's eyes exist as cameras inside that cockpit,
and its motor output moves physical controls. The indirection is the point —
it makes the fly a *pilot*, and it makes the cockpit a stage for maximum
retro-futurist spectacle.

As of Sept 2026 nobody in the fly-connectome wave (cobanov/awesome-fly, ~50
projects) has done the cockpit/pilot framing, and no published example was
found anywhere of "game screen rendered inside a physics sim, sampled by a
virtual camera feeding an agent". This configuration is the novel bit.

## Layers

1. **Game** — `ALE/BattleZone-v5` (gymnasium + ale-py). ROMs ship legally in
   the pip package. Discrete(18) actions, 210x160 RGB frames. First-person
   tank view: the fiction nests (cockpit in a cockpit).
2. **Cockpit** — MuJoCo scene (MJCF). The game frame is uploaded into a model
   texture on a curved-CRT screen geom each step (passive viewer
   `update_texture(texid)` / `mjr_uploadTexture`). Aesthetic goals: emissive
   neon trim, toggle banks, warning lights wired to real game events (damage →
   red flashers — which also feed the fly's looming detectors), fly-scale
   details (a sugar-water feeder line, a tiny mission patch).
3. **Eyes** — offscreen renders from two eye cameras at the pilot's head
   position → hexagonal ommatidia binning (flygym `Retina`: 721
   ommatidia/eye, green+blue channels; aimbug's hex-column geometry ~4.8°
   spacing as an alternative) → photoreceptor drive currents (DOOMFLY-style
   R1–R6 brightness + R8 colour).
4. **Brain** — LIF over the real wiring. Parameters from Shiu et al. 2024:
   0.275 mV/synapse, tau_m 20 ms, tau_s 5 ms, signed by predicted
   neurotransmitter. Dataset: MaleCNS v1.0 (has the full sensory→motor loop)
   with FlyWire v783 as fallback. Backends: sparse SciPy/NumPy (Mac dev),
   PyTorch (MPS on Mac / CUDA on 4090). Real time is not required on the Mac.
5. **Motor** — descending-neuron readout table (per aimbug/Fly64 precedent):
   - DNa02 left-minus-right rate → steering
   - DNg100 → forward drive
   - pIP10 courtship-song bouts → FIRE ("the more aroused the fly, the harder
     it tries to shoot" — proven viral)
   - DNp01 giant fiber → startle/reverse
   In **wired** mode these rates set lever/pedal positions directly (levers
   animate, positions map to ALE actions). In **embodied** mode they drive
   flybody leg actuators and the levers respond to contact forces.
6. **Learning (later)** — DOOMFLY's dopamine hack: PPL101 aversive dopamine
   onto KC→MBON synapses when the tank takes damage. Label honestly: the
   scene's norm (DOOMFLY, Ommatid) is that closed-loop connectomes don't
   demonstrably learn yet. Choreographed demo footage vs closed-loop runs must
   be labelled as such (lyra's Beat Saber fly was replay — don't repeat that
   ambiguity).

## Mode flag

`--pilot wired|embodied`. Wired is the baseline that always works; embodied is
the spectacle. Same brain, eyes, cockpit, game in both.

## Milestones

- [x] M0 — repo, scaffold, smoke tests
- [x] M1 — **the crux**: BattleZone frame → MuJoCo screen texture → eye-camera
      offscreen render sees the game (proven by `scripts/smoke_cockpit_screen.py`,
      including a right-way-up orientation gate)
- [x] M2 — closed loop with a FAKE brain (hand-written reflexes) through the
      full pipeline: frame → retina → stub → levers → actions. ~160 Hz on the
      Mac (10x headroom over the 15 Hz game rate). `scripts/run_m2.py`.
- [x] M2.5 — the fly is IN the cockpit: DeepMind/Janelia `flybody` model
      (fetched by `scripts/fetch_assets.py`, ~137 MB, gitignored) seated via
      MjSpec attach (freejoint deleted, stance keyframe applied by joint name,
      kinematic puppet). Cockpit geometry is derived from its measured anatomy
      (levers under the front claws at ±0.087/0.050/-0.054, screen at head
      height). Retina feeds from "contact-lens" cameras just ahead of the
      antennae (behind that the fly's own head occludes the screen — found
      the fun way); the anatomical 140° fisheye eye cams ship with flybody
      and are kept as `pov_camera` for visuals until M3 maps ommatidia by
      real ray directions. Falls back to the bare cockpit when assets are
      missing. ~21 Hz on the Mac with the fly (still above the 15 Hz game
      rate; renders are the cost).
- [ ] M3 — connectome download + LIF core (start with the looming→giant-fiber
      escape circuit ~600 neurons; validate against Shiu et al. sugar→proboscis
      result before scaling to whole brain)
- [ ] M4 — whole-brain pilot in wired mode; dashboard (Fly64-style local web
      view: eyes, spikes, controls)
- [ ] M5 — cockpit art pass (CRT shader-baked textures, neon, warning lights)
- [ ] M6 — embodied mode: flybody seated, contact-driven controls (4090)
- [ ] M7 — learning experiments + the showcase video

## Interface decisions (M2 review, 2026-09-13)

- `dn.py` owns the DN contract: names, Hz units, per-DN calibration to
  normalized [0,1] drives. Brains emit raw Hz; controls consume normalized.
  Swapping in the M3 connectome = editing calibration entries only. DNp01 is
  calibrated as an event neuron (one spike per tick saturates).
- Brain contract: `step(drives, dt, signals) -> DN rates` + `reset()`.
  `dt` has one source of truth (`BattleZone.dt`, derived from frameskip).
  `signals` = {damage, reward, episode_end} — the M7 dopamine hook.
- Retina owns photoreceptor calibration: `photoreceptors()` returns (n, 2)
  drives in [0,1] (R1-6 broadband, R8 blue) against `WHITE_REF` — after any
  art pass (M5), re-measure that one constant instead of retuning the brain.
- Sticky actions OFF (determinism for circuit validation); episode auto-reset
  is a stated decision ("the fly keeps living") with temporal state cleared at
  the boundary; one-step motor latency is deliberate (reaction time).
- The damage lamp lives above the screen, inside the eyes' FOV — a flash is a
  real stimulus, not decoration.
- Dev gotcha: pip's editable-install `.pth` can end up with the macOS hidden
  flag set, and Python 3.14's site module silently skips hidden `.pth` files
  (`ModuleNotFoundError: noflyzone`). `chflags nohidden` works but the flag
  has come back; the durable fix in this venv is a direct symlink:
  `ln -sfn "$(pwd)/src/noflyzone" .venv/lib/python3.14/site-packages/noflyzone`.

## Platform decisions (verified 2026-09-13)

- **Isaac Sim: rejected.** No macOS support (Ubuntu/Windows + RTX-with-RT-cores
  only). GR00T is NVIDIA's humanoid VLA model, not a simulator. Nobody in the
  fly wave used Isaac; the embodied projects all use MuJoCo.
- **MuJoCo: chosen.** Native Apple-silicon wheels, runs identically on the
  4090 box, `update_texture` solves screen-in-scene, flybody/flygym are
  MuJoCo-native. macOS constraint: GL on the main thread → single-threaded
  loop (step → upload → render → brain → act).
- **Emulator: ale-py.** Legal bundled ROMs, arm64 wheels. stable-retro kept in
  reserve for Cybernator (SNES) as a private-use later target; BizHawk (no
  macOS), RetroArch UDP (no input injection), MAMEToolkit (Linux-only) ruled out.
- Full source landscape and citations: The_Magi
  `projects/fly-connectome/research/` (landscape.md, feasibility.md).
