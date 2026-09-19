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
- [x] M3 — connectome download + LIF core. Shipped 2026-09-14 as
      `brain_lif.py`, a sparse NumPy/SciPy port of the Shiu et al. 2024 model
      over the full FlyWire v783 brain (138,639 neurons, 15.1M edges); the
      planned ~600-neuron escape-circuit stage was skipped. `validate_brain.py`
      gates silence, sugar GRNs → MN9 and LC4 → DNp01 (Mac: MN9 101 Hz, giant
      fibre 66/64 Hz). Gates are qualitative and single-seed; CPU only.
- [~] M4 — whole-brain pilot in wired mode; dashboard. Partial as of
      2026-09-14: `brain_connectome.py` + `run_m3.py` drive the game from real
      DN readouts with a spike-cloud panel. Missing: `run_m3.py` forces an idle
      forward-right scan above the action deadbands, so nothing yet shows the
      connectome steering; the only gate is "brain not silent"; the DNp01
      startle only fires when both giant-fibre cells spike in the same tick
      (the readout averages the pair); no eyes panel; offline mp4 compositor
      rather than the Fly64-style web view.
- [~] M5 — cockpit art pass. Partial as of 2026-09-14: neon trim, lighting,
      set dressing and Pillow-painted decals shipped (ab1c764, a3a9ba8).
      Missing: any CRT treatment (the screen is a flat box with the raw
      frame), warning lights beyond the single damage lamp, and that lamp sits
      about 35° above the eye cameras' optical axis, outside their view.
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

## CUDA backend (2026-09-14)

`brain_cuda.ShiuBrainCUDA` subclasses the reference and keeps its API. Per
0.1 ms step it launches four things on the current stream: zero the ring slot
about to be written; a per-neuron Triton kernel (decay for live neurons,
strict threshold, reset of v and g, refractory bookkeeping with the
activated-neuron exemption, Philox Poisson kick, per-neuron count, append to
the delay ring through an atomic counter); a delivery kernel with one program
per recorded spike that atomic-adds the spike's outgoing CSC column into g
exactly DELAY_STEPS later; and a step-counter increment. The ring has
DELAY_STEPS + 1 slots so the read and write slots never coincide. A whole game
tick (667 steps) is captured once as a CUDA graph and replayed, so only the
spike counts cross to the host. Randomness is Philox on (seed, step * n +
neuron): stateless, reproducible after reset(), not bit-identical to NumPy's
stream. Delivery ordering differs at the float32 ulp, which the recurrent
network amplifies over hundreds of steps; `scripts/compare_backends.py`
checks exact per-step agreement for 500 steps and 1% aggregate agreement
over 0.3 s. Each ring row holds one entry per neuron, so it cannot overflow.
Triton prints harmless `_POSIX_C_SOURCE` redefinition warnings when
it builds its driver stub against the uv-managed Python headers.

## Related work (2026-09-14)

Ten repos from the September 2026 fly-connectome wave were read for
techniques worth borrowing (shuffled-connectome controls, DN screens,
provenance hashes, MaleCNS loaders, plasticity controls); the notes are kept
outside this repository. The short version: nothing surveyed runs a whole
spiking brain faster than this backend at Shiu fidelity, and no surveyed
project shows a connectome doing what a random sparse network would not,
which is the bar this project holds itself to in `docs/research/`.

## Status audit (2026-09-14)

Findings from reading the code on the workstation clone, before anything was
run there. Each is a place where this document or the README says more than
the code delivers. Fix the code or the wording, then delete the line.

- Backends: resolved 2026-09-14 for CUDA. `brain_cuda.py` (Triton kernels,
  CUDA graphs, `pip install -e ".[cuda]"`) is step-exact against the SciPy
  reference for 500 steps and within 1% over chaotic horizons; there is still
  no MPS backend, so the Mac stays on SciPy.
- Dataset: FlyWire v783 from the Shiu GitHub repo is the default
  (`Completeness_783.csv`, `Connectivity_783.parquet`). Since 2026-09-18
  `brain_lif.DATASETS` also names MaleCNS v1.0, converted by
  `scripts/build_malecns.py` from the pinned flat-connectome feathers into
  the same three shapes (166,700 neurons, 25.6M signed pairs, annotations in
  the FlyWire vocabulary, `data/malecns/cache/manifest.json`); every script
  takes `--dataset malecns`. At Shiu gains the male brain is over-excited
  (about 11,000 spiking neurons per tick against 2,000 for FlyWire on the
  same stimuli) and its usable weight window is narrow; see
  `docs/research/wiring-controls-2026-09-18.md`. The Zenodo dump is still
  not fetched.
- Mode flag: `--pilot wired|embodied` exists nowhere. `run_m2.py` and
  `run_m3.py` take only `--steps` and `--video`.
- FIRE channel: pIP10 is absent from the female v783 brain;
  `brain_connectome.py` reads oviDN under the `pIP10` key (and pIP10 itself
  where a dataset has it, as MaleCNS does) and the dashboard bar still says
  pIP10.
- Motor swap: the M3 hand-over changed `controls.py` (idle_throttle,
  idle_steer) rather than only `dn.py` calibration entries, contrary to the
  M2 interface decision above. The `dn.py` spans were tuned for the stub and
  never revisited for real firing rates.
- DNp01: "one spike per tick saturates" is weakened by averaging spike counts
  over both DNp01 cells; a single-cell spike gives drive 0.5 and `controls.py`
  requires more than 0.5, so only a same-tick spike in both cells startles
  (seen in the workstation M3 run).
- Damage lamp: with the eye cameras' default 45° fovy the lamp centre is about
  35° above the optical axis, so "a flash is a real stimulus" is not yet true.
  It is also a solid 0.5 s latch with no emissive material.
- Levers and physics: nothing calls `mj_step`; `set_controls` writes qpos and
  calls `mj_forward`. Levers are a visual puppet of the action, not the cause
  of it. There are no pedals.
- Stance keyframe: the fetched `fruitfly.xml` has no `<key>`, so applying
  `key_qpos[0]` by joint name is a no-op. `pov_camera` is dead code at HEAD.
- Retina: `WHITE_REF` is exactly 255.0 and has not been re-measured since the
  lighting changes; the retina is a from-scratch hex lattice, flygym is not
  imported; ommatidia are not mapped by real ray directions (deferred, now
  unclaimed by any milestone).
- Tests: `pyproject.toml` declares a pytest extra but there is no `tests/`
  directory; `smoke_battlezone.py` has no assertions.
- Versions: `mujoco>=3.2` is floored but `MjSpec.delete()` needs 3.3.4+, and
  `cockpit.py` reaches into the private `Renderer._mjr_context`; `game.py`
  reads the private `env.unwrapped._frameskip`. Known-good set on the
  workstation (2026-09-14): mujoco 3.13.0, ale-py 0.12.1, gymnasium 1.3.0,
  numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, pyarrow 25.0.1, pillow 12.3.0.
  Not yet pinned in `pyproject.toml`.
- Loop rate, measured 2026-09-14 on the workstation: `run_m2` about 233 Hz
  with the fly (Mac: about 21 Hz); `run_m3 --backend cpu` about 2 Hz, 7.7x
  slower than real time, brain 492 ms per tick (Mac: about 2 Hz);
  `run_m3 --backend cuda` about 80 Hz, 5.2x faster than real time over 450
  ticks, brain 8.9 ms per tick. One isolated tick is 5.5 ms on the GPU versus
  263 ms on the CPU.
- Model corrections (2026-09-14): the reference model's sign convention
  (dopamine, octopamine, serotonin as fast +1; Kenyon cells predicted
  dopaminergic) yields a self-sustaining Kenyon-cell/PAM/antennal-lobe state
  after any big visual event, persisting in darkness. `aminergic="zero"` and
  `sensory_input="external_only"` in `brain_lif.ShiuBrain` remove it with the
  validation gates intact; both are off by default so `validate_brain.py`
  still tests the reference, and every run records them in `brain.meta`.
- Attach conflict (MuJoCo 3.13): seating the fly warns because the flybody's
  option block (0.1 ms timestep, gravity -981 in centimetre units, noslip,
  density, viscosity, cone) differs from the cockpit's; parent values win.
  Harmless while the fly is a kinematic puppet; embodied mode (M6) must pick
  one unit system and timestep deliberately.
- Housekeeping: `run_m3.py` never calls `loop.close()`;
  `PilotLoop.latency_report()` divides by zero before the first step; the
  macOS `.pth` note below is Mac-specific and does not apply on Linux.
