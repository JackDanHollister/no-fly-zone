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
- [ ] M1 — **the crux**: BattleZone frame → MuJoCo screen texture → eye-camera
      offscreen render sees the game (proven by `scripts/smoke_cockpit_screen.py`)
- [ ] M2 — closed loop with a FAKE brain (random walk / hand-written reflexes)
      through the full pipeline: frame → retina → stub → levers → actions.
      Proves latency and plumbing before any connectome data.
- [ ] M3 — connectome download + LIF core (start with the looming→giant-fiber
      escape circuit ~600 neurons; validate against Shiu et al. sugar→proboscis
      result before scaling to whole brain)
- [ ] M4 — whole-brain pilot in wired mode; dashboard (Fly64-style local web
      view: eyes, spikes, controls)
- [ ] M5 — cockpit art pass (CRT shader-baked textures, neon, warning lights)
- [ ] M6 — embodied mode: flybody seated, contact-driven controls (4090)
- [ ] M7 — learning experiments + the showcase video

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
