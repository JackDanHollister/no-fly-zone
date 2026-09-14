# Changelog

## [2026-09-14] - Persistent olfactory state found and corrected; honest benchmark

- Long runs settled into fixed pool rates regardless of the screen. Cutting
  input to darkness after a big event left 9,654 neurons (Kenyon cells, PAM,
  antennal-lobe local neurons, receptor neurons, APL/DPM at 400+ Hz) spiking
  forever in the reference model; the rewired control goes silent in two
  ticks. Cause: dopamine/octopamine/serotonin signed as fast +1 synapses (and
  Kenyon cells predicted dopaminergic), closing a KC-PAM-KC loop.
- Loader options `aminergic="zero"` (333,211 modulatory edges lose fast
  weight, literature labels first) and `sensory_input="external_only"` (no
  synapses onto primary sensory neurons); `signs="known_first"` also
  available. Gates unchanged (MN9 97-102 Hz, GF 66/68). Options recorded in
  `brain.meta`, `describe()` and the HUD; flags on `run_m3.py` and
  `bench_pilots.py`.
- `PilotLoop(eye_offset)`: each eye gets its own sideways-shifted retina
  window; both cameras render the same CRT, so without it left/right readouts
  cannot lateralise. Loom pool narrowed to DNp12 + DNp01; vision is now also
  gated during the startle reverse.
- Corrected pilot (100 s): event-driven, eight turns toward targets, three
  kills, seven deaths. Ten-seed benchmark, 30 s episodes: connectome 10/10
  kills, rewired 10/10, silenced 0, stub 0; on the reference model the
  rewired control had beaten the connectome 10 to 3 because the persistent
  state jammed the intact brain. Conclusion recorded in
  `docs/research/cascade-probe-2026-09-14.md`: the loop and readouts work,
  but the wiring is not yet shown to matter versus a random graph.

## [2026-09-14] - Steps 2 and 3: the fly plays (reflexively)

- Named injection maps (`INJECTION_MAPS` v0-v4, `--injection`), a retina
  `zoom` (`--zoom 2` puts the 721 ommatidia over the central half of the eye
  so a horizon tank spans several), and `scripts/screen_dns.py`, a DN screen
  that measures which descending types lateralise with the eye an object is
  shown to. v4 (full T4/T5 columnar inputs plus T3 for LC11) makes a small
  object visible without firing the escape.
- `READOUT_MAPS["pool4"]` from the screen: steer pools {DNp07, DNb05, DNg52,
  DNb06, DNa02} per side, DNg100 drive, {DNp12, DNbe001, DNp01} startle, FIRE
  when both steer pools fire; `dn.CALIBRATION_MAPS["pool4"]`.
- Motor programs in `controls.py`, all named on the HUD: idle scan now off by
  default (`--idle-scan`), saccadic gating (`--saccade-hold`), startle
  refractory (`--startle-refractory`), lateralisation-index steering
  (`--steer-mode li`). `--ablate rewired|silenced` wiring controls in the
  loader; `--trace` per-tick logs; `latency_report` no longer divides by zero.
- First connectome-steered kill: both steer pools lit with a tank ahead, the
  fly fired that tick, the shell landed six ticks later. `scripts/bench_pilots.py`
  same-seed benchmark: connectome 1 kill in 3 seeds, rewired, silenced and
  the reflex stub 0; rewired keeps a comparable firing regime.
- Findings and tables in `docs/research/cascade-probe-2026-09-14.md`.

## [2026-09-14] - Step 1 of getting the fly to play: where the visual drive dies

- Added three diagnostics on the CUDA backend: `scripts/probe_cascade.py`
  (stimulate one cell type per eye and read every stage plus the responding
  descending neurons, left/right), `scripts/audit_dn_inputs.py` (top
  presynaptic types with signs, super-class shares, BFS hop counts from each
  retina stage) and `scripts/probe_frontend.py` (synthetic discs, loom, moving
  discs and bars through the game's own injection path). All three run in
  seconds; results in `out/` and `docs/research/cascade-probe-2026-09-14.md`.
- Findings: photoreceptor output is mis-signed excitatory (predicted
  transmitter, not histamine) and L1 is inhibitory, so the ON pathway is dead
  from the retina; from the medulla down everything propagates when driven
  hard enough; LC4/LPLC2 to DNp01 escape is strong; LC10a to DNa02 is the only
  strongly lateralised steering channel (157 vs 0 Hz) but LC10a is
  inhibition-dominated and reaches only 5.5 Hz from its medulla inputs; LC11
  is reachable at 133 Hz from T2/T3/T2a/Tm21/Tm25. Through the current game
  injection a tank-sized disc is invisible, T4/T5 stay under 5 Hz, and DNa02
  gives one or two spikes per tick uncorrelated with stimulus side.

## [2026-09-14] - Survey of the fly-connectome wave

- Prompted by the "fly solves a Rubik's cube" videos, shallow-cloned and read
  ten public repos (flycube-public, connectome_rubiks_cube, Fly.exe, FastFly,
  flyverse-core, nfly, fly-drone, fly-brain-billiards, NeuroFly, FlyPong)
  with one reader each and a synthesis pass; the notes are kept outside this
  repository.
- The cube fly is a frozen tanh rate reservoir on an 8,192-neuron MaleCNS
  subgraph with a trained encoder and policy head taught by a classical
  solver; 0 of 100 random cubes; every checkpoint trained with propagation
  reversed. Nothing to borrow for the pilot or the engine; the honesty
  scaffolding and MaleCNS loader are worth taking.
- Recorded thirteen ranked borrowables (sensory assay and DN input audit,
  shuffled-connectome control, DN-gated actuation, measured DN calibration,
  numerics tests, provenance hashes, live viewer, ablation ladder, MaleCNS
  loader, M7 plasticity recipe with pre-registered controls, render fix,
  batched brains, embodied recipes), a speed comparison showing our backend
  is the fastest whole-brain spiking model in the set at Shiu fidelity, the
  pinned MaleCNS v1.0 file hashes, and licence and hype cautions.
- Re-ordered the README next steps to match.

## [2026-09-14] - CUDA brain backend: the connectome flies faster than real time

- Added `src/noflyzone/brain_cuda.py`: `ShiuBrainCUDA`, a drop-in subclass of
  the SciPy reference with GPU state, three Triton kernels (per-neuron update,
  atomic-append delay ring, per-spike CSC delivery) and one CUDA graph per
  game tick. Added `brain_backend.py` (`make_brain(backend)`), a `--backend
  cpu|cuda|auto` flag on `run_m3.py` and `validate_brain.py`, a `cuda`
  optional dependency (torch) in `pyproject.toml`, and `ConnectomeBrain.step`
  now uses `run_counts()` when the brain offers it.
- Added `scripts/compare_backends.py`: per-step spike sets identical to the
  CPU for 500 steps under a deterministic kick-every-step drive (0 of 1500
  traced steps differ once FP fusion is disabled in the neuron kernel; with
  fusion on, the first differing step was 877); 0.3 s aggregates within 1%
  (float32 atomic summation order, ~1e-6 mV, amplified by the recurrent
  network; a reordered-summation CPU control never diverges); validation
  gates agree (MN9 100-102 vs 101 Hz, giant fibre 66-70 Hz on both).
- Adversarial review (three lenses plus a judge) found no semantic
  divergence; applied its fixes: snapshot restore in a `finally` and the
  default stream restored on a failed capture, the delay ring sized to one
  entry per neuron with a grid-stride delivery kernel (no overflow path),
  int64 step counter, graphs keyed by seed, launches pinned to the brain's
  device, sorted `step()` output, FP fusion off.
- Measured on the RTX 4090 after the review fixes: one tick 5.5 ms versus
  263 ms on the CPU; `validate_brain --backend cuda` 1 s of sim in 0.1-0.2 s;
  `run_m3 --backend cuda` 450 ticks in 5.8 s wall, 5.2x real time, brain
  8.9 ms of a 12.5 ms tick, loop about 80 Hz. The CPU path is unchanged and
  remains the reference.
- Fixed the `run_m3.py` summary line to report speed as a multiple of real
  time in both directions.

## [2026-09-14] - Workstation environment and full validation ladder

- Created `.venv` from the uv-managed Python 3.12.14 (system 3.14 cannot
  bootstrap pip) and installed the package: mujoco 3.13.0, ale-py 0.12.1,
  gymnasium 1.3.0, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, pyarrow 25.0.1,
  pillow 12.3.0, imageio 2.37.4.
- Fetched the flybody model, Shiu brain model and flywire annotations (138 MB
  assets, 421 MB data) with `scripts/fetch_assets.py`.
- Ran the whole ladder under `MUJOCO_GL=egl`, all PASS: `smoke_battlezone`,
  `smoke_cockpit_screen` bare and with the fly, `run_m2 --video` at about
  233 Hz, `validate_brain` reproducing the Mac numbers exactly (MN9 101 Hz,
  giant fibre 66/64 Hz, 1 s sim per 5 s wall), and `run_m3 --steps 90 --video`
  at 7.7x slower than real time with the brain at 492 ms per tick.
- Confirmed CRT whites reach 255 in the eye renders, so `WHITE_REF` stands on
  the NVIDIA GL path. Noted the MuJoCo 3.13 attach-conflict warning caused by
  the flybody's option block (0.1 ms timestep, centimetre gravity), harmless
  for the kinematic puppet but a decision point for embodied mode.
- Corrected the audit: the DNp01 startle does fire when both giant-fibre cells
  spike in the same tick (seen in the M3 run), not never.
- Recorded results and versions in `README.md` and `docs/architecture.md`.

## [2026-09-14] - Workstation clone and code audit

- Cloned `JackDanHollister/no-fly-zone` onto the Ubuntu 26.04 / RTX 4090
  workstation at `projects/no-fly-zone/` and registered it in The Magi Home
  Edition workspace (roster, repo-boundary note, gitignore).
- Audited every module and script against the milestone list and recorded the
  verified state in `README.md` (Status, Next Steps, workstation setup) and in
  the milestone checklist and new status-audit section of
  `docs/architecture.md`.
- Result: M0 to M3 done, M4 and M5 partial, M6 and M7 not started. Recorded the
  gaps between docs and code: no GPU backend exists, only FlyWire v783 is
  loaded, no `--pilot` flag, oviDN stands in for pIP10, the damage lamp is
  outside the eye cameras' field of view, and the idle scan in `run_m3.py`
  masks whether the connectome steers.
- Nothing has been executed on this machine; no environment, assets or data
  are present yet.

## [2026-09-14] - Whole-brain LIF core and the connectome flies the tank (Mac)

- Cockpit art pass in two commits: deck plating, starboard wall, gantry, neon
  trim, console details, nameplate and CRT-glow lighting; then armour glacis,
  rivets, periscope, ammo crate, helmet, shell casings, dog tags, kill tally
  and war-poster decals.
- Added `brain_lif.py`, a sparse NumPy/SciPy port of the Shiu et al. 2024
  whole-brain LIF model over FlyWire v783 (138,639 neurons, 15.1M edges), and
  `scripts/validate_brain.py`. Recorded on the Mac: silence clean, sugar GRNs
  drive MN9 at 101 Hz, LC4 drives the giant fibre at 66/64 Hz, one second of
  simulation in about five seconds of wall time.
- Added `brain_connectome.py`: retinotopic Poisson injection into R1-6 and
  Mi1/Tm1/Tm2, real descending-neuron readout (oviDN stands in for pIP10 in
  the female brain) and a spike-cloud panel; `scripts/run_m3.py` drives
  Battlezone from the connectome at about 2 Hz loop rate on the Mac.
- Dashboard footer names the pilot per script; `fetch_assets.py` became the
  one-command fetch for flybody, the Shiu model and flywire annotations
  (about 570 MB); README quickstart written.

## [2026-09-13] - Scaffold, screen-in-cockpit, closed loop and seated fly (Mac)

- Scaffolded the src-layout package, BattleZone wrapper and MuJoCo cockpit
  with the game frame uploaded as a CRT texture; smoke scripts prove an eye
  camera sees the game (M0, M1).
- Closed the loop with a hand-written reflex brain: hex retina with 721
  ommatidia, DN contract, motor mapping, `PilotLoop` and `run_m2.py`. Review
  fixes added the right-way-up orientation gate, a single source of truth for
  `dt`, photoreceptor calibration, damage and episode signals, and sticky
  actions off (M2).
- Seated the flybody model in the cockpit via MjSpec attach with contact-lens
  eye cameras and a bare-cockpit fallback; first `fetch_assets.py` (M2.5).
- Added the split-screen dashboard compositor and raised the offscreen
  framebuffer for hero renders.
