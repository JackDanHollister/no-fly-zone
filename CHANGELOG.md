# Changelog

## [2026-09-18] - Night session: the male in the game, and direction through the wiring

- `READOUT_MAPS["pool_male"]` from the male's own screen at weights x0.6
  (steer DNp04, DNb05, DNp02, DNp20, DNp07; loom DNg33, DNp12, DNp01).
  `bench_pilots.py --dataset malecns`, ten independent episodes: the male
  kills 1.2 and dies 2.7 per 100 s (female 2.7 / 5.7), reverses on half its
  ticks, steers toward the target 10 of 12 gated looks; its shuffles never
  cross the gate. `make_video.py sidebyside --preset fight` cuts seed 0 of
  each into `out/no-fly-zone-female-vs-male.mp4` (the tool now streams
  frames instead of re-decoding per frame).
- `scripts/probe_motion.py`: bars in four directions, orientation-free test
  of T4/T5 subtype opponency. Null for v4 and v5 (new map: v4 plus Mi4 as
  low-passed luminance and Mi9 as low-passed darkness, tau 150 ms): every
  subtype at 1.5-4.5 Hz in all directions in every brain. A point LIF with
  one fixed delay does not do the T4 computation.
- Injection map v6: a Hassenstein-Reichardt correlator over neighbouring
  ommatidia, four ON directions injected at T4a-d and four OFF at T5a-d,
  replacing exactly that stage and labelled engineered. Downstream, the
  measured FlyWire wiring makes HSE/HSN/HSS horizontally selective
  (93-116 Hz versus 0-11) and VS1-4 vertically selective (silent for the
  null direction), and ten descending types direction-selective at both
  sweep speeds (DNg41, DNp15, DNp22, DNae002, DNa15, DNae010, DNa10, DNa02
  at 48 Hz, DNa16, DNge033); the within-hemisphere shuffle leaves HS/VS
  flat and has no consistent descending type. The male at x0.6 replicates
  it (HSE 168 / 0.6 Hz, H2 opposite to HS as in the fly; 47 direction-
  selective descending types against 8). Selectivity screen with v6:
  20 direction-selective descending types to a bar against 1-6. In the
  game with pool4, v6 changes nothing (2.5 kills, 45 of 47 correct steers):
  the readout reads position, not motion. `lattice_neighbours` and
  `reichardt` are module functions with a no-data unit test.
- `probe_motion.py --yaw` (both eyes swept together) and injection map v6m
  (right eye's horizontal channels mirrored, as the fly's eyes are): the
  two HS systems oppose (HSE left 113 / 0 Hz, right 11 / 140), H2 the other
  way, and 14 descending types flip their right-minus-left rate with yaw
  direction (DNp15 to 98 Hz, DNg41, DNa02 43 / 3 versus 1 / 22, DNae002,
  DNa15, ...) against 3 weak ones in the shuffle; with v6 (no mirror) only
  4. `READOUT_MAPS["yaw"]` reads the five strongest, sides swapped to steer
  toward motion. In the game the yaw pilot kills 2.4 and dies 4.3 per
  100 s against pool4's 2.2 and 5.9 on v6m; the position-based steering
  metric cannot score a motion readout (a motion-signed metric is next).
  `run_m3.py` HUD now shows the loaded dataset and its neuron count
  instead of the FlyWire constants.

## [2026-09-18] - Selectivity screen, MaleCNS loader, second-individual replication

- `scripts/screen_selectivity.py`: every descending type's ipsilateral
  response to a small moving disc, a tank-sized disc, a loom, the same loom
  played backwards (receding disc, after nsfm/fly-afterlife withdrew a loom
  claim that was reading brightness rather than expansion), a full-field
  flash and bars in both directions; indices for size tuning, expansion
  versus contraction, loom versus flash, loom versus bar and direction;
  every brain gets the identical procedure. FlyWire: 58 responsive DN types
  against about 355 in three within-hemisphere rewirings (287-312 when
  gain-matched), small-object drive up to 47 Hz (DNg100) against 14, and
  14 expansion-selective types (DNp12 among them) against 0-2. No brain
  has size tuning or direction selectivity: a front-end limit.
- `scripts/build_malecns.py`: MaleCNS v1.0 flat connectome (hash-pinned,
  CC-BY 4.0) converted into the loader's three files: 166,700 neurons,
  25,582,938 signed pairs, annotations in the FlyWire vocabulary, "R1-R6"
  renamed "R1-6", manifest with every decision. `brain_lif.DATASETS` and
  `annotations(version)`; `--dataset malecns` on validate_brain, run_m3,
  bench_pilots, screen_dns and screen_selectivity; the v0 readout reads
  pIP10 itself where it exists. `tests/test_malecns.py`.
- MaleCNS at Shiu gains is over-excited (11,000 spiking neurons per tick
  against FlyWire's 2,000 on the same stimuli; pools saturated at 74 Hz)
  with a narrow window, as flybench reports; at weights x0.6 it matches
  FlyWire's regime and replicates the pattern: 84 responsive DN types
  against 412-424 in its own shuffles, DNg100 the top small-object
  responder in both flies, DNp12 expansion-selective in both, DNp07
  lateralised in both. The female pool4 does not transfer as a whole.
- Sweep of the fly-connectome wave since the 14th, kept privately in the
  workspace: another group finds a degree-preserving rewiring recruits 8x
  more cells at matched rate; no one else has a within-hemisphere control.

## [2026-09-18] - Wiring controls: the connectome keeps left from right; a random graph does not

- `ShiuBrain(ablate="rewired_ipsi")`: targets permuted within each
  hemisphere (annotation `side`), so gross laterality survives and only the
  within-hemisphere circuit is destroyed; `weight_scale` multiplies every
  weight for gain-matched controls. Both recorded in `meta` and on the HUD;
  `run_m3.py` gains `--ablate rewired_ipsi`, `--ablate-seed`,
  `--weight-scale`. `tests/test_controls_wiring.py` checks the shuffle keeps
  every cell's total output and each side's total input.
- `screen_dns.py` takes the loader flags and prints a summary: responsive
  DN types, how many lateralise consistently, the pool4 steer pools' LI and
  the firing regime. Measured brain: four types at |LI| > 0.8, pools LI
  +0.92. Three whole-brain rewirings: 0-1 types, pools LI +0.03 / -0.11 /
  +0.21; scaling their weights by 0.7 matches the measured regime (2,193
  versus 2,188 spiking neurons per tick) and changes nothing. The
  within-hemisphere rewiring: eight types at LI +1.00.
- `bench_pilots.py`: `--ablate-seeds`, `--rewired-scales`, per-condition
  tokens (`rewired@1x0.7`, `rewired_ipsi:pool_ipsi0`), and a steering
  correctness metric: on look ticks where the retina's transient energy is
  clearly on one side and the pools cross the steering gate, does the sign
  of the pool LI match? Pooled over ten 100 s episodes: measured wiring
  62/63 correct (binomial p 7e-18); five rewirings 66/165, each at chance
  or below with its own fixed bias; gain-matched rewiring 15/15 but always
  answering right with no left events; the within-hemisphere control with
  the measured pools crosses the gate twice, with its own screened pools
  (`READOUT_MAPS["pool_ipsi0"]`) 17/17. Kills do not separate any of them
  (1.9-2.7 per episode against 2.7). Structurally, each measured steer pool
  gets its two-hop visual input from its own eye at about 100:1; every
  rewiring is balanced. Write-up: `docs/research/wiring-controls-2026-09-18.md`.
- Found that Battlezone ignores the gym seed (identical frames for seeds 0
  and 1), so the 2026-09-14 ten-seed benchmarks were ten replicas of one
  episode per deterministic pilot. `bench_pilots.py` now resets the brain
  and opens each episode with a seeded burst of 15-60 random ticks. A first
  rerun with replicas had made one rewiring look right 27/32; with
  independent episodes it is 11/29.

## [2026-09-14] - Learning results: readouts learn to survive, not to kill; the mushroom body cannot see

- PPO readouts (410k ticks each): kills flat at about one per episode for
  the connectome, rewired and retina heads; deaths fell from about three to
  1.2, 0.7 and 1.4. Greedy evaluation on four fresh seeds
  (`scripts/eval_readout.py`): connectome head 2.25 kills / 0.25 deaths,
  rewired 0.38 / 0.38, retina 0.88 / 0.38, random actions 2.25 / 3.38, reflex
  pilot 1.25 / 2.62. The connectome head alone kept its kill rate while
  surviving; one training seed, so suggestive only.
- Mushroom-body plasticity, rule on versus LR-zero, 48 episodes each: 74 of
  62,261 synapses changed, behaviour identical within noise. Cause measured:
  the calyx-projecting visual neurons that fire (aMe12, 7 cells) give each
  visual Kenyon cell about 2 mV against a 7 mV threshold, so the mushroom
  body cannot see the game at Shiu gains. Written up in
  `docs/research/learning-2026-09-14.md`.

## [2026-09-14] - Learning infrastructure: batched brain, PPO readout, mushroom-body plasticity

- `brain_cuda.py` now batches: `batch=B` runs B flies over one connectivity
  allocation with per-fly state, Poisson rates and delay ring; fly 0 of a
  batch is bit-identical to the unbatched run (new section [D] of
  `compare_backends.py`); `reset_rows()` restarts individual flies;
  `set_edge_weights()` writes into the shared weights in place between graph
  replays. Three flies cost 4.2 ms per fly-tick, less than one alone.
- `rl_env.FlySquad`: B game-and-cockpit loops through one batched brain,
  observations as spike counts pooled by cell type over the descending,
  visual-projection and visual-centrifugal neurons (907 features), or the
  photoreceptor drives directly (the no-brain ceiling); `PilotLoop.sense()`
  splits sensing from the brain step.
- `scripts/train_readout.py`: PPO on a small actor-critic over those
  features, with `--brain connectome|rewired|silenced` and `--features
  brain|retina`; `scripts/play_readout.py` plays a trained head with a
  TRAINED READOUT footer. Runs launched: connectome, rewired and retina,
  eight flies each, 400 updates.
- `plasticity.py` (M7): dopamine-gated three-factor rule on the 62,261
  measured KC to MBON edges, teaching pulses into the real PAM (kill) and
  PPL1 (hit) populations through new extra Poisson sites on
  `ConnectomeBrain`, presynaptic eligibility, centred dopamine, bounded
  weights, `rule=lr0` and `edge_set=shuffled` controls;
  `scripts/train_plasticity.py` runs the reflex pilot while the mushroom body
  learns. `scripts/audit_mb_route.py` showed the substrate exists (visual
  Kenyon cells two hops from the injection, MBONs one hop from the steer
  pools). Unit check: a reward pulse drives PAM to 100 Hz and moves weights.
- Finding: the mushroom body is barely in the visual loop at Shiu
  parameters. A full-field flash activates about 5 of 5,177 Kenyon cells and
  no MBON; a loom plus a moving bar activates one. Any learning effect is
  bounded by that, and the runs are expected to report it.

## [2026-09-14] - Wall art laid out in view, and a sweetheart portrait

- The starboard wall art was placed by guess and the beauty camera saw only
  part of it: one war poster was half off the right edge, the other entirely
  off-screen, and the nameplate and tally sat on top of each other. The wall
  is now laid out in three bands inside the camera's view (nameplate; the
  three posters; tally and aux screens), the beauty lens is 50 degrees
  instead of 45, and nothing overlaps or clips.
- New decal: a pixel-art portrait of a female fly, pinned as a sepia photo
  with a brass frame, two hearts and "COME HOME SAFE", painted in code like
  the other decals (`_SWEETHEART` pixel map in `cockpit.py`). The one she is
  fighting for. Hero image and demo media refreshed.

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
