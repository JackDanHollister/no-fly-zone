# NO-FLY-ZONE — architecture and reproduction notes

Current implementation, reviewed 2026-11-19. The original design evolved
substantially: the default is FlyWire, CUDA exists, idle scanning is optional,
physical controls work, and learning has been tested. For that chronology,
read [JOURNEY.md](../JOURNEY.md); for individual changes, [CHANGELOG.md](../CHANGELOG.md).

## The loop

```mermaid
flowchart LR
    Game[Battlezone / ALE] -->|RGB frame| Screen[CRT texture in MuJoCo]
    Screen --> Eyes[Two eye-camera renders]
    Eyes --> Retina[Hex retinas: 721 samples per eye]
    Retina -->|Engineered injection| Brain[Whole-brain LIF model]
    Brain -->|Descending-neuron rates| Controller[Readout and motor rules]
    Controller -->|Wired mode| Action[ALE action]
    Controller -->|Embodied mode| Legs[IK and flybody servos]
    Legs --> Levers[Contact-driven levers and button]
    Levers -->|Measured positions| Action
    Action --> Game
```

`PilotLoop` owns the loop. The game advances at 15 control ticks per second
(4 emulator frames at 60 Hz); each tick applies the action selected from the
previous percept. GL rendering stays on one thread for macOS compatibility.
The brain never receives the emulator's object coordinates or a target detector.
Game reward and damage are available separately to the learning machinery.

| Component | Responsibility | Main implementation |
|---|---|---|
| Game | ALE registration, actions, RGB frames, reward, life loss and automatic game-over reset | `src/noflyzone/game.py` |
| Cockpit | Scene, textured screen, body attachment, rendering and optional physical controls | `src/noflyzone/cockpit.py` |
| Retina | Hexagonal image sampling, luminance and transient drives | `src/noflyzone/retina.py` |
| Brain engine | Sparse connection matrix, LIF state, delayed spikes and Poisson input | `brain_lif.py`, `brain_cuda.py`, `brain_backend.py` |
| Brain interface | Cell annotations, input maps and descending-neuron pools | `src/noflyzone/brain_connectome.py` |
| Controls | Rate calibration, steering, startle, movement gating and action mapping | `dn.py`, `controls.py` |
| Experiment layer | Batched flies, PPO feature readouts and three-factor plasticity | `rl_env.py`, `plasticity.py`, scripts |
| Inspection | Dashboard, live window, trace files and provenance | `dashboard.py`, `liveview.py`, `provenance.py` |

## Launching and recording

`./no-fly-zone` resolves the project environment from its own location,
then calls `scripts/pilot.py`. It checks the required files and Python
packages, tests a real MuJoCo render and ALE start, and checks the video
encoder before constructing the full brain. Linux uses EGL unless the
caller selected a different `MUJOCO_GL` backend.

The default is a 60-second live recording on a desktop, or a headless
recording without a display. `watch`, `record`, `doctor` and `status` are
explicit modes. Recordings and traces receive unique UTC filenames in
`out/`; the status command reads the latest campaign's durable manifest
and completed episode records without loading the model or using the GPU.

The launcher calls `scripts/run_m3.py --preset active` by default (pure frozen
connectome reflex, zero RL), which sets `--injection v11 --readout pool4
--zoom 1.75 --eye-offset 0.21 --center-y 0.48 --eye-fovy 54 --converge-eyes
--suppress-turn-transients --saccade-hold 1 --startle-ticks 5
--startle-refractory 18 --startle-thresh 0.80 --steer-mode li
--steer-min-hz 8 --sharp-steer-hz 6 --search-steer 1 --search-delay 2
--patrol-sweep 6 --patrol-stride 4 --fovea-li 0.18 --pip-lock-drive 0.50
--no-off-axis-fire --direct-steer --aminergic zero --sensory external_only`.
Pass `--controller legacy` (or `--preset pilot`) for the historical September
25 configuration, or `--controller learned` to opt into a saved GRU readout
checkpoint. Explicit options override preset values. The runner's bare defaults
still use the early reference configuration. `--seed` reaches both the
single-pilot game and neural backend; the learning protocol separately manages
its independent game/neural roots.

Live playback uses `--realtime` to cap speed at 15 game ticks per second;
it does not drop neural ticks when computation is slower. Record-only
mode runs as fast as the hardware permits. Both save video at 15 fps.
Closing a live window or pressing Q/Escape ends the loop, and the trace
records actual completed steps alongside the requested duration. Ctrl-C
during play also preserves completed work. An exit stack closes the video
writer, window and cockpit even if an error interrupts the run.

## Eyes and neural input

The retina is a project-local hex sampler; it does not import flygym's
retina. Eye cameras are fixed in the cockpit near head height. The sampler
uses image-plane windows, not anatomical rays through every ommatidium.
`WHITE_REF=255` was checked on the workstation's NVIDIA render path.

The default `--preset active` configuration uses injection `v11` (`v4` plus
signed 2D `R8 − R1-6` chromatic opponency into FlyWire's `Tm20 / Tm5a / Tm5b`
medulla neurons and `T3`), converged `eye_fovy = 54°` cameras, `zoom = 1.75`,
`eye_offset = 0.21` and `center_y = 0.48`, aligning each eye's anterior `LC10a`
fovea cleanly on either side of the crosshair with binocular overlap on
`x = 80` and zero bezel bleed. The historical `--preset pilot` (`--controller
legacy`) keeps `v4` with parallel `fovy = 45°` cameras, `zoom = 2` and
`eye_offset = 0.15`. Input rates are Poisson drives into named visual cell
populations. The medulla channels bypass limitations of the earliest visual
pathway in this model.

| Map | Purpose | Interpretation |
|---|---|---|
| `v0` | Early photoreceptor and medulla injection | Retained as the advanced runner's bare default/reference configuration |
| `v4` | Expanded ON/OFF and small-object-relay inputs | Pure achromatic baseline for tracking and historical reflex comparisons |
| `v5` | Additional slow Mi4/Mi9 channels | Did not establish T4/T5 direction selectivity |
| `v6`, `v6m` | Retinal motion correlator injected at T4/T5, optionally mirroring the right eye | Direction is engineered at the input; downstream routing is tested |
| `v7` | Retinal centre-surround processing | Insufficient for small-versus-loom selectivity |
| `v8` | Surround plus wide-field suppression across channels | Also creates selectivity in shuffles and suppresses looming |
| `v9` | Small-object channel restricted to T2/T2a/T3 | Measured wiring selectively routes this supplied signal |
| `v10` | `v4` plus `R8` blue and wide-field-opponent `T2` | Experimental feature front end for bilateral GRU readouts |
| `v11` | `v4` plus signed 2D `R8 − R1-6` blue-ON opponency at `Tm20/Tm5a/Tm5b` and `T3` | Default active-reflex front end; models histaminergic `R8/Dm9` chromatic surround so stationary targets stay visible (`0.0 Hz` in tested shuffles) |

These are alternative experimental configurations, not a single cumulative
biological reconstruction. `FlySquad` uses the baseline input path; do not
assume it implements every newer map available in the standalone pilot.

## Datasets and model

FlyWire v783 is loaded from the connectivity distributed with the Shiu
reference implementation. There are 138,639 neurons and 15,091,983 input
connection records. Each record represents a directed connection weighted
by anatomical synapse count. Earlier notes sometimes called the record
count “synapses”; it is not the number of individual anatomical synapses.

MaleCNS v1.0 is converted into the same index, connectivity and annotation
formats by `scripts/build_malecns.py`: 166,700 retained neurons and
25,582,938 directed pairs. The loader verifies the three source files
against pinned SHA-256 hashes and records conversion decisions in a manifest.
It includes VNC data, but the physical controller does not use a neural VNC.

Both engines implement the Shiu-derived point-neuron LIF dynamics with
0.1 ms steps, a 20 ms membrane time constant, 5 ms synaptic time constant,
1.8 ms delay and 0.275 mV per anatomical synapse. Transmitter labels supply
signs; anatomical counts supply the baseline magnitudes. This does not
recover all physiological properties of the measured animal.

Two playing-mode corrections are explicit options:

- `--aminergic zero`: aminergic connections have no fast synaptic weight.
  This prevents the persistent olfactory-region state observed in the
  uncorrected sign convention. Literature transmitter labels are used for
  identifying these cells.
- `--sensory external_only`: incoming recurrent connections to primary
  sensory neurons carry zero weight; those neurons receive injected input.

The reference settings remain the loader default so the reference gates can
still be run. Settings and dataset are stored in brain metadata.

## Readouts and controls

The baseline `pool4` readout uses screened descending types for left/right
steering, DNg100 for drive and a pool for looming/startle. Coincident steer
pool activity supplies its fire signal. The older `v0` mapping reads oviDN
under the legacy `pIP10` contract key in the female, which lacks pIP10;
MaleCNS can read pIP10 itself. That old label is not a biological explanation
of the pooled pilot's firing behaviour.

`dn.py` owns Hz-to-drive calibration. `controls.py` supplies lateralisation
index steering, a three-tick movement suppression period and a 15-tick
startle refractory in the playing configuration. These are engineered motor
rules. Idle scanning is off unless `--idle-scan` is supplied.

Fixed `pool4` calibration is provisional but remains the baseline. Adaptive
calibration tracks each pool's decaying running maximum with a 4 Hz floor.
It did not rescue the harmful learned brains and did not improve the
baseline in the recorded calibration comparison.

## CPU and CUDA

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

Batched execution shares the connection matrix, with separate neuronal
state and input for each fly. `reset_rows()` resets selected neural states;
`set_edge_weights()` updates shared weights between graph replays.

The equivalence gates establish the tested short-horizon spike matches and
aggregate agreement. They are not a claim of bit-identical trajectories
for arbitrary durations or Poisson generators.

Historical measurements on Ubuntu 26.04, Python 3.12.14, RTX 4090,
192 GB system RAM, headless EGL, 2026-09-14:

| Gate or run | Recorded result |
|---|---|
| CPU feeding / escape gates | MN9 101 Hz; giant fibre 66/64 Hz, matching the Mac run |
| CUDA feeding / escape gates | MN9 about 102 Hz; giant fibre 66–70 Hz |
| CPU pilot, 90 ticks | Brain 492 ms per tick; about 7.7× slower than real time |
| CUDA pilot, 450 ticks | 5.8 s wall time for 30 s game time; brain 8.9 ms per tick |
| Isolated brain tick | 263 ms CPU versus 5.5 ms CUDA |
| Three-fly batch | About 4.2 ms per fly-tick |
| Embodied pilot, September 20 | About real time; physics roughly 36–37 ms per tick |

The early timing runs predate some later rendering and controller features.
Use run-specific wall-clock reports when comparing performance. Videos are
encoded at game speed.

## Embodied mode (M6, 2026-09-20)

`Cockpit(physics=True)`, reached by `run_m3.py --pilot embodied`. What is
physical and what is not, in order of the loop:

1. The descending neurons' rates go through the same controller as in
   wired mode and give an intended steer and throttle.
2. The intent is turned into a target angle for each lever (a full command
   is 0.3 rad) and into T1 leg joint targets by a damped least-squares
   inverse kinematics on the claw site, so the claw follows the knob's arc.
   Those targets go to the flybody's own leg position servos (gain 0.8).
3. MuJoCo steps 667 times at 0.1 ms per game tick with the flybody's option
   block. The thorax is on the seat frame (a tethered fly); the T1 claws
   adhere (the model's adhesion actuators, on); the levers are fly-scale
   bodies (density 1.18, a 0.05 return spring) that collide only with the
   claws. The fly's own body-part collision geoms are off, tarsi and claws
   excepted, because the thorax mesh interpenetrates head and coxae at the
   stance and a welded fly chatters at 1,000 rad/s from it.
4. The lever hinge angles are read, zeroed where the legs' neutral
   command leaves them, normalised by the 0.08 rad a fly's push achieves
   (measured; the hinge travels 0.55), and become `controls.measured`; the
   ALE action is taken from those, not from the intent, through a dead
   zone of 0.35 of the push range, because a pushed lever rests 0.15-0.26
   from zero when released (a real hysteresis of claw, adhesion and
   servo; the return spring does not change it) and without the dead zone
   that rest sat on the reverse threshold and the fly backed up for 940 of
   1,500 ticks. The treads on the dashboard show the physical levers.
5. Fire is physical too: the button is a sprung slide placed where the
   fully extended proboscis reaches (1 mm into the tip's path, spring
   8 against a gravity load of 0.005 and a servo of 0.1), colliding only
   with the rostrum, haustellum and labrum geoms; the fire intent
   extends the rostrum and haustellum to their limits and the shot is
   taken when the button reads 0.35 of its travel (rest 0.08, pressed
   0.57). The fly fires by extending its proboscis onto the button. The
   wings, whose joints are torque motors with no command, are held at the
   stance by a weak spring.

Measured: the gate `scripts/validate_embodied.py` moves the levers to
+1.0 forward, -0.7 back, and (+0.9, -0.7) for a turn, with the claws in
contact throughout and no chatter; physics costs about 32 ms per 66.7 ms
game tick on the workstation CPU, so the embodied pilot runs at about
real time with the CUDA brain. A MuJoCo 3.13 quirk found on the way: with
the levers' contact flags written in the XML on every geom of the body,
the claw contacts vanished; set after compile they are kept.

Played: on seed 0, 100 s, with the levers physical and fire still
puppeted, the embodied pilot scored 3000 with seven lives lost and the
same action mix as the wired pilot (NOOP 1,335, DOWN 111, UP 20, FIRE 21
against 1,325 / 108 / 29 / 28); with fire physical too it scores 2000
with three lives lost and 16 fire ticks against the wired pilot's 28,
the proboscis being slower to the button than a puppeted switch. Both at
about real time with physics at 36-37 ms per tick; the levers reach
-0.75 to +1.0 of the push range. Before the proportional loop and the
dead zone the fly held reverse for 940, then 1,399, of 1,500 ticks.

Open: the achieved lever angle is about a quarter of the IK target (servo
gain against spring and contact), which the normalisation absorbs; the
VNC for the male; the fly's hind legs do nothing.

## Reproducing the comparisons

Start with the [README setup and full pilot command](../README.md#run-it).
The `run_m3.py` defaults intentionally retain the early `v0` reference path;
use the complete playing flags for the recorded baseline.

The corrected pilot benchmark varies the start using a seeded burst of
15–60 random actions. It supports whole-brain shuffles, within-hemisphere
shuffles, silencing and gain scaling. See its `--help` and the commands in
[the wiring note](research/wiring-controls-2026-09-18.md).
A bare `--seed` change in `run_m3.py` does not introduce that warm-up.

For the male, obtain the v1.0 flat-connectome files from the
[official download page](https://male-cns.janelia.org/download/), place
these in `data/malecns/`, and run `.venv/bin/python scripts/build_malecns.py`:

- `body-annotations-male-cns-v1.0-minconf-0.5.feather`
- `body-neurotransmitters-male-cns-v1.0.feather`
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather`

The converter checks the version by hash; it is not a downloader. Add
`--dataset malecns --weight-scale 0.6` to supported scripts and use
`--readout pool_male` for the recorded male pilot. The corrected placement
uses annotated hex columns and partner-derived columns for missing cells.
The residual photoreceptor count imbalance remains documented.

## Learning protocol and historical limitations

`train_readout.py` trains a PPO action readout on a frozen connectome,
a shuffled connectome or retinal features. `train_plasticity.py` instead
changes selected connections in the spiking model while the fixed reflex
controller plays. Rule-off and size-matched shuffled-edge controls are
available; the initial mushroom-body experiment did not run the shuffled
control after finding negligible activity.

**The September 22 review found that the independent-start fix in
`bench_pilots.py` had not reached `rl_env.FlySquad`.** Eight requested seed
labels produced identical frame sequences under the same 450 actions.
Training labels also did not reach the neural RNG. The historical training
runs have different realised histories, but their labels do not establish
independent neural seeds or held-out game starts.

Furthermore, a fixed tick-budget boundary reset episode bookkeeping without
resetting the game; neural rows reset only on game over. Thus 48 historical
“episodes” are not necessarily 48 independently reset games. Episode-level
standard errors in the historical tables should not be treated as accounting
for all dependence.

The September 25 implementation uses `independent-starts-v1`:

- Each game root and episode index derives a reset seed and a separate
  random-action warm-up seed. Warm-up lasts 75–150 game ticks (5–10 seconds),
  beyond the input-insensitive opening that made two shorter warm-ups
  produce identical action traces. Warm-up rewards are excluded.
- Every episode records its warm-up actions, initial lives, image and RAM
  hashes, neural seed and row. `eval_plasticity.py` audits the exact planned
  starts for reproducibility and distinct action traces before scoring.
- The neural root is explicit; both training scripts forward `--seed`.
  Per-episode neural seeds and random-counter origins live in device tensors
  read by the CUDA graph, so a row restarts its stream independently of
  other flies. Unseeded `reset_rows()` retains the older continuing stream.
- Game over and the tick limit both reset the game, neural dynamics,
  sensory history, teaching rates and controller. The terminal frame is
  sensed before resetting; `last_counts`/`last_dn` retain terminal activity
  for a learning update, while returned observations describe the next
  episode. A reset does not consume an unreported NOOP tick.
- The plasticity trainer conservatively clears its pooled traces, dopamine
  baseline and pending pulses when any row ends. Shared learned weights
  remain. Independent training histories still require separate runs.
- Frozen comparisons use exact per-fly episode quotas and no external
  teaching pulses. The latter differs from historical rule-off runs and is
  explicitly recorded. Completed rows no longer accumulate extra episodes.

Historical observations and checkpoint weights are preserved. See the
[learning audit](research/learning-2026-09-19.md#evaluation-protocol-audit-2026-09-22)
and the September 25 protocol recorded immediately after it. A corrected
evaluation of old weights does not fix the independence of their training.

With the historical checkpoint files available locally, reproduce the
seven-condition frozen comparison with a new output filename:

```bash
MUJOCO_GL=egl .venv/bin/python scripts/eval_plasticity.py \
  --out out/eval_plasticity_reproduction.json
```

The default is four roots (900–903), four reset episodes per root and a
900-tick horizon. `--seeds`, `--neural-seed` and `--weights` are explicit
overrides. Existing outputs are refused rather than overwritten. The
checkpoint files remain local; the review evidence includes their hashes.

The fresh [September 25 bounded campaign](research/learning-campaign-2026-09-25.md)
uses `run_learning_campaign.py` and a JSON plan hashed before training.
Four independent histories each run latched visual-to-descending plasticity,
rule off, and size-matched shuffled-edge plasticity. The runner verifies
training quotas, recorded starts and unchanged rule-off arrays, then tests
the eight changed checkpoints plus measured weights on two reserved panels.
Source/plan changes, errors and the two-hour wall limit stop it as incomplete;
there is no automatic retry or selection of the best checkpoint.

All 288 training and 288 evaluation episodes completed in 80.25 minutes.
Every predeclared gate failed in both panels; each learned history had lower
utility than both measured weights and its matched shuffled-edge control.
This specific campaign is closed.

The November 17 continuation ([research note](research/active-vision-and-tracking-2026-11-17.md))
adds:

- **Converged binocular eye geometry (`Cockpit(eye_fovy=54.0, converge_eyes=True)`):**
  converges both eye cameras onto the CRT centre (`(0, 0.25, 0.10)`) with
  `zoom=1.15, eye_offset=0.11, center_y=0.46`, eliminating the central crosshair
  blind spot (`x ≈ 74..86`) and bringing the top radar and cockpit lamp edge
  into both retinal lattices with zero bezel bleed. Legacy defaults remain
  unchanged unless requested (`--preset active` or `--converge-eyes`).
- **Active saccadic foraging (`ACTIVE_PRESET` / `scripts/bench_tracking.py`):**
  combines 1-tick post-turn fixation (`saccade_hold=1`), self-turn transient
  gating, active search when steer pools are quiet (`search_steer=1.0`), and a
  binocular crosshair fovea (`fovea_li=0.20`).
- **Bilateral recurrent GRU readout (`features="brain_bilateral"`, `RecurrentActorCritic`):**
  separates `(cell_type, side)` into `1,799` pools on FlyWire v783 (instead of
  collapsing left/right into `908` symmetric features), applies a variance floor
  (`var_floor=1.0`) in `RunningNorm` so near-silent pools do not amplify stray
  Poisson spikes, and feeds a `256`-unit `GRUCell` with motor efference copy and
  `3 × 3 × 2` factored `(throttle, steer, fire)` heads. `./no-fly-zone` exposes
  `--controller {auto,learned,active,legacy}` in both wired and `--embodied`
  modes, labelling the footer `TRAINED READOUT (GRU)` whenever a trained
  checkpoint drives the controls.

## Tests, artefacts and remaining limits

The November 17 full suite passes all 54 tests with
`MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest`.
This includes real ALE start diversity/reproduction, independent episode
boundaries, explicit GPU stream resets, CPU/GPU equivalence, connectome
gates, the physical lever/proboscis test, launch presets, early window-close
cleanup and the campaign's control/consistency decision rules. The flybody attachment emitted
its known non-fatal constraint/keyframe-limit warning. A short two-fly,
two-episode run also passed the real frozen evaluator's start-matching gates.

New run artefacts use `provenance.py` for commit, dirty state, settings,
package versions and an identity hash. Older artefacts predate that change.
The frozen learning runners additionally record source and checkpoint
SHA-256 values, identifying the uncommitted code used for those experiments.
The small [review evidence bundle](research/evidence/) contains final-learning
rows and the seed audit; the complete raw run archive, models, fetched data
and full videos remain in gitignored `out/`, `data/` and `assets/`.

Remaining implementation limits worth carrying into future work:

- The damage lamp is outside the default eye view. The game's own damage
  flash can reach the retina; these are different stimuli.
- The CRT is a flat textured box without a curvature/shader treatment.
- Validation gates test silence and selected response pathways, with limited
  seeds. MaleCNS has no labelled sugar-GRN set for the same feeding gate.
- The advanced wired runner may fall back to a bare cockpit if assets are
  absent. The new launcher and physical validation script require the flybody.
- There is no Apple MPS brain backend. macOS uses SciPy; keep GL on the
  main thread. A historical Mac editable-install issue is recorded in Git
  history; it is not a required Linux setup step.
- The project uses private MuJoCo renderer and ALE frameskip attributes.
  Use the tested constraints; broad dependency lower bounds are not a
  guarantee that every older combination works.
- The fly is tethered. Its legs use IK, not a learned biological motor
  circuit; hind legs are not used as controls.
