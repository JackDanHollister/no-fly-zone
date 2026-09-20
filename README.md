# NO-FLY-ZONE

A real fruit-fly connectome fighting for its life in a tank it was never
licensed to drive.

![The connectome pilot firing at a tank, scoring, and backing off from the loom](docs/media/demo.gif)

*Real time, seed 0, the measured FlyWire wiring with the corrections listed
below. Left: the fly seated at the CRT, with the one she is fighting for
pinned to the wall. Middle: Atari Battlezone. Right: 138,639 neurons, spikes
glowing. Bottom: the descending neurons it is read out from, and, under the
brain, what each eye sampled this tick, the two retinal mosaics with their
transient energy; everything on the footer is a named, measured mapping.*

**The complete FlyWire v783 female fruit-fly brain, 138,639 neurons and 15.1
million synapses, spiking on a CUDA backend at five times real time, pilots
an Atari Battlezone tank through its own eyes and its own descending
neurons.** The Google/Janelia male map (MaleCNS v1.0, 166,700 neurons with
the ventral nerve cord) loads through the same code with `--dataset malecns`
and is used as a second individual for the controls; the pilot itself still
runs on FlyWire.

**Status, honestly: it fires at tanks that come into view, backs off from
loom, turns toward targets, and scores about one kill per 30 seconds. A
degree-preserving random rewiring of the same brain scores the same number
of kills, but it cannot tell which side a target is on: the measured wiring
steers toward the target on 62 of 63 gated look ticks over ten 100 s
episodes, five whole-brain rewirings on 66 of 165. A rewiring confined to
each hemisphere keeps the side by anatomy, but not the rest: the measured
wiring delivers vision to 58 of 460 descending types where the shuffles
light about 355, at up to 47 Hz where they reach 14, and 14 of its types
tell an expanding object from a contracting one where the shuffles have at
most two. The male brain (MaleCNS v1.0, now loadable) replicates this at
matched gain with the same cells. The model cannot compute motion direction
at T4 (a point-neuron limit, measured); given direction there by an
engineered correlator, the wiring builds the fly's horizontal and vertical
motion systems (HS cells 113 Hz one way, 0 the other; VS cells likewise)
and ten direction-selective descending neurons including DNa02, in both
flies, where the shuffle builds none. A trained readout on top of it learns
to survive but not to kill.**

The brain runs as a leaky integrate-and-fire network over the measured
wiring and is seated as the **pilot** inside a retro-futuristic 3D cockpit.
The cockpit's screen shows Atari **Battlezone**; the fly's compound eyes watch
that screen; its descending motor neurons work the cockpit controls; the
controls drive the game. Fly → cockpit → tank → robot war.

## What is measured, what is modelled, what is not claimed

- **Measured:** every neuron and synapse count of FlyWire v783, the
  literature transmitter of each cell where known, the descending-neuron
  types that fire on the same side as an object (found by a screen, not
  chosen), and every number in this README, on the machine that ran it.
- **Modelled:** Shiu et al. 2024 LIF dynamics with predicted synapse signs;
  two corrections, both off by default and recorded on every run: aminergic
  synapses carry no fast weight (otherwise the reference model has a
  self-sustaining olfactory state), and primary sensory neurons fire only
  from injected input. The retina is a hex lattice over the CRT; the
  medulla injection map, the readout pools, the calibration spans, the
  saccadic gating and the steering rule are engineered and named on the HUD.
  Levers are a kinematic puppet.
- **Not claimed:** that the fly learns (the two learning experiments and
  their controls are in `docs/research/learning-2026-09-14.md`), that the
  wiring beats a random graph on kills (it does not), that a readout trained on it plays better than one
  trained on a rewired brain or on the retina (it does not, three seeds,
  `docs/research/learning-2026-09-19.md`), that the
  within-hemisphere circuit matters beyond keeping the two sides apart
  (`docs/research/wiring-controls-2026-09-18.md`), that the mushroom body
  is in the loop, or anything about consciousness. Every video is closed-loop unless
  its footer says otherwise; there is no replay footage in this repository.
- **No token, no coin, ever.**

## Status

Last audited 2026-09-20 against the code, not the commit messages.

The wired pipeline is complete end to end: game → CRT texture → eye renders →
hex retina → brain → descending-neuron rates → levers → ALE action. Two brains
satisfy the same contract: the hand-written reflex stub and a sparse
NumPy/SciPy port of the Shiu et al. 2024 whole-brain LIF model over FlyWire
v783. `scripts/run_m3.py` drives Battlezone from real descending neurons with a
spike-cloud dashboard panel. Developed on the Apple-silicon Mac; on
2026-09-14 the full script ladder was run on the Ubuntu workstation and every
gate passed with the same brain numbers. The same day the brain gained a CUDA
backend (`brain_cuda.py`: Triton kernels, one CUDA graph per game tick) that
matches the SciPy reference step for step and runs the whole-brain pilot at
about 5x faster than real time on the RTX 4090, versus 8x slower on the CPU.
`--backend cpu|cuda|auto` selects it in `run_m3.py` and `validate_brain.py`;
`scripts/compare_backends.py` is the equivalence gate.

Later the same day the pilot was made to play: named injection maps, a
retina zoom and per-eye windows, descending-neuron readout pools chosen by a
screen, saccadic gating and a startle refractory, lateralisation-index
steering, and rewired/silenced wiring controls. Along the way the reference
model turned out to have a self-sustaining olfactory state (dopamine signed
as fast excitation) that a loader option removes with the validation gates
intact. The fly now fires at tanks that come into view, backs off from loom,
turns toward off-axis targets, and scores about one kill per 30 s episode;
so does a degree-preserving random rewiring of the same brain. Details and
the honest benchmark: `docs/research/cascade-probe-2026-09-14.md`.

On 2026-09-18 the wiring question was measured directly
(`docs/research/wiring-controls-2026-09-18.md`). A DN screen on the same
stimuli shows the measured graph delivering the side of a small object to
four descending types at LI above 0.8, and three independent whole-brain
rewirings delivering none (pool LI +0.03, -0.11, +0.21), gain-matched or
not. In the game, over ten independent 100 s episodes, the measured
brain's steering sign matched the retina's side on 62 of 63 gated look
ticks; five rewirings scored 66 of 165, each at chance or below with its
own fixed bias, while killing as often as the connectome. A new control
that shuffles targets within each hemisphere (`--ablate rewired_ipsi`)
keeps the side and, with its own screened pools, steers correctly too.
Beyond laterality, `scripts/screen_selectivity.py` gives every brain the
same battery (small disc, tank-sized disc, loom, receding disc, flash, bars)
and finds three things the shuffles cannot inherit: the measured wiring
routes vision to 58 descending types rather than about 355, drives a few of
them hard (DNg100 at 47 Hz against 14 for the shuffles' best), and has 14
types that prefer expansion to contraction, DNp12 among them, against at
most two. Size tuning and direction selectivity are absent in every brain,
a front-end limit. MaleCNS v1.0, converted by `scripts/build_malecns.py`,
is over-excited at Shiu gains (11,000 spiking neurons per tick against
2,000) and has a narrow window, as flybench also found; at weight scale 0.6
it replicates the pattern with the same cells (DNg100, DNp12, DNp07; under
the column placement of 2026-09-20, 12 expansion-selective descending
types against 1-4 in the shuffles). In
the game with its own screened pools (`pool_male`) the male reverses on
half its ticks, kills 1.2 and dies 2.7 times per 100 s against the
female's 2.7 and 5.7, and steers toward the target 10 times in 12;
`out/no-fly-zone-female-vs-male.mp4` is seed 0 of each, side by side.
Those male numbers were measured with its cells placed on the retina by
soma position, and on 2026-09-20 that turned out to be the reason the
male favoured one side: its photoreceptors have no soma positions (13 of
3,377 do), so the luminance channel reached 13 cells. Cells are now
placed by their annotated hex column, and a cell without one takes the
column of its synaptic partners (`docs/research/male-retinotopy-2026-09-20.md`).
The male's right-over-left drive asymmetry falls from 7.2 to 2.05 (the
rest is the annotation's 2:1 photoreceptor count), and the male pilot
kills 2.2 and dies 2.3 per 100 s, steering toward the target on 0.90 of
gated looks on both sides.
`scripts/probe_motion.py` then showed that no injection at the columnar
inputs makes T4/T5 direction-selective in this LIF (v5 adds the slow Mi4
and Mi9 channels; still flat), so injection map v6 computes direction with a
Reichardt correlator at the retina and injects it at the T4 and T5
subtypes, replacing that one stage. Downstream the wiring does the rest:
HS cells at 93-116 Hz for one horizontal direction and 0-11 for the other,
VS cells for one vertical direction and silent for the other, and ten
descending types direction-selective at both sweep speeds (DNa02 among
them, at 48 Hz), against flat HS/VS and no consistent descending type in
the within-hemisphere shuffle; the male reproduces it with H2 opposite to
HS as in the fly. The same day's audit found that Battlezone ignores the
gym seed, so the earlier "ten-seed" benchmarks were replicas;
`bench_pilots.py` now opens each episode with a seeded random warm-up.

Two learning routes were then tried (`docs/research/learning-2026-09-14.md`):
a PPO readout trained on the frozen brain's pooled spike counts learned to
survive, from three deaths per minute to under one, but not to kill more than
random play does, and the same head on a rewired brain learned survival just
as well; dopamine-gated plasticity on the measured Kenyon-cell to MBON
synapses changed 74 of 62,261 and nothing else, because at these gains the
mushroom body cannot see the game. Both are labelled as such wherever shown.

On 2026-09-19 the same three-factor rule was moved to where the game does
reach, the synapses onto the descending neurons (visual inputs, 24,110, or
every input, 332,213; `docs/research/learning-2026-09-19.md`). It now
moves 40% of what it is given, and every brain it produces dies more on
unseen games than the measured one (2.5-2.8 lives per episode against
1.9, six of six across mixed, hits-only and kills-only teaching), while
size-matched shuffled controls do not. Hits depress the loom pathway to a
quarter of its measured weight, because a hit follows a loom and the rule
punishes the response as the cause; kills potentiate the steer and escape
inputs, and the fly dies just as much. An adaptive readout whose spans
follow each pool's own running peak (2026-09-20) rescues none of them,
and the giant fibre's peak rate sits at the floor in every hits-trained
brain: the loom pathway is gone, not rescaled. The harm is the rule's
credit assignment, which attributes each reinforcer to the correct
response that preceded it. `run_m3.py --learned-weights` plays any of
these brains and `--learn vis_dn` runs the rule live, both labelled on
the footer. The PPO readout was re-run the same day with three seeds per
condition and a kill-shaped reward (a kill worth ten lives): on four
unseen games the heads on the measured connectome, the rewired brain and
the bare retina are indistinguishable (1.5, 1.4, 1.3 kills per 900 ticks,
errors 0.2-0.3) and none beats random play. The measured wiring is not a
better feature map for this policy than a random graph with the same
degrees, and the earlier one-seed hint that it was does not replicate.

| Milestone | State | Note |
|---|---|---|
| M0 scaffold, smoke scripts | done | Smoke scripts plus a `tests/` pytest suite (17 tests; the brain gates, controls, correlator, MaleCNS loader and plasticity sets are marked `data`/`cuda` and skip without them). |
| M1 game frame → CRT texture → eye camera | done | The orientation gate guards vertical flips only. |
| M2 closed loop with the reflex stub | done | Levers are a kinematic puppet derived from the action, not causal. |
| M2.5 flybody seated in the cockpit | done | Kinematic attach; eye cameras are world-fixed at head height; silent fallback to the bare cockpit. |
| M3 whole-brain LIF core, validated | done | Gates are qualitative (MN9 > 5 Hz, any giant-fibre spike), single seed. CPU (SciPy) and CUDA (Triton) backends agree. MaleCNS v1.0 loads through the same path (`--dataset malecns`); its giant-fibre gate passes, its sugar gate cannot be run (no labelled sugar GRNs), and it needs weights x0.6 to sit in FlyWire's firing regime. |
| M4 connectome pilots in wired mode, dashboard | partial | Idle scan now off by default and labelled when on; pooled readouts, saccadic gating and LI steering give event-driven play with real turns and kills; the wiring is shown to carry target side to the descending neurons where a whole-brain rewiring does not, but a within-hemisphere rewiring also keeps it; eyes panel (the two retinal mosaics and their transient energy) on the dashboard, and `run_m3.py --live` shows it in a window as the fly plays (2026-09-20). |
| M5 cockpit art pass | partial | Neon, lighting, set dressing and decals done; the screen is a flat box showing the raw frame; the single damage lamp sits about 35° above the eye cameras' axis, so the retina never sees it. |
| M6 embodied mode, contact-driven controls | baseline done | `run_m3.py --pilot embodied` (2026-09-20): the flybody's own leg servos, driven by inverse kinematics from the descending neurons' intent, push fly-scale physical levers with adhering claws; the action reads the levers. Gate `scripts/validate_embodied.py`. About real time. Fire still puppeted; wings droop. |
| M7 learning and showcase video | partial | Batched brain, PPO readout trainer, three-factor plasticity on the mushroom body and on the descending neurons' inputs, with LR-zero and shuffled controls and a frozen test on unseen games. Readouts learn survival, not kills; the mushroom body cannot see the game; the descending-input rule changes behaviour reproducibly and for the worse. Learned brains and live learning play in `run_m3.py`, labelled. No showcase cut. |

Doc-versus-code gaps are listed in `docs/architecture.md` under "Status audit".
The important ones: only FlyWire v783 is loaded, and pIP10 does not exist in
the female brain, so oviDN is read out under that name.

## Next Steps

1. Done 2026-09-14: the whole ladder passes on the workstation with the
   versions recorded under Machines; `constraints.txt` pins the exact set.
2. Done 2026-09-14: the CUDA brain backend; the `tests/` suite wraps the
   gates and the CPU/CUDA equivalence. Done 2026-09-20: `run_m3.py --live`,
   a window on the dashboard as the fly plays, real time on the 4090.
3. Done 2026-09-14 in spirit: cascade probes, DN audit and screen, `--ablate`
   controls, idle scan off, pooled readouts, `bench_pilots.py`
   (`docs/research/cascade-probe-2026-09-14.md`). Still open: DN calibration
   from measured closed-loop distributions (the `pool4` spans are
   provisional). Done 2026-09-20: a provenance stamp on every artefact
   (`noflyzone/provenance.py`: commit and dirty flag, settings, brain
   metadata, package versions, and an id hashed from the parts that
   determine the result; the pilot's footer shows the commit).
4. Done 2026-09-18: longer episodes, a balanced left/right steering
   metric, a gain-matched rewired control, three rewiring seeds and a
   within-hemisphere control (`docs/research/wiring-controls-2026-09-18.md`).
   Result: the wiring keeps the target's side; the within-hemisphere circuit
   is not yet shown to matter. Also done the same day: the selectivity
   screen (specificity, gain and expansion selectivity separate the wiring
   from a within-hemisphere shuffle; size tuning and direction selectivity
   are absent everywhere). Then the motion probe and injection map v6
   (direction from a correlator, injected at T4/T5, labelled engineered):
   the wiring downstream builds HS/VS and direction-selective descending
   neurons; with the right eye mirrored (v6m) yaw optic flow gives 14
   descending types a direction-signed left/right difference (DNa02, DNp15,
   DNg41, ...), and `READOUT_MAPS["yaw"]` pilots on five of them (kills
   2.4, dies 4.3 per 100 s against pool4's 2.2 and 5.9). Scored on
   2026-09-20 with a motion-signed steering metric: the yaw pools cross
   the steering gate on 2-4 look ticks per 1,000 s, because the fly turns
   2-8 times an episode and a translating tank on static ground is a
   position signal, not yaw flow; the yaw pilot's lower death rate is
   passivity (addendum in `docs/research/wiring-controls-2026-09-18.md`).
   Still open: a task that turns the fly, if motion steering is ever to be
   tested; size tuning, still absent everywhere; the eye-render settings
   fix.
5. Done 2026-09-18: MaleCNS v1.0 loader (`scripts/build_malecns.py`,
   `--dataset malecns`), used as a second individual for the controls.
   Done 2026-09-20: the male's own hex-column retinotopy, columns
   inherited from partners for the cells without one (the photoreceptors
   above all), which removes most of the male's one-sidedness. Still
   open: a per-side normalisation of the photoreceptor drive if the
   annotation's 2:1 count is to be undone (an engineered step, so stated),
   male-screened readout pools re-done under the new placement, and the
   VNC for M6.
6. Learning, continued. Done 2026-09-19: the three-factor rule on the
   descending neurons' inputs, with controls and a frozen test (it learns,
   and what it learns is harmful; see Status), and the PPO readout at
   three seeds per condition with the kill-shaped reward (no condition
   beats another or random play), and on 2026-09-20 the adaptive-readout
   test that ruled out calibration as the cause. Still open, for in-brain
   learning: a rule whose eligibility is tied to the fly's actions rather
   than its inputs, or a readout learned jointly with the synapses; a
   deliberate, labelled calyx-gain experiment if the mushroom body is ever
   to learn.
7. M6, baseline done 2026-09-20 (`docs/architecture.md`, "Embodied mode").
   Still open: contact-driven fire, the wings, and the VNC for the male.

## Architecture

```
[ALE BattleZone-v5] ←— joystick actions —— [cockpit control state]
        |                                          ↑
    RGB frames                            control physics (MuJoCo)
        ↓                                          ↑
[MuJoCo cockpit: CRT screen geom ← frame texture upload]
        ↓  eye cameras (offscreen render)
[retina: hex ommatidia binning → photoreceptor drive]
        ↓
[LIF spiking brain over the real connectome (Shiu et al. 2024 params)]
        ↓  descending neurons (DNa02 steer, pIP10 fire, DNp01 startle, ...)
[motor mapping → levers / pedals / trigger]
```

Two pilot modes, same brain and eyes:
- **wired** — descending-neuron rates drive animated cockpit controls directly
  (the working baseline)
- **embodied** — the TuragaLab `flybody` model physically seated in the
  cockpit, motor neurons → leg actuators → contact forces on the levers
  (the over-the-top goal)

## Quickstart (any machine)

```bash
git clone https://github.com/JackDanHollister/no-fly-zone
cd no-fly-zone
python3 -m venv .venv                 # python >= 3.11
.venv/bin/pip install -e .            # ale-py bundles the Atari ROMs legally
.venv/bin/python scripts/fetch_assets.py   # fly body + connectome data, ~570 MB
.venv/bin/python scripts/smoke_battlezone.py
.venv/bin/python scripts/smoke_cockpit_screen.py
.venv/bin/python scripts/run_m2.py --video          # reflex-stub pilot, fast
.venv/bin/python scripts/validate_brain.py          # LIF core vs published results
.venv/bin/python scripts/run_m3.py --steps 150 --video  # the real connectome plays
```

Videos and renders land in `out/`. macOS-only gotcha: if `import noflyzone`
fails after installing, see the `.pth` note in `docs/architecture.md`.

## Machines

- **Dev**: Apple-silicon Mac — everything runs here (MuJoCo has native arm64
  wheels; ALE bundles the Atari ROMs legally; the brain sim is sparse
  NumPy/SciPy on the CPU, so `run_m3.py` runs in slow motion).
- **Final**: workstation (Ubuntu 26.04, RTX 4090, 192 GB RAM) — lives in The
  Magi Home Edition workspace as `projects/no-fly-zone/`; set up and validated
  2026-09-14 with Python 3.12.14, mujoco 3.13.0, ale-py 0.12.1, gymnasium
  1.3.0, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, pyarrow 25.0.1, pillow
  12.3.0, imageio 2.37.4, plus torch 2.14.0+cu130 with triton 3.8.0 for the
  CUDA brain backend (`pip install -e ".[cuda]"`). All components are pure
  Python; only backends swap.

Linux workstation setup, with the CUDA backend:

```bash
python3.12 -m venv .venv            # any Python 3.11-3.13 with pip; uv works too
.venv/bin/pip install -e ".[cuda]"  # torch with bundled CUDA; Triton comes with it
export MUJOCO_GL=egl                # for every script that renders when not on a desktop terminal
.venv/bin/python scripts/smoke_battlezone.py
```

`imageio[ffmpeg]` bundles its own ffmpeg, so `--video` works without apt
ffmpeg. `fetch_assets.py` needs git 2.25 or newer and github.com access.

Measured on the workstation, 2026-09-14, all under `MUJOCO_GL=egl`:

| Script | Result |
|---|---|
| `smoke_battlezone.py` | 500 random steps in 0.3 s |
| `smoke_cockpit_screen.py` | PASS bare and with the fly; CRT whites reach 255 in the eye renders, so `WHITE_REF` stands |
| `run_m2.py --video` | PASS, 600 steps, about 233 Hz with the fly attached (Mac: about 21 Hz) |
| `validate_brain.py` | PASS, load 1 s, MN9 101 Hz, giant fibre 66/64 Hz, 1 s sim per 5 s wall (identical to the Mac) |
| `run_m3.py --steps 90 --video` (cpu) | PASS, 7.7x slower than real time; brain 492 ms of the 499 ms tick |
| `run_m3.py --backend cuda --steps 450` | PASS, 30 s of game in 5.8 s wall (5.2x real time); brain 8.9 ms, loop about 80 Hz |
| `validate_brain.py --backend cuda` | PASS, MN9 102 Hz, giant fibre 66/64 Hz; 1 s sim in 0.2 s wall |
| `compare_backends.py` | PASS: per-step spike sets identical for 500 steps; 0.3 s aggregates within 1%; tick 5.5 ms on CUDA vs 263 ms on CPU |

MuJoCo 3.13 prints an attach-conflict warning when seating the fly: the
flybody's option block (0.1 ms timestep, gravity -981 in centimetre units)
differs from the cockpit's and the parent values are kept. Harmless for the
kinematic puppet, but the unit and timestep mismatch must be resolved before
embodied mode.

macOS note: keep the step→upload-texture→render loop single-threaded (GL must
stay on the main thread).

## Layout

- `src/noflyzone/` — package (game bridge, cockpit scene, retina, brain;
  `brain_lif.py` is the SciPy reference, `brain_cuda.py` the Triton/CUDA
  backend, `brain_backend.py` picks one; `plasticity.py` the learning
  rules, `rl_env.py` the batched squad for trained readouts,
  `provenance.py` the stamp every artefact carries)
- `scripts/` — runnable smoke tests and demos
- `docs/architecture.md` — full design notes, milestone checklist and status audit
- `CHANGELOG.md` — dated session log
- `assets/` — flybody model (gitignored, fetched by `scripts/fetch_assets.py`)
- `data/` — connectome downloads (gitignored, same script)
- `out/` — renders and videos (gitignored)

## Data & reference implementations

- MaleCNS v1.0 (Janelia/Google, CC-BY): https://male-cns.janelia.org —
  loaded by `scripts/build_malecns.py` and `--dataset malecns`
- FlyWire v783 dump (CC-BY, no login): https://zenodo.org/records/10676866 —
  the code loads v783 from the Shiu model repository instead of this dump
- LIF parameters: Shiu et al., Nature 634:210 (2024) — github.com/philshiu/Drosophila_brain_model
- Eye sampling: flygym Retina (NeLy-EPFL), aimbug (slickdomi/aimbug, MIT),
  DOOMFLY (nftechie/doomfly, MIT)
- Body model: TuragaLab/flybody (MuJoCo)

## Licence and data

Code: MIT (see `LICENSE`). The data this fetches is not in the repository and
carries its own terms: the Shiu et al. brain model code is MIT; the FlyWire
v783 connectivity it carries is CC-BY 4.0 (Zenodo record 10676866, Dorkenwald
et al. 2024, and cite the FlyWire papers if you use it); the cell-type
annotations are from Schlegel et al. 2024 (CC-BY 4.0); MaleCNS v1.0 is CC-BY
4.0 from Janelia; the flybody model is Apache-2.0 from MuJoCo Menagerie; the
Atari ROM ships legally inside `ale-py`.

## Running it

```bash
git clone https://github.com/JackDanHollister/no-fly-zone && cd no-fly-zone
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]" -c constraints.txt
.venv/bin/pytest                          # no data or GPU needed for the unit tests
.venv/bin/python scripts/fetch_assets.py  # ~570 MB: fly body, connectome, annotations
.venv/bin/pytest -m data                  # the brain gates, CPU or CUDA
```

`constraints.txt` is the exact package set the workstation results were
measured with; without it, recent `mujoco`, `ale-py` and `gymnasium` should
still work. The CUDA backend needs `pip install -e ".[cuda]"` and an NVIDIA
GPU; everything else runs on CPU, slowly.

## How this was built

Designed and steered by Jack Hollister, with Claude Code as the pair
programmer for the code, the probes and the write-ups. Every number in this
README and in `docs/` was measured on the machine that ran it, and every
claim about the fly's behaviour is checked against rewired and silenced
controls before it is made. Where the answer is "not yet shown", it says so.
