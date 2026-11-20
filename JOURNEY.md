# A fly, a tank, and the experiments in between

*A build journal by Jack Daniel Hollister. September 13–22, 2026.
Draft prepared for the author's review; see the [credits](README.md#credits-and-licence).*

I put a simulated fruit fly in a tank cockpit, gave it a screen showing
Battlezone, and made its brain operate the controls.

There is a whole-brain spiking simulation behind the little pilot. There
is also a framed picture of her sweetheart on the wall. Both took work.

The project grew from getting a game frame onto a screen into a GPU
engine, a physical control system and a series of experiments about
what measured neural wiring actually contributes. It has first kills,
broken assumptions, a second fly, learning rules that made things worse,
and a promising result that shrank when I ran more trials.

This is the route through it. The [changelog](CHANGELOG.md) keeps the
implementation history; the linked lab notes keep the methods and tables.

![The cockpit with its wall art and tiny pilot](docs/media/cockpit_beauty.png)

## September 13: build the world before asking the brain to understand it

The first job was to close a loop I could inspect. Battlezone supplied
frames through Arcade Learning Environment. MuJoCo displayed them as a
texture on a cockpit screen. Two cameras near the fly's head rendered
that screen, and a hexagonal retina reduced each image to 721 samples.

That extra trip through the cockpit gives the pilot an actual viewpoint.
Camera position, field of view, screen brightness and the size of a target
all affect its input. It also creates several opportunities to get the
image upside down. The early validation included an orientation check.

A hand-written reflex brain came first. It established the interface:
retinal drive in, descending-neuron-style firing rates out, then a
controller mapping rates to actions. The game runs at 15 control ticks
per second, with one tick of sensorimotor latency.

I attached the anatomical flybody model to the seat and positioned the
cameras just ahead of the antennae. Behind them, the fly's own head blocked
the screen. The initial levers followed the chosen action directly. That
was enough to prove the visual loop before introducing physical contact.

**Reached:** a working game → cockpit → eyes → controller → game loop,
with the fly seated inside it.

## September 14: the whole brain, then the speed to investigate it

The first neural backend was a sparse NumPy/SciPy implementation of the
Shiu et al. leaky integrate-and-fire model. It loaded the FlyWire v783
female brain: 138,639 neurons and 15,091,983 directed connection records,
weighted from anatomical synapse counts and transmitter signs.

The model passed the initial qualitative gates: silence without input,
a feeding response from sugar-sensing input, and an escape response from
LC4 visual neurons to the giant fibre. Those gates exercise particular
routes through the model; they do not validate every computation of a fly.

It ran on the Apple-silicon development machine. On the Ubuntu workstation,
the CPU pilot took roughly eight seconds of computation for one second of
game time. That was enough for a proof of concept, but a poor way to run
many controls.

So I built a CUDA backend: Triton neuron-update and sparse-delivery kernels,
a delayed-spike ring, and one captured CUDA graph per game tick. The graph
replays hundreds of simulation steps without repeatedly returning to Python.
Only the resulting spike counts need to come back to the host.

The validation caught a small floating-point trap. Fused operations could
change a threshold crossing; disabling fusion restored exact short-horizon
agreement on the deterministic comparison. Over longer runs, parallel
summation order still matters in a recurrent network. The check therefore
records both exact short runs and aggregate tolerance on longer ones.

| Workstation measurement, September 14 | CPU | RTX 4090 CUDA |
|---|---:|---:|
| Isolated brain tick | 263 ms | 5.5 ms |
| End-to-end pilot | About 8× slower than real time | 30 seconds of game in 5.8 seconds |

Those are particular measured configurations, not promises for every
rendering mode. Physical controls add their own cost later. The improvement
made whole-brain experiments practical on one workstation.

The engine also gained batching: several flies share the connection matrix
while retaining separate activity and input state. Three flies cost about
4.2 ms per fly-tick in the recorded benchmark. That became the basis for
the learning experiments.

**Reached:** a validated CPU reference, a much faster GPU implementation,
and a usable experimental engine. [Technical details](docs/architecture.md#cpu-and-cuda).

## September 14: a lit-up brain is only the beginning

The first whole-brain demo had an idle scan that kept the tank moving.
That made something happen on screen, but it obscured whether neural
activity was steering it. The scan needed to become optional and visible
in the run settings.

The next work was diagnostic. I injected stimuli at different visual
stages, inspected the inputs to candidate motor neurons, and sent synthetic
discs, bars, flashes and expanding objects through the same retinal path
used by the game.

A tank-sized object barely reached the original steering readout. The
injection map omitted several inputs that the relevant visual relays
needed. Predicted transmitter signs also left the early visual path behaving
poorly in this simplified model. I broadened the explicitly engineered
injection, enlarged the retinal sampling window over the target, and
screened descending neurons for useful responses.

The resulting `pool4` pilot reads groups of neurons for steering, forward
drive and escape. It briefly suppresses visual drive after movement and
has a refractory period after a startle. The two eyes use offset windows
so that target side can affect their relative input. These choices are
part of the controller and are recorded alongside the wiring.

Then it fired at a tank. Both steering pools became active with the tank
ahead, a shot followed, and the shell landed six ticks later.

There was another problem hiding in longer runs. After a large visual event,
about 9,654 neurons could remain active even in darkness. The reference
model treated dopamine and other aminergic outputs as fast excitation,
sustaining an olfactory-region feedback loop. Options to remove those fast
weights and restrict primary sensory neurons to injected input cleared the
persistent state while keeping the validation gates intact.

**Reached:** an event-driven pilot that turns, fires and reverses, with
named sensory and control mappings. [Cascade investigation](docs/research/cascade-probe-2026-09-14.md).

## September 14: give the pilot somewhere to come home from

Alongside the neural work, the cockpit acquired deck plating, armour,
rivets, gantries, neon strips, auxiliary screens, shell casings, a helmet,
dog tags, propaganda posters and a kill tally. The dashboard put the
cockpit beside the actual game and a cloud of spiking neurons.

The wall needed a second layout pass because the camera clipped the art
and the nameplate overlapped the tally. A pixel-art sweetheart portrait
went into a brass frame under the words “COME HOME SAFE”. The decals were
painted in code, including the portrait.

![The sweetheart portrait](docs/media/sweetheart_portrait.png)

It gives the project a character while the dashboard gives the viewer
something concrete to inspect. Later, the two retinal mosaics joined the
footer, so it became possible to see what the fly actually sampled on
that tick.

## September 18: ask a harder question than whether it scores

A randomly rewired version of the brain could score kills too. That
became one of the central results of the project. In this game, firing
while an enemy approaches can succeed without sophisticated aiming.

The early ten-seed benchmark had another flaw: Battlezone ignored the
requested Gym seed in this configuration, producing repeated starts.
I changed the benchmark to begin with a seeded sequence of random actions
and reset the brain before measurement. The corrected comparisons used
different game histories rather than merely different seed labels.

With those starts, the measured brain's steering sign matched the target's
side on **62 of 63 gated look ticks** across ten 100-second episodes.
Five whole-brain shuffles managed **66 of 165**. These counts refer to the
subset of ticks with a visible lateralised signal and a crossed steering
gate, so they should not be advertised as whole-game aiming accuracy.

A stronger control shuffled targets only within each hemisphere. That
preserved the broad left/right organisation, and with its own screened
readout pools it could steer correctly too. The first result therefore
could not establish the importance of the detailed local circuit.

A broader stimulus battery found differences beyond side. The measured
brain sent visual responses to 58 of 460 descending types, while the
within-hemisphere shuffles activated roughly 355. Its strongest small-object
response reached about 47 Hz versus 14 Hz for the shuffles. Fourteen types
preferred expansion to contraction; the tested shuffles had at most two.

The measured wiring was routing signals selectively. Its kill count still
did not separate it from the whole-brain shuffled controls.

**Reached:** evidence about specific computations, with controls that
narrow the claim. [Wiring and selectivity experiments](docs/research/wiring-controls-2026-09-18.md).

## September 18–20: another fly, and two missing computations

The MaleCNS v1.0 loader brought a second individual into the same engine:
166,700 neurons, with a different annotation scheme and firing regime.
At the original gain it was overactive. At a weight scale of 0.6, several
selectivity results appeared in the same named cell types as the female.

Its game behaviour was strongly one-sided. The eventual cause was in the
input mapping: only 13 of its 3,377 R1–6 photoreceptors had soma positions,
so a position-based luminance map reached only those 13 cells. Switching
to annotated retinal columns, and inheriting missing columns from synaptic
partners, placed all 16,095 injected cells. The male's measured kill rate
rose from 1.2 to 2.2 per 100 seconds in that comparison, and its side bias
fell substantially. Some imbalance remains in the annotation counts.

Motion needed a different investigation. The point-neuron model did not
produce useful T4/T5 direction selectivity under the tested inputs,
even with additional slow visual channels. I added a labelled engineered
motion correlator at that stage. Downstream, the measured wiring produced
horizontal and vertical motion-selective systems and ten consistently
direction-selective descending types in the tested female assay. The
within-hemisphere shuffle did not reproduce that pattern.

Small-object selectivity followed a similar route. A surround filter alone
was insufficient. Broadly suppressing wide-field activity created a signal
in shuffled brains too and removed looming responses. Restricting that
engineered small-object channel to T2/T2a/T3 allowed the measured wiring
to route it to 14 selective descending types; five tested shuffles routed
it to none. The male showed a smaller version of the same separation.

Neither change automatically made a better Battlezone pilot. The motion
readout rarely crossed its steering gate. The small-object pilot's apparent
kill improvement did not separate clearly over twenty starts and came
with more deaths. Both remain experimental alternatives.

**Reached:** a second connectome, a repaired input mapping, and a distinction
between supplying a signal and the wiring routing it.
[Male mapping](docs/research/male-retinotopy-2026-09-20.md) ·
[Motion](docs/research/wiring-controls-2026-09-18.md#can-the-wiring-compute-direction-no-not-as-a-point-neuron-lif) ·
[Small objects](docs/research/size-tuning-2026-09-20.md).

## September 20: make the fly actually move the controls

The physical cockpit was a second engineering project inside the first.
The early animated levers became hinged bodies that the fly's front claws
could push. Descending-neuron intent sets lever targets; inverse kinematics
turns those into leg-joint targets; the flybody's servos and adhering claws
supply the force. The game reads the resulting lever angles.

The first attempts exposed scale and contact mistakes. Levers were too
heavy, some geometry intersected the deck, claws started inside the knobs,
and self-collision made the tethered body chatter. Releasing a lever left
it short of neutral, which made the tank keep reversing. A measured dead
zone and proportional correction made the control usable.

Fire became physical as well: an extended proboscis presses a sprung
button. The initial spring was overwhelmed by gravity. After correction,
the button's measured travel triggered the shot and releasing it stopped
firing. The wings were held in their resting stance with springs.

In the recorded 100-second seed-0 comparison, physical levers with a
puppeted fire switch matched the wired pilot's 3,000 points. With the
fire button physical too, the pilot scored 2,000 points and lost three
lives, versus 3,000 points and seven lives for wired control. That is a
single demonstration, not a claim of superior gameplay.

It runs at about real time on the workstation. The controls are physical;
the neural-to-leg transformation is still engineered inverse kinematics.
Connecting a ventral nerve cord to those muscles is future work.

**Reached:** a fly pushing its controls and firing with its proboscis,
verified by a contact and action-path gate. [Embodiment details](docs/architecture.md#embodied-mode-m6-2026-09-20).

## September 14–20: teaching it turned out to be the difficult part

There were three main learning routes.

First, a PPO policy learned from pooled activity in a frozen brain. The
same policy architecture also trained on a rewired brain and on retinal
features directly. Survival improved, but a promising initial result for
the measured wiring did not survive the later three-run comparison. The
recorded kill scores were similar across the three feature sources and
none demonstrated an advantage over random play.

Second, a dopamine-gated rule operated on 62,261 Kenyon-cell-to-MBON
connections in the mushroom body. Only 74 changed. The visual drive at
these gains barely activated that circuit. Installing a learning rule
was insufficient when the relevant activity did not reach it.

Third, the rule moved to the descending neurons' inputs: 24,110 visual
connections, or 332,213 inputs in the broader set. This time substantial
weight changes occurred. The resulting pilots consistently performed
worse in the first frozen evaluations. Learning had become visible and
harmful.

I tested calibration, because a changed firing range might simply confuse
the fixed controller. An adaptive readout did not rescue the brains. I
also tried a reward-prediction-error signal; its critic failed to predict
the hits under the tested settings.

Inspecting the fight traces revealed a timing problem. The punishment
pulse arrived while the brain was responding to the hit's own screen
flash. The rule could depress escape-related activity caused by the
outcome itself. Latching eligibility from before the event stopped that
particular form of blame.

The first two latched training runs produced brains with lower death rates.
Two more produced worse brains. The pooled result no longer supported
an overall improvement. The sequence stays in the lab notes, including
the earlier heading that the later results corrected.

A shell-visibility probe also found that incoming shells were poorly
resolved by the 721-sample retinal windows. That supports a sensory
bottleneck in the tested configuration; it does not prove that no possible
predictor could anticipate a hit.

**Reached:** working plasticity machinery, several rejected explanations,
and a narrower learning claim. [First learning experiments](docs/research/learning-2026-09-14.md) ·
[Later experiments and revisions](docs/research/learning-2026-09-19.md).

## September 20, recovered September 22: the experiment at the end of the queue

The better and worse learned brains differed conspicuously in the
LC4/LPLC4 inputs to an escape-related pool. Increasing or decreasing those
483 connections in the measured brain did not reproduce either outcome.

The last queued experiment went further: swap that pathway between a
better learned brain and a worse one. Four frozen evaluations finished
at 15:53 on September 20, after the last commit. Their results were still
in the local logs when the project was reopened on September 22.

| Recorded configuration | Mean lives lost, evaluation label 700–703 | Mean lives lost, label 800–803 |
|---|---:|---:|
| Better learned brain, seed-0 label | 1.38 | 1.52 |
| Same brain with the worse brain's pathway | 1.42 | 1.38 |
| Worse learned brain, seed-2 label | 2.48 | 2.56 |
| Same brain with the better brain's pathway | 2.56 | 2.56 |

Each value summarises 48 logged episode records. The better brain retained
its lower death rate and the worse brain was not rescued. This intervention
does not support that pathway alone explaining the observed split.

Preparing this write-up exposed a further qualification: the random-start
fix from the pilot benchmark was never added to the batched learning
wrapper. Changing its Gym seed labels is not evidence of independent game
starts. The training label also is not forwarded as the neural RNG seed.
The old descriptions of these batches as “unseen games” and independent
training seeds therefore go beyond what the code establishes.

The logs remain useful observations of changed weights and behaviour.
Demonstrating generalisation requires a corrected evaluation protocol and
new runs. This also qualifies the earlier session recap. The source notes
now carry the correction at the top, so it cannot be missed by reading an
old result in isolation.

[Final pathway-swap results and evaluation audit](docs/research/learning-2026-09-19.md).

## September 25: repair the evaluation and repeat the comparisons

The learning wrapper now starts a fresh game at every episode boundary,
including its time limit, and resets each fly's neural state and random
stream. Game starts have recorded action lists and image hashes. The old
short warm-up still produced identical gameplay for two seed labels; a
longer warm-up reached controllable play and passed the diversity checks.
All 38 tests passed, including real game starts, CPU/GPU equivalence and
the physical controls.

The first corrected comparison covered 112 episodes: measured weights,
four saved learned brains and both pathway swaps on 16 matched starts each.
The learned brains averaged 1.27 kills and 2.91 life losses per episode,
against 1.38 and 2.63 for the baseline. Their average did not improve it.
All weights were frozen and external teaching pulses were off, a recorded
difference from the historical evaluations.

One swap looked promising: brain 0 lost fewer lives with brain 2's escape
pathway. A focused 48-episode follow-up used new starts and neural seeds.
The original brain then lost 1.88 lives per episode and the chimera 1.94;
the apparent gain from the swap did not repeat. Both raw result sets are
preserved with their source and checkpoint hashes. The next learning runs
will need fresh independent training histories; repairing evaluation cannot
retroactively repair the independence of the old training labels.

[Corrected protocol, results and follow-up](docs/research/learning-2026-09-19.md#corrected-evaluation-protocol-2026-09-25).

## September 25: make it usable, then give learning a stopping point

The working pilot now launches with `./no-fly-zone`. It checks its local
installation, opens a live dashboard or records without a display, and
saves a uniquely named video and trace. Physical lever controls are one
option away. Closing the window keeps the completed recording and releases
the renderer. All 47 tests pass, and real recordings verify both direct and
physical controls.

Learning received one fixed campaign: four fresh training histories, each
with rule-off and shuffled-edge controls, followed by frozen evaluation on
two reserved start sets. The budget and success criteria were recorded
before training. All 288 training and 288 evaluation episodes finished in
80 minutes, with their starts, neural seeds and source hashes checked.

The result was negative in both test sets. Each learned history had lower
utility than the measured baseline and its own shuffled-edge control. The
learned group averaged fewer kills and more life losses. The stopping rule
therefore closes this Battlezone plasticity attempt; there is no extra
selected seed or parameter search after seeing that result.

[Fixed campaign, complete evidence and interpretation](docs/research/learning-campaign-2026-09-25.md).

## What is here now

The repo contains the game bridge, cockpit, retina, CPU and GPU brain
engines, readouts, physical controller, live viewer, learning machinery,
validation suite and the probes used to ask these questions. A 66-second
showcase combines the pilot, physical controls and two flies with labelled
findings; the full recordings remain local review material.

I am proud of the scale of the working system and of how much can be
inspected: from the pixels arriving at one eye to a descending-neuron pool,
a moving leg, a lever angle and an action in the game. The experiments
also made the project more specific. There is evidence for selective
routing through measured wiring, and a substantial engineering build
around it. Reliable learning and a game-score advantage remain open.

The evaluation repairs, corrected comparisons and fresh controlled campaign
are complete. The pilot is now a usable local package.

## November 17: why the fly could not see its own crosshair

Reopening the recorded 60-second demo trace (`pilot_20260925_200200_883034.json`)
explained why the pilot had looked hesitant and why measured wiring had never
outscored a shuffled graph in Battlezone: the tank output `NOOP` on 787 of 900
ticks and turned five times in a minute.

The culprit was optical geometry, not the connectome. Both eye cameras pointed
parallel (`+y`, `fovy = 45°`) at a close CRT while the zoomed retinal windows
were shifted outward (`eye_offset = 0.15`). Projecting the screen centre
(`x = 80`, the Battlezone aiming sight) through both cameras landed outside
both 721-ommatidium lattices — **zero ommatidia in either eye saw the
crosshair**, and the top radar was cropped off as well. Turning toward a tank
made it vanish right as it lined up with the barrel, starving `pool4`'s
binocular fire rule (`min(DNa02_L, DNa02_R)`). Meanwhile, `FlySquad`'s readout
wrapper pooled `cell_type` without `side`, averaging left and right hemispheres
together so PPO could not tell left from right, and divided `1,192` near-silent
pools by `sqrt(var + 1e-8)`, amplifying stray single spikes more than tenfold.

Converging both eyes on the CRT centre (`eye_fovy = 54°`, `converge_eyes = True`,
`zoom = 1.15`, `eye_offset = 0.11`) put binocular overlap directly on the
crosshair and brought the radar into view with zero bezel bleed. Three results
followed immediately:

1. **Moving-target tracking (`scripts/bench_tracking.py`):** across eight
   reserved trajectories, measured FlyWire wiring holds a compact target at
   `16.9°` RMS error (`168` turns), beating whole-brain shuffles (`25.4–28.7°`),
   `silenced` (`22.3°`) and `rewired_ipsi` on its own pools (`26.9°`) — while
   also showing honestly that `rewired_ipsi` reading `pool4` (`16.1–20.2°`) can
   track a *lone* bright dot on a blank dark screen by hemisphere segregation
   alone.
2. **Cluttered Battlezone combat (`--preset active`):** once mountains,
   obstacles and horizon lines fill both eyes at once, gross hemisphere
   segregation is no longer enough. Across ten 100-second episodes
   (`out/bench_active_wiring10.json`), the frozen measured connectome scores
   **2.10 kills** (`342.6` aimed shots) against **0.60** for three whole-brain
   shuffles, **0.80–1.10** for within-hemisphere shuffles, and **0.10** when
   silenced.
3. **Bilateral recurrent GRU readout (`TRAINED READOUT (GRU)`):** splitting
   `(cell_type, side)` into `1,799` bilateral pools with a variance floor and a
   `256`-unit GRU reached `2.19` kills and `2.50` lives lost per 60 s (`+0.94`
   utility) across 16 held-out starts (`out/eval_readout_2026-11-17.json`), and
   `6,000` points with `1` life lost on seed 0.

## November 19: catching the RL shortcut — and finding where the fly actually looks

Watching that `6,000`-point GRU video back was a great sanity check. On paper
the numbers looked brilliant; on screen, the tank just seemed to drift
relentlessly to the right instead of hunting left and right or reacting to
enemies.

Opening the 900-tick trace (`pilot_20261117_174132_735739.json`) caught a
classic RL shortcut red-handed. Across 60 seconds, the PPO GRU held `UP` on
**823 of 900 ticks**, pulsed `RIGHT` **445 times** (almost literally every
second tick in a 2-tick `UP` ↔ `UPRIGHTFIRE` metronome), and turned **`LEFT`
only 3 times in the entire game**. In 1980 Atari Battlezone, enemy tanks shoot
at where you *were*, without leading a curving path. Driving forward in a
steady clockwise spiral slides you off incoming shells while sweeping your gun
across 360° for occasional hit-and-run kills — and because a recurrent GRU has
its own hidden state and previous-action feedback, it can run that clockwise
metronome largely open-loop without even relying on the fly's vision.

The smoking gun was already sitting in our own control table
(`out/eval_readout_2026-11-17.json`): training the exact same from-scratch PPO
GRU on a **randomly scrambled (`rewired`) fly brain** actually *beat* the real
connectome (`2.31` kills and `+0.78` utility vs `1.69` kills and `+0.41`
utility). RL had quietly sidelined the connectome and gamed Battlezone's
physics.

So I pulled the trained GRU off the default launcher (`./no-fly-zone` now runs
the pure frozen connectome reflex with **zero RL**, keeping `--controller
learned` strictly opt-in) and went back to look at what the real fly brain was
doing on its own.

Sweeping a single spot across the left and right eyes separately revealed two
genuinely biological quirks hiding in FlyWire v783:

1. **The fly's tracking fovea sits on the anterior rim of each eye, not the
   centre.** The `LC10a → DNp07 / DNb05 / DNa02` steering pool barely fires for
   lateral spots (`2.5–10.5 Hz`), then spikes to **`79 Hz` on the inner edge of
   the left eye** and **`60 Hz` on the inner edge of the right eye**. At
   `zoom = 1.15, eye_offset = 0.11`, those two anterior foveas were actually
   crossed past each other over the centreline — the fly was slightly
   cross-eyed.
2. **This particular female fly is slightly right-handed.** On an identical
   bilateral flash, `DNb06` fires `105 Hz` on the right against `15 Hz` on the
   left, giving the outer right eye a `7–10.5 Hz` background floor on passing
   mountains that kept dragging the old low-threshold patrol clockwise.

Uncrossing the two anterior foveas onto **`zoom = 1.75, eye_offset = 0.21,
center_y = 0.48`** (still zero bezel bleed) snapped left targets cleanly onto
the left fovea (`90.8 Hz`, `LI = -0.85`), right targets onto the right fovea
(`69.2 Hz`, `LI = +0.87`), and binocular `pIP10` fire lock dead onto the
crosshair (`69.2 Hz`, `LI = -0.08`). Pairing that with a `10 Hz` steering gate
above the right-eye background floor, directional giant-fibre looming escapes
(`DNp01 > 0.60`), and balanced alternating left/right patrol sweeps changed
everything — with **zero learned weights**:

- **Moving-target tracking (`out/bench_tracking_2026-11-19.json`):** measured
  FlyWire wiring dropped from `16.9°` down to **`8.36°` RMS error (`5.97°` MAE,
  `77%` crosshair lock, `0.99` turn-sign accuracy)** — separating the real
  connectome cleanly from **every** control, including within-hemisphere
  shuffles (`rewired_ipsi`: `12.75–14.45°` RMS, `38–55%` lock) for the first
  time. Keeping left and right hemispheres separate is no longer enough once
  you test the anterior-rim fovea's sharp retinotopy.

The steering now looked right, yet watching the `v4` replay showed the fly was
still hesitant to finish tanks off. Auditing the eight enemy encounters tick by
tick exposed two final quirks — one biological, one pure 1980 Atari hardware:

1. **The fly had "T-Rex vision" for stopped tanks.** Because `v4` only injected
   frame-to-frame luminance changes, the instant an enemy tank stopped to aim,
   every descending neuron dropped to `0.0 Hz` and the fly turned away. Yet
   `HexRetina` had been computing the fly's **`R8` blue channel** all along
   (`3×` stronger contrast on blue tanks against green ground than greyscale
   luminance), and FlyWire v783 has **`3,630` measured excitatory synapses**
   from the real R8-downstream chromatic medulla neurons (`Tm20`, `Tm5a`,
   `Tm5b`) straight into `LC10a / LC11 / LC4`. Feeding pure signed `R8 − R1-6`
   blue-ON centre-surround opponency into those chromatic cells (`v11`) gave
   `0.000` false alarms across the entire 360° sky, sunset and mountain horizon
   while letting a **stationary** enemy tank light up `pool4` cleanly — whereas
   both whole-brain and within-hemisphere shuffles stayed dead silent (`0.0 Hz`).
2. **A missed Battlezone shot jams the cannon for `29` ticks (`~2` seconds).**
   Checking the emulator proved that an early off-axis shot locks out the gun
   while the shell flies to the horizon, causing Atari to ignore later
   dead-centre trigger pulls. Holding fire until the target sits tightly inside
   the binocular fovea (`|LI| <= 0.18`) keeps the barrel loaded for the real
   kill shot.

Putting uncrossed foveas, real `R8` chromatic opponency (`v11`) and tight
crosshair fire discipline together delivered the cleanest zero-RL combat runs
of the project:

- **Cluttered Battlezone (`out/bench_active_wiring10_v11_2026-11-19.json`):**
  across ten 100-second seeded episodes, the frozen measured connectome reached
  **`3.40` kills / 100 s** (scoring in 10 of 10 episodes, up to 6 kills), cut
  deaths nearly in half to **`3.60` lives lost / 100 s** (fewer deaths than
  *every* control, including `silenced`'s `4.20`), and hit **`+1.60` utility**
  with zero RL — beating three whole-brain shuffles (`0.90–1.60` kills, `-0.35`
  to `-1.60` utility), both within-hemisphere shuffles (`1.40–2.00` kills,
  `-1.15` to `-1.25`), and `silenced` (`0.70` non-firing baseline kills, `0`
  shots, `-1.40`).
- **Default 60-second recordings (`./no-fly-zone record`):** wired mode scores
  **`3,000` points with `3` lives lost** (first two kills at `3.5 s` and
  `5.0 s`) and razor-even hunting (**`159` `LEFT` / `158` `RIGHT`**, `271`
  `UP`, `80` `DOWN`, `133` `FIRE`). The physical-lever `--embodied` fly pushes
  its cockpit levers (`-0.90..+1.00` left, `-0.85..+1.00` right) and presses
  the fire button with its proboscis for **`3,000` points and only `1` life
  lost in 60 seconds** (three kills inside the first `10.3` seconds).

[Active binocular vision, target tracking, PPO shortcut analysis, uncrossed foveas and R8 chromatic vision](docs/research/active-vision-and-tracking-2026-11-17.md).
