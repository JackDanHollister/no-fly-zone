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

The next step is to make the learning evaluation independently reproducible,
then identify which changes in the learned brains actually transfer a useful
behaviour. That is where the journey currently stands.
