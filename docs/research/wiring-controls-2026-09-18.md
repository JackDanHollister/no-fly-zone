# Does the wiring matter? Lateralisation screen and in-game controls, 2026-09-18

The open question from `cascade-probe-2026-09-14.md`: on a ten-seed, 30 s
benchmark the measured FlyWire v783 wiring and a degree-preserving random
rewiring both scored one kill per episode, so the topology was not shown to
do anything. That benchmark counted kills, and the first kill in Battlezone
is a head-on target that any dense graph fires at. This note measures the
thing steering actually needs, whether the descending neurons know which
side an object is on, and runs the in-game benchmark for longer, with more
controls and a metric that scores direction rather than kills.

All brains are the corrected model (`--aminergic zero --sensory
external_only`), injection map v4, CUDA backend. Scripts:
`scripts/screen_dns.py` (now takes the loader flags) and
`scripts/bench_pilots.py` (now takes `--ablate-seeds`, `--rewired-scales`,
and per-condition tokens such as `rewired@0x0.7` or
`rewired_ipsi:pool_ipsi0`). Raw results: `out/screen_dns_v4_*.json`,
`out/bench_wiring10.json`.

## Controls

- **rewired**: every edge's postsynaptic target permuted over the whole
  brain, seeded (`--ablate-seed 0/1/2`). Each presynaptic cell keeps its
  out-degree, synapse counts and sign; all topology, including which
  hemisphere an edge lands in, is destroyed.
- **rewired x0.7**: the same rewiring with every weight scaled by 0.7, the
  gain-matched control. The rewired brain is *more* excitable than the
  measured one under the screen's stimuli (3,568 versus 2,188 spiking
  neurons per tick, mean over the stimulus set); 0.7 brings it to 2,193.
  Any difference in behaviour is therefore not a firing-regime artefact,
  and the direction of the mismatch means gain cannot rescue the control.
- **rewired_ipsi** (new, `ablate="rewired_ipsi"`): targets permuted only
  among cells on the same side of the brain (annotation `side`: left,
  right, center, na), so every hemisphere keeps exactly its total input and
  the measured 14-16% of weight that crosses the midline, and only the
  within-hemisphere circuit is destroyed. This is the control that
  separates "the wiring keeps left and right apart" (gross anatomy) from
  "the wiring computes something". Verified by `tests/test_controls_wiring.py`:
  per-cell total output preserved, per-side totals preserved, targets moved.
- **pool_ipsi0** readout: the rewired_ipsi brain gets the *same screening
  procedure* the measured brain got, not the measured brain's pools. Its
  five best ipsilateral small-object DN types (DNp103, DNge041, DNge104,
  DNa10, DNae009) and its two loom-selective types (DNp102, DNg84) form its
  own readout map, with pool4's calibration spans.

## The screen: does side information reach the descending neurons?

`screen_dns.py`: a small moving disc, a moving bar and a loom, shown to
each eye in turn through the game's own injection path; every DN type's
rate by side; LI = (ipsi - contra) / (ipsi + contra); a type counts as
responsive and consistent when it exceeds its dark rate by 5 Hz and the
side of its response agrees for both eyes.

| Brain | DN types responsive and consistent | of which \|LI\| > 0.8 | pool4 steer pools, small object ipsi / contra (Hz) | pool4 LI | pool4 bar LI | spiking per tick (mean over stimuli) |
|---|---|---|---|---|---|---|
| measured | 4 | 4 | 9.5 / 0.4 | +0.92 | +0.81 | 2,188 |
| rewired, seed 0 | 0 | 0 | 0.6 / 0.5 | +0.03 | -0.01 | 3,568 |
| rewired, seed 1 | 1 | 0 | 0.5 / 0.6 | -0.11 | +0.03 | 3,516 |
| rewired, seed 2 | 1 | 0 | 2.5 / 1.6 | +0.21 | +0.19 | 3,571 |
| rewired seed 0, weights x0.7 | 0 | 0 | 0.3 / 0.2 | +0.29 | -0.02 | 2,193 |
| rewired seed 0, weights x0.5 | 0 | 0 | 0.1 / 0.0 | (+0.60 on 0.1 Hz) | -0.07 | 1,458 |
| rewired_ipsi, seed 0 | 8 | 8 | 1.7 / 0.0 | +1.00 | +0.87 | 3,663 |

Reading:

- The measured wiring delivers the side of a small object to the
  descending neurons: four types (DNp07, DNb05, DNg52, DNb06) fire at
  11-14 Hz on the object's side and 0-1 Hz on the other, and the pooled
  readout has LI +0.92.
- A whole-brain rewiring delivers none of it. Over three independent
  rewirings the pooled readout's LI is +0.03, -0.11 and +0.21 at 0.5-2.5 Hz,
  and no DN type passes the screen with |LI| > 0.8. Scaling the weights
  down to match the measured firing regime does not change this. This is
  the control the 2026-09-14 benchmark used, and it explains that result:
  the rewired pilot fires at head-on targets because both pools light
  together for any large event, and its turns are noise.
- The within-hemisphere rewiring keeps the side, as it must: eight DN types
  pass with LI +1.00 and its pools lateralise perfectly. What it loses is
  gain and selectivity: a small object drives the measured brain's pools at
  9.5 Hz and the control's own best types at 6-9 Hz, but the measured
  brain's pools at 1.7 Hz; its loom-selective types (DNp102, DNg84) are
  different cells from the measured brain's (DNp12, DNbe001, DNb05, DNp07).
  So whether the wiring does more than keep left and right apart has to
  be answered in the game, with each brain given its own screened pools.

## Where the side information travels

Direct and two-hop synaptic weight from each eye's injected cells (the v4
map's types, by side) onto each steer pool (the pool4 types, by side),
signed sums in mV over the corrected model:

| Brain | direct, pool L from inj L / R | direct, pool R from inj L / R | two-hop, pool L from inj L / R | two-hop, pool R from inj L / R |
|---|---|---|---|---|
| measured | 0 / 0 | 0 / 0 | 58,358 / 598 | 297 / 50,170 |
| rewired seed 0 | 114 / 153 | 147 / 165 | -27,795 / -29,431 | -12,509 / -14,089 |
| rewired seed 1 | 127 / 147 | 165 / 166 | -7,000 / -8,255 | -13,439 / -16,555 |
| rewired seed 2 | 143 / 147 | 145 / 183 | -7,120 / -7,324 | -1,890 / 2,567 |

In the measured brain no injected cell touches a steer pool directly, and
the two-hop route is ipsilateral at about 100 to 1 (through the lobula
columnar and tangential relays the cascade probe found). In every rewiring
the pools get a few hundred mV of direct input from each eye, balanced to
within 10-30%, and a net-inhibitory, balanced two-hop input. There is no
structural side channel in a rewired graph, and none is expected.

## A flaw in the earlier benchmarks

Battlezone ignores the gym seed: with sticky actions off, 120 ticks of the
same action give identical frames for seeds 0 and 1. The 2026-09-14
"ten-seed" benchmark therefore ran ten copies of one episode for every
deterministic pilot (the reflex stub's rows are identical across seeds in
`out/bench_pilots_fixed10.log`), and its connectome rows differed only
because the brain's state was carried over from one run into the next.
`bench_pilots.py` now resets the brain at the start of every run and opens
each episode with a seeded burst of 15-60 random joystick ticks before
measurement, so the ten episodes are ten different games. The 10/10 kill
counts on 2026-09-14 stand as "the first head-on enemy is always killed",
which is what they were.

## In-game benchmark

`bench_pilots.py`, ten episodes of 1,500 ticks (100 s) each, seeded random
warm-up, `v4 / pool4 / zoom 2 / eyes 0.15 / saccade 3 / refractory 15 / LI
steering`, corrected model, CUDA. Steering correctness: on every
unsuppressed look tick where the retina's transient energy (summed
|delta luminance| per eye) is at least 0.5 and lateralised by 0.3 or more,
the tick is a *lateralised target*; if the steer pools also cross the
8 Hz steering gate it is a *gated event*, and it is *correct* when the sign
of the pool LI matches the side of the energy. The binomial p is one-sided
against 0.5 over the pooled gated events. Full rows, per-tick events and
action histograms are in `out/bench_wiring10.json`.

| Condition | kills / ep | deaths / ep | turns / ep | lateralised targets / ep | gated events (L / R) | correct L | correct R | correct pooled | p | sign of LI, mean |
|---|---|---|---|---|---|---|---|---|---|---|
| measured wiring | 2.7 | 5.7 | 6.2 | 71 | 63 (14 / 49) | 0.93 | 1.00 | 62 / 63 | 7e-18 | +0.57 |
| rewired seed 0 | 2.3 | 5.9 | 12.4 | 72 | 58 (12 / 46) | 0.00 | 0.52 | 24 / 58 | 0.93 | +0.26 |
| rewired seed 1 | 2.5 | 6.8 | 3.2 | 118 | 29 (12 / 17) | 0.75 | 0.12 | 11 / 29 | 0.93 | -0.69 |
| rewired seed 2 | 2.7 | 6.3 | 1.5 | 96 | 17 (5 / 12) | 0.60 | 0.50 | 9 / 17 | 0.50 | -0.06 |
| rewired seed 3 | 2.3 | 7.4 | 7.6 | 94 | 43 (33 / 10) | 0.58 | 0.00 | 19 / 43 | 0.82 | -0.37 |
| rewired seed 4 | 2.4 | 7.1 | 1.4 | 139 | 18 (4 / 14) | 0.50 | 0.07 | 3 / 18 | 1.00 | -0.72 |
| all five rewirings | 2.4 | 6.7 | 5.2 | 104 | 165 (66 / 99) | | | 66 / 165 | 1.00 | |
| rewired seed 0, weights x0.7 | 2.0 | 6.0 | 8.2 | 67 | 15 (0 / 15) | - | 1.00 | 15 / 15 | (one-sided) | +1.00 |
| rewired_ipsi, measured pools | 1.9 | 8.1 | 0.0 | 158 | 2 (2 / 0) | 0.50 | - | 1 / 2 | - | 0.00 |
| rewired_ipsi, own pools | 0.5 | 2.7 | 2.7 | 13 | 17 (11 / 6) | 1.00 | 1.00 | 17 / 17 | 8e-6 | -0.29 |

Reading:

- **Kills do not measure the wiring.** Every whole-brain condition kills
  1.9-2.7 times per 100 s against the measured brain's 2.7, and dies about
  as often. The first enemy always comes head on and dies to whoever
  presses fire; this is the 2026-09-14 result again, now with independent
  episodes.
- **Steering direction does.** The measured brain's pools crossed the gate
  63 times with a lateralised target in view and pointed at it 62 times.
  The five rewirings crossed it 165 times and pointed at it 66 times, below
  chance in aggregate and at chance or below for each graph. Each rewiring
  has its own fixed bias (mean sign of LI from -0.72 to +0.26) that does
  not track the target, which is why an earlier run of this benchmark with
  replica episodes made one rewiring look right 27 times in 32: it was one
  episode counted ten times with a lucky bias.
- **Gain matching does not rescue the control.** At 0.7 the rewired brain's
  pools crossed the gate 15 times, always with the target on the right and
  always answering right; with no left events and a mean LI sign of +1.00
  that is a one-sided reflex, not steering.
- **The within-hemisphere control keeps the side.** With the measured
  brain's pools it barely crosses the gate (2 events in 1,000 s: those
  types fire at 1.7 Hz in that graph). With its own screened pools it
  crosses 17 times and is right 17 times. It also plays a different game:
  its loom types fire for its own motion, so it reverses on 57% of ticks,
  dies half as often and kills a fifth as often.

## Beyond laterality: what the within-hemisphere shuffle cannot inherit

`scripts/screen_selectivity.py` (same day, later): every descending type's
ipsilateral response to a small moving disc, a tank-sized static disc, a
loom, a full-field flash and a moving bar in each direction, through the
v4 injection, for the measured brain, three within-hemisphere rewirings, a
whole-brain rewiring, and gain-matched versions of the shuffles. Every
brain gets the identical procedure. A type is responsive when some
stimulus lifts it 5 Hz above dark; an index is (a - b) / (a + b) and
"selective" means above 1/3 (the preferred stimulus at least twice the
other). Raw: `out/screen_selectivity*.json`.

| Brain | spiking / tick | responsive DN types (of 460) | small-object responders | best small-object rate (Hz) | lateralised, \|LI\| > 0.8 | loom vs flash selective | loom vs bar selective | direction selective (bar) |
|---|---|---|---|---|---|---|---|---|
| measured | 1,972 | 58 | 6 | 46.7 (DNg100) | 4 | 3 | 8 | 0 |
| (expansion versus contraction, rerun with the receding disc: measured 14 types, rewired_ipsi seed 0: 2, seed 0 x0.7: 2, seed 1 x0.6: 0) | | | | | | | | |
| rewired_ipsi seed 0 | 3,396 | 356 | 10 | 9.0 | 10 | 0 | 55 | 5 |
| rewired_ipsi seed 1 | 3,485 | 359 | 15 | 13.9 | 13 | 0 | 79 | 4 |
| rewired_ipsi seed 2 | 3,416 | 350 | 10 | 10.1 | 9 | 0 | 44 | 5 |
| rewired_ipsi seed 0, x0.7 | 2,259 | 312 | 1 | 5.6 | 1 | 0 | 87 | 0 |
| rewired_ipsi seed 1, x0.6 | 1,942 | 287 | 2 | 9.9 | 2 | 0 | 84 | 0 |
| rewired_ipsi seed 0, x0.5 | 1,591 | 245 | 0 | 2.4 | 0 | 0 | 71 | 1 |
| rewired seed 0 | 3,356 | 243 | 1 | 8.1 | 0 | 0 | 37 | 0 |
| rewired seed 0, x0.7 | 2,118 | 181 | 1 | 5.3 | 0 | 0 | 31 | 0 |

Three things separate the measured wiring from a hemisphere-respecting
random graph, and none of them is gain:

- **Specificity.** Visual input reaches 58 descending types in the measured
  brain and about 355 in every within-hemisphere shuffle; at the measured
  firing regime the shuffles still light 287-312. The measured wiring
  routes vision to a small set of descending neurons; a shuffle sprays it
  over three quarters of them.
- **Gain onto those few.** A small moving object drives six measured types
  at 6-47 Hz (DNg100 47, DNp07 14, DNb05 13, DNbe001 12, DNg52 11, DNb06
  6). The shuffles' best small-object types reach 9-14 Hz at their own
  higher regime and 2-10 Hz when gain-matched, with one or two responders.
  This is why the rewired_ipsi pilot with the measured pools never crossed
  the steering gate in the game, and why its own pools only did so 17
  times in 1,000 s against the measured brain's 63.
- **Expansion.** Three measured types respond more to an expanding disc
  than to a full-field flash of the same final extent (DNae010 9.5 versus
  0 Hz, DNa16 20.5 versus 7.5, DNg100 38 versus 16) and no shuffle at any
  gain has one (best index 0.14). But loom-versus-flash confounds
  expansion with "sustained edge input versus one transient" (a lesson
  from nsfm/fly-afterlife, which withdrew a loom claim on exactly this),
  so the screen was rerun with a **receding disc**: the loom played
  backwards, same edges, opposite direction. Fourteen measured types
  prefer expansion to contraction by at least 2 to 1 (DNg79 0.76, DNg41
  0.65, DNp15 0.65, DNp12 0.59, DNae010 0.58, DNp10 0.56, ...), and
  DNp12, the startle pool's loom cell, is among them at 23 versus 6 Hz.
  DNg100 is not (38 versus 43 Hz): its loom preference was sustained
  input. The within-hemisphere shuffles have 2, 2 and 0 such types at
  best index 0.37, 0.50 and 0.29, and no type in any brain prefers
  contraction. Expansion selectivity is in the wiring; a random graph
  inside a hemisphere sees a loom as an edge moving.

What does not separate them, and is a limit of the model rather than of
the controls: no type in any brain prefers a small object to a loom (size
tuning; the LC11 route carries too little at these gains), and the measured
brain has no direction-selective descending type (the front end injects
rectified transients with no motion computation, as the cascade probe
found for HS cells), while the shuffles produce four or five by chance.
Loom-versus-bar selectivity is common everywhere and says nothing.

## A second individual: MaleCNS v1.0

Would more neurons help? MaleCNS v1.0 (Janelia/Google, CC-BY 4.0) is a
different fly, male, with the ventral nerve cord attached: 166,700
neurons and 25.6 million signed pairs after `scripts/build_malecns.py`
converts the pinned flat-connectome feathers into the three files the
loader reads (`data/malecns/cache/manifest.json`; `--dataset malecns` on
every script). It does not add visual neurons, both brains carry two
complete optic lobes; what it adds is a real pIP10, the VNC and motor
neurons M6 needs, and, for this note, a replication in a second brain
whose wiring was reconstructed by different people from a different
animal.

Gates: silent with no input; LC4 drive fires the giant fibre at 164/82 Hz
(FlyWire 66/64). The Shiu sugar-to-MN9 gate cannot be run, the male
annotations do not label the labellar sugar GRNs by type (flybench
reports the same, and that 13% of them are annotated glutamatergic).

At Shiu gains the male model is over-excited: the same stimuli that light
about 2,000 neurons per tick in FlyWire light 11,000, the pool4 pools
saturate at 74 Hz on both sides (LI 0.00), and 179 descending types
respond. Scaling every weight by 0.6 brings the regime to 1,954, matching
FlyWire's 1,972; at 0.5 the pools are at 1.6 Hz and at 0.35 nothing
lateralises. flybench found the same narrow window and no gain at which
taste and vision both work; our loader confirms it. All male numbers
below are at 0.6, with the male's own shuffles at the same scale.

| Brain (MaleCNS, weights x0.6) | spiking / tick | responsive DN types (of 472) | small-object responders | best small-object rate (Hz) | lateralised, \|LI\| > 0.8 | expansion vs contraction selective | loom vs flash selective |
|---|---|---|---|---|---|---|---|
| measured | 1,954 | 84 | 12 | 31.3 (DNg100) | 2 | 5 | 1 |
| rewired_ipsi seed 0 | 3,762 | 424 | 2 | 16.0 | 2 | 0 | 0 |
| rewired_ipsi seed 1 | 3,810 | 412 | 0 | 4.1 | 0 | 0 | 0 |
| rewired seed 0 | 3,728 | 338 | 0 | 3.4 | 0 | 0 | 0 |

The pattern replicates, cell for cell in places:

- Specificity: 84 responsive descending types against 412-424 in the
  within-hemisphere shuffles and 338 in the whole-brain one.
- Gain onto the same few: DNg100 is the strongest small-object responder
  in both brains (47 Hz in FlyWire, 31 Hz in the male) and it is the
  drive channel the pilot already reads; 12 male types respond to a small
  object against 0-2 in the shuffles.
- Expansion: five male types prefer an expanding disc to a contracting
  one (DNp103 1.00, DNpe025 0.86, DNpe056 0.57, DNb03 0.43, DNp12 0.37)
  against none in any shuffle; DNp12, the startle pool's loom cell, is
  expansion-selective in both flies.
- Laterality: DNp04 and DNp07 lateralise at LI +1.00 in the male (7-8 Hz
  on the object's side, 0 on the other); DNp07 is in the FlyWire steer
  pool. The female pool4 as a whole does not transfer (small-object LI
  +0.03, bar +0.59): DNb05, DNg52, DNb06 and DNa02 do not lateralise in
  the male at these gains, so a male pilot would need its own screened
  pools, which is how the controls were treated above.

Caveats specific to the male run: the retinotopic injection uses the
FlyWire PCA heuristic (`_pca2d`, axes signed against FAFB x and y) on male
coordinates, and the male responds asymmetrically to it (443 spiking
neurons per tick for a small object on the left eye, 1,123 on the right),
so the male's own hex-column assignments (`assignedOlHex1/2`, kept in the
converted table) should replace the heuristic before any male pilot run;
the weight scale 0.6 was chosen to match the female regime, not fitted to
any result; and the male has six contraction-selective and seven
small-object direction-selective types where the female has none, at
rates too low to read anything into yet.

So: more neurons do not help with what is limiting the pilot (the front
end and the model's gain window), but a second brain does something more
useful, it shows that the specificity, the DNg100 gain, the DNp12
expansion signal and the DNp07 lateralisation are properties of fly
wiring rather than of one reconstruction.

## The male in the game, and the fight

`READOUT_MAPS["pool_male"]`: the same recipe as pool_ipsi0, applied to the
male's own screen at weights x0.6. Steer pools = the descending types
lateralised (LI > 0.5) on both a small object and a bar, ranked by
response: DNp04, DNb05, DNp02, DNp20, DNp07 (DNb05 and DNp07 are in the
female pool too). Loom pool = its loudest loom-selective types plus the
giant fibre: DNg33, DNp12, DNp01. pool4's calibration spans. Ten
independent 100 s episodes, same seeds as the female run, the male's own
shuffles at the same weight scale:

| Pilot | kills / ep | deaths / ep | moves / ep | gated events (L / R) | correct | spiking / tick |
|---|---|---|---|---|---|---|
| FlyWire female, pool4 (from above) | 2.7 | 5.7 | 133 | 63 (14 / 49) | 62 / 63 | 11,900 |
| MaleCNS x0.6, pool_male | 1.2 | 2.7 | 805 | 12 (2 / 10) | 10 / 12 | 3,990 |
| MaleCNS rewired_ipsi x0.6, pool_male | 2.1 | 8.4 | 151 | 3 | 3 / 3 | 10,630 |
| MaleCNS rewired x0.6, pool_male | 1.9 | 7.6 | 141 | 4 | 0 / 4 | 10,090 |

The male plays a different game: its loom pool (DNg33 fires at 49 Hz to a
loom and 0 to a small object on the screen, but also to its own motion in
the game) keeps it reversing on 54% of ticks, so it dies half as often as
the female and kills half as often, and it looks at the world on far fewer
ticks (12 gated events against 63). When it does respond it steers toward
the target 10 times in 12. Its shuffles kill about as often as it does,
die three times as often, and never cross the steering gate with the
target in view. So the male replicates the female's qualitative result
(a random graph does not steer, the measured one does) with weaker
statistics, and the two flies are not comparable on kills because their
screened pools give them different reflexes. `out/no-fly-zone-female-vs-male.mp4`
shows seed 0 of each, 100 s, side by side, with the caveats on the cards.

## Can the wiring compute direction? No, not as a point-neuron LIF

`scripts/probe_motion.py`: a bar swept across the left eye in four
directions (two speeds), every T4 and T5 subtype, the HS cells and every
descending type read out; the test is orientation-free (do the a and b
subtypes prefer opposite horizontal directions, and c and d opposite
vertical ones), so it does not depend on the PCA retinotopy's sign.
Injection v4, and v5 = v4 plus the slow inhibitory columnar inputs of T4
(Mi4 as low-passed luminance, Mi9 as low-passed darkness, tau 150 ms, 80 Hz
at full scale), which in the fly arrive from neighbouring columns and make
the offsets that produce direction selectivity.

Result: none. Every T4 and T5 subtype fires at 1.5-4.5 Hz, the same in all
four directions, in the measured brain and in the shuffle, with v4 and with
v5, at both speeds (|DSI| below 0.05 for all eight subtypes in all eight
runs). The single HS cells wander (HSN 0 versus 22 Hz in one run, 35 versus
35 in the next), which is one cell's Poisson noise. The T4 mechanism is a
sub-threshold, dendritic, multi-time-constant computation; a point LIF with
one 1.8 ms delay, forced Poisson spiking at the inputs and a 67 ms sampling
tick does not have it, and adding the right cell types with the right
kinetics at the input does not conjure it. nsfm/fly-afterlife reached the
same place from the other side and distilled flyvis onto the optic lobe.

So direction has to be computed outside the wiring, and the honest way to
do that is to replace exactly one stage. Injection map v6 = v4 plus a
Hassenstein-Reichardt correlator over neighbouring ommatidia (this tick's
transient at the neighbour times last tick's at the cell, minus the
reverse), four directions for ON injected at T4a-d and four for OFF at
T5a-d, gain 20,000 Hz per unit product, capped at 400 Hz, with the
letter-to-direction assignment (a +x, b -x, c +y, d -y) recorded as a
convention. Everything downstream of T4/T5, the tangential cells, the
lobula columnar cells and the descending neurons, stays on the connectome,
and the question becomes whether *that* wiring routes direction to the
descending neurons where a shuffle does not.

## With direction supplied at T4/T5, the wiring builds the horizontal and vertical systems

`probe_motion.py --injection v6`, the same bars, FlyWire, measured against
the within-hemisphere shuffle. By construction T4a/T4b and T5a/T5b now
prefer opposite horizontal directions and T4c/d, T5c/d opposite vertical
ones, in both brains (DSI 0.7-0.9 at the slow sweep): the input is the
same. Everything after it is not:

| Cell | measured, slow sweep (LtoR / RtoL / DtoU / UtoD, Hz) | DSI h / v | shuffle | DSI h / v |
|---|---|---|---|---|
| HSE | 113 / 0 / 11 / 4 | +1.00 / +0.42 | 11 / 13 / 9 / 11 | -0.11 / -0.12 |
| HSN | 93 / 1 / 11 / 4 | +0.97 / +0.50 | 9 / 14 / 10 / 8 | -0.21 / +0.10 |
| HSS | 116 / 11 / 33 / 24 | +0.82 / +0.16 | 13 / 13 / 11 / 13 | 0.00 / -0.05 |
| VS1 | 1 / 63 / 0 / 33 | -0.96 / -1.00 | 21 / 18 / 18 / 18 | +0.10 / 0.00 |
| VS2 | 20 / 28 / 0 / 82 | -0.17 / -1.00 | 8 / 10 / 9 / 6 | -0.10 / +0.20 |
| VS3 | 29 / 6 / 0 / 53 | +0.64 / -1.00 | 0 / 1 / 0 / 3 | (silent) |
| VS4 | 19 / 2 / 0 / 31 | +0.82 / -1.00 | 16 / 18 / 14 / 18 | -0.05 / -0.10 |

The three HS cells (the fly's horizontal system, named for this) fire at
93-116 Hz for one horizontal direction and 0-11 Hz for the other; the
four VS cells (the vertical system) fire for one vertical direction and
are silent for the other. In the shuffle every one of them sits at 8-21 Hz
in all four directions. At the faster sweep the HS preference holds
(+0.55 to +0.77) and VS1 holds (-1.00), the smaller VS cells fade.

Descending neurons: ten types are direction-selective with the same sign
at both speeds in the measured brain and none in the shuffle. DNg41 (h,
1.00 / 0.93, 43 Hz), DNp15 (h, 1.00 / 1.00, 28 Hz), DNp22 (v, -1.00 /
-1.00, 38 Hz), DNae002, DNa15, DNae010, DNa10, **DNa02** (h, 0.77 / 0.53,
48 Hz), DNa16, DNge033. DNa02 is the fly's best-known steering descending
neuron and the cell the M3 pilot first read; with direction available it
carries horizontal motion at 48 Hz. flybench's task #24 (right-yaw optic
flow to left DNp15 only) is the same observation from another group.

The male does the same, louder: at x0.6 with v6, HSE 168 / 0.6 Hz, HSN
156 / 1.9, HSS 181 / 17 for the two horizontal directions, and H2, the
contralateral horizontal cell that in the fly prefers the opposite
direction to HS, 0 / 114 Hz, opposite to HS. Its shuffle has all four at
13-34 Hz in every direction. Direction-selective descending types at the
slow sweep: 47 in the male against 8 in its shuffle (single speed, so a
looser count than the female's ten-at-both-speeds).

The selectivity screen rerun with v6 on FlyWire gives the measured brain
20 direction-selective descending types to a bar and 8 to a small moving
object, against 1-6 and 0-2 in the shuffles at any gain; the expansion
count rises from 14 to 19 and loom-versus-flash from 3 to 8; the
responsive count stays sparse (74 against 287-358).

In the game, v6 changes nothing for the pilot as read out today: with
pool4 the measured brain kills 2.5 and dies 5.8 per 100 s and steers toward
the target on 45 of 47 gated looks (v4: 2.7, 5.7, 62 of 63); the
whole-brain rewiring 26 of 52. The steer pools read object position, not
motion, so direction at T4/T5 is available to the brain and unused by the
readout. A readout that uses it is the next thing to build.

**Yaw, and the optomotor steering signal.** `probe_motion.py --yaw` sweeps
the bar across both eyes at once, the screen rotating past the fly. With
v6 nothing lateralises: both eyes share the screen's direction convention,
so HS on both sides prefer the same sweep (HSE left 113 / 0 Hz, right
145 / 14) and only four descending types flip their right-minus-left rate
with sweep direction. The fly's eyes are mirror images, so v6m mirrors the
right eye's horizontal channels (a means the same body-relative direction
on both sides; which screen direction is front-to-back is a recorded
convention). Under v6m the two HS systems oppose each other (HSE left
113 / 0, right 11 / 140; H2 left 0 / 119, right 63 / 0), and fourteen
descending types flip sign with yaw direction at peak rates of 6-98 Hz:
DNp15 (48 / 0 left, 5 / 98 right), DNg41 (53 / 1, 3 / 64), DNb03, DNa02
(43 / 3, 1 / 22), DNae002, DNa04, DNa16, DNa11, DNg13, DNae005, DNa15,
DNb02, DNge127, DNb01. The within-hemisphere shuffle has three, all under
8 Hz. That is a lateralised, direction-signed steering signal at the
descending neurons for whole-field rotation, from the wiring, given
direction at T4/T5. `READOUT_MAPS["yaw"]` reads the five strongest
(DNp15, DNg41, DNae002, DNa02, DNa15) with the sides swapped so the pilot
steers toward the motion.

In the game (ten independent 100 s episodes, v6m): the yaw pilot kills 2.4
and dies 4.3 per episode against pool4's 2.2 and 5.9 on the same map, with
6.2 turns per episode to pool4's 5.5. The steering-correctness metric
cannot score it (0 gated events): it grades the pool's side against the
retina's *position* energy, and a motion readout fires for scene motion,
which is not lateralised in that measure. The within-hemisphere shuffle
with the same readout kills 0.8 and answers "right" on all 28 of its gated
looks regardless of the target. So the yaw readout is a working pilot
that dies less, and whether its turns follow the scene's motion needs a
motion-signed metric, which is the first thing to build next.

So the two-stage answer to "can the wiring compute direction": the T4
stage cannot, as a point LIF; everything from T4 onward can, and does, and
it builds the known horizontal and vertical motion pathways, in both
flies, and delivers them to named steering descending neurons where a
hemisphere-respecting random graph delivers nothing. v6 is an engineered
stage and is labelled as one wherever it is used.

## Conclusion

The measured FlyWire wiring delivers which side a target is on to the
descending neurons, through a two-hop ipsilateral route with about 100 to
1 selectivity, and the closed-loop pilot steers by it on 62 of 63 gated
looks. A degree-preserving whole-brain rewiring delivers nothing of the
sort, on a screen (three graphs, pool LI within ±0.21), structurally
(balanced direct and two-hop input from both eyes) and in the game (five
graphs, 66 of 165), and matching its firing regime does not change that.
So the wiring matters, and the 2026-09-14 kill benchmark could not see it
because kills are decided by the first head-on enemy.

Laterality alone is anatomy: a rewiring confined to each side of the brain
keeps the pools lateralised (screen LI +1.00, game 17 of 17 with its own
pools), because one eye's optic lobe and most of that hemisphere's
descending neurons share a side. The selectivity screen is what separates
"keeps left from right" from "computes something": the measured wiring
delivers vision to 58 descending types rather than 355, at up to 47 Hz
rather than 14, and fourteen of those types tell an expanding object from
a contracting one where the shuffles have at most two. What the model does not
reproduce in any brain, size tuning and direction selectivity, is a
front-end limit (rectified transients at the medulla, no motion detector)
and is listed as such. So the claim now supported is: the FlyWire wiring
carries a target's side to the descending neurons and concentrates visual
drive, including a loom-specific signal, onto a specific few of them, and
neither a whole-brain nor a within-hemisphere random graph does either.
The same holds in a second individual, MaleCNS, at matched firing regime,
with the same cells (DNg100, DNp12, DNp07) carrying the same signals.
