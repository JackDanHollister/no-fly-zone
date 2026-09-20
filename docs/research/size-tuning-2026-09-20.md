# Size tuning: an engineered surround does not give the wiring it, 2026-09-20

## The question

Every screen so far (2026-09-18, female and male, every control) found no
descending type that prefers a small moving disc to a loom. The 18th's
note called this a front-end limit: injection map v4 drives the medulla
columnar types with each ommatidium's own ON and OFF transients, and a
point LIF over columnar inputs has no lateral antagonism to build a
small-object channel from. Direction selectivity had the same shape of
problem and was answered by injection map v6: compute the stage the LIF
cannot (a correlator) at the retina, inject it where the fly has it (T4,
T5), and ask whether the wiring downstream builds what the fly builds
(it did: HS, VS, direction-selective descending neurons). This is the
same experiment for size.

## The front end

Injection map v7 is v4 with the medulla transient channels
surround-antagonised at the retina: each ommatidium's ON (or OFF)
transient minus the mean transient of the ommatidia within three lattice
spacings, rectified (`surround_matrix`, `centre_surround`; modes `on_cs`,
`off_cs`, `onoff_cs`; radius wider than the screen's small disc, which has
radius two, so a small disc keeps 0.6 of its drive at its centre while a
uniform interior is silent). The fly has such surrounds in the lamina and
medulla (L2 and C3, Mi4, Mi9, CT1); here they are computed, not wired,
and labelled ENGINEERED like v6. Injected at Mi1, Tm3 (ON), Tm1, Tm2,
Tm4, Tm9 (OFF) and T3 (both), the same cells as v4.

## Result

`screen_selectivity.py --injection v7`, the measured brain against two
within-hemisphere rewirings and one whole-brain rewiring, the same
battery as before:

| brain | injection | responsive DN types | small-selective vs loom | small-selective vs flash | loom-selective vs flash | best small-vs-loom index |
|---|---|---|---|---|---|---|
| measured | v4 | 58 | 0 | 1 | 3 | 0.10 |
| measured | v7 | 31 | 0 | 6 | 23 | 0.07 (DNg100) |
| rewired_ipsi seed 0 | v4 / v7 | 356 / 209 | 0 / 0 | 0 / 10 | 0 / 119 | -0.27 / - |
| rewired_ipsi seed 1 | v4 / v7 | 359 / 206 | 0 / 0 | 0 / 12 | 0 / 121 | -0.46 / - |
| rewired seed 0 | v4 / v7 | 243 / 84 | 0 / 0 | 0 / 1 | 0 / 30 | -0.63 / - |

No brain gains a descending type that prefers the small disc to the
loom. What v7 does create, small-versus-flash and loom-versus-flash
selectivity in dozens of types, it creates in every brain, the shuffles
included, and for a reason that has nothing to do with wiring: a
surround silences a uniform field, so the full flash stops driving
anything, and every type that responds to a loom or a disc is now
"selective against the flash". That is the front end talking, not the
connectome.

Why the small disc does not win against the loom under a surround: the
loom is an expanding edge, and an edge is exactly what a surround
detector passes. At every moment the loom presents a ring of ommatidia
whose centres are lit and whose outer surround is not, and that ring
carries as much surround-antagonised drive as the small disc does, at
more places. The measured brain's best small-versus-loom type is DNg100
at an index of 0.07, and it responds to everything. In the fly the
small-object detectors (LC11, LC18 and their kind) do not get their size
tuning from a retinal surround alone; they combine it with the lobula's
own wide-field inhibition against large moving fields and with the
motion of the object against a still background, and none of that is
supplied by a static centre-surround at the input. So the front-end
limit stands, but it is not the surround: the next engineered stage, if
this is pursued, is wide-field motion opponency in the lobula (LPi-like
inhibition), which is a different injection map and a different claim.

## The second stage: wide-field opponency (v8)

Injection map v8 adds the stage the note above asked for: the
surround-antagonised transient minus the eye's mean surround energy,
scaled by the fraction of the eye a small object covers (0.10), rectified
(`wide_field_opponent`). On the retina a lone small disc keeps 0.82 of
its drive, a tank-sized one 0.90, and a loom's edge ring past radius 5
keeps nothing: the ring is everywhere at once and suppresses itself.
Same battery, same brains:

| brain | responsive | small-selective vs loom | loom-responsive types | expansion-selective | best small-vs-loom |
|---|---|---|---|---|---|
| measured, v7 / v8 | 31 / 28 | 0 / 4 | 23 / 2 | 13 / 0 | 0.07 / 0.86 (DNp07) |
| rewired_ipsi seed 0, v7 / v8 | 209 / 173 | 0 / 7 | 125 / 7 | 15 / 0 | - / - |
| rewired_ipsi seed 1, v7 / v8 | 206 / 176 | 0 / 4 | 127 / 11 | 6 / 0 | - / - |
| rewired seed 0, v7 / v8 | 84 / 63 | 0 / 0 | 34 / 0 | 8 / 0 | - / - |

Size-tuned descending neurons appear: in the measured brain DNp07 at
0.86, DNb05 at 0.52, DNg100 at 0.48 and DNg52 at 0.44 prefer the small
disc to the loom. But two things say this is the front end and not the
wiring. The within-hemisphere shuffles have as many such types (7 and 4),
so the connectome adds nothing to what any side-preserving graph passes
through; and every brain has lost its loom, from 23-127 loom-responsive
types to 2-11 and from 6-15 expansion-selective types to none, because a
front end that suppresses wide-field motion suppresses a loom before any
wiring sees it. The fly does not pay that price: its loom detectors (LC4,
LPLC2) take the un-opposed signal and its small-object detectors (LC11,
LC18) the opposed one, on different columnar inputs.

## The mixed map (v9)

So the last version keeps v4's raw transients on the medulla ON and OFF
channels that feed the loom pathway and puts the small-object channel
(v8's computation) only on T2, T2a and T3, the columnar inputs of the
lobula's small-object detectors. The question becomes the one the
direction experiment answered: given the computed signal at the cells
that carry it, does the measured wiring deliver it to particular
descending neurons where a shuffle does not, and keep its loom?

### Result

| brain | injection | responsive | small-selective vs loom | loom-responsive | expansion-selective | lateralised |
|---|---|---|---|---|---|---|
| measured | v9 | 70 | 14 | 42 | 18 | 6 |
| rewired_ipsi seed 0 | v9 | 342 | 0 | 245 | 3 | 34 |
| rewired_ipsi seed 1 | v9 | 345 | 0 | 258 | 4 | 36 |
| rewired seed 0 | v9 | 236 | 0 | 121 | 2 | 0 |
| rewired_ipsi seed 2 | v9 | 331 | 0 | 244 | - | 26 |
| rewired seed 1 | v9 | 236 | 0 | 114 | - | 0 |
| male (MaleCNS x0.6, columns) | v9 | 86 | 2 (DNg16 0.45, DNg100 0.43) | 52 | - | 4 |
| male, rewired_ipsi seed 0 | v9 | 412 | 0 | 293 | - | 5 |
| male, rewired seed 0 | v9 | 310 | 0 | 153 | - | 0 |

This is the direction experiment's answer, for size. Given the
small-object signal at the cells that carry it in the fly, the measured
wiring delivers it to 14 descending types that prefer the small disc to
the loom, DNpe053 (index 0.64), DNge035 (0.61), DNg40 (0.48, 22 Hz to the
disc against 8 to the loom), DNge053, DNpe042, DNg75, DNp70 (58 against
23 Hz) among them, and keeps its loom: 42 loom-responsive types and 18
expansion-selective ones, as with v4. The three rewirings deliver it
nowhere, and neither do a third within-hemisphere rewiring nor a second
whole-brain one (0 and 0). The male at weights x0.6 under its column
placement replicates the direction with two types (DNg16, DNg100)
against none in its shuffles. The rewirings' T2 and T3 outputs, scattered by the shuffle, dilute into
the 236-345 types they light for anything; not one prefers the small
disc. The within-hemisphere shuffles keep laterality (34-36 lateralised
types, more than the measured brain's 6, since everything they touch is
lateralised by anatomy) and nothing else.

So the statement for the README is the same as for direction: the point
LIF cannot build a small-object channel from columnar transients (v4), a
front end that builds it for every cell hands size tuning to every brain
and takes away the loom (v8), and a front end that builds it where the
fly builds it (v9) lets the measured wiring, and only the measured
wiring, route it to specific descending neurons while keeping the loom.
The wiring's contribution is the routing, as it was for direction.

## In the game

`READOUT_MAPS["pool_v9"]`, pool4's recipe on the v9 screen (steer DNp07,
DNp05, DNp02, DNp04, DNp64; loom DNp12, DNge141, DNp26, DNp18), ten 100 s
episodes on v9, against pool4 on v9 and pool4 on v4:

| pilot | injection | kills / lives lost per 100 s | gated looks (L / R) | steer toward target |
|---|---|---|---|---|
| pool4 | v4 | 2.7 / 5.7 | 14 / 49 | 0.96 |
| pool4 | v9 | 3.1 / 4.4 | 91 / 17 | 0.64 |
| pool_v9 | v9 | 2.5 / 4.8 | 7 / 4 | 1.00 |

Nothing to choose on kills with ten seeds. The small-object pools steer
perfectly on the looks they take and take almost none (the engineered
channel is quiet in the game: a tank against Battlezone's static ground
crosses the gate on eleven looks in a thousand seconds); pool4 under v9
takes many more looks and steers worse on the left. The engineered
channel is a screen result, not yet a pilot.

## Kept

v7, v8 and v9 are in `INJECTION_MAPS` for the record, with tests
(`tests/test_surround.py`); `pool_v9` is a named readout. No pilot uses
them by default.
