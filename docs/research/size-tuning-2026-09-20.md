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

## Kept

v7 is in `INJECTION_MAPS` for the record, with its test
(`tests/test_surround.py`). It is not used by any pilot.
