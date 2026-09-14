# Where the visual drive dies: cascade probes, 2026-09-14

Three diagnostics on the FlyWire v783 LIF (CUDA backend, seconds each), run
before any motor tuning. Scripts: `scripts/probe_cascade.py` (stimulate one
cell type, left eye, Poisson at 150/300 Hz for 0.3 s, read every stage),
`scripts/audit_dn_inputs.py` (top presynaptic types and BFS hop counts),
`scripts/probe_frontend.py` (synthetic patterns through the game's own
injection path). Raw results in `out/probe_cascade.json`,
`out/probe_lc_inputs.json`, `out/audit_dn_inputs.json`,
`out/probe_frontend.json`.

## Signs

The Shiu file signs every edge by the presynaptic cell's *predicted*
transmitter. R1-6 photoreceptors are histaminergic (inhibitory onto L1/L2 in
the fly) but are predicted as acetylcholine (5,343 cells), glutamate (1,675),
GABA (530), serotonin and octopamine, so 81% of their 26,928 output edges are
excitatory. L1 is correctly glutamatergic (26,775 of 26,894 edges negative).
Net effect: light excites L1, L1 inhibits Mi1, so the ON pathway never fires
from the retina; L2 is signed excitatory and does drive Tm1/Tm2 weakly. This
is why the original design injected at Mi1/Tm1/Tm2 and bypassed the lamina.

## Stage-by-stage propagation (left eye, all cells of the type)

Mean Hz over all left cells of each downstream type; DN columns are single
cells so 15 Hz is one spike per 66.7 ms.

| Stimulus at 150 Hz | Mi1 | Tm1 | T4a | T5a | HSE | LC4 | LPLC2 | LC10a | LC11 | DNa02 L/R | DNg100 | DNp01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1-6 | 0 | 2.4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 0 |
| L1+L2+L3 | 0 | 24.6 | 0 | 0 | 0 | 0.1 | 0 | 0 | 0 | 0/0 | 0 | 0 |
| Mi1 | 149 | 0 | 6.3 | 0 | 3.3 | 0 | 0 | 0 | 0 | 10/? | 33 | 0 |
| Mi1+Tm3 | 149 | 0 | 11.2 | 0 | 50 | 0.2 | 0 | 0 | 0 | 63/? | 13 | 0 |
| Tm1+Tm2+Tm4+Tm9 | 0 | 151 | 0 | 7.4 | 60 | 71 | 0 | 0 | 0 | 27/? | 3 | 7 |
| T4a-d+T5a-d | 0 | 0 | 151 | 149 | 363 | 0.5 | 30 | 0 | 0 | 70/? | 23 | 13 |
| LC4 | 0 | 0 | 0 | 0 | 0 | 150 | 0 | 0 | 0 | 0/0 | 0 | 137 |
| LPLC2 | 0 | 0 | 0 | 0 | 0 | 0.1 | 150 | 0 | 0 | 0/0 | 0 | 180 |
| LC10a | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 151 | 0 | 157/0 | 47 | 0 |
| LC11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 151 | 33/53 | 40 | 0 |
| HSE+HSN+HSS | 0 | 0 | 0 | 0 | 137 | 0 | 0 | 0 | 0 | 27/0 | 0 | 0 |

At 300 Hz the same pathways roughly double; Mi1 alone then reaches T4a 35 Hz,
HSE 160 Hz and DNa02 67/7 Hz. The escape pathway is intact and strong (LC4
and LPLC2 to DNp01, plus DNp04, DNp02, DNp11). The only strongly lateralised
steering channel is LC10a to DNa02 (157 Hz same side, 0 opposite), through
the AOTU, the fly's target-pursuit circuit; HS to DNa02 is weaker but also
ipsilateral. DNa02's inputs are 84% central brain (PFL3, LAL, PS, VES) and
only 3% visual projection, the same picture FlyPong found in MaleCNS, but the
two-hop route exists and works.

## Inputs to the relays that matter

| Target | Top excitatory inputs (synapses) | Top inhibitory |
|---|---|---|
| LC10a (115 per side) | LC9 2,024, Tm5f 1,308, Tm20 1,067, LC10c 806, Tm4 715, Tm36 688 | TuTuAa 3,576, AOTU042 2,342, LT52 1,011, Li01 835 |
| LC11 (66 per side) | T3 20,900, T2 12,922, T2a 8,686, Tm21 6,308, Tm25 4,951 | Li15 4,757, cL21 3,707 |
| LC4 (104) | T2 22,194, TmY3 19,737, Tm4 14,234, Tm2 12,409, Tm3 6,550 | Li22 5,198 |
| LPLC2 (210) | Tm5f 14,102, T5c 8,950, T4c 8,551, T5b 8,449, T4b 6,704, Tm20 6,602 | PVLP011 9,660 |
| T4a (1,462) | Mi1 69,000, Tm3 22,672 | Mi9 19,387, CT1 15,723, Y1 13,448, Mi4 10,206 |
| T5a (1,484) | Tm9 37,099, Tm2 35,273, Tm1 27,701, Tm4 15,355 | CT1 31,312 |
| DNp01 (2) | LPLC2 1,080, LC4 805, DNp70 587 | CB3707 633 |

Stimulating those inputs at 150 Hz: T2+T3+T2a+Tm21+Tm25 drives LC11 to
133 Hz; TmY3 drives LC4 to 105 Hz and DNp01 to 100 Hz; T2 drives LC4 to 84 Hz;
but Tm5f+Tm20+Tm4+Tm36+LC9 together drive LC10a to only 5.5 Hz. LC10a is
inhibition-dominated in this model and will not be a practical channel
without either much stronger drive or silencing TuTuAa/AOTU042.

Hop counts (edges with at least one synapse): from Mi1+Tm3 or the OFF
medulla set, every lobula relay (LC4, LPLC2, LC10a, LC11, T4, T5) is one hop
and DNa02, DNg100 and DNp01 are two hops; oviDN is three. Structure is not
the obstacle. Drive is.

## What the game's injection actually delivers

Synthetic patterns on the left eye through `ConnectomeBrain.step` as the loop
calls it (R1-6 luminance at up to 150 Hz; Mi1 ON and Tm1/Tm2 OFF transients
at gain 3,000/3,000/2,000 Hz per unit change, capped at 400 Hz). Peak rates
across ticks, left side:

| Pattern | lit ommatidia | Mi1 | Tm1 | T4a | T5a | HSS | LC4 | LC11 | DNa02 L/R | DNg100 | DNp01 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| disc r=1.5 (tank-sized) | 7 | 4 | 4 | 0 | 0 | 0 | 1 | 0 | 0/0 | 0 | 15 |
| disc r=3 | 31 | 13 | 15 | 0 | 1 | 0 | 6 | 2 | 0/0 | 22 | 38 |
| disc r=6 | 121 | 47 | 61 | 0 | 5 | 0 | 21 | 4 | 15/15 | 75 | 30 |
| whole eye | 721 | 397 | 406 | 0 | 0 | 180 | 95 | 0 | 30/30 | 90 | 82 |
| loom 1 to 14 | 685 | 76 | 388 | 0 | 0 | 165 | 96 | 0 | 30/15 | 82 | 68 |
| small disc moving L to R | 16 | 10 | 13 | 2 | 0 | 90 | 4 | 2 | 30/30 | 52 | 45 |
| small disc moving R to L | 16 | 13 | 13 | 2 | 0 | 90 | 4 | 0 | 0/30 | 52 | 15 |
| bar L to R | 93 | 29 | 30 | 4 | 0 | 135 | 8 | 3 | 30/45 | 82 | 60 |
| bar R to L | 93 | 30 | 35 | 0 | 0 | 120 | 9 | 0 | 45/60 | 82 | 60 |

Conclusions:

- A tank-sized object is invisible: seven ommatidia of transient drive reach
  Mi1/Tm1/Tm2 at the cap but nothing downstream integrates it. LC11, the
  small-object detector, needs T2/T3/T2a, which the injection never touches.
- T4/T5 barely fire (5 Hz or less) because the drive is transient-only at
  Mi1 and Tm1/Tm2, with no Tm3, Tm4 or Tm9, and no sustained component.
  HS cells still reach 90-180 Hz by summing thousands of weak T4/T5 inputs,
  but without direction selectivity.
- DNa02 responses are one or two spikes per tick and do not follow stimulus
  side or direction. The steering signal the M3 demo was reading is noise on
  top of the idle scan.
- Loom and whole-field flashes reach LC4/LPLC2 and DNp01 at 70-80 Hz. The
  damage lamp, if it were inside the eye field of view, would trigger a real
  escape.

## Step 2 plan

Broaden the injection to the cell types the relays actually read: Tm3 on the
ON channel; Tm4 and Tm9 on the OFF channel; T2, T3 and T2a with rectified
transients for LC11; TmY3 for LC4. Then re-run `probe_frontend.py` and judge
by three numbers: LC11 and T4/T5 response to a tank-sized moving disc, DNa02
lateralisation with stimulus side, and DNp01 on loom. Keep R1-6 luminance as
context. LC10a stays out until its inhibition is understood. Every variant is
recorded as a named injection map so footage can say which one it used.

## Step 2 results: injection maps

`brain_connectome.INJECTION_MAPS` now holds named maps; `run_m3.py
--injection` selects one and the HUD prints it. Front-end probe peaks (left
eye, Hz), the numbers that matter:

| Map | Adds | tank-sized disc: LC11 | small moving disc: LC11 / LC4 / DNp01 | loom: LC4 / DNp01 | small disc fires escape? |
|---|---|---|---|---|---|
| v0 | (M3 map) | 0 | 2 / 4 / 45 | 95 / 60 | yes |
| v1 | Tm3 (ON), Tm4, Tm9 (OFF) | 0 | 1 / 8 / 38 | 183 / 38 | yes |
| v2 | v1 + T2, T3, T2a, TmY3 | 11 | 22 / 27 / 98 | 284 / 142 | yes, strongly |
| v3 | v1 + T3, T2a | 10 | 19 / 5 / 60 | 155 / 90 | yes (T2a via LC15/LC17) |
| v4 | v1 + T3 | 7 | 13 / 6 / 0 | 163 / 38 | no |

v4 is the working map: a small object reaches LC11 on the correct eye and
does not fire the giant fibre; loom still does. LC10a stays unreachable.

## Step 3 results: the DN screen

`scripts/screen_dns.py` presents a small moving disc to each eye and a loom
to each eye through a named map, accumulates rates for every descending type
by side, and reports the lateralisation index LI = (ipsi - contra) /
(ipsi + contra), requiring the sign to agree for both eyes.

With v4, the types that fire on the same side as a small object, all four
with LI above 0.8, and none on the opposite side:

| Type | small object ipsi / contra (Hz) | bar ipsi / contra | loom |
|---|---|---|---|
| DNp07 | 12.9 / 0.0 | 46.3 / 2.2 | 28.2 |
| DNb05 | 12.4 / 0.0 | 76.7 / 1.1 | 33.5 |
| DNg52 | 11.6 / 0.8 | 22.0 / 4.8 | 9.0 |
| DNb06 | 5.4 / 0.6 | 17.8 / 9.8 | 16.2 |
| DNa02 (the M3 readout) | 4.3 / 0.6 | 17.6 / 3.6 | 11.5 |

With v3 the picture inverts: a dozen types respond on the opposite side
(DNa01 at 28 Hz contra, DNg13, DNb01, DNa03, DNa13, DNa02 with LI -0.55),
which is why the M3 demo's single-cell DNa02 "steering" pointed the wrong way
for one eye and nowhere for the other. Loom-selective types: DNp12 (loom
24 Hz, small object 0.6 Hz), DNbe001, DNb05, DNp07. DNg100 fires at 50-80 Hz
for anything visual on either eye and is kept as the drive channel. oviDN,
the pIP10 stand-in, is silent under every visual stimulus.

`READOUT_MAPS["pool4"]`: steer pools = {DNp07, DNb05, DNg52, DNb06, DNa02}
per side (6 cells each), drive = DNg100, startle = {DNp12, DNbe001, DNp01},
FIRE = min of the two steer pools (both eyes see it: object ahead). The
contract keys are unchanged so `controls.py` is untouched;
`dn.CALIBRATION_MAPS["pool4"]` holds provisional spans from the screen.

## Getting the fly to play: what it took

A real Battlezone tank on the horizon is one ommatidium wide at 721
ommatidia over the whole screen (`out/retina_mosaic_120.png`), so
`HexRetina(zoom=2)` lays the lattice over the central half of the eye. Then,
with `run_m3.py --injection v4 --readout pool4 --zoom 2` and the idle scan
off (now the default; `--idle-scan` re-enables it, labelled on the HUD):

1. First closed-loop run: silent for 15 s while nothing moved, then an enemy
   appeared, both steer pools and the loom pool fired, the fly fired and
   backed off, scored 1,000 (a kill), lost a life, and then locked into
   reverse-and-fire for the remaining 16 s because its own motion lit every
   pool at 70-140 Hz and re-triggered the startle every tick.
2. Saccadic gating (`--saccade-hold 3`: after any tick that moved the tank,
   visual drives are ignored for three ticks; the fly's own efference-copy
   trick) plus a startle refractory (`--startle-refractory 15`) removed the
   lock-in: 358 of 450 ticks idle, 27 fire ticks instead of 245, and the same
   kill, this time traceable: at tick 212 both pools lit with a tank ahead,
   the fly fired that tick, the shell landed at tick 218.
3. It never turned. With a 24 Hz span both pools saturate on any strong
   event, so the normalised difference is zero while the lateral signal is a
   ratio. `--steer-mode li` steers from (R - L) / (R + L) gated on R + L of
   at least 8 Hz; a one-sided input now gives a left saccade, three ticks of
   fixation, and another. In that seed every event was straight ahead (one
   tick with |LI| above 0.5), so the mechanism had nothing to turn toward.

Same-seed benchmark, `scripts/bench_pilots.py`, three seeds, 450 ticks,
v4 / pool4 / zoom 2 / saccade 3 / refractory 15, difference steering:

| Condition | reward (mean) | lives lost | fire ticks | moves | turns | spiking per tick |
|---|---|---|---|---|---|---|
| connectome | 333 (1,000 / 0 / 0) | 1.0 | 81 | 215 | 0 | 19,136 |
| rewired (targets permuted) | 0 | 1.7 | 6 | 52 | 12 | 15,130 |
| silenced | 0 | 1.0 | 0 | 0 | 0 | 8,253 |
| reflex stub | 0 | 2.0 | 20 | 143 | 6 | n/a |

The rewired control keeps a similar firing regime (15k vs 19k spiking
neurons per tick), so no gain matching was needed. One kill in three seeds
against none for the controls is suggestive, not evidence; the ten-seed run
with LI steering is below.

Honesty notes: the pools, the FIRE rule, the saccade program, the refractory
and the LI normalisation are all designed mappings from real descending
neuron activity, chosen from measurements, and every run prints and burns
them into the video footer. The connectome computes which neurons fire; it
does not choose the mapping. The startle back-off is the fly's own loom
escape; the fire-when-both-eyes-see-it rule is ours.

## The persistent state, and two model corrections

Every long run settled into pool rates of exactly 70 / 56 / 48 Hz whatever
was on screen. Cutting the input to darkness after a big visual event left
9,654 neurons spiking indefinitely in the reference model: 2,142 KCg-m and
1,608 KCab Kenyon cells, PAM dopamine neurons, antennal-lobe local neurons,
olfactory receptor neurons, and APL, DPM and MBON05 at 400+ Hz. The rewired
control falls silent within two ticks. The loop is an artifact of the sign
convention: the Shiu file signs every edge by the presynaptic cell's
*predicted* transmitter and treats dopamine, octopamine and serotonin as fast
+1 synapses, so the ~300 PAM cells close a Kenyon cell to PAM to Kenyon cell
positive feedback loop. It is reachable at every injection cap down to 100 Hz,
so drive strength is not the cause. (The prediction also calls every Kenyon
cell dopaminergic; they are cholinergic, and `known_nt` says so.)

Two loader options, both recorded in `brain.meta` and on the HUD:

- `aminergic="zero"`: fast weight removed from the 333,211 edges whose
  presynaptic cell's literature transmitter is dopamine, octopamine or
  serotonin (labels from `known_nt` first, so Kenyon cells keep their
  cholinergic outputs). Persistent state: gone, 0 neurons in darkness.
  Gates: sugar to MN9 97-102 Hz (reference 101), LC4 to giant fibre 66/68
  (reference 66/64).
- `sensory_input="external_only"`: every synapse onto the 16,933 primary
  sensory neurons zeroed, so receptor neurons fire only from injected input.
  Alone it removes the receptor neurons from the loop but not the loop.
- `signs="known_first"` (re-sign 14.8M edges by literature transmitter) was
  tried and is not needed; it leaves a 122-neuron lateral-horn remnant.

With `--aminergic zero --sensory external_only` the pilot is event-driven:
pools quiet on 1,185 of 1,220 look ticks in a 100 s run, bursts only when a
tank is close, eight real turns (the frame `out/m3_fixed_turn.png`: tank
right of centre, right pool 12 Hz, left 2 Hz, next action UPRIGHT), three
kills, but seven deaths: a fly that waits gets shot at close range, and a
distant tank is still two ommatidia wide.

Both eye cameras render the same CRT, so `PilotLoop(eye_offset=0.15)` now
shifts each eye's retina window sideways with binocular overlap in front;
without it a left/right readout can never lateralise.

## The honest benchmark

`bench_pilots.py`, ten seeds, 450 ticks (30 s) each, `v4 / pool4 / zoom 2 /
eyes 0.15 / saccade 3 / refractory 15 / LI steering`:

| Brain | Condition | kills / 10 | lives lost | fire ticks | turns |
|---|---|---|---|---|---|
| reference model (before the corrections) | connectome | 3 | 1.0 | 84 | 0.3 |
| | rewired | 10 | 1.8 | 7 | 4.1 |
| corrected (aminergic zero, sensory external-only) | connectome | 10 | 1.3 | 5.6 | 1.7 |
| | rewired | 10 | 1.1 | 4.4 | 2.9 |
| either | silenced | 0 | 1.0 | 0 | 0 |
| n/a | reflex stub | 0 | 2.0 | 19 | 8 |

Reading: on this task, at this level, the measured wiring does not do
anything a degree-preserving random graph does not. The kill is the first
enemy approaching head-on; when it fills enough of the eye, many descending
cells fire in any dense graph, both steer pools cross the FIRE rule, and the
shot lands. The reference model was *worse* than its control because the
persistent state jammed its pools after the first event. The silenced brain
and the stub never score, so the loop, the readouts and the motor programs
are doing real work; the claim they cannot support yet is that the
connectome's topology matters. What would test that: longer episodes with
off-axis targets (the connectome's turns are toward the target, the rewired
graph's are noise, but 1.7 versus 2.9 turns per episode is not a measurement),
a balanced-success metric over left and right targets, and a gain-matched
rewired control (the shuffled graph's firing regime must match the intact
one before behaviour is compared).

Recommended pilot command:

```bash
MUJOCO_GL=egl .venv/bin/python scripts/run_m3.py --backend cuda --steps 1500 --video \
  --injection v4 --readout pool4 --zoom 2 --eye-offset 0.15 \
  --saccade-hold 3 --startle-refractory 15 --steer-mode li \
  --aminergic zero --sensory external_only --trace my_run
```
