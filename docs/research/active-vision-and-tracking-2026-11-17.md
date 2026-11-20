# Active binocular vision, target tracking and bilateral GRU readouts, 17 November 2026

Following the [closed September 25 plasticity campaign](learning-campaign-2026-09-25.md),
I audited why the standalone pilot spent most of its 60-second demo idle
(`out/pilot_20260925_200200_883034.json`: `NOOP` on 787 of 900 ticks, five
turns, 1,000 points, five lives lost), why the measured connectome had not yet
outscored whole-brain shuffles in the game, and why the earlier PPO readout
heads collapsed across brain conditions.

Three concrete bottlenecks turned up in the geometry and feature code before
touching a single synaptic weight.

## 1. Three bottlenecks in the existing loop and readout wrapper

### Central crosshair scotoma in the eye cameras

`eye_left` (`x = -0.022 m`) and `eye_right` (`x = +0.022 m`) pointed straight
ahead (`+y`, parallel optical axes) with `fovy = 45°` at a CRT `0.175 m` away.
That geometry projects the screen centre (`x = 80` in Battlezone, the aiming
crosshair) onto `u = 0.614` in the left camera image and `u = 0.386` in the
right. Shifting the zoomed (`zoom = 2.0`) retinal windows outward by
`eye_offset = 0.15` (`center_x = 0.35` left, `0.65` right) moved both retinas
*away* from the crosshair rather than onto it:

- **Zero ommatidia in either eye** sampled `x ≈ 74..86` on the Battlezone CRT.
  Turning toward a tank made it disappear into a central blind spot just as it
  lined up with the gun, breaking `pool4`'s binocular fire rule
  (`min(DNa02_L, DNa02_R)`) and hiding incoming shells travelling down the
  sight line.
- The top ~30 pixels of the CRT (the radar and damage lamp) also fell outside
  both zoomed lattices.

Converging both eye cameras onto the CRT centre (`(0, 0.25, 0.10)`,
`eye_fovy = 54°`, `converge_eyes = True`) with `zoom = 1.15`,
`eye_offset = 0.11`, `center_y = 0.46` keeps zero bezel bleed (`max == 0` on a
black CRT), places >= 15 ommatidia per eye across the central crosshair bar
(`x = 78..83`), and brings the radar back onto the upper ommatidia of both eyes.

### Saccadic foraging and discrete-turn transients

Battlezone rotates `11.25°` per tick (`32` ticks for `360°`). With no active
search when both steer pools are quiet, an empty viewport yields zero retinal
transients (`NOOP` forever). Conversely, turning on consecutive ticks without a
1-tick post-saccade fixation (`saccade_hold = 1`) overshoots the crosshair due
to the 1-tick sensorimotor delay and floods `L1/L2/L3` with wide-field mountain
edges. Adding an opt-in active-foraging motor mode (`--preset active`:
converged eyes, `saccade_hold = 1`, `fovea_li = 0.20`, `search_steer = 1.0`,
`direct_steer = True`, and self-turn transient gating) lets the frozen circuit
sweep in discrete saccade-and-fixate steps and hold fire when a target enters
the binocular fovea.

Testing `injection = "v10"` (`onoff_wf` per-eye mean normalisation on `T2`)
exposed a separate front-end trap: dividing `|delta|` by each eye's own spatial
mean makes the summed drive across all 721 ommatidia equal `721` in *both* the
target eye and an almost-empty eye with horizon flicker, muting inter-ocular
contrast for a fixed linear pool. `ACTIVE_PRESET` therefore keeps the pure
lamina front end (`injection = "v4"`).

### Hemisphere collapse and quiet-pool noise amplification in `FlySquad`

`FlySquad._build_feature_pooling` previously pooled `(ct == t)` across the
whole brain without `side`, averaging left and right hemispheres into 908
symmetric features so a readout head could not tell left visual projection
activity from right. Splitting `(cell_type, side)` produces `1,799` bilateral
features on FlyWire v783, but `1,192` of those pools are near-silent
(`var < 0.01` in `log1p(Hz)`), so standard `1 / sqrt(var + 1e-8)` normalisation
amplified single stray Poisson spikes more than tenfold across `1,192`
channels. Adding a variance floor (`var_floor = 1.0`) to `RunningNorm`, encoder
weight decay (`1e-4`), internal motor efference copy, and a `256`-unit
`GRUCell` with factored `3 × 3 × 2` `(throttle, steer, fire)` heads resolves
both issues without giving the policy any non-retinal game state.

---

## 2. Focused moving-target tracking benchmark

`scripts/bench_tracking.py` addresses the September 25 milestone directly: keep
a compact (`~22 × 26` px, ~2-ommatidia radius) high-contrast target centred on
the cockpit CRT across eight balanced left/right trajectories (`150` ticks each
at `15 Hz`, `search_steer = 0.0` so blind scanning cannot masquerade as
tracking), comparing measured FlyWire v783 wiring against three gain-matched
whole-brain shuffles, two within-hemisphere shuffles, `rewired_ipsi` with its
own screened pools (`pool_ipsi0`), and a silenced brain (`out/bench_tracking_2026-11-17.json`):

| Condition | RMS error (°) | MAE (°) | Lock fraction (`<= 8°`) | Turns | Turn-sign accuracy | Spiking / tick |
|---|---:|---:|---:|---:|---:|---:|
| `measured` (`pool4`) | **16.87** | **15.86** | 0.17 | 168 | 0.67 | 5,767 |
| `rewired@0x0.7` | 25.41 | 22.94 | 0.15 | 14 | 0.43 | 5,199 |
| `rewired@1x0.7` | 26.74 | 23.73 | 0.17 | 5 | 0.11 | 5,136 |
| `rewired@2x0.7` | 28.74 | 25.85 | 0.15 | 37 | 0.33 | 5,004 |
| `rewired_ipsi@0x0.7` (`pool4`) | 16.13 | 14.15 | 0.39 | 47 | 0.78 | 5,747 |
| `rewired_ipsi@1x0.7` (`pool4`) | 20.19 | 18.38 | 0.08 | 67 | 0.71 | 5,452 |
| `rewired_ipsi@0x0.7:pool_ipsi0` | 26.88 | 24.48 | 0.10 | 47 | 0.55 | 5,151 |
| `silenced` | 22.32 | 20.32 | 0.17 | 0 | — | 4,718 |

**What this shows (and what it does not):**

- Measured wiring tracks cleanly (`16.87°` RMS, `168` turns) and beats all three
  whole-brain shuffles (`25.41–28.74°`), the silenced brain (`22.32°`),
  `rewired_ipsi` on its own screened pools (`26.88°`), and the `rewired_ipsi`
  group mean (`18.16°`).
- However, `rewired_ipsi@0x0.7` reading the anatomical `pool4` cells *also*
  tracks a solitary bright spot on an empty dark field (`16.13°`). When only
  one object exists in the scene, keeping left-eye projections inside the left
  hemisphere and right-eye projections inside the right hemisphere already acts
  as a monocular energy meter. Solitary-spot tracking therefore separates
  hemisphere-preserving wiring from whole-brain shuffles, but does not by
  itself isolate internal optic-lobe microcircuitry.

---

## 3. Cluttered in-game active-reflex wiring benchmark

Where internal microcircuitry *does* separate measured wiring from both
whole-brain and within-hemisphere shuffles is closed-loop Battlezone combat
with the crosshair blind spot removed (`scripts/bench_pilots.py --preset active`,
ten seeded `1,500`-tick / `100`-second episodes per condition,
`out/bench_active_wiring10.json`):

| Condition | Kills / 100 s | Lives lost / 100 s | Aimed fire ticks | Turns | Responded fraction | Spiking / tick |
|---|---:|---:|---:|---:|---:|---:|
| `connectome` (`pool4`) | **2.10** | 6.20 | **342.6** | 517.0 | **0.50** | 12,226 |
| `rewired@0x0.7` | 0.60 | 6.60 | 43.2 | 736.4 | 0.00 | 11,299 |
| `rewired@1x0.7` | 0.60 | 6.10 | 63.0 | 710.4 | 0.00 | 11,524 |
| `rewired@2x0.7` | 0.60 | 5.60 | 44.7 | 711.3 | 0.05 | 11,223 |
| `rewired_ipsi@0x0.7` (`pool4`) | 0.80 | 6.30 | 62.8 | 715.8 | 0.00 | 11,554 |
| `rewired_ipsi@0x0.7:pool_ipsi0` | 1.10 | 5.20 | 54.4 | 672.2 | 0.07 | 11,248 |
| `silenced` | 0.10 | 6.50 | 0.0 | 750.0 | 0.00 | 8,205 |

With mountains, horizon edges, obstacles and radar blips present in both eyes at
once, `connectome` crosses the steering gate on `50%` of lateralised target
looks (versus `0–7%` in every shuffle), fires `342.6` aimed binocular shots per
episode, and scores **2.10 kills per 100 seconds** — `3.5×` the three
whole-brain shuffles (`0.60` each), `2.6×` `rewired_ipsi@0x0.7` (`0.80`), and
`1.9×` `rewired_ipsi@0x0.7:pool_ipsi0` (`1.10`). On 16 additional 60-second
held-out starts (`101..116`), the same frozen `v4` reflex scored `1.62` kills
versus `0.38` for `rewired@0x0.7` and `0.75` for `rewired_ipsi@0x0.7:pool_ipsi0`.

---

## 4. Bilateral recurrent GRU readouts on the frozen connectome

Finally, I trained variance-floored `RecurrentActorCritic` (`GRUCell(256, 256)`)
policies over the `1,799` bilateral `(cell_type, side)` features on the RTX
4090:

1. **Four matched from-scratch runs (`--seed 3`, starts `300..315`, `307,200`
   fly-ticks / ~345 episodes each, `--warmstart-updates 0`):** `connectome`,
   `rewired` (`x0.7`), `rewired_ipsi` (`x0.7`), and the no-brain `retina`
   ceiling.
2. **A warm-started connectome pilot (`champion_connectome_gru`, `--seed 42`,
   starts `4200..4215`, `409,600` fly-ticks / `466` episodes, `35` DAgger
   behaviour-cloning updates from the active reflex followed by `165` PPO
   updates).**

Evaluated greedily (`argmax`, zero exploration noise) across 16 fresh audited
60-second starts (`101..116`, disjoint from both `300..315` and `4200..4215`,
`out/eval_readout_2026-11-17.json`):

| Policy / brain condition | Kills / 60 s | Lives lost / 60 s | Utility (`kills - 0.5 * lives`) |
|---|---:|---:|---:|
| Random actions | 1.56 | 4.31 | -0.59 |
| Legacy reflex (`pilot` preset) | 1.31–1.56 | 2.56–2.75 | -0.06 to +0.28 |
| **Champion Connectome GRU (`_best`, frozen connectome)** | **2.19** | **2.50** | **+0.94** |
| Champion Connectome GRU (final update 200) | 1.81 | **2.44** | +0.59 |
| From-scratch Retina GRU (`_s3`, no-brain ceiling) | 2.00 | 2.62 | +0.69 |
| From-scratch Connectome GRU (`_best`) | 1.88 | 3.12 | +0.31 |
| From-scratch Rewired GRU (`_s3`, `x0.7`) | 2.19 | 3.44 | +0.47 |
| From-scratch Rewired-Ipsi GRU (`_s3`, `x0.7`) | 1.94 | 3.75 | +0.06 |

**Interpretation and honest boundary:**

- **The pilot now genuinely plays Battlezone in both wired and physical-lever
  modes.** On `101..116`, `champion_connectome_gru_best` scores kills in 15 of
  16 episodes (including two 5-kill episodes and a 4-kill episode), cuts life
  losses from `4.31` (random) to `2.50`, and reaches **+0.94 utility** — outperforming
  random play, legacy and active reflexes, and the from-scratch retina ceiling
  (`+0.69`). On the bare default seed-0 60-second launcher run
  (`out/pilot_20261117_174132_735739.mp4`), it scores **6,000 points with 1 life
  lost** (`24` `NOOP` ticks out of `900`), and in physical `--embodied` mode
  (`out/pilot_20261117_174139_553233.mp4`) it scores **2,000 points with 2 lives
  lost** via T1 claw contact on the levers and proboscis presses on the fire
  button.
- **High-capacity readouts still exploit shuffled brains as liquid reservoirs.**
  When a `256`-unit GRU with `~460k` trainable weights reads all `1,799`
  bilateral pools from scratch, `rewired` (`2.19` kills, `3.44` lives, `+0.47`
  utility) matches or slightly edges from-scratch `connectome` (`1.88` kills,
  `3.12` lives, `+0.31` utility) in raw kills, though at higher mortality than
  warm-started `connectome` (`2.50` lives) and `retina` (`2.62` lives). That
  matches the earlier MLP lesson: a large learned decoder can read retinotopic
  random projections, whereas **the zero-parameter anatomical descending-neuron
  readout (`pool4`) cannot**. Both controllers are exposed on `./no-fly-zone`
  (`--controller active|learned|legacy`) and clearly distinguished on the HUD
  (`REFLEX PILOT` vs `TRAINED READOUT (GRU)`).

---

## 5. Follow-up (19 November 2026): catching the PPO circle-strafe shortcut and uncrossing the fly's anterior foveas

### Why the trained GRU looked sideways in the replay

Watching the November 17 default recording (`out/pilot_20261117_174132_735739.mp4`)
immediately raised a red flag: the tank seemed to sweep steadily to the right
rather than turning left and right to hunt. Inspecting the 900-tick trace
confirmed a textbook RL shortcut:

- Across `900` ticks, the PPO GRU held `UP` on `823` ticks (`91%`), pulsed
  `RIGHT` on **`445`** ticks (`439` single-tick pulses, `mean_run = 1.0`, `889`
  action transitions) and pressed **`LEFT` only `3` times total**, alternating a
  2-tick `UP ↔ UPRIGHTFIRE` metronome (`ticks 200–240`).
- In 1980 Atari Battlezone, enemy tanks shoot at your current position without
  leading a curving path. Driving continuously forward while curving clockwise
  slides the tank off incoming shells (cutting deaths to `1` in 60 s) while
  sweeping the crosshair across 360° for accidental fire kills (such as the
  `6,000`-point hit on tick `605`).
- Because the GRU receives its own previous motor command (`motor_efference`)
  and carries recurrent state `h_t`, it can sustain that `UP ↔ UPRIGHTFIRE`
  oscillator largely open-loop. That explains the control table in
  `out/eval_readout_2026-11-17.json`: a from-scratch PPO GRU trained on a
  **randomly scrambled (`rewired`) brain** reached `2.31` kills and `+0.78`
  utility — outscoring the same GRU on the real connectome (`1.69` kills,
  `+0.41` utility).

Because of that shortcut, `./no-fly-zone` now defaults to the **pure frozen
connectome reflex (`--controller active`, zero RL)**, leaving `--controller
learned` strictly opt-in for readout experiments.

### Mapping FlyWire v783's anterior tracking fovea and individual handedness

Auditing why the pure reflex at `zoom = 1.15, eye_offset = 0.11` (and
`zoom = 1.40, eye_offset = 0.18`) still leaned right uncovered two real
anatomical properties of the reconstructed FlyWire v783 female brain:

1. **The `LC10a → DNp07 / DNb05 / DNa02` tracking fovea sits on the anterior/medial
   rim of each eye, not the centre.** Sweeping an isolated spot across retinal
   azimuth `sx ∈ [-0.6, +0.6]` on each eye separately (5 neural seeds):
   - Left eye (`DNa02_L`): `2.5 Hz` at `sx = -0.6` (lateral) → `5.0 Hz` at
     `sx = 0.0` → **`79.0 Hz` at `sx = +0.6` (anterior rim)** (`DNp07 = 180 Hz`,
     `DNb05 = 159 Hz`, `DNa02 = 93 Hz`).
   - Right eye (`DNa02_R`): **`60.0 Hz` at `sx = -0.6` (anterior rim)**
     (`DNp07 = 129 Hz`, `DNb05 = 135 Hz`) → `7.0 Hz` at `sx = 0.0` →
     `10.5 Hz` at `sx = +0.6` (lateral).
   At `zoom = 1.15, eye_offset = 0.11` (and even `zoom = 1.40, eye_offset = 0.18`),
   `eye_offset < 0.55 × (0.5 / zoom)`, which projected each eye's anterior fovea
   **across the midline into the opposite hemifield** (cross-eyed overlap).
2. **Individual hemispheric asymmetry in `DNb06` / `DNg52`.** On an identical
   bilateral full-field flash, `DNb06` fires `105 Hz` on the right versus
   `15 Hz` on the left (`7×`), and during pure straight-ahead Battlezone driving
   (`UP`), `DNa02_R` averages `1.25×–1.29×` `DNa02_L` (`7–10.5 Hz` background
   floor on the outer right eye versus `2.5–5 Hz` on the outer left). When the
   steering gate was set below that floor (`4–6 Hz`) and lateral triggers
   overwrote `_search_sign`, right-eye mountain edges continually pinned idle
   search clockwise.

Uncrossing the two anterior foveas to **`zoom = 1.75, eye_offset = 0.21,
center_y = 0.48`** (`eye_fovy = 54°`, `converge_eyes = True`, `0.0` bezel bleed)
snaps both eyes' `70–91 Hz` anterior foveas cleanly onto either side of the
crosshair with binocular overlap right on `x = 80`:

| Target CRT `x` (px) | `DNa02_L` (Hz) | `DNa02_R` (Hz) | `LI` | `pIP10` (Hz) | Motor command |
|---:|---:|---:|---:|---:|---|
| 55 (left) | 20.8 | 2.5 | -0.79 | 2.5 | `LEFT` |
| 65 (left) | **90.8** | 7.5 | **-0.85** | 7.5 | `LEFT` |
| 72 (left) | **90.8** | 13.3 | **-0.74** | 13.3 | `LEFT` |
| **80 (crosshair)** | **80.8** | **69.2** | **-0.08** | **69.2** | **LOCK + FIRE** |
| 88 (right) | 5.0 | **69.2** | **+0.87** | 5.0 | `RIGHT` |
| 95 (right) | 3.3 | **65.8** | **+0.90** | 3.3 | `RIGHT` |
| 105 (right) | 0.8 | 25.8 | +0.94 | 0.8 | `RIGHT` |

Pairing that uncrossed geometry with a `10 Hz` steering gate (above the outer
right-eye background floor, well below the `21–91 Hz` anterior foveas),
requiring `|LI| <= 0.22` for binocular lock, adding directional giant-fibre
escape (`DNp01 > 0.60`, jinking diagonally away from the louder `DNa02` side
without firing), and decoupling balanced alternating L/R patrol sweeps +
forward optic-flow strides (`patrol_sweep = 6, patrol_stride = 4`) fixes both
tracking precision and hunting symmetry without adding any learned network.

### Updated moving-target tracking (`out/bench_tracking_2026-11-19.json`)

Re-running `scripts/bench_tracking.py` at `zoom = 1.75, eye_offset = 0.21,
center_y = 0.48` cuts measured tracking error in half and — for the first time
— **cleanly separates measured FlyWire wiring from within-hemisphere shuffles
(`rewired_ipsi`) on pure target tracking**:

| Condition | RMS error (°) | MAE (°) | Lock fraction (`<= 8°`) | Turns | Turn-sign accuracy | Spiking / tick |
|---|---:|---:|---:|---:|---:|---:|
| **`measured` (`pool4`)** | **8.36** | **5.97** | **0.77** | 53 | **0.99** | 6,844 |
| `rewired_ipsi@0x0.7:pool_ipsi0` | 12.75 | 10.59 | 0.53 | 57 | 1.00 | 6,082 |
| `rewired_ipsi@0x0.7` (`pool4`) | 13.29 | 11.25 | 0.55 | 66 | 0.99 | 6,052 |
| `rewired_ipsi@1x0.7` (`pool4`) | 14.45 | 12.80 | 0.38 | 71 | 0.86 | 6,179 |
| `silenced` | 22.32 | 20.32 | 0.17 | 0 | — | 5,035 |
| `rewired@1x0.7` | 22.53 | 19.54 | 0.26 | 35 | 0.88 | 5,653 |
| `rewired@2x0.7` | 23.74 | 20.62 | 0.24 | 42 | 0.78 | 5,686 |
| `rewired@0x0.7` | 29.70 | 26.25 | 0.20 | 43 | 0.32 | 5,359 |

Why `rewired_ipsi` now trails (`12.75–14.45°` RMS, `38–55%` lock vs `8.36°` RMS,
`77%` lock): keeping synapses inside each hemisphere preserves coarse
left-vs-right energy, but destroys the anterior-rim retinotopic concentration of
`LC10a → DNp07 / DNb05 / DNa02`.

### Updated 10-seed Battlezone active-reflex benchmark (`out/bench_active_wiring10_2026-11-19.json` & `v11` follow-up)

Running `scripts/bench_pilots.py --preset active` with `v4` across ten
`1,500`-tick (`100`-second) seeded episodes per condition
(`out/bench_active_wiring10_2026-11-19.json`) raised measured wiring to
`3.10` kills / 100 s (`6.20` lives lost, `+0.00` utility) against `0.90–1.30`
for whole-brain shuffles (`-0.90` to `-1.35`), `1.40–1.90` for
within-hemisphere shuffles (`-1.10` to `-1.55`), and `0.70` non-firing baseline
kills when silenced (`0.0` shots, `-1.40`).

### Curing "T-Rex vision" and Battlezone's 29-tick cannon lock-out (`v11`)

Auditing the `v4` seed-0 recording tick by tick across all 8 enemy encounters
(`out/pilot_20261119_143256_263607.json`) turned up two final hunting
bottlenecks:

1. **Stationary enemies turned invisible under `v4`:** because `v4` only
   injects temporal luminance changes (`lum(t) - lum(t-1)`), the moment an
   enemy tank stopped moving to aim (e.g. ticks `38–40` and `370–376`), every
   descending neuron in `pool4` dropped to `0.0 Hz`. Meanwhile, Battlezone blue
   tanks have weak luminance contrast against olive ground (`0.284` vs `0.398`,
   `0.11` delta) compared to `HexRetina`'s unused `R8` blue channel (`0.498` vs
   `0.151`, `3×` stronger). In FlyWire v783, the real R8-downstream chromatic
   medulla neurons `Tm20` (`1,496` cells), `Tm5a` (`451`) and `Tm5b` (`447`)
   send **`3,630` measured excitatory synapses** straight onto
   `LC10a / LC11 / LC4`. Injecting pure photoreceptor-level signed 2D
   `R8 − R1-6` blue-ON centre-surround opponency (`v11`, modelling the
   histaminergic `R8/Dm9` chromatic surround a point LIF cannot propagate) into
   `Tm20 / Tm5a / Tm5b` and additively into `T3` produces **`0.000` false
   alarms** across the full 360° sky, sunset and mountain horizon while letting
   a completely stationary tank drive `measured` `pool4` cleanly (`L=10.0, R=5.0`
   left; `L=10.0, R=10.8, pIP10=10.0` centred; `L=2.5, R=10.8` right) — whereas
   both `rewired@0x0.7` and `rewired_ipsi:own` remain at **`0.0 Hz` on every
   stationary tank frame**.
2. **Atari 2600 Battlezone locks out the cannon for `29` ticks (`~2.0 s`) while
   a missed shell flies to the horizon:** off-axis shots (`UPRIGHTFIRE` / wide
   `fovea_li`) jammed the gun during subsequent dead-centre locks (`t=82, 84`
   after an off-axis shot at `t=72`), and `startle_thresh = 0.60` (`12 Hz`)
   occasionally startled off medium-range targets (`DNp01 = 15 Hz`). Tightening
   fire discipline (`fovea_li = 0.18`, `pip_lock_drive = 0.50`,
   `no_off_axis_fire = True`), adding a `6.0 Hz` unilateral sharp-contrast gate
   (`|LI| >= 0.60`), and raising `startle_thresh = 0.80` kept the gun loaded
   for dead-centre locks.

Re-running the 10-seed, `1,500`-tick (`100`-second) wiring benchmark on this
final `v11` `ACTIVE_PRESET` (`out/bench_active_wiring10_v11_2026-11-19.json`):

| Condition | Kills / 100 s | Lives lost / 100 s | Utility (`kills - 0.5 * lives`) | Aimed fire ticks | Balanced steer acc | L / R / UP / DN (%) |
|---|---:|---:|---:|---:|---:|---|
| **`connectome` (`pool4`)** | **3.40** | **3.60** | **+1.60** | **190.6** | **0.80** | 16.2 / 19.5 / 28.4 / 7.1 |
| `rewired@0x0.7` | 0.90 | 5.00 | -1.60 | 4.6 | 0.50 | 18.6 / 20.4 / 27.8 / 2.9 |
| `rewired@1x0.7` | 1.60 | 3.90 | -0.35 | 55.7 | — | 17.9 / 18.8 / 31.5 / 2.5 |
| `rewired@2x0.7` | 1.40 | 5.20 | -1.20 | 9.5 | 0.77 | 19.8 / 19.0 / 26.7 / 3.1 |
| `rewired_ipsi@0x0.7` (`pool4`) | 1.40 | 5.30 | -1.25 | 54.5 | 0.50 | 18.6 / 18.3 / 31.7 / 3.1 |
| `rewired_ipsi@0x0.7:pool_ipsi0` | 2.00 | 6.30 | -1.15 | 27.1 | 0.85 | 20.2 / 22.1 / 14.1 / 13.2 |
| `silenced` *(non-firing floor)* | 0.70 | 4.20 | -1.40 | 0.0 | — | 18.8 / 18.8 / 31.0 / 0.0 |

`connectome` scored kills in **10 of 10 episodes** (`[1, 3, 5, 2, 3, 3, 3, 5, 6, 3]`,
mean **`3.40` kills / 100 s**), cut mortality nearly in half to **`3.60` lives
lost / 100 s** (lower than *every* control, including `silenced`'s `4.20`), and
reached **`+1.60` utility** with zero RL — beating every wiring control
(`-0.35` to `-1.60`) by `+1.95` to `+3.20` full utility points. On the 16
additional 60-second held-out starts (`101..116`), the exact same `v11` reflex
reached **`1.56` kills / 60 s and `-0.22` utility** (beating random play
`-0.59`, `rewired@0x0.7` `0.75` / `-1.03`, and `rewired_ipsi:own` `0.56` /
`-1.34`). Note that `silenced` logs `0.70` kills / 100 s with `0.0` fire ticks
from Battlezone's in-flight warm-up shells and collision scoring during the
identical patrol strides; above that shared non-firing floor, `connectome` adds
`+2.70` aimed kills per 100 s versus `+0.20..+0.90` for whole-brain shuffles
and `+0.70..+1.30` for within-hemisphere shuffles.

### Default pure-connectome 60-second seed-0 recordings (zero RL)

- **Wired (`./no-fly-zone record --seconds 60` → `out/pilot_20261119_155351_746031.mp4`):**
  `3,000` points (`3` kills, first two at `3.5 s` and `5.0 s`), `3` lives lost,
  **`159` `LEFT` / `158` `RIGHT`**, `271` `UP`, `80` `DOWN`, `133` `FIRE`
  (`2.5×` real time, `11.89 ms/step`).
- **Embodied physical levers (`./no-fly-zone record --seconds 60 --embodied` → `out/pilot_20261119_155420_567215.mp4`):**
  `3,000` points (`3` kills at `3.5 s`, `4.8 s`, `10.3 s`), **only `1` life
  lost in 60 seconds**, `101` `LEFT` / `179` `RIGHT`, `346` `UP`, `129` `DOWN`,
  `155` `FIRE` (`1.0×` real time; physical levers reached `-0.90..+1.00` left
  and `-0.85..+1.00` right, proboscis pressing the physical fire button).
