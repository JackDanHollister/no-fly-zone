# Review draft — updated 20 November 2026

This is the author's reading copy before a public release. The GitHub
repository remains **PRIVATE** for review before making it public.

The September 25 continuation added a one-command local pilot launcher,
repaired the learning protocol, and completed the 576-episode
[bounded learning campaign](research/learning-campaign-2026-09-25.md), closing
that Battlezone synaptic-plasticity attempt under its stopping rule.

The November 17–20 continuation ([research note](research/active-vision-and-tracking-2026-11-17.md))
fixed the parallel-camera crosshair blind spot, audited the PPO GRU readout
(catching its open-loop clockwise circle-strafe shortcut — `445` `RIGHT` vs `3`
`LEFT` turns, where a scrambled brain beat the real connectome — and returning
`./no-fly-zone` to the pure frozen-connectome reflex by default with zero RL),
aligned the converged eye cameras onto FlyWire v783's real anterior `LC10a`
tracking foveas (`zoom = 1.75, eye_offset = 0.21, center_y = 0.48`), wired
signed `R8` chromatic opponency (`v11` → `Tm20 / Tm5a / Tm5b` + `T3`) with
tight crosshair fire discipline (`|LI| <= 0.18`, avoiding Battlezone's 29-tick
missed-shell lock-out), refreshed the README hero GIF and kill still from the
new physical-lever `v11` run, and simplified `README.md` for GitHub readers.
Under that preset, measured wiring drops moving-target tracking error to
**`8.36°` RMS (`77%` lock, `0.99` turn-sign accuracy)** — separating measured
microcircuitry even from within-hemisphere shuffles (`12.75–14.45°` RMS) — and
reaches **`3.40` kills and `3.60` lives lost / 100 s (`+1.60` utility, kills in
10/10 episodes)** in Battlezone against `0.90–1.60` kills (`-0.35` to `-1.60`
utility) for whole-brain shuffles and `1.40–2.00` kills (`-1.15` to `-1.25`)
for within-hemisphere shuffles. Default 60-second seed-0 recordings score
`3,000` points with `3` lives lost wired (`159` L / `158` R) and `3,000` points
with **`1` life lost** using physical levers + proboscis. All 56 tests pass.

## Read in this order

1. [README](../README.md): the front page, demo, achievements, main findings,
   setup and credits. Check whether the opening sounds like you.
2. [JOURNEY](../JOURNEY.md): the complete build story, written in your voice
   for review. It covers the cockpit, CPU/GPU work, first kills, stronger
   controls, male brain, physical controls and successive learning attempts.
3. [Learning audit](research/learning-2026-09-19.md#evaluation-protocol-audit-2026-09-22):
   the material qualification discovered while preparing this draft.
4. [Architecture](architecture.md) and [research notes](research/): the
   technical backing for the story. Historical interpretations remain in the
   lab notes, with corrections signposted at the top of the learning notes.

## What changed for this draft

The front page now gives the working system room to speak for itself, with
images and concrete measurements. The full chronology has its own document,
and the architecture reflects the current implementation. Unsupported
priority/speed rankings against other projects have been removed from the
architecture; the workstation's own measured performance remains.

The 15.1 million figure is now described as directed connection records,
weighted by anatomical synapse counts. The physical controller is correctly
described as inverse kinematics and servos driven by neural intent.

The last pathway-swap results were recovered from completed local logs.
Fourteen final-learning evaluations, 672 episode records in total, are
included in [an evidence bundle](research/evidence/learning-final-2026-09-22.json)
with source hashes and recomputed means. The full videos, trained weights,
connectome downloads and other raw outputs remain local.

The September 22 code review found that learning evaluations did not implement the
pilot benchmark's random-start fix, do not forward the training label to
the neural RNG, and can log adjacent windows from a continuing game as
separate episodes. A direct emulator check confirmed identical frames for
eight requested seeds under the same actions. The draft therefore avoids
claiming generalisation to unseen games from those historical results.
The [September 25 correction and completed reruns](research/learning-2026-09-19.md#corrected-evaluation-protocol-2026-09-25)
repair the evaluator; they cannot retroactively repair old training histories.

## Media to inspect locally

The repository includes the small GIF and stills used in the README and
journey. The full recordings available on the workstation are:

| Local file under `out/` | Content |
|---|---|
| `no-fly-zone-showcase.mp4` | 66-second edited overview: wired pilot, physical controls, two flies, findings and caveats |
| `no-fly-zone-highlights.mp4` | Shorter gameplay highlight cut |
| `no-fly-zone-female-vs-male.mp4` | Side-by-side recorded female and corrected male pilots |
| `no-fly-zone-vs-rewired.mp4` | Earlier measured/shuffled comparison |
| `learned_latch_seed0.mp4` | A learned brain playing, labelled LEARNED WEIGHTS |
| `pilot_20260925_200200_883034.mp4` | Historical September 25 60-second reflex pilot recording (`1,000` pts, `5` lives lost) |
| `pilot_20260925_200407_665054.mp4` | Five-second live check using the physical levers |
| `pilot_20261117_174132_735739.mp4` | Historical Nov 17 wired PPO GRU recording (`6,000` pts, `1` life lost; exposed clockwise circle-strafe shortcut: `445` R / `3` L) |
| `pilot_20261117_174139_553233.mp4` | Historical Nov 17 `--embodied` PPO GRU recording (`2,000` pts, `2` lives lost) |
| `pilot_20261117_174349_035394.mp4` | Historical Nov 17 `--controller active` recording at `zoom = 1.15, eye_offset = 0.11` (`2,000` pts, `4` lives lost) |
| `pilot_20261119_143256_263607.mp4` | Nov 19 pure-connectome `v4` wired recording (`zoom = 1.75, eye_offset = 0.21`: `3,000` pts, `162` L / `145` R) |
| `pilot_20261119_143325_186962.mp4` | Nov 19 pure-connectome `v4` `--embodied` recording (`3,000` pts, `2` lives lost, `118` L / `121` R) |
| `pilot_20261119_155351_746031.mp4` | Default 60-second seed-0 pure-connectome `v11` wired recording (zero RL: `3,000` pts, `3` lives lost, `159` L / `158` R) |
| `pilot_20261119_155420_567215.mp4` | Default 60-second seed-0 pure-connectome `v11` `--embodied` physical-lever + proboscis recording (zero RL: `3,000` pts, `1` life lost) |

These videos have not been added to Git or uploaded as releases. Review their
existing cards alongside the corrected learning wording before selecting a
public showcase. The `seed0` in the learned video's filename is the game
recording label; the installed learned weights are from training label 1.

## September 22 validation of the original draft

- Existing selected tests: 21 passed, 11 data/CUDA tests deselected.
  Includes the physical lever and fire-button test; one known MuJoCo
  attachment warning.
- Seed check: eight labels, 450 identical actions each; one unique frame
  sequence, with hashes and package versions saved in the repo.
- Final-learning means: recomputed from the existing episode rows. The
  copied rows and original file hashes can be checked independently.
- Documentation links, command-line options and Git whitespace checked
  during preparation. No application code or learned weights changed.

## September 25 launcher validation

All 47 tests pass, including the model's data/CUDA gates, physical controls,
campaign decision rules, preset overrides and cleanup after an early window
close. The known MuJoCo attachment warning remains. A real 60-second video
contains exactly 900 frames and 900 trace rows. Actual live windows completed
five seconds with physical controls and three seconds with direct controls;
both recordings have matching frame counts and traces. Live playback is
capped at game speed, while record-only mode can run faster.

The new campaign's full 576 episode quotas, 501,692 scored ticks, source and
checkpoint hashes, matched game/neural starts and group rates were checked
after completion. Its complete evidence is linked from the campaign note.

## Repository handoff

The repository is [JackDanHollister/no-fly-zone](https://github.com/JackDanHollister/no-fly-zone).
GitHub device login was restored on September 22, and the repository was
verified as **PRIVATE**. After fetching the remote and confirming that it
had no divergent commits, all 46 pending commits through review draft
`b9430f8` were pushed normally to `main`, preserving the development history.

The source, README, journey, lab notes, tests, evidence and small media are
available there for review. The full local video and data archive remains
outside Git. No repository visibility change, release or announcement was
made.

After your editorial review, public release remains a separate action.
The intended publication copy consists of the source, tests, story, research
notes, evidence and small media already in this repository. Private workspace
surveys about other projects have not been imported into it.
