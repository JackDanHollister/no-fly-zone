# Review draft — 22 September 2026

This is the author's reading copy before a public release. No visibility
change or public announcement is part of this preparation.

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

The new code review found that learning evaluations do not implement the
pilot benchmark's random-start fix, do not forward the training label to
the neural RNG, and can log adjacent windows from a continuing game as
separate episodes. A direct emulator check confirmed identical frames for
eight requested seeds under the same actions. The draft therefore avoids
claiming generalisation to unseen games. Fixing and rerunning that protocol
is the next research task, not a result already obtained.

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

These videos have not been added to Git or uploaded as releases. Review their
existing cards alongside the corrected learning wording before selecting a
public showcase. The `seed0` in the learned video's filename is the game
recording label; the installed learned weights are from training label 1.

## Validation of this draft

- Existing selected tests: 21 passed, 11 data/CUDA tests deselected.
  Includes the physical lever and fire-button test; one known MuJoCo
  attachment warning.
- Seed check: eight labels, 450 identical actions each; one unique frame
  sequence, with hashes and package versions saved in the repo.
- Final-learning means: recomputed from the existing episode rows. The
  copied rows and original file hashes can be checked independently.
- Documentation links, command-line options and Git whitespace checked
  during preparation. No application code or learned weights changed.

## Repository handoff

The existing repository is `JackDanHollister/no-fly-zone`, with the source
and development history already in local Git. At the start of this session
`main` was 45 commits beyond the last fetched `origin/main` reference.
The saved GitHub login returned HTTP 401, so current remote visibility and
branch state could not yet be rechecked. The draft can be committed locally;
remote synchronisation requires refreshed authentication and confirmation
that the destination is private.

After your editorial review, public release remains a separate action.
The intended publication copy consists of the source, tests, story, research
notes, evidence and small media already in this repository. Private workspace
surveys about other projects have not been imported into it.
