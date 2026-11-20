# Bounded learning campaign, 25 September 2026

The September 25 review agreed two next steps: package the working pilot
as an easy-to-use first version, and give Battlezone plasticity one fresh,
controlled screen with a stopping rule fixed before training. This is a
local predeclared plan, not an externally registered study.

The [machine-readable plan](evidence/learning-campaign-2026-09-25-plan.json)
is the specification. The runner copies it and records its SHA-256 before
training, along with source hashes, package versions, commands and all
episode records. It refuses to overwrite an existing campaign.

## Fixed protocol

Four independent training histories each run three conditions: latched
event plasticity on the 24,110 visual-to-descending connections; the same
calculation with the learning rate effectively zero; and the same learning
rule on an equal-sized shuffled edge set. Game and neural starts match
within a history. Each condition starts from measured weights, with a
four-fly batch sharing one weight matrix. That batch is one training
history, not four independently learned brains.

Each run gets six episodes per fly, at most 900 ticks each. Across 12 runs
that is 288 training episodes. The rule is fixed at eta 0.00002, Hebbian
eligibility capped at 50, relative weight changes, latched pre-event
eligibility, and both kill and hit teaching pulses. The readout and
calibration stay at `pool4`; no critic or controller adaptation is trained.

At every episode boundary the game, neural row, controller and sensory
history reset. The existing pooled learner clears all eligibility,
dopamine traces and pending teaching pulses whenever any fly ends. This
also discards terminal-event pulses; they do not teach the next episode.
This conservative boundary policy is part of the screened rule.

Before training, every planned training and evaluation reset is repeated
and its 450-action trace checked for uniqueness across the whole schedule.
Training logs must match that audit and their paired conditions' neural
starts. Final checkpoints alone are evaluated, with frozen weights and
external teaching off, on two reserved 16-start panels. All four rule-off
checkpoints must be bit-identical to measured weights before the measured
evaluation can stand in for them. Thus nine distinct conditions per panel
give another 288 episodes. Both panels run regardless of the first result.

## Decision fixed before training

The primary score is `(kills - 0.5 × life losses) × 900 / scored ticks`,
calculated for each checkpoint and then averaged equally over the four
training histories. Each reserved panel must independently pass every gate:

- Mean learned utility exceeds both measured and shuffled controls by at
  least 0.25 per 900 scored ticks.
- Mean kill rate is at least 95% of each control's rate, and mean life-loss
  rate is no higher than either control's rate.
- At least three of the four histories beat both measured weights and
  their own matched shuffled-edge control in utility.

This is a practical screen with four training histories, not a powered
statistical demonstration. A pass would justify a larger independent
confirmation. A failure closes this Battlezone plasticity campaign without
tuning or selected extra tests; the next research milestone becomes a
focused visual task, while the working pilot remains a useful artifact.

The ceiling is 518,400 scored ticks and two hours of campaign wall time.
An error, changed source or plan, failed audit, or timeout means incomplete,
not a positive or negative learning result. There are no automatic retries.

## Execution

The campaign started at 20:55:32 BST on September 25 with plan SHA-256
`44edebc7b44be01802f4ac0290671cd2cf00e577866badbc1fcd37217987bbe8`.
All 128 scheduled starts passed the repeatability and distinct-trace audit.
The 32 reserved traces also differ from all 32 used in the earlier corrected
evaluations.

Training completed at 21:37:13 BST: 288 episodes, 250,031 scored ticks,
12 final checkpoints, and 96 distinct episode neural seeds across four
histories. The phase took 41.69 minutes including its initial audit. Every
checkpoint has 24,110 selected edges; all final weights preserve their signs
and bounds. All four rule-off arrays are byte-identical to their measured
weights. Both reserved panels then completed all 144 episodes each.

The full campaign finished in **80.25 minutes**, below its two-hour ceiling:
288 training and 288 evaluation episodes, with 501,692 scored ticks against
the maximum 518,400. No retries, extra training, selected checkpoints or
changed parameters were used. The original source and checkpoint hashes
still match; all episode quotas, matched starts and reported rates were
independently checked after completion.

```bash
mkdir -p out
MUJOCO_GL=egl .venv/bin/python scripts/run_learning_campaign.py
```

The complete local record is
`out/campaign_bounded_20260925/manifest.json`; individual process logs,
the frozen plan, start audit and evaluation JSON files live beside it.

## Result: the campaign did not meet its success criterion

**Every gate failed in both panels.** The learned histories averaged fewer
kill events and more life losses than measured weights and shuffled-edge
learning. All four learned histories had lower utility than both measured
weights and their own matched shuffled-edge control, in each panel.

All rates below are per 900 scored ticks. Learned and shuffled rows are
equal-weight means of four checkpoint rates, as specified in the plan;
each checkpoint has 16 episodes per panel. The measured baseline has 16
episodes per panel and represents the four verified unchanged rule-off
checkpoints. Utility is kills minus half the life-loss rate.

| Panel | Brain group | Kill events | Life losses | Utility |
|---|---|---:|---:|---:|
| A | Measured / rule off | 1.039 | 2.859 | -0.390 |
| A | Learned visual-to-descending | 0.836 | 3.654 | -0.992 |
| A | Shuffled-edge learning | 1.249 | 2.799 | -0.151 |
| B | Measured / rule off | 2.094 | 3.075 | 0.556 |
| B | Learned visual-to-descending | 0.891 | 3.740 | -0.979 |
| B | Shuffled-edge learning | 1.719 | 2.913 | 0.262 |

The individual utility scores show that the negative group result does not
come from a single poor training history:

| History | A: learned | A: matched shuffled | B: learned | B: matched shuffled |
|---|---:|---:|---:|---:|
| 0 | -1.254 | -0.126 | -0.596 | 0.345 |
| 1 | -0.868 | -0.096 | -0.839 | 0.191 |
| 2 | -0.781 | -0.319 | -1.217 | 0.161 |
| 3 | -1.063 | -0.063 | -1.264 | 0.352 |

This is a negative result for the tested pooled, latched rule and its fixed
training budget. It does not establish that all connectome-based learning
is impossible, or identify a unique causal mechanism for the loss of
performance. Unlike the earlier saved checkpoints, these histories have
explicit independent neural seeds and fully reset episodes.

## Evidence and the next milestone

The repository preserves exact copies of the completed records:

- [Original plan](evidence/learning-campaign-2026-09-25-plan.json) and
  [128-start audit](evidence/learning-campaign-2026-09-25-starts.json).
- [Full manifest](evidence/learning-campaign-2026-09-25.json), including all
  288 training episode rows, checkpoint hashes, source hashes and decisions.
- [Panel A](evidence/learning-campaign-2026-09-25-a.json) and
  [panel B](evidence/learning-campaign-2026-09-25-b.json), including all
  288 evaluation episode rows, start records and frozen-weight checks.

The stopping rule is now applied: **this Battlezone plasticity campaign is
closed**, with no further parameter tuning or selected follow-ups. The
working pilot remains available through `./no-fly-zone`.

The next research milestone is a focused moving-target tracking benchmark.
It should measure tracking error in controlled scenes, compare measured,
whole-brain shuffled and within-hemisphere shuffled wiring with identical
visual inputs and motor rules, and reserve separate test trajectories before
calibration. First establish whether the frozen circuit gives a useful
behavioural advantage; learning comes after that. This benchmark is the next
planned implementation, not a result claimed by this campaign.
