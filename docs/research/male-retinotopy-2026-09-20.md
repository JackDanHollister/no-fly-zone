# The male's own retinotopy, 2026-09-20

## The problem

MaleCNS v1.0 was loaded on 2026-09-18 through the same code as FlyWire,
and its cells were placed on the retina the same way: each injected cell
type's soma positions projected onto their principal plane and snapped to
the nearest ommatidium (`_pca2d`). The male responded asymmetrically to
that: in the descending-neuron screen a small object on the right drove
the right-side pools seven times harder than one on the left drove the
left-side pools (summed over responsive types), and the male pilot
steered toward the target on 10 of 12 gated looks with all but two of
those looks on one side.

The cause, measured today: the male's photoreceptors have no soma
positions. Of 1,112 left and 2,265 right R1-6 bodies, 4 and 9 have a
`somaLocation`, so the luminance channel of injection map v4 was being
injected into 13 cells, 4 on the left and 9 on the right, and everything
downstream of the lamina saw a right-heavy, nearly absent luminance
drive. (FlyWire's R1-6 have positions, 4,044 and 3,888 of them, which is
why the female never showed this.)

## The fix

MaleCNS annotates 23,720 optic-lobe intrinsic cells with a hex column
(`assignedOlHex1/2`), one column identity shared across the 12 or so
columnar types that live in it (879 columns on the left, 892 on the
right). `ConnectomeBrain` now places a cell by that column where it has
one: the axial coordinates go onto a hex lattice in the plane
(`hex_to_plane`; any axial basis gives the same lattice up to an
isometry), the lattice is oriented to the type's soma-PCA frame by an
orthogonal Procrustes fit so the axes keep their convention, and the
result is normalised and snapped to the nearest ommatidium as before.

A cell without a column, a photoreceptor above all, inherits the
synapse-weighted majority column of its same-side partners that have one
(`inherit_hex`): a column in the fly is defined by its connectivity, so a
photoreceptor is placed at the lamina column it drives, Tm3 and T3 at the
column of the Mi1 and L cells they connect to. With that, all 16,095
injected male cells are on columns, 8,232 of them by inheritance, and
none by PCA. FlyWire has no column annotation and takes the old path
unchanged; `retinotopy="pca"` restores the 2026-09-18 behaviour for the
male as a control.

## What changed

Descending-neuron screen (`screen_dns.py`, v4, weights x0.6, small
object on one side):

| placement | R-object/R-pool over L-object/L-pool, summed over responsive types | neurons spiking, small object L / R |
|---|---|---|
| soma PCA (2026-09-18) | 7.2 | 443 / 1,123 |
| hex where annotated, PCA otherwise | 5.5 | 618 / 1,282 |
| hex, columns inherited | 2.05 | 656 / 791 |

The residual factor of two matches the annotation: the right eye has
2,237 placed photoreceptors to the left's 1,107, so each right column
gets about twice the luminance drive. That is a property of the male
data (the left retina is less completely traced), and equalising it would
be an engineered per-side normalisation, not a placement; it is left as
it is and stated.

The male pilot (`bench_pilots.py`, ten 100 s episodes, `pool_male`,
weights x0.6):

| placement | kills / lives lost per 100 s | steer toward target, pooled gated looks | gated looks L / R |
|---|---|---|---|
| soma PCA (2026-09-18) | 1.2 / 2.7 | 0.70 (L 0.50, R 0.90) | 2 / 10 |
| hex, columns inherited | 2.2 / 2.3 | 0.90 (L 0.80, R 1.00) | 5 / 9 |

For comparison the female on the same settings kills 2.7 and dies 5.7.
The male now sees both sides, kills nearly twice as often as before, and
dies less; its steering is correct on both sides rather than one.

## The selectivity screen under the new placement

`screen_selectivity.py`, v4, weights x0.6, the measured male against two
within-hemisphere rewirings and one whole-brain rewiring, PCA placement
(2026-09-18) against column placement (today):

| brain | responsive DN types | lateralised (LI > 0.8) | expansion-selective | loom-selective vs flash | best small-object rate (Hz) |
|---|---|---|---|---|---|
| measured, PCA | 84 | 2 | 5 | 1 | 31 |
| measured, columns | 85 | 4 | 12 | 6 | 27 |
| rewired_ipsi seed 0, PCA / columns | 424 / 426 | 2 / 0 | 0 / 1 | 0 / 0 | 16 / 5 |
| rewired_ipsi seed 1, PCA / columns | 412 / 416 | 0 / 4 | 0 / 2 | 0 / 0 | 4 / 9 |
| rewired, PCA / columns | 338 / 343 | 0 / 0 | 0 / 4 | 0 / 0 | 3 / 3 |

The 2026-09-18 replication stands and sharpens: the measured male wiring
routes vision to 85 descending types where the shuffles light 343-426,
and under the column placement it has 12 expansion-selective types
against 1-4 in the shuffles (5 against 0 before), 6 loom-selective types
against none, and 4 lateralised types. The placement changed which
inputs each column receives, not the conclusion about the wiring.

## The pools re-screened

`pool_male` was screened under the PCA placement. Re-screened under the
column placement by the same recipe (`pool_male_hex`: steer DNp04, DNp02,
DNp15, DNp07, DNb05, the five lateralised types with the highest
small-object rate; loom DNg33, DNg36_a, DNp12, DNge141), it does not do
better: 1.6 kills and 2.4 lives lost per 100 s, steering 0.96 correct on
its gated looks but only one of those fourteen looks on the left, against
2.2 / 2.3 and 0.90 with five left looks for `pool_male`. Ten seeds are
too few to rank the two on kills; the old pools stay the male's default
and the new set is kept as a named alternative.

## Still open

The annotation's
2:1 photoreceptor count is left as it is; and FlyWire could be given the
same treatment from its own column annotation (Matsliah et al. 2024) if
that is ever added to the data this loads.
