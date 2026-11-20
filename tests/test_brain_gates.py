"""The published validation gates and the CPU/CUDA equivalence, on real data."""

import numpy as np
import pytest

from noflyzone.brain_backend import make_brain
from noflyzone.brain_lif import DT

SUGAR = [720575940611875570, 720575940612670570, 720575940616885538, 720575940617000768,
         720575940617937543, 720575940620900446, 720575940621502051, 720575940621754367,
         720575940624963786, 720575940628853239, 720575940629176663, 720575940630233916,
         720575940630797113, 720575940632425919, 720575940632889389, 720575940633143833,
         720575940637568838, 720575940638202345, 720575940639198653, 720575940639332736,
         720575940640649691]
MN9 = 720575940660219265


def gates(brain):
    brain.reset()
    assert brain.run(0.2).sum() == 0, "spontaneously active"
    sugar, _ = brain.indices_of(SUGAR)
    mn9 = brain.indices_of([MN9])[0][0]
    brain.reset(); brain.set_poisson({int(i): 150.0 for i in sugar})
    assert brain.run(1.0)[mn9] > 5
    lc4 = brain.indices_for_type("LC4")
    gf = np.concatenate([brain.indices_for_type("DNp01", side=s) for s in ("left", "right")])
    brain.reset(); brain.set_poisson({int(i): 100.0 for i in lc4})
    assert brain.run(0.5)[gf].max() > 0


@pytest.mark.data
def test_cpu_gates():
    gates(make_brain("cpu"))


@pytest.mark.data
@pytest.mark.cuda
def test_cuda_gates_and_corrections():
    gates(make_brain("cuda"))
    gates(make_brain("cuda", aminergic="zero", sensory_input="external_only"))


@pytest.mark.data
@pytest.mark.cuda
def test_cuda_matches_cpu_step_for_step():
    cpu, gpu = make_brain("cpu"), make_brain("cuda")
    sugar, _ = cpu.indices_of(SUGAR)
    for b in (cpu, gpu):
        b.reset(); b.set_poisson({int(i): 1.0 / DT for i in sugar[:3]})
    for _ in range(300):
        assert np.array_equal(cpu.step(), gpu.step())


@pytest.mark.data
@pytest.mark.cuda
def test_batched_fly_zero_matches_single():
    single, trio = make_brain("cuda"), make_brain("cuda", batch=3)
    sugar, _ = single.indices_of(SUGAR)
    for b in (single, trio):
        b.reset(); b.set_poisson({int(i): 150.0 for i in sugar})
    assert np.array_equal(single.run_counts(300), trio.run_counts(300)[0])


@pytest.mark.data
@pytest.mark.cuda
def test_seeded_row_reset_restarts_rng_without_touching_other_flies():
    brain = make_brain("cuda", batch=2, ablate="silenced")
    sugar, _ = brain.indices_of(SUGAR)
    rates = np.full((2, len(sugar)), 150.0, np.float32)
    brain.set_poisson({int(i): 150.0 for i in sugar})
    brain.reset_rows([0, 1], seeds=[123, 456])
    brain.update_rates(rates)
    initial = brain.run_counts(600).copy()
    peer = brain._v[1].clone()
    brain.reset_rows([0], seeds=[123])
    assert (brain._v[1] == peer).all()
    brain.update_rates(rates)
    replay = brain.run_counts(600)
    np.testing.assert_array_equal(replay[0], initial[0])
    brain.reset_rows([0], seeds=[789])
    brain.update_rates(rates)
    assert not np.array_equal(brain.run_counts(600)[0], initial[0])
