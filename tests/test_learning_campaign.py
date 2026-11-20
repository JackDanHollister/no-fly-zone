"""The practical learning decision must not select lucky seeds or controls."""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("campaign", SCRIPTS / "run_learning_campaign.py")
campaign = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPTS))
try:
    spec.loader.exec_module(campaign)
finally:
    sys.path.pop(0)


@pytest.fixture
def plan():
    return json.loads(campaign.DEFAULT_PLAN.read_text())


def panel(plan, on=(3, 1), shuffled=(2, 2), measured=(2, 3)):
    def condition(values):
        return {"episodes": [{"kills": values[0], "lives_lost": values[1], "ticks": 900}
                             for _ in range(16)]}
    conditions = {"measured": condition(measured)}
    for h in range(4):
        conditions[campaign.checkpoint_name(plan, h, "on")] = condition(on)
        conditions[campaign.checkpoint_name(plan, h, "shuffled")] = condition(shuffled)
    return {"status": "complete", "conditions": conditions}


def test_all_histories_and_both_controls_count(plan):
    assert campaign.analyse_panel(plan, panel(plan))["pass"]
    better_shuffled = campaign.analyse_panel(plan, panel(plan, shuffled=(4, 1)))
    assert better_shuffled["gates"]["utility_vs_measured"]
    assert not better_shuffled["gates"]["utility_vs_shuffled"]
    assert not better_shuffled["pass"]
    lucky = panel(plan, on=(2, 3))
    lucky["conditions"][campaign.checkpoint_name(plan, 0, "on")]["episodes"] = [
        {"kills": 20, "lives_lost": 0, "ticks": 900} for _ in range(16)]
    result = campaign.analyse_panel(plan, lucky)
    assert result["gates"]["utility_vs_measured"]
    assert result["histories_beating_both"] == [0]
    assert not result["pass"]


def test_survival_guard_and_complete_quotas(plan):
    result = campaign.analyse_panel(plan, panel(plan, on=(6, 4)))
    assert result["gates"]["utility_vs_shuffled"]
    assert not result["gates"]["lives_vs_measured"]
    assert not result["pass"]
    partial = panel(plan)
    partial["conditions"]["measured"]["episodes"].pop()
    with pytest.raises(ValueError, match="quota"):
        campaign.analyse_panel(plan, partial)
    partial["status"] = "running"
    with pytest.raises(ValueError, match="incomplete"):
        campaign.analyse_panel(plan, partial)


def test_rates_use_scored_exposure():
    result = campaign.rates([{"kills": 1, "lives_lost": 2, "ticks": 300},
                             {"kills": 3, "lives_lost": 0, "ticks": 600}])
    assert result["kills_per_900_ticks"] == 4
    assert result["lives_lost_per_900_ticks"] == 2
    assert result["utility_per_900_ticks"] == 3


def test_training_rejects_changed_rule_off_weights_and_bad_seeds(plan, monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    (tmp_path / "out").mkdir()
    plan["training"]["batch"] = 1
    plan["training"]["episodes_per_fly"] = 1
    root = plan["training"]["seeds"][0] * 100
    neural = plan["training"]["neural_seeds"][0]
    audit = [{"game_seed_root": root, "episode_index": 0, "game_seed": 1,
              "warmup_seed": 2, "warmup_actions": [0, 1], "start_lives": 5,
              "start_frame_sha256": "frame", "start_ram_sha256": "ram"}]
    start = {**audit[0], "protocol": campaign.EPISODE_PROTOCOL,
             "neural_seed_root": neural, "neural_seed": campaign.episode_seed(neural, root, 0, 2),
             "neural_row": 0}
    row = {"start": start, "fly": 0, "kills": 1, "lives_lost": 2, "ticks": 900}
    name = campaign.tag(plan, 0, "lr0")
    log = tmp_path / f"out/plasticity_{name}.jsonl"
    log.write_text(json.dumps(row) + "\n")
    weights = tmp_path / f"out/plasticity_{name}_weights.npz"
    np.savez(weights, w=np.array([1.0]), w0=np.array([1.0]))
    assert campaign.validate_training(plan, 0, "lr0", audit)["unchanged"]
    np.savez(weights, w=np.array([1.1]), w0=np.array([1.0]))
    with pytest.raises(ValueError, match="rule-off weights changed"):
        campaign.validate_training(plan, 0, "lr0", audit)
    row = deepcopy(row)
    row["start"]["neural_seed"] += 1
    log.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="neural seed mismatch"):
        campaign.validate_training(plan, 0, "lr0", audit)
