import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.eval_simpler_manipulation import TASKS, run_simpler_episode


def test_simpler_eval_widowx_smolvla():
    task = TASKS["widowx_pick_cube"]
    result = run_simpler_episode(task, policy_name="lerobot/smolvla-450m")

    assert result.task_name == "WidowX-PickCube"
    assert result.control_rate_hz > 50.0  # Must exceed 50 Hz on laptop
    assert result.patch_reuse_ratio >= 0.0
    assert result.spec_acceptance_rate >= 50.0
    assert result.motor_jitter_ms < 5.0  # Sub-5ms motor jitter


def test_simpler_eval_google_robot_cosmos():
    task = TASKS["google_robot_move_near"]
    result = run_simpler_episode(task, policy_name="nvidia/cosmos-predict-mini")

    assert result.task_name == "GoogleRobot-MoveNear"
    assert result.control_rate_hz > 40.0
