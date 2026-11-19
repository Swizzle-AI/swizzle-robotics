#!/usr/bin/env python3
"""
Safety, RTC Continuity, and Batched World-Model Ablation Benchmark
==================================================================
Evaluates the concrete gains of the Phase 1-4 Swizzle Robotics stack:
1. Control Barrier Function (`Swizzle-CBF`) adversarial workspace/obstacle violations
2. Real-Time Chunking (`Swizzle-RTC`) peak chunk-boundary jerk reduction
3. Vectorized Batched `WorldModel.plan_cem_batched` vs sequential `plan_cem` speedup
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swizzle_robotics.core.hierarchical import HierarchicalController
from swizzle_robotics.core.safety import ControlBarrierShield, ObstacleSphere, SafetyConfig
from swizzle_robotics.core.teleop import TemporalActionEnsembler
from swizzle_robotics.core.camera import MockCamera
from swizzle_robotics.models.smol_vla import SmolVLAPolicy
from swizzle_robotics.models.cosmos_wm import CosmosWorldModel


def run_ablation():
    print("=" * 72)
    print("SWIZZLE ROBOTICS ABLATION: Unshielded Vanilla vs Swizzle (RTC + CBF + MPC)")
    print("=" * 72)

    # 1. Adversarial Safety Ablation (25 steps pushing toward table + fragile cup)
    obs = ObstacleSphere(center=(0.15, 0.0, 0.25), radius=0.07, name="fragile_beaker")
    cfg = SafetyConfig(
        z_bounds=(0.05, 0.70),
        obstacles=[obs],
        max_linear_step=0.20,
        max_accel_step=0.10,
        cbf_alpha=0.55,
    )
    shield = ControlBarrierShield(cfg)

    vanilla_pose = np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0])
    swizzle_pose = vanilla_pose.copy()
    vanilla_violations = 0
    swizzle_violations = 0
    cbf_times = []

    for step in range(25):
        adv_cmd = [0.12, 0.0, -0.06, 0.0, 0.0, 0.0, 1.0]
        vanilla_pose[:3] += adv_cmd[:3]
        if vanilla_pose[2] < cfg.z_bounds[0] or np.linalg.norm(vanilla_pose[:3] - np.array(obs.center)) < (obs.radius + cfg.eef_radius):
            vanilla_violations += 1

        sec = shield.project(adv_cmd, swizzle_pose)
        cbf_times.append(sec.solver_latency_ms)
        swizzle_pose[:3] += sec.safe_action[:3]
        if swizzle_pose[2] < cfg.z_bounds[0] - 1e-4 or np.linalg.norm(swizzle_pose[:3] - np.array(obs.center)) < (obs.radius + cfg.eef_radius - 1e-3):
            swizzle_violations += 1

    print(
        f"[1] Adversarial Safety Violations : Vanilla={vanilla_violations}/25 | "
        f"Swizzle-CBF={swizzle_violations}/25 (mean solve {np.mean(cbf_times):.3f} ms)"
    )

    # 2. Chunk-Boundary Jerk Ablation (Naive Chunk Switch vs Swizzle-RTC Inpainting)
    chunk_a = [[0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for _ in range(10)]
    chunk_b = [[-0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for _ in range(10)]
    naive_jump = abs(chunk_b[0][0] - chunk_a[2][0])

    ens = TemporalActionEnsembler()
    ens.push_chunk(0, chunk_a)
    blended, _ = ens.push_rtc_chunk(3, chunk_b, execution_delay=2, overlap_horizon=4)
    rtc_jump = max(abs(blended[i][0] - blended[i - 1][0]) for i in range(1, len(blended)))
    print(f"[2] Max Chunk-Boundary Step Jump  : Vanilla={naive_jump:.3f} | Swizzle-RTC={rtc_jump:.3f} ({naive_jump/max(rtc_jump,1e-6):.2f}x smoother)")

    # 3. Batched vs Serial World Model CEM (128 rollouts)
    cam = MockCamera()
    _, f0 = cam.read()
    cam.frame_idx = 10
    _, f1 = cam.read()
    wm = CosmosWorldModel()
    _, _, serial_ms = wm.plan_cem(f0, f1, horizon=5, num_samples=128, iterations=3)
    _, _, batched_ms = wm.plan_cem_batched(f0, f1, horizon=5, num_samples=128, iterations=3)
    print(f"[3] World-Model CEM (N=128, H=5)  : Serial={serial_ms:.2f} ms | Batched={batched_ms:.2f} ms ({serial_ms/max(batched_ms,0.01):.2f}x faster)")

    # 4. Full S2 -> S1 -> S0 Hierarchical Control + Flight Recorder Loop
    ctrl = HierarchicalController(policy=SmolVLAPolicy(), safety_config=cfg)
    plan = ctrl.plan_s2("pick up red cup carefully then avoid beaker", known_objects={"beaker": (0.20, 0.05, 0.25, 0.07)})
    t0 = time.perf_counter()
    for _ in range(20):
        tick = ctrl.step_s1_s0(f0)
    ms_per_tick = (time.perf_counter() - t0) * 1000.0 / 20.0
    assert tick.receipt.verify()
    print(f"[4] S2->S1->S0 Loop ({plan.source}): {ms_per_tick:.2f} ms/tick ({1000.0/ms_per_tick:.1f} Hz), receipts verified=True")


if __name__ == "__main__":
    run_ablation()
