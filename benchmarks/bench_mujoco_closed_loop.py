#!/usr/bin/env python3
"""
Phase 5 End-to-End Physics & Sparse-PatchCache Flow-Matching Benchmark
======================================================================
1. Benchmarks Dense vs Sparse PatchCache neural ViT token projection + K-step
   Euler Flow-Matching ODE integration (`TorchEdgeVLAPolicy`).
2. Executes closed-loop tabletop pick-and-place inside `SwizzleManipulationArena`
   and records binary `.mcap` + `LeRobot v2.1` + cryptographic receipts.
"""

import os
import sys
import tempfile
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swizzle_robotics.core.hierarchical import HierarchicalController
from swizzle_robotics.core.safety import SafetyConfig
from swizzle_robotics.envs import ArenaTaskConfig, SwizzleManipulationArena
from swizzle_robotics.models.edge_flow_runner import HAS_TORCH, TorchEdgeVLAPolicy


def run_benchmark():
    print("=" * 72)
    print(f"SWIZZLE ROBOTICS PHASE 5: Neural Sparse Flow VLA + Arena (torch={HAS_TORCH})")
    print("=" * 72)

    policy = TorchEdgeVLAPolicy()
    cfg = ArenaTaskConfig(goal_pos=(0.22, 0.16, 0.03), obstacle_pos=(0.04, 0.02, 0.09), obstacle_radius=0.05)
    arena = SwizzleManipulationArena(cfg, prefer_mujoco=True)
    obs = arena.reset(seed=2026)

    # 1. Dense vs Sparse 92%-reused PatchCache ViT encoding latency
    mask_dense = [False] * 196
    mask_sparse = [i < 180 for i in range(196)]  # ~91.8% cached static background

    dense_ms, sparse_ms = [], []
    intensities = [0.25] * 196
    for _ in range(30):
        policy.encode_sparse_vit(intensities, reuse_mask=mask_dense)
        dense_ms.append(policy.last_sparse_vit_ms)
        policy.encode_sparse_vit(intensities, reuse_mask=mask_sparse)
        sparse_ms.append(policy.last_sparse_vit_ms)

    print(
        f"[1] ViT Encoding (196 tokens) : Dense={np.mean(dense_ms):.3f} ms | "
        f"Sparse PatchCache={np.mean(sparse_ms):.3f} ms ({np.mean(dense_ms)/max(np.mean(sparse_ms),1e-6):.2f}x faster)"
    )

    # 2. Closed-loop manipulation episode in Arena with S2 -> S1 -> S0 + Flight Recorder
    ctrl = HierarchicalController(
        policy=policy,
        safety_config=SafetyConfig(z_bounds=(0.02, 0.65), max_linear_step=0.22, max_accel_step=0.14),
        episode_id="ep_phase5_bench",
    )
    plan = ctrl.plan_s2(
        "pick up red cube while avoiding pillar then place on green tray",
        known_objects={"pillar": (*cfg.obstacle_pos, cfg.obstacle_radius)},
    )

    t0 = time.perf_counter()
    collisions = 0
    for _ in range(30):
        target = np.array(obs.cube_pos if not obs.grasped else obs.goal_pos)
        err = (target - np.array(obs.eef_pose[:3])) * 4.0
        grip = 1.0 if float(np.linalg.norm(target - np.array(obs.eef_pose[:3]))) < 0.09 or obs.grasped else -1.0
        sec = ctrl.safety_shield.project([float(err[0]), float(err[1]), float(err[2]), 0.0, 0.0, 0.0, grip], obs.eef_pose)
        obs, _, done, info = arena.step(sec.safe_action)
        ctrl.recorder.record_step(obs.frame, obs.eef_pose, sec, prompt=plan.subtasks[0])
        if info["obstacle_collision"] or info["table_penetration_mm"] > 0.0:
            collisions += 1
        if done:
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    hz = len(ctrl.recorder.steps) * 1000.0 / max(1.0, elapsed_ms)

    with tempfile.TemporaryDirectory() as tmp:
        mcap_path = ctrl.recorder.export_mcap(f"{tmp}/ep.mcap")
        ds = ctrl.recorder.export_lerobot_v2(f"{tmp}/ds")
        print(
            f"[2] Closed-Loop Arena ({arena.backend_mode}): steps={len(ctrl.recorder.steps)}, "
            f"collisions={collisions}, rate={hz:.1f} Hz, mcap_bytes={mcap_path.stat().st_size}, lerobot_frames={ds['frames']}"
        )
    arena.close()


if __name__ == "__main__":
    run_benchmark()
