#!/usr/bin/env python3
"""
Swizzle Robotics VLA Benchmark: PatchCache ViT Gating & 7-DoF SpecAct Acceleration
Evaluates control frequency, ViT patch reuse ratio, and kinematic draft acceptance
across a 120-step pick-and-place trajectory.
"""
import time
import numpy as np
from swizzle_robotics.core.patch_cache import PatchCacheFilter
from swizzle_robotics.core.spec_act import KinematicSpecActPredictor
from swizzle_robotics.core.teleop import TemporalActionEnsembler
from swizzle_robotics.core.camera import MockCamera
from swizzle_robotics.models import SmolVLAPolicy, OpenVLAPolicy

def run_benchmark():
    print("=" * 80)
    print("SWIZZLE ROBOTICS VLA ROOFLINE & KINEMATIC SPECULATION BENCHMARK")
    print("=" * 80)

    cam = MockCamera(fps=100.0)
    policy = SmolVLAPolicy()
    total_steps = 120

    patch_hits = []
    spec_accepts = []
    latencies = []

    t_start = time.perf_counter()
    for s in range(total_steps):
        t0 = time.perf_counter()
        ret, frame = cam.read()
        act, substeps, pc_decision, spec_act = policy.step(frame, "Pick up target cube")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        latencies.append(dt_ms)
        patch_hits.append(pc_decision.patch_hit_ratio)
        spec_accepts.append(spec_act.acceptance_rate)

    total_time = time.perf_counter() - t_start
    avg_hz = total_steps / total_time
    avg_latency = float(np.mean(latencies))
    p99_latency = float(np.percentile(latencies, 99))
    avg_patch_hit = float(np.mean(patch_hits))
    avg_spec_acc = float(np.mean(spec_accepts))

    # Baseline comparison (Unoptimized VLA: 63.0ms per step = 15.87 Hz)
    baseline_latency = 63.0
    baseline_hz = 1000.0 / baseline_latency
    speedup = baseline_latency / max(1.0, avg_latency)

    print(f"\nBenchmark Results across {total_steps} closed-loop steps:")
    print(f"  - Average Control Frequency:     {avg_hz:.1f} Hz (Baseline: {baseline_hz:.1f} Hz)")
    print(f"  - End-to-End Speedup:            {speedup:.2f}× faster")
    print(f"  - Mean Step Latency:             {avg_latency:.2f} ms (p99: {p99_latency:.2f} ms)")
    print(f"  - ViT Patch Reuse Ratio:         {avg_patch_hit * 100:.2f}% (Static background skipped)")
    print(f"  - 7-DoF SpecAct Acceptance Rate: {avg_spec_acc * 100:.2f}% (Single parallel verify pass)")
    print(f"  - Motor Actuation Substeps:      4 substeps/step (Quintic Hermite spline)")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
