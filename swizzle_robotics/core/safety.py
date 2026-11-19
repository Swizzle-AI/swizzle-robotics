"""
Swizzle-CBF: Sub-Millisecond Control Barrier Function Safety Shield
===================================================================
Enforces deterministic physical safety invariants on Vision-Language-Action (VLA)
and World-Model action commands at every 50-200 Hz control tick:

1. Workspace Geofence Barriers (Discrete ECBF): keeps end-effector inside safe bounds.
2. Obstacle & Human-Proximity Avoidance: smooth half-space projection around spherical keep-out zones.
3. Velocity & Acceleration/Jerk Slew-Rate Limits: prevents chunk-boundary mechanical spikes.
4. Network Watchdog Deadman Decay: ramps commands smoothly to zero on stale tunnel heartbeats.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..types import clamp_unit


@dataclass
class ObstacleSphere:
    """Spherical keep-out zone (e.g., fragile fixture, sensor rig, or human hand)."""

    center: Tuple[float, float, float]
    radius: float
    name: str = "obstacle"


@dataclass
class SafetyConfig:
    """Configuration for the Swizzle-CBF real-time safety shield."""

    x_bounds: Tuple[float, float] = (-0.60, 0.60)
    y_bounds: Tuple[float, float] = (-0.60, 0.60)
    z_bounds: Tuple[float, float] = (0.02, 0.75)
    max_linear_step: float = 0.25
    max_angular_step: float = 0.35
    max_accel_step: float = 0.15
    cbf_alpha: float = 0.65
    eef_radius: float = 0.04
    obstacles: List[ObstacleSphere] = field(default_factory=list)
    watchdog_timeout_steps: int = 5


@dataclass
class SafetyTelemetry:
    """Per-tick verification and projection telemetry emitted by ControlBarrierShield."""

    safe_action: List[float]
    raw_action: List[float]
    cbf_intervened: bool
    min_barrier_margin: float
    clamped_axes: List[str]
    intervention_reasons: List[str]
    solver_latency_ms: float
    watchdog_active: bool = False


class ControlBarrierShield:
    """
    Deterministic discrete-time Control Barrier Function (CBF) safety filter.
    Projects nominal 7-DoF policy deltas onto the safe admissible control set in <0.2 ms.
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self._prev_delta = np.zeros(6, dtype=np.float64)
        self._stale_steps = 0

    def reset(self) -> None:
        self._prev_delta = np.zeros(6, dtype=np.float64)
        self._stale_steps = 0

    def apply_watchdog_decay(self, stale_steps: int) -> List[float]:
        """Smoothly damps previous motor command toward rest when remote inference stalls."""
        timeout = max(1, int(self.config.watchdog_timeout_steps))
        ratio = max(0.0, 1.0 - float(stale_steps) / float(timeout))
        decayed = self._prev_delta * ratio
        self._prev_delta = decayed
        return [round(float(x), 4) for x in decayed] + [0.0]

    def project(
        self,
        raw_action: Sequence[float],
        current_pose: Sequence[float] = (0.0, 0.0, 0.25, 0.0, 0.0, 0.0),
        heartbeat_ok: bool = True,
    ) -> SafetyTelemetry:
        """
        Projects 7-DoF action [dx, dy, dz, dr, dp, dyaw, gripper] onto CBF-safe constraints
        given current end-effector Cartesian pose [x, y, z, r, p, yaw].
        """
        t0 = time.perf_counter()
        raw = [float(x) for x in raw_action[:7]]
        while len(raw) < 7:
            raw.append(0.0)

        if not heartbeat_ok:
            self._stale_steps += 1
            safe_watchdog = self.apply_watchdog_decay(self._stale_steps)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return SafetyTelemetry(
                safe_action=safe_watchdog,
                raw_action=[round(x, 4) for x in raw],
                cbf_intervened=True,
                min_barrier_margin=0.0,
                clamped_axes=["dx", "dy", "dz"],
                intervention_reasons=["watchdog_stale_heartbeat"],
                solver_latency_ms=round(dt_ms, 4),
                watchdog_active=True,
            )

        self._stale_steps = 0
        pose = np.array([float(x) for x in current_pose[:6]], dtype=np.float64)
        if pose.shape[0] < 6:
            pose = np.pad(pose, (0, 6 - pose.shape[0]))

        u = np.array(raw[:6], dtype=np.float64)
        reasons: List[str] = []
        clamped_axes: List[str] = []

        # 1. Acceleration / jerk slew-rate constraint relative to previous command
        slew_diff = u - self._prev_delta
        max_acc = float(self.config.max_accel_step)
        if np.any(np.abs(slew_diff) > max_acc):
            clipped = np.clip(slew_diff, -max_acc, max_acc)
            for idx, axis in enumerate(["dx", "dy", "dz", "droll", "dpitch", "dyaw"]):
                if abs(slew_diff[idx]) > max_acc + 1e-9:
                    clamped_axes.append(axis)
            u = self._prev_delta + clipped
            reasons.append("slew_rate_jerk_limit")

        # 2. Euclidean linear and angular velocity bounds
        lin_norm = float(np.linalg.norm(u[:3]))
        if lin_norm > self.config.max_linear_step and lin_norm > 1e-9:
            u[:3] *= self.config.max_linear_step / lin_norm
            reasons.append("linear_velocity_limit")
            for axis in ("dx", "dy", "dz"):
                if axis not in clamped_axes:
                    clamped_axes.append(axis)

        ang_norm = float(np.linalg.norm(u[3:6]))
        if ang_norm > self.config.max_angular_step and ang_norm > 1e-9:
            u[3:6] *= self.config.max_angular_step / ang_norm
            reasons.append("angular_velocity_limit")

        # 3. Discrete Workspace Geofence CBF: x_next in [lo, hi] with decay rate alpha
        alpha = max(0.05, min(1.0, float(self.config.cbf_alpha)))
        bounds = [self.config.x_bounds, self.config.y_bounds, self.config.z_bounds]
        axis_names = ["dx", "dy", "dz"]
        min_margin = float("inf")

        for dim_idx, (lo, hi) in enumerate(bounds):
            p = pose[dim_idx]
            h_lo = p - lo
            h_hi = hi - p
            min_margin = min(min_margin, h_lo, h_hi)

            # Enforce discrete CBF invariance: p + u_d >= lo + (1 - alpha) * h_lo => u_d >= -alpha * h_lo
            u_min = -alpha * max(0.0, h_lo) if h_lo >= 0.0 else (lo - p)
            u_max = alpha * max(0.0, h_hi) if h_hi >= 0.0 else (hi - p)
            if u_min > u_max:
                mid = 0.5 * (u_min + u_max)
                u[dim_idx] = mid
                reasons.append(f"workspace_recover_{axis_names[dim_idx]}")
            elif u[dim_idx] < u_min - 1e-9:
                u[dim_idx] = u_min
                reasons.append(f"cbf_lower_bound_{axis_names[dim_idx]}")
                if axis_names[dim_idx] not in clamped_axes:
                    clamped_axes.append(axis_names[dim_idx])
            elif u[dim_idx] > u_max + 1e-9:
                u[dim_idx] = u_max
                reasons.append(f"cbf_upper_bound_{axis_names[dim_idx]}")
                if axis_names[dim_idx] not in clamped_axes:
                    clamped_axes.append(axis_names[dim_idx])

        # 4. Spherical Obstacle Avoidance CBF: ||p - c|| - R_safe >= 0
        for obs in self.config.obstacles:
            center = np.array(obs.center, dtype=np.float64)
            diff = pose[:3] - center
            dist = float(np.linalg.norm(diff))
            r_safe = float(obs.radius) + float(self.config.eef_radius)
            signed_dist = dist - r_safe
            min_margin = min(min_margin, signed_dist)

            normal = diff / dist if dist > 1e-7 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
            # First-order signed-distance CBF condition: n^T u + alpha * signed_dist >= 0
            lhs = float(np.dot(normal, u[:3])) + alpha * signed_dist
            if lhs < 0.0:
                u[:3] -= lhs * normal
                reasons.append(f"cbf_obstacle_{obs.name}")
                for axis in ("dx", "dy", "dz"):
                    if axis not in clamped_axes:
                        clamped_axes.append(axis)

        if math.isinf(min_margin):
            min_margin = 1.0

        self._prev_delta = u.copy()
        gripper = clamp_unit(raw[6])
        safe_vec = [round(clamp_unit(float(val)), 4) for val in u] + [round(gripper, 4)]

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return SafetyTelemetry(
            safe_action=safe_vec,
            raw_action=[round(float(x), 4) for x in raw],
            cbf_intervened=len(reasons) > 0,
            min_barrier_margin=round(float(min_margin), 5),
            clamped_axes=clamped_axes,
            intervention_reasons=reasons,
            solver_latency_ms=round(dt_ms, 4),
            watchdog_active=False,
        )
