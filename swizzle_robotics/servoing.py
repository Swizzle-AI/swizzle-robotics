from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from .types import PatchCacheDecision, SpecActTelemetry
from .core.camera import LiveCamera
from .core.patch_cache import extract_patch_intensities_from_frame
from .core.safety import ControlBarrierShield, SafetyTelemetry
from .models.base import BaseVLAPolicy
from .models import get_policy


@dataclass
class ServoingConfig:
    """Configuration for closed-loop visual servoing."""

    target_hz: float = 80.5
    grid_size: int = 14
    enable_cbf_shield: bool = True


@dataclass
class ServoingTelemetry:
    step: int
    dt_ms: float
    effective_hz: float
    action_7dof: List[float]
    substeps_count: int
    patch_hit_ratio: float
    spec_act_rate: float
    prefill_speedup: float
    cbf_intervened: bool = False
    min_barrier_margin: float = 1.0


class VisualServoingLoop:
    """
    80.5 Hz Real-Time Visual Servoing Runtime.
    Drives continuous camera ingestion, spatio-temporal patch caching,
    model inference, CBF safety verification, and quintic spline trajectory smoothing.
    """

    def __init__(
        self,
        config_or_policy: Optional[Any] = None,
        camera: Optional[LiveCamera] = None,
        target_hz: float = 80.5,
        policy: Optional[BaseVLAPolicy] = None,
        safety_shield: Optional[ControlBarrierShield] = None,
    ):
        if isinstance(config_or_policy, ServoingConfig):
            self.config = config_or_policy
            target_hz = config_or_policy.target_hz
            resolved_policy = policy
        else:
            self.config = ServoingConfig(target_hz=target_hz)
            resolved_policy = policy or config_or_policy

        self.policy = resolved_policy or get_policy("lerobot/smolvla-450m")
        self.camera = camera or LiveCamera()
        self.target_hz = target_hz
        self.interval = 1.0 / target_hz
        self.safety_shield = safety_shield or (
            ControlBarrierShield() if self.config.enable_cbf_shield else None
        )
        self.pose = [0.0, 0.0, 0.25, 0.0, 0.0, 0.0]
        self.step_idx = 0
        self._latencies: List[float] = []

    def extract_patch_luminance(self, frame: Any) -> np.ndarray:
        """Extracts a normalized `(grid_size, grid_size)` patch luminance grid."""
        g = self.config.grid_size
        vals = extract_patch_intensities_from_frame(frame, grid_size=g)
        return np.array(vals, dtype=np.float32).reshape(g, g)

    def step(
        self, prompt_or_frame: Any = "Pick up target object", goal: Optional[Any] = None
    ) -> Any:
        t0 = time.perf_counter()
        if isinstance(prompt_or_frame, str) and goal is None:
            _, frame = self.camera.read()
            prompt = prompt_or_frame
            return_legacy_tuple = False
        else:
            frame = prompt_or_frame
            prompt = "Visual servo target alignment"
            return_legacy_tuple = True

        action, substeps, pc_decision, spec_act = self.policy.step(frame, prompt)
        cbf_intervened = False
        min_margin = 1.0
        if self.safety_shield is not None:
            sec = self.safety_shield.project(action, self.pose)
            action = sec.safe_action
            cbf_intervened = sec.cbf_intervened
            min_margin = sec.min_barrier_margin
            for d in range(6):
                self.pose[d] += action[d] * 0.02

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._latencies.append(elapsed_ms)
        if len(self._latencies) > 100:
            self._latencies.pop(0)

        self.step_idx += 1
        if return_legacy_tuple:
            return action[:6], round(elapsed_ms, 2)

        fps = 1000.0 / max(1.0, elapsed_ms)
        return ServoingTelemetry(
            step=self.step_idx,
            dt_ms=round(elapsed_ms, 2),
            effective_hz=round(fps, 1),
            action_7dof=action,
            substeps_count=len(substeps),
            patch_hit_ratio=pc_decision.patch_hit_ratio,
            spec_act_rate=spec_act.acceptance_rate,
            prefill_speedup=pc_decision.prefill_speedup,
            cbf_intervened=cbf_intervened,
            min_barrier_margin=min_margin,
        )

    def run_n_steps(self, n: int = 50, prompt: str = "Pick up target object") -> List[ServoingTelemetry]:
        results = []
        for _ in range(n):
            res = self.step(prompt)
            assert isinstance(res, ServoingTelemetry)
            results.append(res)
        return results
