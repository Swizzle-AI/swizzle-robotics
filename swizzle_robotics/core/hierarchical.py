"""
Swizzle Dual-Rate Hierarchical Control (`S2 -> S1 -> S0`)
==========================================================
Aligns with modern generalist humanoid / manipulation stacks (Helix S0-S2, Gemini Robotics ER):

- S2 (Slow VLM Grounded Task Planner): Queries `swizzle-rt serve` `/v1/chat/completions`
  with `response_format` JSON Schema (or uses built-in deterministic spatial grounding)
  to emit structured subtasks and inject dynamic `ObstacleSphere` keep-out barriers.
- S1 (20-50 Hz Policy Chunker): Executes `BaseVLAPolicy` + `Swizzle-PatchCache` + `Swizzle-RTC`
  prefix-frozen chunk inpainting to maintain continuous trajectory intent.
- S0 (50-200 Hz Safe Actuation Loop): Applies `ControlBarrierShield` + `QuinticSplineInterpolator`
  and logs every tick to `EmbodiedFlightRecorder` (`MCAP` + `LeRobot v2.1` + receipts).
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..models.base import BaseVLAPolicy
from .recorder import EmbodiedFlightRecorder, PhysicalReceipt
from .safety import ControlBarrierShield, ObstacleSphere, SafetyConfig, SafetyTelemetry


S2_TASK_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "subtasks": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "avoid_zones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "center": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "radius": {"type": "number"},
                },
                "required": ["name", "center", "radius"],
            },
        },
        "max_speed_scale": {"type": "number"},
    },
    "required": ["subtasks", "avoid_zones", "max_speed_scale"],
    "additionalProperties": False,
}


@dataclass
class S2TaskPlan:
    instruction: str
    subtasks: List[str]
    avoid_obstacles: List[ObstacleSphere] = field(default_factory=list)
    max_speed_scale: float = 1.0
    source: str = "deterministic-grounding"


@dataclass
class HierarchicalTick:
    step: int
    subtask: str
    safe_action: List[float]
    raw_action: List[float]
    substeps_50hz: List[List[float]]
    safety: SafetyTelemetry
    receipt: PhysicalReceipt
    patch_hit_ratio: float
    speed_scale: float


class HierarchicalController:
    """
    Couples S2 schema-constrained semantic task planning with S1 Real-Time Chunking (RTC)
    and S0 Control Barrier Function (CBF) safety verification + flight recording.
    """

    def __init__(
        self,
        policy: BaseVLAPolicy,
        safety_config: Optional[SafetyConfig] = None,
        swizzle_rt_url: Optional[str] = None,
        episode_id: str = "ep_hierarchical_000",
    ):
        self.policy = policy
        self.safety_shield = ControlBarrierShield(safety_config)
        self.swizzle_rt_url = swizzle_rt_url.rstrip("/") if swizzle_rt_url else None
        self.recorder = EmbodiedFlightRecorder(episode_id=episode_id, model_id=policy.model_name)
        self.pose = [0.0, 0.0, 0.25, 0.0, 0.0, 0.0]
        self.active_plan: Optional[S2TaskPlan] = None
        self.step_count = 0

    def plan_s2(self, instruction: str, known_objects: Optional[Dict[str, TupleLike]] = None) -> S2TaskPlan:
        """
        Generates a structured `S2TaskPlan` via `swizzle-rt serve` JSON Schema endpoint if
        configured/reachable, otherwise deterministically decomposes instruction & constraints.
        """
        if self.swizzle_rt_url:
            plan = self._query_swizzle_rt(instruction)
            if plan is not None:
                self._apply_plan(plan)
                return plan

        subtasks = self._decompose_instruction(instruction)
        avoid: List[ObstacleSphere] = []
        speed_scale = 1.0
        low = instruction.lower()
        if any(w in low for w in ("careful", "slow", "fragile", "human", "glass")):
            speed_scale = 0.5

        if known_objects:
            for name, pos in known_objects.items():
                if f"avoid {name.lower()}" in low or f"avoiding {name.lower()}" in low or "fragile" in low:
                    avoid.append(
                        ObstacleSphere(
                            center=(float(pos[0]), float(pos[1]), float(pos[2])),
                            radius=float(pos[3]) if len(pos) > 3 else 0.08,
                            name=name,
                        )
                    )

        plan = S2TaskPlan(
            instruction=instruction,
            subtasks=subtasks,
            avoid_obstacles=avoid,
            max_speed_scale=speed_scale,
            source="deterministic-grounding",
        )
        self._apply_plan(plan)
        return plan

    def _apply_plan(self, plan: S2TaskPlan) -> None:
        self.active_plan = plan
        existing = {obs.name: obs for obs in self.safety_shield.config.obstacles}
        for obs in plan.avoid_obstacles:
            existing[obs.name] = obs
        self.safety_shield.config.obstacles = list(existing.values())

    def _query_swizzle_rt(self, instruction: str) -> Optional[S2TaskPlan]:
        body = json.dumps(
            {
                "model": "swizzle-rt",
                "temperature": 0.0,
                "messages": [
                    {
                        "role": "system",
                        "content": "Decompose robot instruction into subtasks, obstacle avoid_zones, and max_speed_scale.",
                    },
                    {"role": "user", "content": instruction},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "s2_task_plan", "strict": True, "schema": S2_TASK_JSON_SCHEMA},
                },
            }
        ).encode("utf-8")
        try:
            req = urllib.request.Request(
                f"{self.swizzle_rt_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                obstacles = [
                    ObstacleSphere(
                        center=(float(z["center"][0]), float(z["center"][1]), float(z["center"][2])),
                        radius=float(z["radius"]),
                        name=str(z["name"]),
                    )
                    for z in parsed.get("avoid_zones", [])
                ]
                return S2TaskPlan(
                    instruction=instruction,
                    subtasks=[str(s) for s in parsed.get("subtasks", [instruction])],
                    avoid_obstacles=obstacles,
                    max_speed_scale=max(0.1, min(1.0, float(parsed.get("max_speed_scale", 1.0)))),
                    source="swizzle-rt-json-schema",
                )
        except Exception:
            return None

    @staticmethod
    def _decompose_instruction(instruction: str) -> List[str]:
        parts = [p.strip() for p in re.split(r"\b(?:then|and then|, then)\b", instruction, flags=re.I) if p.strip()]
        return parts or [instruction]

    def step_s1_s0(
        self,
        frame: Any,
        subtask_idx: int = 0,
        execution_delay: int = 2,
        heartbeat_ok: bool = True,
        trace_id: str = "local",
    ) -> HierarchicalTick:
        if self.active_plan is None:
            self.plan_s2("execute manipulation task")
        assert self.active_plan is not None

        subtasks = self.active_plan.subtasks
        subtask = subtasks[min(max(0, subtask_idx), len(subtasks) - 1)]
        scale = float(self.active_plan.max_speed_scale)

        self.step_count += 1
        chunk = self.policy.forward_action_chunk(frame, subtask, horizon=16)
        _, _, pc_decision, _ = self.policy.step(frame, subtask)
        inpainted, substeps = self.policy.teleop.push_rtc_chunk(
            self.step_count, chunk.actions, execution_delay=execution_delay, overlap_horizon=4
        )
        raw_scaled = [inpainted[0][d] * scale for d in range(6)] + [inpainted[0][6]]

        safety = self.safety_shield.project(raw_scaled, self.pose, heartbeat_ok=heartbeat_ok)
        for d in range(6):
            self.pose[d] += safety.safe_action[d] * 0.05

        receipt = self.recorder.record_step(
            frame=frame,
            pose=self.pose,
            safety=safety,
            prompt=subtask,
            trace_id=trace_id,
        )
        return HierarchicalTick(
            step=self.step_count,
            subtask=subtask,
            safe_action=safety.safe_action,
            raw_action=safety.raw_action,
            substeps_50hz=substeps,
            safety=safety,
            receipt=receipt,
            patch_hit_ratio=pc_decision.patch_hit_ratio,
            speed_scale=scale,
        )


TupleLike = Sequence[float]
