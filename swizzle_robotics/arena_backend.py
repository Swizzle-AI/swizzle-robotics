"""
Swizzle Real-Life Robot Arena Backend (FastAPI + WebSocket + Edge Runtime)
==========================================================================
Unified control plane for connecting real physical robots (SO-ARM100, Franka,
UR5e, Unitree G1 via WebSocket SDK or Eclipse Zenoh / ROS 2) or running the
closed-loop MuJoCo hardware-in-the-loop twin with hot-reloadable runtime tuning:

- Live parameter tuning over REST (`PATCH /api/arena/params`) & WebSocket (`/ws/studio`):
  - `Swizzle-CBF`: `cbf_alpha`, `max_linear_step`, `max_accel_step`, `eef_radius`,
    workspace bounds (`x_bounds`, `y_bounds`, `z_bounds`), `watchdog_timeout_steps`,
    and dynamic `ObstacleSphere` keep-out zones.
  - `Swizzle-RTC` + `PatchCache`: `policy_model`, `target_hz`, `decay_lambda`,
    `rtc_substeps`, `patch_delta_threshold`, `action_gain`.
- Physical Robot Ingress (`WS /ws/robot/{robot_id}`):
  Accepts live proprioception & visual/patch frames from real hardware, computes
  or filters 7-DoF actions through `ControlBarrierShield`, enforces deadman/E-Stop
  invariants, seals SHA-256 `PhysicalReceipt`s, and returns safe commands in <2 ms.
- Flight Recording Export (`POST /api/arena/export`):
  Writes Foxglove binary `\\x89MCAP0` logs, Hugging Face `LeRobot v2.1` datasets,
  and a ready-to-deploy `swizzle_deploy_profile.json`.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .core.patch_cache import extract_patch_intensities_from_frame
from .core.recorder import EmbodiedFlightRecorder
from .core.safety import ControlBarrierShield, ObstacleSphere, SafetyConfig
from .core.zenoh_bridge import ZenohBridgeConfig, ZenohRoboticsBridge
from .envs import ArenaTaskConfig, SwizzleManipulationArena
from .hardware import HardwareDetector
from .models import MODEL_REGISTRY, get_policy


class ObstacleSpec(BaseModel):
    name: str = Field(..., min_length=1, max_length=48)
    center: Tuple[float, float, float]
    radius: float = Field(..., ge=0.02, le=0.40)


class ArenaTuningParams(BaseModel):
    cbf_enabled: bool = True
    cbf_alpha: float = Field(0.65, ge=0.05, le=1.0)
    max_linear_step: float = Field(0.22, ge=0.02, le=0.50)
    max_angular_step: float = Field(0.35, ge=0.05, le=0.80)
    max_accel_step: float = Field(0.14, ge=0.02, le=0.50)
    eef_radius: float = Field(0.045, ge=0.01, le=0.15)
    x_bounds: Tuple[float, float] = (-0.55, 0.55)
    y_bounds: Tuple[float, float] = (-0.55, 0.55)
    z_bounds: Tuple[float, float] = (0.03, 0.70)
    watchdog_timeout_steps: int = Field(5, ge=1, le=20)
    policy_model: str = "swizzle/edge-flow-vla"
    target_hz: float = Field(60.0, ge=10.0, le=200.0)
    decay_lambda: float = Field(0.06, ge=0.0, le=0.50)
    rtc_substeps: int = Field(4, ge=1, le=8)
    patch_delta_threshold: float = Field(0.025, ge=0.002, le=0.15)
    action_gain: float = Field(1.0, ge=0.1, le=2.0)
    instruction: str = "pick up target cube while avoiding obstacle pillar then place on goal tray"
    obstacles: List[ObstacleSpec] = Field(
        default_factory=lambda: [
            ObstacleSpec(name="pillar", center=(0.0, 0.0, 0.24), radius=0.065),
        ]
    )


class ArenaParamsPatch(BaseModel):
    cbf_enabled: Optional[bool] = None
    cbf_alpha: Optional[float] = Field(None, ge=0.05, le=1.0)
    max_linear_step: Optional[float] = Field(None, ge=0.02, le=0.50)
    max_angular_step: Optional[float] = Field(None, ge=0.05, le=0.80)
    max_accel_step: Optional[float] = Field(None, ge=0.02, le=0.50)
    eef_radius: Optional[float] = Field(None, ge=0.01, le=0.15)
    x_bounds: Optional[Tuple[float, float]] = None
    y_bounds: Optional[Tuple[float, float]] = None
    z_bounds: Optional[Tuple[float, float]] = None
    watchdog_timeout_steps: Optional[int] = Field(None, ge=1, le=20)
    policy_model: Optional[str] = None
    target_hz: Optional[float] = Field(None, ge=10.0, le=200.0)
    decay_lambda: Optional[float] = Field(None, ge=0.0, le=0.50)
    rtc_substeps: Optional[int] = Field(None, ge=1, le=8)
    patch_delta_threshold: Optional[float] = Field(None, ge=0.002, le=0.15)
    action_gain: Optional[float] = Field(None, ge=0.1, le=2.0)
    instruction: Optional[str] = None
    obstacles: Optional[List[ObstacleSpec]] = None


class RobotConnectRequest(BaseModel):
    robot_id: str = Field(..., min_length=2, max_length=48)
    name: str = Field(..., min_length=2, max_length=64)
    adapter: str = Field(
        "websocket_sdk",
        pattern="^(mujoco_hil|websocket_sdk|ros2_zenoh|serial_so100|ur_rtde|franka_fci)$",
    )
    endpoint: str = "ws://127.0.0.1:8787/ws/robot/cell-01"
    dof: int = Field(7, ge=4, le=14)
    hardware_preset: str = "custom"


class ControlActionRequest(BaseModel):
    command: str = Field(..., pattern="^(start|pause|reset|nudge|set_target|estop|clear_estop)$")
    delta: Optional[List[float]] = None
    target: Optional[List[float]] = None


class ExportRequest(BaseModel):
    out_dir: Optional[str] = None
    episode_label: str = "real_arena_session"


ROBOT_ADAPTER_TEMPLATES = [
    {
        "preset_id": "so100",
        "name": "LeRobot SO-ARM100",
        "adapter": "serial_so100",
        "default_endpoint": "serial://COM4@1000000",
        "dof": 6,
        "recommended_cbf_alpha": 0.55,
        "recommended_max_linear_step": 0.16,
    },
    {
        "preset_id": "franka_fr3",
        "name": "Franka Robotics FR3 (FCI)",
        "adapter": "franka_fci",
        "default_endpoint": "tcp://172.16.0.2:7447",
        "dof": 7,
        "recommended_cbf_alpha": 0.65,
        "recommended_max_linear_step": 0.22,
    },
    {
        "preset_id": "ur5e",
        "name": "Universal Robots UR5e (RTDE)",
        "adapter": "ur_rtde",
        "default_endpoint": "rtde://192.168.1.102:30004",
        "dof": 7,
        "recommended_cbf_alpha": 0.60,
        "recommended_max_linear_step": 0.20,
    },
    {
        "preset_id": "unitree_g1",
        "name": "Unitree G1 (Zenoh S2->S1->S0)",
        "adapter": "ros2_zenoh",
        "default_endpoint": "tcp/192.168.123.164:7447",
        "dof": 14,
        "recommended_cbf_alpha": 0.72,
        "recommended_max_linear_step": 0.25,
    },
]


class RealLifeArenaEngine:
    """
    Shared state & real-time control engine powering the Swizzle Real-Life Robot Arena.
    Synchronizes tunable safety/RTC/PatchCache parameters across MuJoCo HIL simulation
    and live physical robots connected via WebSocket SDK or Zenoh/ROS 2 bridges.
    """

    def __init__(self, prefer_mujoco: bool = True):
        self.params = ArenaTuningParams()
        self.task_cfg = ArenaTaskConfig(max_steps=500)
        self.arena = SwizzleManipulationArena(self.task_cfg, prefer_mujoco=prefer_mujoco)
        self.obs = self.arena.reset(seed=42)
        self.policy = get_policy(self.params.policy_model)
        self.shield = ControlBarrierShield(self._build_safety_config())
        self.recorder = EmbodiedFlightRecorder(episode_id="ep_live_000", model_id=self.params.policy_model)
        self.zenoh = ZenohRoboticsBridge(ZenohBridgeConfig(), use_live_zenoh=False)
        self.hardware = HardwareDetector.detect()

        self.running = True
        self.estop_active = False
        self.step_index = 0
        self.episode_index = 0
        self.collision_count = 0
        self.interventions_count = 0
        self.active_robot_id = "local-mujoco-hil"
        self.manual_target: Optional[List[float]] = None
        self.pending_impulse: Optional[List[float]] = None
        self.last_telemetry: Dict[str, Any] = {}
        self.recent_events: List[Dict[str, Any]] = []
        self._last_tick_ts = time.perf_counter()

        self.robots: Dict[str, Dict[str, Any]] = {
            "local-mujoco-hil": {
                "robot_id": "local-mujoco-hil",
                "name": "Swizzle Workcell Twin (MuJoCo HIL)",
                "adapter": "mujoco_hil",
                "endpoint": f"local://{self.arena.backend_mode}",
                "dof": 7,
                "hardware_preset": "mujoco_hil",
                "connected": True,
                "live_ws": False,
                "last_seen": time.time(),
                "last_pose": list(self.obs.eef_pose),
                "ticks_served": 0,
            }
        }
        self._robot_sockets: Dict[str, WebSocket] = {}
        self._studio_sockets: List[WebSocket] = []
        self._apply_params_to_runtime(self.params)
        self._push_event("SYSTEM", f"Arena engine initialized ({self.arena.backend_mode} backend).")

    def _build_safety_config(self) -> SafetyConfig:
        obstacles = [
            ObstacleSphere(center=o.center, radius=o.radius, name=o.name)
            for o in self.params.obstacles
        ]
        return SafetyConfig(
            x_bounds=tuple(self.params.x_bounds),
            y_bounds=tuple(self.params.y_bounds),
            z_bounds=tuple(self.params.z_bounds),
            max_linear_step=float(self.params.max_linear_step),
            max_angular_step=float(self.params.max_angular_step),
            max_accel_step=float(self.params.max_accel_step),
            cbf_alpha=float(self.params.cbf_alpha),
            eef_radius=float(self.params.eef_radius),
            obstacles=obstacles,
            watchdog_timeout_steps=int(self.params.watchdog_timeout_steps),
        )

    def _apply_params_to_runtime(self, params: ArenaTuningParams) -> None:
        if self.policy.model_name != params.policy_model:
            self.policy = get_policy(params.policy_model)
            self.recorder.model_id = params.policy_model
            self._push_event("MODEL", f"Hot-swapped policy -> {params.policy_model}")

        self.shield.config = self._build_safety_config()
        self.policy.patch_cache.delta_threshold = float(params.patch_delta_threshold)
        self.policy.teleop.decay_lambda = float(params.decay_lambda)
        self.policy.teleop.substeps = int(params.rtc_substeps)

        if params.obstacles:
            p = params.obstacles[0]
            self.task_cfg.obstacle_pos = tuple(p.center)
            self.task_cfg.obstacle_radius = float(p.radius)

    def _push_event(self, kind: str, message: str) -> None:
        self.recent_events.insert(
            0,
            {
                "ts": round(time.time(), 3),
                "step": self.step_index,
                "kind": kind,
                "message": message,
            },
        )
        del self.recent_events[40:]

    def update_params(self, patch: ArenaParamsPatch) -> ArenaTuningParams:
        current = self.params.model_dump()
        updates = patch.model_dump(exclude_none=True)
        for key, value in updates.items():
            if key == "obstacles":
                current["obstacles"] = [ObstacleSpec(**o) if isinstance(o, dict) else o for o in value]
            else:
                current[key] = value
        # Sanity enforce bounds ordering
        for b_key in ("x_bounds", "y_bounds", "z_bounds"):
            lo, hi = current[b_key]
            if lo >= hi:
                raise ValueError(f"{b_key} lower bound ({lo}) must be < upper bound ({hi})")
        self.params = ArenaTuningParams(**current)
        self._apply_params_to_runtime(self.params)
        self._push_event("TUNE", f"Updated {', '.join(sorted(updates.keys())) or 'params'}")
        return self.params

    def upsert_obstacle(self, obs: ObstacleSpec) -> List[ObstacleSpec]:
        existing = [o for o in self.params.obstacles if o.name != obs.name]
        existing.append(obs)
        self.params.obstacles = existing
        self._apply_params_to_runtime(self.params)
        self._push_event("OBSTACLE", f"Set zone '{obs.name}' r={obs.radius:.3f}m @ {obs.center}")
        return self.params.obstacles

    def remove_obstacle(self, name: str) -> List[ObstacleSpec]:
        self.params.obstacles = [o for o in self.params.obstacles if o.name != name]
        self._apply_params_to_runtime(self.params)
        self._push_event("OBSTACLE", f"Removed keep-out zone '{name}'")
        return self.params.obstacles

    def register_robot(self, req: RobotConnectRequest) -> Dict[str, Any]:
        entry = {
            "robot_id": req.robot_id,
            "name": req.name,
            "adapter": req.adapter,
            "endpoint": req.endpoint,
            "dof": req.dof,
            "hardware_preset": req.hardware_preset,
            "connected": req.adapter == "mujoco_hil",
            "live_ws": False,
            "last_seen": time.time(),
            "last_pose": [0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
            "ticks_served": 0,
        }
        self.robots[req.robot_id] = entry
        self.active_robot_id = req.robot_id
        if req.adapter == "ros2_zenoh":
            self.zenoh = ZenohRoboticsBridge(
                ZenohBridgeConfig(locator=req.endpoint),
                use_live_zenoh=True,
            )
        self._push_event("ROBOT", f"Registered {req.name} ({req.adapter}) -> {req.endpoint}")
        return entry

    def select_robot(self, robot_id: str) -> Dict[str, Any]:
        if robot_id not in self.robots:
            raise KeyError(f"Unknown robot_id: {robot_id}")
        self.active_robot_id = robot_id
        self._push_event("ROBOT", f"Switched active workcell to '{robot_id}'")
        return self.robots[robot_id]

    def disconnect_robot(self, robot_id: str) -> None:
        if robot_id == "local-mujoco-hil":
            raise ValueError("Cannot remove built-in local MuJoCo HIL twin")
        self.robots.pop(robot_id, None)
        self._robot_sockets.pop(robot_id, None)
        if self.active_robot_id == robot_id:
            self.active_robot_id = "local-mujoco-hil"
        self._push_event("ROBOT", f"Disconnected '{robot_id}'")

    def handle_control(self, req: ControlActionRequest) -> Dict[str, Any]:
        cmd = req.command
        if cmd == "estop":
            self.estop_active = True
            self.running = False
            self.shield.reset()
            self._push_event("ESTOP", "HARD E-STOP ENGAGED — all motor outputs latched to zero")
        elif cmd == "clear_estop":
            self.estop_active = False
            self._push_event("SAFETY", "E-STOP cleared by operator (paused until Resume)")
        elif cmd == "start":
            if self.estop_active:
                raise RuntimeError("Clear E-STOP before starting control loop")
            self.running = True
            self._push_event("LOOP", "Closed-loop execution resumed")
        elif cmd == "pause":
            self.running = False
            self._push_event("LOOP", "Closed-loop execution paused")
        elif cmd == "reset":
            self.episode_index += 1
            self.step_index = 0
            self.collision_count = 0
            self.interventions_count = 0
            self.manual_target = None
            self.pending_impulse = None
            self.shield.reset()
            self.obs = self.arena.reset(seed=42 + self.episode_index)
            self.recorder = EmbodiedFlightRecorder(
                episode_id=f"ep_live_{self.episode_index:03d}",
                model_id=self.params.policy_model,
            )
            self._push_event("RESET", f"Reset episode -> ep_live_{self.episode_index:03d}")
        elif cmd == "nudge":
            delta = list(req.delta or [0.45, 0.0, 0.0])
            while len(delta) < 3:
                delta.append(0.0)
            self.pending_impulse = delta[:3]
            self._push_event("TEST", f"Injected synthetic disturbance delta={delta[:3]}")
        elif cmd == "set_target":
            if req.target and len(req.target) >= 3:
                self.manual_target = [float(req.target[0]), float(req.target[1]), float(req.target[2])]
                self._push_event("TARGET", f"Manual setpoint updated -> {self.manual_target}")
        return self.snapshot()

    def _compute_safe_control(
        self,
        raw_action: Sequence[float],
        pose: Sequence[float],
        frame_data: Any,
        heartbeat_ok: bool = True,
        trace_id: str = "arena-studio",
    ) -> Dict[str, Any]:
        gain = float(self.params.action_gain)
        raw = [float(v) * gain for v in list(raw_action)[:7]]
        while len(raw) < 7:
            raw.append(0.0)
        raw[6] = max(-1.0, min(1.0, float(raw_action[6]) if len(raw_action) > 6 else 0.0))

        if self.estop_active:
            safety = self.shield.project([0.0] * 7, pose, heartbeat_ok=False)
            safety.safe_action = [0.0] * 7
            safety.cbf_intervened = True
            safety.intervention_reasons = ["hard_estop_latched"]
            safety.watchdog_active = True
        elif self.params.cbf_enabled:
            safety = self.shield.project(raw, pose, heartbeat_ok=heartbeat_ok)
        else:
            # Nominal unshielded mode (still evaluate margin passively for telemetry readout)
            probe = self.shield.project(raw, pose, heartbeat_ok=heartbeat_ok)
            safety = probe
            safety.safe_action = [round(max(-1.0, min(1.0, float(v))), 4) for v in raw]
            safety.cbf_intervened = False
            safety.clamped_axes = []
            safety.intervention_reasons = ["unshielded_bypass"]
            safety.solver_latency_ms = 0.0

        if safety.cbf_intervened and not self.estop_active:
            self.interventions_count += 1

        receipt = self.recorder.record_step(
            frame=frame_data,
            pose=list(pose),
            safety=safety,
            prompt=self.params.instruction,
            trace_id=trace_id,
        )
        return {
            "safety": safety,
            "receipt": receipt,
        }

    def step_tick(self) -> Dict[str, Any]:
        """Advances the active local MuJoCo/physics HIL arena by one control tick."""
        now = time.perf_counter()
        dt = max(1e-4, now - self._last_tick_ts)
        self._last_tick_ts = now
        self.step_index += 1

        # Determine goal for policy + servo target
        target = np.array(
            self.manual_target
            if self.manual_target is not None
            else (self.obs.cube_pos if not self.obs.grasped else self.obs.goal_pos),
            dtype=np.float64,
        )
        eef = np.array(self.obs.eef_pose[:3], dtype=np.float64)
        err = (target - eef) * 3.8

        # Combine policy chunk evaluation (for real PatchCache + RTC telemetry) with tracking vector
        _, substeps, pc_decision, spec_act = self.policy.step(self.obs.frame, self.params.instruction)
        grip = 1.0 if float(np.linalg.norm(target - eef)) < 0.09 or self.obs.grasped else -1.0

        nominal = [
            float(err[0]),
            float(err[1]),
            float(err[2]),
            0.0,
            0.0,
            0.0,
            grip,
        ]
        if self.pending_impulse is not None:
            nominal[0] += float(self.pending_impulse[0])
            nominal[1] += float(self.pending_impulse[1])
            nominal[2] += float(self.pending_impulse[2])
            self.pending_impulse = None

        res = self._compute_safe_control(
            raw_action=nominal,
            pose=self.obs.eef_pose,
            frame_data=self.obs.frame,
            heartbeat_ok=True,
            trace_id=f"studio-{self.episode_index:03d}-{self.step_index:05d}",
        )
        safety = res["safety"]
        receipt = res["receipt"]

        if not self.estop_active:
            next_obs, _, done, info = self.arena.step(safety.safe_action)
            self.obs = next_obs
            if info.get("obstacle_collision") or info.get("table_penetration_mm", 0.0) > 0.0:
                self.collision_count += 1
        else:
            done = False
            info = {"obstacle_collision": False, "table_penetration_mm": 0.0, "success": False}

        # Update local robot status
        local_bot = self.robots["local-mujoco-hil"]
        local_bot["last_seen"] = time.time()
        local_bot["last_pose"] = [round(float(v), 4) for v in self.obs.eef_pose]
        local_bot["ticks_served"] += 1

        frame = {
            "type": "telemetry",
            "robot_id": self.active_robot_id,
            "backend_mode": self.arena.backend_mode,
            "transport_mode": self.zenoh.transport_mode,
            "episode_id": self.recorder.episode_id,
            "step_index": self.step_index,
            "running": self.running,
            "estop_active": self.estop_active,
            "effective_hz": round(min(999.0, 1.0 / dt), 1),
            "pose": [round(float(v), 4) for v in self.obs.eef_pose],
            "target": [round(float(v), 4) for v in target],
            "cube_pos": [round(float(v), 4) for v in self.obs.cube_pos],
            "goal_pos": [round(float(v), 4) for v in self.obs.goal_pos],
            "grasped": bool(self.obs.grasped),
            "raw_action": safety.raw_action,
            "safe_action": safety.safe_action,
            "rtc_substeps": len(substeps),
            "cbf_intervened": safety.cbf_intervened,
            "min_barrier_margin": safety.min_barrier_margin,
            "clamped_axes": safety.clamped_axes,
            "intervention_reasons": safety.intervention_reasons,
            "solver_latency_ms": safety.solver_latency_ms,
            "patch_hit_ratio": round(float(pc_decision.patch_hit_ratio), 4),
            "spec_act_rate": round(float(spec_act.acceptance_rate), 4),
            "collisions": self.collision_count,
            "interventions_count": self.interventions_count,
            "receipt_digest": receipt.receipt_digest,
            "dagger_flagged": self.recorder.dagger_flagged,
            "obstacles": [o.model_dump() for o in self.params.obstacles],
            "info": info,
        }
        if done and self.step_index > 10:
            self._push_event("SUCCESS", f"Episode ep_live_{self.episode_index:03d} completed task cleanly!")
        self.last_telemetry = frame
        return frame

    def ingest_robot_telemetry(self, robot_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a live frame + proprioceptive pose sent by an external physical robot
        connected over `WS /ws/robot/{robot_id}` and returns the CBF-safe actuation command.
        """
        now = time.perf_counter()
        dt = max(1e-4, now - self._last_tick_ts)
        self._last_tick_ts = now
        self.step_index += 1

        if robot_id not in self.robots:
            self.register_robot(
                RobotConnectRequest(
                    robot_id=robot_id,
                    name=payload.get("name", robot_id),
                    adapter="websocket_sdk",
                    endpoint=f"ws://connected/{robot_id}",
                    dof=int(payload.get("dof", 7)),
                )
            )
        bot = self.robots[robot_id]
        bot["connected"] = True
        bot["live_ws"] = True
        bot["last_seen"] = time.time()
        bot["ticks_served"] += 1

        pose = [float(x) for x in payload.get("pose", bot.get("last_pose", [0.0, 0.0, 0.25, 0.0, 0.0, 0.0]))]
        while len(pose) < 6:
            pose.append(0.0)
        pose = pose[:6]
        bot["last_pose"] = [round(v, 4) for v in pose]

        frame_input = payload.get("frame_patches")
        if frame_input is None:
            frame_input = extract_patch_intensities_from_frame(self.obs.frame, grid_size=14)

        _, substeps, pc_decision, spec_act = self.policy.step(frame_input, payload.get("prompt", self.params.instruction))

        if "raw_action" in payload and isinstance(payload["raw_action"], list):
            nominal = [float(x) for x in payload["raw_action"][:7]]
        else:
            # Closed-loop policy output from active Swizzle VLA
            nominal = self.policy.forward_action_chunk(frame_input, self.params.instruction, horizon=8).actions[0]

        heartbeat_ok = bool(payload.get("heartbeat_ok", True))
        res = self._compute_safe_control(
            raw_action=nominal,
            pose=pose,
            frame_data=frame_input,
            heartbeat_ok=heartbeat_ok,
            trace_id=f"hw-{robot_id}-{self.step_index:06d}",
        )
        safety = res["safety"]
        receipt = res["receipt"]

        if safety.min_barrier_margin < -1e-4:
            self.collision_count += 1

        telemetry = {
            "type": "telemetry",
            "robot_id": robot_id,
            "backend_mode": f"physical:{bot['adapter']}",
            "transport_mode": "websocket_sdk",
            "episode_id": self.recorder.episode_id,
            "step_index": self.step_index,
            "running": True,
            "estop_active": self.estop_active,
            "effective_hz": round(min(999.0, 1.0 / dt), 1),
            "pose": [round(v, 4) for v in pose],
            "target": [round(float(v), 4) for v in (self.manual_target or payload.get("target", [0.2, 0.0, 0.15]))],
            "cube_pos": [round(float(v), 4) for v in payload.get("object_pos", [0.2, 0.0, 0.05])],
            "goal_pos": [round(float(v), 4) for v in payload.get("goal_pos", [-0.15, 0.18, 0.05])],
            "grasped": bool(payload.get("grasped", False)),
            "raw_action": safety.raw_action,
            "safe_action": safety.safe_action,
            "rtc_substeps": len(substeps),
            "cbf_intervened": safety.cbf_intervened,
            "min_barrier_margin": safety.min_barrier_margin,
            "clamped_axes": safety.clamped_axes,
            "intervention_reasons": safety.intervention_reasons,
            "solver_latency_ms": safety.solver_latency_ms,
            "patch_hit_ratio": round(float(pc_decision.patch_hit_ratio), 4),
            "spec_act_rate": round(float(spec_act.acceptance_rate), 4),
            "collisions": self.collision_count,
            "interventions_count": self.interventions_count,
            "receipt_digest": receipt.receipt_digest,
            "dagger_flagged": self.recorder.dagger_flagged,
            "obstacles": [o.model_dump() for o in self.params.obstacles],
            "info": {"physical_hardware": True},
        }
        self.last_telemetry = telemetry
        return {
            "command_type": "safe_actuation",
            "safe_action": safety.safe_action,
            "estop_active": self.estop_active,
            "cbf_intervened": safety.cbf_intervened,
            "min_barrier_margin": safety.min_barrier_margin,
            "solver_latency_ms": safety.solver_latency_ms,
            "receipt_digest": receipt.receipt_digest,
            "params": self.params.model_dump(),
            "telemetry": telemetry,
        }

    def export_session(self, out_dir: Optional[str] = None, label: str = "real_arena_session") -> Dict[str, Any]:
        root = Path(out_dir) if out_dir else (Path.cwd() / "artifacts" / f"{label}_{int(time.time())}")
        root.mkdir(parents=True, exist_ok=True)
        mcap_path = root / f"{self.recorder.episode_id}.mcap"
        lerobot_dir = root / "lerobot_dataset"
        profile_path = root / "swizzle_deploy_profile.json"

        if self.recorder.steps:
            written_mcap = self.recorder.export_mcap(mcap_path)
            mcap_bytes = int(Path(written_mcap).stat().st_size)
            raw_lerobot = self.recorder.export_lerobot_v2(lerobot_dir, episode_index=self.episode_index)
            lerobot_summary = {
                k: (str(v) if isinstance(v, Path) else v) for k, v in raw_lerobot.items()
            }
        else:
            mcap_bytes = 0
            lerobot_summary = {"episodes": 0, "steps": 0}

        deploy_profile = {
            "schema": "swizzle.robotics.deploy_profile.v1",
            "generated_at": time.time(),
            "active_robot": self.robots.get(self.active_robot_id),
            "hardware_host": asdict(self.hardware),
            "tuned_params": self.params.model_dump(),
            "session_stats": {
                "steps_recorded": len(self.recorder.steps),
                "collisions": self.collision_count,
                "interventions": self.interventions_count,
                "dagger_flagged": self.recorder.dagger_flagged,
                "mcap_bytes": mcap_bytes,
                "latest_receipt_digest": self.recorder.steps[-1].receipt.receipt_digest if self.recorder.steps else None,
            },
        }
        profile_path.write_text(json.dumps(deploy_profile, indent=2), encoding="utf-8")
        self._push_event("EXPORT", f"Saved deploy profile + {mcap_bytes} B MCAP -> {root}")
        return {
            "export_dir": str(root),
            "mcap_path": str(mcap_path),
            "mcap_bytes": mcap_bytes,
            "lerobot_summary": lerobot_summary,
            "profile_path": str(profile_path),
            "deploy_profile": deploy_profile,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "active_robot_id": self.active_robot_id,
            "running": self.running,
            "estop_active": self.estop_active,
            "step_index": self.step_index,
            "episode_index": self.episode_index,
            "collisions": self.collision_count,
            "interventions_count": self.interventions_count,
            "recorded_steps": len(self.recorder.steps),
            "backend_mode": self.arena.backend_mode,
            "hardware": asdict(self.hardware),
            "available_models": sorted(MODEL_REGISTRY.keys()),
            "adapter_templates": ROBOT_ADAPTER_TEMPLATES,
            "robots": list(self.robots.values()),
            "params": self.params.model_dump(),
            "last_telemetry": self.last_telemetry,
            "events": self.recent_events[:15],
        }


def create_arena_app(
    engine: Optional[RealLifeArenaEngine] = None,
    enable_background_loop: bool = True,
) -> FastAPI:
    arena_engine = engine or RealLifeArenaEngine(prefer_mujoco=True)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        loop_task = None
        if enable_background_loop:

            async def _runner() -> None:
                while True:
                    target_hz = max(10.0, min(120.0, float(arena_engine.params.target_hz)))
                    active_bot = arena_engine.robots.get(arena_engine.active_robot_id, {})
                    is_live_ws_robot = bool(active_bot.get("live_ws"))
                    if arena_engine.running and not is_live_ws_robot:
                        tick = arena_engine.step_tick()
                        await _broadcast_studio(tick)
                    await asyncio.sleep(1.0 / min(target_hz, 30.0))

            loop_task = asyncio.create_task(_runner())
        try:
            yield
        finally:
            if loop_task is not None:
                loop_task.cancel()
            arena_engine.arena.close()

    app = FastAPI(
        title="Swizzle Real-Life Robot Arena",
        version="0.3.0",
        description="Real-time Physical AI connection, parameter tuning, CBF shield & MCAP flight studio",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.engine = arena_engine

    async def _broadcast_studio(payload: Dict[str, Any]) -> None:
        if not arena_engine._studio_sockets:
            return
        dead: List[WebSocket] = []
        encoded = json.dumps(payload)
        for ws in list(arena_engine._studio_sockets):
            try:
                await ws.send_text(encoded)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in arena_engine._studio_sockets:
                arena_engine._studio_sockets.remove(ws)

    async def _push_params_to_robots() -> None:
        msg = json.dumps({"command_type": "params_update", "params": arena_engine.params.model_dump()})
        for rid, ws in list(arena_engine._robot_sockets.items()):
            try:
                await ws.send_text(msg)
            except Exception:
                arena_engine._robot_sockets.pop(rid, None)

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "service": "swizzle-real-life-arena",
            "backend_mode": arena_engine.arena.backend_mode,
            "estop_active": arena_engine.estop_active,
        }

    @app.get("/api/arena/state")
    def get_state() -> Dict[str, Any]:
        return arena_engine.snapshot()

    @app.get("/api/arena/params")
    def get_params() -> Dict[str, Any]:
        return arena_engine.params.model_dump()

    @app.patch("/api/arena/params")
    async def patch_params(patch: ArenaParamsPatch) -> Dict[str, Any]:
        try:
            updated = arena_engine.update_params(patch)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await _push_params_to_robots()
        await _broadcast_studio({"type": "state_sync", "snapshot": arena_engine.snapshot()})
        return updated.model_dump()

    @app.post("/api/arena/obstacles")
    async def upsert_obstacle(obs: ObstacleSpec) -> Dict[str, Any]:
        zones = arena_engine.upsert_obstacle(obs)
        await _push_params_to_robots()
        await _broadcast_studio({"type": "state_sync", "snapshot": arena_engine.snapshot()})
        return {"obstacles": [z.model_dump() for z in zones]}

    @app.delete("/api/arena/obstacles/{name}")
    async def delete_obstacle(name: str) -> Dict[str, Any]:
        zones = arena_engine.remove_obstacle(name)
        await _push_params_to_robots()
        await _broadcast_studio({"type": "state_sync", "snapshot": arena_engine.snapshot()})
        return {"obstacles": [z.model_dump() for z in zones]}

    @app.get("/api/arena/robots")
    def list_robots() -> Dict[str, Any]:
        return {
            "active_robot_id": arena_engine.active_robot_id,
            "robots": list(arena_engine.robots.values()),
            "adapter_templates": ROBOT_ADAPTER_TEMPLATES,
        }

    @app.post("/api/arena/robots/connect")
    async def connect_robot(req: RobotConnectRequest) -> Dict[str, Any]:
        bot = arena_engine.register_robot(req)
        await _broadcast_studio({"type": "state_sync", "snapshot": arena_engine.snapshot()})
        return {"robot": bot, "ws_ingest_path": f"/ws/robot/{req.robot_id}"}

    @app.post("/api/arena/robots/{robot_id}/select")
    async def select_robot(robot_id: str) -> Dict[str, Any]:
        try:
            bot = arena_engine.select_robot(robot_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        await _broadcast_studio({"type": "state_sync", "snapshot": arena_engine.snapshot()})
        return {"active_robot_id": robot_id, "robot": bot}

    @app.post("/api/arena/robots/{robot_id}/disconnect")
    async def disconnect_robot(robot_id: str) -> Dict[str, Any]:
        try:
            arena_engine.disconnect_robot(robot_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await _broadcast_studio({"type": "state_sync", "snapshot": arena_engine.snapshot()})
        return {"disconnected": robot_id, "active_robot_id": arena_engine.active_robot_id}

    @app.post("/api/arena/control")
    async def control_loop(req: ControlActionRequest) -> Dict[str, Any]:
        try:
            snap = arena_engine.handle_control(req)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if req.command == "estop":
            for rid, ws in list(arena_engine._robot_sockets.items()):
                try:
                    await ws.send_text(json.dumps({"command_type": "estop", "safe_action": [0.0] * 7, "estop_active": True}))
                except Exception:
                    arena_engine._robot_sockets.pop(rid, None)
        await _broadcast_studio({"type": "state_sync", "snapshot": snap})
        return snap

    @app.post("/api/arena/export")
    def export_session(req: ExportRequest) -> Dict[str, Any]:
        return arena_engine.export_session(out_dir=req.out_dir, label=req.episode_label)

    @app.websocket("/ws/studio")
    async def studio_ws(ws: WebSocket) -> None:
        await ws.accept()
        arena_engine._studio_sockets.append(ws)
        try:
            await ws.send_text(json.dumps({"type": "state_sync", "snapshot": arena_engine.snapshot()}))
            while True:
                raw_msg = await ws.receive_text()
                data = json.loads(raw_msg)
                kind = data.get("type")
                if kind == "patch_params":
                    patch = ArenaParamsPatch(**data.get("params", {}))
                    arena_engine.update_params(patch)
                    await _push_params_to_robots()
                    await _broadcast_studio({"type": "state_sync", "snapshot": arena_engine.snapshot()})
                elif kind == "control":
                    ctrl = ControlActionRequest(**data.get("payload", {}))
                    snap = arena_engine.handle_control(ctrl)
                    await _broadcast_studio({"type": "state_sync", "snapshot": snap})
        except WebSocketDisconnect:
            pass
        finally:
            if ws in arena_engine._studio_sockets:
                arena_engine._studio_sockets.remove(ws)

    @app.websocket("/ws/robot/{robot_id}")
    async def robot_ws(ws: WebSocket, robot_id: str) -> None:
        await ws.accept()
        arena_engine._robot_sockets[robot_id] = ws
        arena_engine._push_event("ROBOT_WS", f"Physical hardware socket connected: {robot_id}")
        try:
            while True:
                raw_msg = await ws.receive_text()
                payload = json.loads(raw_msg)
                response = arena_engine.ingest_robot_telemetry(robot_id, payload)
                telemetry = response.pop("telemetry")
                await ws.send_text(json.dumps(response))
                await _broadcast_studio(telemetry)
        except WebSocketDisconnect:
            if robot_id in arena_engine.robots:
                arena_engine.robots[robot_id]["live_ws"] = False
                arena_engine.robots[robot_id]["connected"] = False
            arena_engine._robot_sockets.pop(robot_id, None)
            arena_engine._push_event("ROBOT_WS", f"Physical hardware socket disconnected: {robot_id}")
            await _broadcast_studio({"type": "state_sync", "snapshot": arena_engine.snapshot()})

    # Optionally serve built frontend when dist exists
    dist_dir = Path(__file__).resolve().parent.parent / "website" / "dist"
    if dist_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="website")

    return app


app = create_arena_app(enable_background_loop=True)


def run_arena_server(port: int = 8787, host: str = "0.0.0.0") -> None:
    import uvicorn

    print(f"Swizzle Real-Life Arena Backend listening on http://{host}:{port}")
    print(f"  - Studio WebSocket:  ws://{host}:{port}/ws/studio")
    print(f"  - Robot HW Bridge:   ws://{host}:{port}/ws/robot/<robot_id>")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_arena_server()
