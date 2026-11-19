"""
Swizzle Manipulation Arena (MuJoCo MJCF + Zero-Dep Rigid-Body Fallback)
========================================================================
Provides a standardized 3D tabletop pick-and-place / obstacle-avoidance physics arena
with real camera image rendering for closing the loop on VLAs, `Swizzle-CBF`,
`Swizzle-RTC`, and `EmbodiedFlightRecorder`:

- Automatically uses `mujoco` (`mujoco.MjModel` + `MjData` + offscreen `Renderer`)
  when installed (`pip install swizzle-robotics[sim]`).
- Falls back cleanly to a deterministic 3D rigid-body contact + perspective projection
  simulator in pure NumPy so CI and minimal student laptops run identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


ARENA_MJCF_XML = """
<mujoco model="swizzle_tabletop_arena">
  <option timestep="0.005" gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <global offwidth="224" offheight="224"/>
  </visual>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="table" type="box" pos="0 0 -0.02" size="0.65 0.65 0.02" rgba="0.18 0.22 0.20 1"/>
    <geom name="tray" type="box" pos="0.22 0.16 0.002" size="0.08 0.08 0.004" rgba="0.10 0.78 0.38 1"/>
    <geom name="obstacle" type="cylinder" pos="0.12 -0.02 0.09" size="0.045 0.09" rgba="0.92 0.55 0.12 1"/>

    <body name="eef" pos="-0.15 -0.15 0.24" mocap="true">
      <geom name="eef_body" type="sphere" size="0.032" rgba="0.79 1.0 0.23 1" contype="0" conaffinity="0"/>
    </body>

    <body name="cube" pos="-0.12 -0.12 0.03">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="0.028 0.028 0.028" density="450" rgba="0.92 0.22 0.22 1"/>
    </body>

    <camera name="overhead" pos="0.0 0.0 0.92" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""


@dataclass
class ArenaTaskConfig:
    table_z: float = 0.0
    cube_half_extent: float = 0.028
    goal_pos: Tuple[float, float, float] = (0.22, 0.16, 0.03)
    obstacle_pos: Tuple[float, float, float] = (0.12, -0.02, 0.09)
    obstacle_radius: float = 0.045
    success_radius: float = 0.055
    max_steps: int = 80
    image_size: int = 84


@dataclass
class ArenaObservation:
    frame: np.ndarray
    eef_pose: List[float]
    cube_pos: List[float]
    goal_pos: List[float]
    obstacle_pos: List[float]
    grasped: bool
    step_count: int


class SwizzleManipulationArena:
    """
    Physical manipulation environment driving either live `mujoco` physics or
    a deterministic 3D rigid-body contact engine with top-down RGB rendering.
    """

    def __init__(self, config: Optional[ArenaTaskConfig] = None, prefer_mujoco: bool = True):
        self.config = config or ArenaTaskConfig()
        self._mujoco = None
        self._mj_model = None
        self._mj_data = None
        self._mj_renderer = None
        self.backend_mode = "numpy-rigid-body"

        if prefer_mujoco:
            self._try_init_mujoco()

        self.eef_pose = np.zeros(6, dtype=np.float64)
        self.cube_pos = np.zeros(3, dtype=np.float64)
        self.cube_vel = np.zeros(3, dtype=np.float64)
        self.prev_delta = np.zeros(3, dtype=np.float64)
        self.grasped = False
        self.step_count = 0

    def _try_init_mujoco(self) -> None:
        try:
            import mujoco  # type: ignore

            model = mujoco.MjModel.from_xml_string(ARENA_MJCF_XML)
            data = mujoco.MjData(model)
            self._mujoco = mujoco
            self._mj_model = model
            self._mj_data = data
            try:
                sz = self.config.image_size
                self._mj_renderer = mujoco.Renderer(model, height=sz, width=sz)
            except Exception:
                self._mj_renderer = None
            self.backend_mode = "mujoco"
        except Exception:
            self._mujoco = None
            self.backend_mode = "numpy-rigid-body"

    def close(self) -> None:
        if self._mj_renderer is not None:
            try:
                self._mj_renderer.close()
            except Exception:
                pass
            self._mj_renderer = None

    def reset(self, seed: int = 42) -> ArenaObservation:
        rng = np.random.default_rng(seed)
        jitter = rng.uniform(-0.025, 0.025, size=2)
        self.eef_pose = np.array([-0.14, -0.14, 0.22, 0.0, 0.0, 0.0], dtype=np.float64)
        self.cube_pos = np.array([-0.14 + jitter[0], -0.14 + jitter[1], self.config.cube_half_extent], dtype=np.float64)
        self.cube_vel = np.zeros(3, dtype=np.float64)
        self.prev_delta = np.zeros(3, dtype=np.float64)
        self.grasped = False
        self.step_count = 0

        if self._mujoco is not None and self._mj_model is not None and self._mj_data is not None:
            self._mujoco.mj_resetData(self._mj_model, self._mj_data)
            self._mj_data.mocap_pos[0] = self.eef_pose[:3]
            self._mj_data.qpos[:3] = self.cube_pos
            self._mujoco.mj_forward(self._mj_model, self._mj_data)

        return self._observe()

    def step(self, action_7dof: Sequence[float]) -> Tuple[ArenaObservation, float, bool, Dict[str, Any]]:
        act = np.array([float(x) for x in action_7dof[:7]], dtype=np.float64)
        if act.shape[0] < 7:
            act = np.pad(act, (0, 7 - act.shape[0]))

        delta_pos = act[:3] * 0.12
        gripper_closed = act[6] > 0.25

        step_accel = float(np.linalg.norm(delta_pos - self.prev_delta))
        self.prev_delta = delta_pos.copy()

        self.eef_pose[:3] += delta_pos
        self.eef_pose[3:6] += act[3:6] * 0.08

        # Rigid body contact & grasping update
        if self._mujoco is not None and self._mj_model is not None and self._mj_data is not None:
            self._mj_data.mocap_pos[0] = self.eef_pose[:3]
            for _ in range(4):
                self._mujoco.mj_step(self._mj_model, self._mj_data)
            self.cube_pos = np.array(self._mj_data.qpos[:3], dtype=np.float64)
            if gripper_closed and float(np.linalg.norm(self.eef_pose[:3] - self.cube_pos)) < 0.095:
                self.grasped = True
                self.cube_pos = self.eef_pose[:3] + np.array([0.0, 0.0, -0.02])
                self._mj_data.qpos[:3] = self.cube_pos
                self._mj_data.qvel[:6] = 0.0
                self._mujoco.mj_forward(self._mj_model, self._mj_data)
            else:
                self.grasped = False
        else:
            dist_to_cube = float(np.linalg.norm(self.eef_pose[:3] - self.cube_pos))
            if gripper_closed and (self.grasped or dist_to_cube < 0.095):
                self.grasped = True
                self.cube_pos = self.eef_pose[:3] + np.array([0.0, 0.0, -0.02])
                self.cube_vel[:] = 0.0
            else:
                self.grasped = False
                # Free gravity & table contact
                if self.cube_pos[2] > self.config.cube_half_extent:
                    self.cube_vel[2] -= 9.81 * 0.02
                    self.cube_pos += self.cube_vel * 0.02
                if self.cube_pos[2] <= self.config.cube_half_extent:
                    self.cube_pos[2] = self.config.cube_half_extent
                    self.cube_vel[:] = 0.0

        table_penetration_mm = max(0.0, (self.config.table_z - float(self.eef_pose[2])) * 1000.0)
        obs_c = np.array(self.config.obstacle_pos, dtype=np.float64)
        obs_dist = float(np.linalg.norm(self.eef_pose[:3] - obs_c))
        obstacle_collision = obs_dist < (self.config.obstacle_radius + 0.032)

        goal = np.array(self.config.goal_pos, dtype=np.float64)
        cube_goal_dist = float(np.linalg.norm(self.cube_pos - goal))
        success = cube_goal_dist <= self.config.success_radius and not obstacle_collision and table_penetration_mm == 0.0

        self.step_count += 1
        done = success or self.step_count >= self.config.max_steps
        reward = -cube_goal_dist - (2.0 if obstacle_collision else 0.0) + (5.0 if success else 0.0)

        info = {
            "backend_mode": self.backend_mode,
            "cube_to_goal_dist": round(cube_goal_dist, 4),
            "table_penetration_mm": round(table_penetration_mm, 3),
            "obstacle_collision": obstacle_collision,
            "step_jerk_delta": round(step_accel, 4),
            "grasped": self.grasped,
            "success": success,
        }
        return self._observe(), round(reward, 4), done, info

    def _observe(self) -> ArenaObservation:
        if self._mj_renderer is not None and self._mj_data is not None:
            self._mj_renderer.update_scene(self._mj_data, camera="overhead")
            frame = self._mj_renderer.render().copy()
        else:
            frame = self._render_synthetic_overhead()
        return ArenaObservation(
            frame=frame,
            eef_pose=[round(float(x), 5) for x in self.eef_pose],
            cube_pos=[round(float(x), 5) for x in self.cube_pos],
            goal_pos=list(self.config.goal_pos),
            obstacle_pos=list(self.config.obstacle_pos),
            grasped=self.grasped,
            step_count=self.step_count,
        )

    def _render_synthetic_overhead(self) -> np.ndarray:
        """Renders an overhead tabletop camera view showing goal tray, obstacle, cube, and EEF."""
        sz = self.config.image_size
        img = np.full((sz, sz, 3), fill_value=36, dtype=np.uint8)
        xs = np.linspace(-0.45, 0.45, sz)
        ys = np.linspace(-0.45, 0.45, sz)
        gx, gy = np.meshgrid(xs, ys)

        # Green goal tray
        tr_mask = (np.abs(gx - self.config.goal_pos[0]) < 0.08) & (np.abs(gy - self.config.goal_pos[1]) < 0.08)
        img[tr_mask] = [28, 195, 96]

        # Amber obstacle cylinder
        ob_mask = ((gx - self.config.obstacle_pos[0]) ** 2 + (gy - self.config.obstacle_pos[1]) ** 2) < (self.config.obstacle_radius ** 2)
        img[ob_mask] = [235, 140, 30]

        # Red target cube
        cu_mask = (np.abs(gx - self.cube_pos[0]) < 0.032) & (np.abs(gy - self.cube_pos[1]) < 0.032)
        img[cu_mask] = [235, 56, 56]

        # Swizzle Lime end-effector marker
        ef_mask = ((gx - self.eef_pose[0]) ** 2 + (gy - self.eef_pose[1]) ** 2) < (0.030 ** 2)
        img[ef_mask] = [201, 255, 59]
        return img
