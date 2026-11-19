import tempfile
import unittest
from pathlib import Path
import numpy as np

from swizzle_robotics.core.hierarchical import HierarchicalController
from swizzle_robotics.core.safety import SafetyConfig
from swizzle_robotics.envs import ArenaTaskConfig, SwizzleManipulationArena
from swizzle_robotics.models.smol_vla import SmolVLAPolicy


class TestMuJoCoArena(unittest.TestCase):
    def test_arena_closed_loop_with_cbf_and_recorder(self):
        cfg = ArenaTaskConfig(goal_pos=(0.22, 0.16, 0.03), obstacle_pos=(0.04, 0.02, 0.09), obstacle_radius=0.05)
        arena = SwizzleManipulationArena(cfg, prefer_mujoco=True)
        obs = arena.reset(seed=7)
        self.assertEqual(obs.frame.shape, (84, 84, 3))

        safety_cfg = SafetyConfig(
            z_bounds=(0.02, 0.65),
            max_linear_step=0.22,
            max_accel_step=0.14,
            cbf_alpha=0.55,
        )
        ctrl = HierarchicalController(policy=SmolVLAPolicy(), safety_config=safety_cfg, episode_id="ep_arena_01")
        plan = ctrl.plan_s2(
            "pick up red cube while avoiding pillar then place on green tray",
            known_objects={"pillar": (*cfg.obstacle_pos, cfg.obstacle_radius)},
        )
        self.assertEqual(len(plan.avoid_obstacles), 1)

        # Drive proportional goal-reaching trajectory inside arena while filtered by S0 CBF + Recorder
        ctrl.pose = list(obs.eef_pose)
        for step in range(30):
            target = np.array(obs.cube_pos if not obs.grasped else obs.goal_pos)
            err = (target - np.array(obs.eef_pose[:3])) * 4.0
            grip = 1.0 if float(np.linalg.norm(target - np.array(obs.eef_pose[:3]))) < 0.09 or obs.grasped else -1.0
            raw_cmd = [float(err[0]), float(err[1]), float(err[2]), 0.0, 0.0, 0.0, grip]

            sec = ctrl.safety_shield.project(raw_cmd, obs.eef_pose)
            obs, reward, done, info = arena.step(sec.safe_action)
            rec = ctrl.recorder.record_step(obs.frame, obs.eef_pose, sec, prompt=plan.subtasks[0], trace_id="arena-test")
            self.assertTrue(rec.verify())
            self.assertEqual(info["table_penetration_mm"], 0.0)
            self.assertFalse(info["obstacle_collision"])
            if done:
                break

        with tempfile.TemporaryDirectory() as tmp:
            mcap_file = ctrl.recorder.export_mcap(Path(tmp) / "arena_ep.mcap")
            self.assertGreater(mcap_file.stat().st_size, 200)
            ds = ctrl.recorder.export_lerobot_v2(Path(tmp) / "arena_lerobot")
            self.assertGreater(ds["frames"], 0)
        arena.close()


if __name__ == "__main__":
    unittest.main()
