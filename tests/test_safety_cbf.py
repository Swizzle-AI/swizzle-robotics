import unittest
import numpy as np
from swizzle_robotics.core.safety import ControlBarrierShield, ObstacleSphere, SafetyConfig


class TestSafetyCBF(unittest.TestCase):
    def test_workspace_geofence_prevents_table_crash(self):
        cfg = SafetyConfig(z_bounds=(0.05, 0.70), max_accel_step=1.0, max_linear_step=1.0, cbf_alpha=0.5)
        shield = ControlBarrierShield(cfg)
        # End-effector close to table (z = 0.06, margin = 0.01) commanded downward -0.50
        res = shield.project([0.0, 0.0, -0.50, 0.0, 0.0, 0.0, 1.0], current_pose=(0.0, 0.0, 0.06, 0.0, 0.0, 0.0))
        self.assertTrue(res.cbf_intervened)
        self.assertIn("dz", res.clamped_axes)
        self.assertGreaterEqual(0.06 + res.safe_action[2], 0.05)
        self.assertLess(res.solver_latency_ms, 5.0)

    def test_spherical_obstacle_repulsion_keeps_margin(self):
        obs = ObstacleSphere(center=(0.20, 0.0, 0.25), radius=0.08, name="human_hand")
        cfg = SafetyConfig(
            obstacles=[obs],
            eef_radius=0.04,
            max_accel_step=1.0,
            max_linear_step=1.0,
            cbf_alpha=0.5,
        )
        shield = ControlBarrierShield(cfg)
        # Move directly toward obstacle along +x from x=0.07 (dist=0.13, r_safe=0.12, signed_dist=0.01)
        res = shield.project([0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], current_pose=(0.07, 0.0, 0.25, 0.0, 0.0, 0.0))
        self.assertTrue(res.cbf_intervened)
        self.assertIn("cbf_obstacle_human_hand", res.intervention_reasons)
        self.assertLessEqual(res.safe_action[0], 0.0055)

    def test_slew_rate_limiter_smooths_adversarial_spike(self):
        cfg = SafetyConfig(max_accel_step=0.05, max_linear_step=1.0)
        shield = ControlBarrierShield(cfg)
        res1 = shield.project([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(res1.safe_action[0], 0.02, places=4)
        # Adversarial sudden spike to +0.90
        res2 = shield.project([0.90, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(res2.cbf_intervened)
        self.assertIn("slew_rate_jerk_limit", res2.intervention_reasons)
        self.assertAlmostEqual(res2.safe_action[0], 0.07, places=4)

    def test_watchdog_deadman_decay_stops_robot_gracefully(self):
        cfg = SafetyConfig(watchdog_timeout_steps=4, max_accel_step=1.0)
        shield = ControlBarrierShield(cfg)
        shield.project([0.20, -0.10, 0.0, 0.0, 0.0, 0.0, 1.0])
        speeds = []
        for _ in range(4):
            res = shield.project([0.20, -0.10, 0.0, 0.0, 0.0, 0.0, 1.0], heartbeat_ok=False)
            self.assertTrue(res.watchdog_active)
            speeds.append(float(np.linalg.norm(res.safe_action[:3])))
        self.assertAlmostEqual(speeds[-1], 0.0, places=4)
        self.assertTrue(all(speeds[i] >= speeds[i + 1] for i in range(len(speeds) - 1)))


if __name__ == "__main__":
    unittest.main()
