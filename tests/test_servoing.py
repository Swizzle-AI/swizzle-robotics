import unittest
from swizzle_robotics.servoing import VisualServoingLoop
from swizzle_robotics.core.camera import MockCamera
from swizzle_robotics.models.smol_vla import SmolVLAPolicy

class TestServoing(unittest.TestCase):
    def test_servoing_loop(self):
        cam = MockCamera(fps=60.0)
        policy = SmolVLAPolicy()
        loop = VisualServoingLoop(policy=policy, camera=cam, target_hz=80.5)

        results = loop.run_n_steps(n=10, prompt="Pick up red block")
        self.assertEqual(len(results), 10)
        for r in results:
            self.assertEqual(len(r.action_7dof), 7)
            self.assertGreater(r.effective_hz, 10.0)
            self.assertGreaterEqual(r.patch_hit_ratio, 0.0)

if __name__ == '__main__':
    unittest.main()
