import unittest
import numpy as np
from swizzle_robotics.servoing import VisualServoingLoop, ServoingConfig
from swizzle_robotics.world_model import LatentWorldModel, MPCPlanner
from swizzle_robotics.spline import QuinticSplineTrajectory

class TestRobotics(unittest.TestCase):
    def test_luminance_patch_extraction(self):
        loop = VisualServoingLoop(ServoingConfig(target_hz=80.5))
        dummy_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        patches = loop.extract_patch_luminance(dummy_frame)
        self.assertEqual(patches.shape, (14, 14))
        self.assertTrue(0.0 <= patches.min() and patches.max() <= 1.0)

    def test_servoing_step(self):
        loop = VisualServoingLoop()
        frame = np.ones((224, 224, 3), dtype=np.uint8) * 128
        goal = np.ones((14, 14), dtype=np.float32) * 0.5
        delta, latency = loop.step(frame, goal)
        self.assertEqual(len(delta), 6)
        self.assertLess(latency, 20.0)

    def test_quintic_spline(self):
        t, pos, vel, acc = QuinticSplineTrajectory.generate(0, 0, 0, 1.0, 0, 0, T=1.0, steps=50)
        self.assertAlmostEqual(pos[0], 0.0, places=4)
        self.assertAlmostEqual(pos[-1], 1.0, places=4)
        self.assertAlmostEqual(vel[0], 0.0, places=4)
        self.assertAlmostEqual(vel[-1], 0.0, places=4)

if __name__ == '__main__':
    unittest.main()
