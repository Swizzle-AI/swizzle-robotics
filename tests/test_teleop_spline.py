import unittest
from swizzle_robotics.core.teleop import QuinticSplineInterpolator, TemporalActionEnsembler

class TestTeleopSpline(unittest.TestCase):
    def test_quintic_interpolation(self):
        p0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        p1 = [0.5, 0.2, -0.1, 0.0, 0.0, 0.0, 1.0]
        substeps = QuinticSplineInterpolator.interpolate(p0, p1, substeps=4)
        self.assertEqual(len(substeps), 4)
        for s in substeps:
            self.assertEqual(len(s), 7)
        # Final substep should approximate p1
        self.assertAlmostEqual(substeps[-1][0], p1[0], delta=0.05)

    def test_temporal_ensembler(self):
        ensembler = TemporalActionEnsembler(decay_lambda=0.05, substeps=4)
        chunk1 = [[0.1 * i, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for i in range(16)]
        act1, subs1 = ensembler.push_chunk(1, chunk1)
        self.assertEqual(len(act1), 7)
        self.assertEqual(len(subs1), 4)

        chunk2 = [[0.1 * (i + 1), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for i in range(16)]
        act2, subs2 = ensembler.push_chunk(2, chunk2)
        self.assertEqual(len(act2), 7)
        self.assertEqual(len(subs2), 4)

if __name__ == '__main__':
    unittest.main()
