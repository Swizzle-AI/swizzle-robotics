import unittest
from swizzle_robotics.core.spec_act import KinematicSpecActPredictor
from swizzle_robotics.types import quantize_action_to_token, dequantize_token_to_action

class TestSpecAct(unittest.TestCase):
    def test_quantization_roundtrip(self):
        for orig in [-1.0, -0.5, 0.0, 0.25, 0.75, 1.0]:
            tok = quantize_action_to_token(orig)
            self.assertTrue(31744 <= tok <= 32000)
            back = dequantize_token_to_action(tok)
            self.assertAlmostEqual(orig, back, delta=0.02)

    def test_kinematic_draft_verification(self):
        predictor = KinematicSpecActPredictor(bin_tolerance=2)
        # Sequence of continuous actions
        t1 = [0.1, 0.2, -0.3, 0.0, 0.0, 0.0, 1.0]
        t2 = [0.15, 0.25, -0.28, 0.0, 0.0, 0.0, 1.0]
        t3 = [0.20, 0.30, -0.26, 0.0, 0.0, 0.0, 1.0]

        res1 = predictor.draft_and_verify(t1)
        res2 = predictor.draft_and_verify(t2)
        res3 = predictor.draft_and_verify(t3)

        self.assertEqual(len(res3.drafted_tokens), 7)
        self.assertEqual(len(res3.verified_tokens), 7)
        self.assertGreaterEqual(res3.accepted_dof_tokens, 5)
        self.assertGreater(res3.decoding_speedup, 1.5)

if __name__ == '__main__':
    unittest.main()
