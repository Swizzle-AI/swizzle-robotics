import unittest
import numpy as np
from swizzle_robotics.core.camera import MockCamera
from swizzle_robotics.core.teleop import TemporalActionEnsembler
from swizzle_robotics.models.cosmos_wm import CosmosWorldModel
from swizzle_robotics.models.vjepa_wm import VJEPAWorldModel


class TestRTCAndBatchedWorldModel(unittest.TestCase):
    def test_rtc_soft_mask_schedule_freezes_delay_prefix(self):
        weights = TemporalActionEnsembler.rtc_soft_mask_weights(
            execution_delay=2, overlap_horizon=4, chunk_horizon=10
        )
        self.assertEqual(len(weights), 10)
        self.assertEqual(weights[0], 1.0)
        self.assertEqual(weights[1], 1.0)
        self.assertTrue(1.0 >= weights[2] > weights[3] > weights[4] > weights[5] > 0.0)
        self.assertEqual(weights[6:], [0.0, 0.0, 0.0, 0.0])

    def test_rtc_inpainting_prevents_boundary_step_jump(self):
        ens = TemporalActionEnsembler(decay_lambda=0.05, substeps=4)
        chunk1 = [[0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for _ in range(12)]
        ens.push_chunk(0, chunk1)

        # Delayed chunk arrives at step 3 predicting a sudden reversal -0.80
        chunk2 = [[-0.80, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for _ in range(10)]
        blended, substeps = ens.push_rtc_chunk(3, chunk2, execution_delay=2, overlap_horizon=4)

        # First 2 steps are frozen from chunk1 (+0.10), transitioning smoothly toward -0.80
        self.assertAlmostEqual(blended[0][0], 0.10, places=3)
        self.assertAlmostEqual(blended[1][0], 0.10, places=3)
        self.assertGreater(blended[2][0], -0.40)
        self.assertAlmostEqual(blended[-1][0], -0.80, places=3)
        self.assertEqual(len(substeps), 4)

    def test_batched_world_model_cem_matches_single_dynamics(self):
        cam = MockCamera()
        _, f0 = cam.read()
        cam.frame_idx = 12
        _, f1 = cam.read()

        for wm in (CosmosWorldModel(), VJEPAWorldModel()):
            z = np.random.randn(4, wm.latent_dim).astype(np.float32)
            a = np.random.randn(4, 7).astype(np.float32) * 0.2
            batched = wm.predict_next_state_batch(z, a)
            serial = np.stack([wm.predict_next_state(z[i], a[i]) for i in range(4)], axis=0)
            np.testing.assert_allclose(batched, serial, atol=1e-5)

            act, traj, lat_ms = wm.plan_cem_batched(f0, f1, horizon=5, num_samples=128, iterations=3)
            self.assertEqual(len(act), 7)
            self.assertEqual(len(traj), 5)
            self.assertLess(lat_ms, 500.0)


if __name__ == "__main__":
    unittest.main()
