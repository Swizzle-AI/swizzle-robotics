import tempfile
import unittest
from pathlib import Path
import numpy as np

from swizzle_robotics.core.camera import MockCamera
from swizzle_robotics.models import get_policy
from swizzle_robotics.models.edge_flow_runner import HAS_TORCH, TorchEdgeVLAPolicy


class TestEdgeFlowRunner(unittest.TestCase):
    def test_sparse_patchcache_flow_runner_and_safetensors_roundtrip(self):
        policy = get_policy("swizzle/edge-flow-vla")
        self.assertIsInstance(policy, TorchEdgeVLAPolicy)

        cam = MockCamera(width=84, height=84)
        _, frame0 = cam.read()
        chunk0 = policy.forward_action_chunk(frame0, "pick up red cube", horizon=16)
        self.assertEqual(chunk0.horizon, 16)
        self.assertEqual(len(chunk0.actions[0]), 7)

        # Evaluate sparse token update when only 4 of 196 patches are dirty
        reuse_mask = [i >= 4 for i in range(196)]
        _, frame1 = cam.read()
        chunk1 = policy.forward_action_chunk(frame1, "pick up red cube", horizon=16, reuse_mask=reuse_mask)
        self.assertEqual(len(chunk1.actions), 16)

        with tempfile.TemporaryDirectory() as tmp:
            ext = "safetensors" if HAS_TORCH else "npz"
            ckpt = policy.save_safetensors(Path(tmp) / f"swizzle_edge_vla.{ext}")
            self.assertTrue(ckpt.exists())
            reloaded = TorchEdgeVLAPolicy()
            reloaded.load_safetensors(ckpt)
            chunk_reloaded = reloaded.forward_action_chunk(frame0, "pick up red cube", horizon=16)
            np.testing.assert_allclose(chunk0.actions, chunk_reloaded.actions, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
