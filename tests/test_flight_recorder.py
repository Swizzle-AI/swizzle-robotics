import json
import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np

from swizzle_robotics.core.recorder import EmbodiedFlightRecorder, MCAP_MAGIC
from swizzle_robotics.core.safety import ControlBarrierShield, SafetyConfig


class TestFlightRecorder(unittest.TestCase):
    def test_mcap_binary_lerobot_v2_and_cryptographic_receipts(self):
        shield = ControlBarrierShield(SafetyConfig(z_bounds=(0.05, 0.70), max_accel_step=1.0, max_linear_step=1.0))
        recorder = EmbodiedFlightRecorder(episode_id="ep_test_001", model_id="lerobot/smolvla-450m")

        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        # Step 0: safe nominal step
        pose0 = [0.0, 0.0, 0.25, 0.0, 0.0, 0.0]
        sec0 = shield.project([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], pose0)
        rec0 = recorder.record_step(frame, pose0, sec0, prompt="pick block", trace_id="trace-1")
        self.assertTrue(rec0.verify())

        # Step 1: unsafe downward move that triggers CBF intervention & DAgger flag
        pose1 = [0.05, 0.0, 0.06, 0.0, 0.0, 0.0]
        sec1 = shield.project([0.0, 0.0, -0.50, 0.0, 0.0, 0.0, 1.0], pose1)
        rec1 = recorder.record_step(frame, pose1, sec1, prompt="pick block", trace_id="trace-1")
        self.assertTrue(rec1.verify())
        self.assertTrue(rec1.cbf_intervened)
        self.assertTrue(recorder.dagger_flagged)
        self.assertEqual(recorder.intervention_count, 1)

        # Tampering with safe_action breaks cryptographic verification
        rec1.safe_action[2] = -0.99
        self.assertFalse(rec1.verify())
        rec1.safe_action[2] = sec1.safe_action[2]
        self.assertTrue(rec1.verify())

        with tempfile.TemporaryDirectory() as tmp:
            mcap_path = recorder.export_mcap(Path(tmp) / "ep_000000.mcap")
            raw_mcap = mcap_path.read_bytes()
            self.assertTrue(raw_mcap.startswith(MCAP_MAGIC))
            self.assertTrue(raw_mcap.endswith(MCAP_MAGIC))
            # Check opcode 0x01 (Header) immediately after 8-byte MCAP magic
            self.assertEqual(struct.unpack("<B", raw_mcap[8:9])[0], 0x01)
            self.assertIn(b"/swizzle/receipt", raw_mcap)
            self.assertIn(b"/action/cbf_safe", raw_mcap)

            ds_info = recorder.export_lerobot_v2(Path(tmp) / "lerobot_ds", episode_index=0, fps=50)
            self.assertTrue(ds_info["dagger_flagged"])
            self.assertEqual(ds_info["frames"], 2)
            info_json = json.loads((Path(tmp) / "lerobot_ds" / "meta" / "info.json").read_text())
            self.assertEqual(info_json["codebase_version"], "v2.1")
            self.assertEqual(info_json["total_frames"], 2)
            stats_json = json.loads((Path(tmp) / "lerobot_ds" / "meta" / "stats.json").read_text())
            self.assertEqual(len(stats_json["action"]["mean"]), 7)


if __name__ == "__main__":
    unittest.main()
