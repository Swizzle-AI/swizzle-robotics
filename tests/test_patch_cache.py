import unittest
import numpy as np
from swizzle_robotics.core.patch_cache import PatchCacheFilter, extract_patch_intensities_from_frame

class TestPatchCache(unittest.TestCase):
    def test_extract_from_sequence(self):
        seq = [0.5] * 196
        res = extract_patch_intensities_from_frame(seq, grid_size=14)
        self.assertEqual(len(res), 196)
        self.assertEqual(res[0], 0.5)

    def test_extract_from_ndarray(self):
        arr = np.ones((224, 224, 3), dtype=np.uint8) * 128
        res = extract_patch_intensities_from_frame(arr, grid_size=14)
        self.assertEqual(len(res), 196)
        self.assertAlmostEqual(res[0], 128 / 255.0, places=2)

    def test_patch_reuse_evaluation(self):
        f = PatchCacheFilter(grid_size=14, delta_threshold=0.03)
        # Frame 1: all 0.5
        d1 = f.evaluate([0.5] * 196)
        self.assertEqual(d1.reused_patches, 0)
        self.assertEqual(d1.total_patches, 196)

        # Frame 2: 90% unchanged, 10% changed
        f2 = [0.5] * 196
        for i in range(20):
            f2[i] = 0.9  # Moved
        d2 = f.evaluate(f2)
        self.assertEqual(d2.reused_patches, 176)
        self.assertEqual(d2.recomputed_patches, 20)
        self.assertAlmostEqual(d2.patch_hit_ratio, 176 / 196.0, places=3)
        self.assertGreater(d2.prefill_speedup, 3.0)

if __name__ == '__main__':
    unittest.main()
