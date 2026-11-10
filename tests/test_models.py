import unittest
import numpy as np
from swizzle_robotics.models import (
    CosmosWorldModel,
    Gr00tPolicy,
    SmolVLAPolicy,
    OpenVLAPolicy,
    Pi0Policy,
    VJEPAWorldModel,
    get_policy,
    get_world_model,
)

class TestModels(unittest.TestCase):
    def setUp(self):
        self.dummy_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    def test_cosmos_world_model(self):
        wm = CosmosWorldModel()
        z0 = wm.encode_observation(self.dummy_frame)
        self.assertEqual(len(z0), 64)

        z1 = wm.predict_next_state(z0, [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(len(z1), 64)

        rollout = wm.rollout(z0, [[0.1] * 7] * 5)
        self.assertEqual(rollout.action_horizon, 5)
        self.assertEqual(len(rollout.predicted_latent_states), 5)

    def test_cosmos_cem_planner(self):
        wm = CosmosWorldModel()
        curr = np.zeros((224, 224, 3), dtype=np.uint8)
        goal = np.ones((224, 224, 3), dtype=np.uint8) * 200
        best_act, full_traj, latency = wm.plan_cem(curr, goal, horizon=4, num_samples=16)
        self.assertEqual(len(best_act), 7)
        self.assertEqual(len(full_traj), 4)
        self.assertLess(latency, 200.0)

    def test_gr00t_policy(self):
        policy = Gr00tPolicy()
        act, subs, pc, spec = policy.step(self.dummy_frame, "pick red cube")
        self.assertEqual(len(act), 7)
        self.assertEqual(len(subs), 4)
        self.assertGreaterEqual(pc.patch_hit_ratio, 0.0)

    def test_smol_vla(self):
        policy = SmolVLAPolicy()
        act, subs, pc, spec = policy.step(self.dummy_frame, "grasp object")
        self.assertEqual(len(act), 7)

    def test_openvla(self):
        policy = OpenVLAPolicy()
        act, subs, pc, spec = policy.step(self.dummy_frame, "lift object")
        self.assertEqual(len(act), 7)

    def test_pi0(self):
        policy = Pi0Policy()
        act, subs, pc, spec = policy.step(self.dummy_frame, "push object")
        self.assertEqual(len(act), 7)

    def test_vjepa_wm(self):
        wm = VJEPAWorldModel()
        z = wm.encode_observation(self.dummy_frame)
        self.assertEqual(len(z), 64)

if __name__ == '__main__':
    unittest.main()
