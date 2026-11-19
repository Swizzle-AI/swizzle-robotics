import time
import unittest
from swizzle_robotics.core.camera import MockCamera
from swizzle_robotics.core.hierarchical import HierarchicalController
from swizzle_robotics.core.safety import SafetyConfig
from swizzle_robotics.core.zenoh_bridge import ZenohBridgeConfig, ZenohRoboticsBridge
from swizzle_robotics.models.smol_vla import SmolVLAPolicy


class TestHierarchicalAndZenoh(unittest.TestCase):
    def test_s2_grounded_planning_injects_cbf_obstacles_and_runs_s1_s0(self):
        ctrl = HierarchicalController(policy=SmolVLAPolicy(), safety_config=SafetyConfig(max_accel_step=0.25))
        plan = ctrl.plan_s2(
            "pick up mug carefully while avoiding soldering_iron then place on tray",
            known_objects={"soldering_iron": (0.12, -0.05, 0.25, 0.08)},
        )
        self.assertEqual(len(plan.subtasks), 2)
        self.assertEqual(plan.max_speed_scale, 0.5)
        self.assertTrue(any(o.name == "soldering_iron" for o in ctrl.safety_shield.config.obstacles))

        cam = MockCamera()
        _, frame = cam.read()
        tick = ctrl.step_s1_s0(frame, subtask_idx=0, trace_id="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
        self.assertTrue(tick.receipt.verify())
        self.assertEqual(len(tick.safe_action), 7)
        self.assertEqual(len(tick.substeps_50hz), 4)

    def test_zenoh_bridge_ring_buffer_and_heartbeat_monitoring(self):
        bridge = ZenohRoboticsBridge(ZenohBridgeConfig(queue_capacity=2, heartbeat_timeout_ms=50.0))
        received = []
        bridge.subscribe("rt/cmd_vel", lambda env: received.append(env.payload))

        bridge.publish("rt/cmd_vel", {"seq": 1}, traceparent="00-abc-123-01")
        bridge.publish("rt/cmd_vel", {"seq": 2}, traceparent="00-abc-123-01")
        bridge.publish("rt/cmd_vel", {"seq": 3}, traceparent="00-abc-123-01")

        self.assertEqual(len(received), 3)
        drained = bridge.drain("rt/cmd_vel")
        # Capacity=2 drops oldest (seq=1), keeping [2, 3]
        self.assertEqual([d.payload["seq"] for d in drained], [2, 3])
        self.assertTrue(bridge.heartbeat_healthy())
        bridge.heartbeat(now_ns=time.time_ns() - int(200e6))
        self.assertFalse(bridge.heartbeat_healthy())


if __name__ == "__main__":
    unittest.main()
