"""
Connect a Physical Robot (or Edge Controller) to the Swizzle Real-Life Arena
============================================================================
Usage:
    # 1. Start the Real-Life Arena Backend:
    swizzle-robotics arena-server --port 8787

    # 2. Connect your physical robot (replace callbacks with Franka/UR5e/SO-100 SDK):
    python examples/connect_real_robot.py --robot-id franka-cell-01 --url ws://127.0.0.1:8787
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import websockets


async def run_robot_bridge(url: str, robot_id: str, hz: float = 30.0) -> None:
    ws_url = f"{url.rstrip('/')}/ws/robot/{robot_id}"
    print(f"Connecting robot '{robot_id}' -> {ws_url}")
    pose = [-0.28, -0.14, 0.22, 0.0, 0.0, 0.0]

    async with websockets.connect(ws_url) as ws:
        print("Connected. Streaming proprioception & receiving CBF-safe actuation commands.")
        dt = 1.0 / max(1.0, hz)
        while True:
            payload = {
                "name": f"Physical Arm ({robot_id})",
                "dof": 7,
                "pose": pose,
                "target": [0.24, 0.16, 0.14],
                "heartbeat_ok": True,
                "prompt": "pick up part avoiding obstacle pillar",
            }
            await ws.send(json.dumps(payload))
            reply = json.loads(await ws.recv())

            if reply.get("command_type") == "params_update":
                print("Live parameter update pushed from Studio:", reply["params"]["cbf_alpha"])
                continue

            if reply.get("estop_active"):
                print("[ESTOP] Motor commands zeroed by operator!")
                await asyncio.sleep(dt)
                continue

            safe_act = reply["safe_action"]
            # Apply safe velocity delta onto real motors here (simulated Euler step below):
            pose[0] += safe_act[0] * dt * 0.35
            pose[1] += safe_act[1] * dt * 0.35
            pose[2] = max(0.03, pose[2] + safe_act[2] * dt * 0.35)
            await asyncio.sleep(dt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real Robot -> Swizzle Arena WebSocket Bridge")
    parser.add_argument("--robot-id", default="franka-cell-01")
    parser.add_argument("--url", default="ws://127.0.0.1:8787")
    parser.add_argument("--hz", type=float, default=30.0)
    args = parser.parse_args()
    asyncio.run(run_robot_bridge(args.url, args.robot_id, hz=args.hz))


if __name__ == "__main__":
    main()
