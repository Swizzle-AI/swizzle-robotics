from pathlib import Path
from fastapi.testclient import TestClient

from swizzle_robotics.arena_backend import RealLifeArenaEngine, create_arena_app


def _make_client():
    engine = RealLifeArenaEngine(prefer_mujoco=False)
    app = create_arena_app(engine=engine, enable_background_loop=False)
    return TestClient(app), engine


def test_arena_state_and_hot_param_tuning():
    client, engine = _make_client()

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    state = client.get("/api/arena/state")
    assert state.status_code == 200
    data = state.json()
    assert data["active_robot_id"] == "local-mujoco-hil"
    assert len(data["robots"]) >= 1

    # Hot-tune CBF + PatchCache + RTC parameters live
    patch = client.patch(
        "/api/arena/params",
        json={
            "cbf_alpha": 0.42,
            "max_linear_step": 0.15,
            "patch_delta_threshold": 0.045,
            "decay_lambda": 0.12,
            "rtc_substeps": 6,
            "action_gain": 0.85,
        },
    )
    assert patch.status_code == 200
    tuned = patch.json()
    assert tuned["cbf_alpha"] == 0.42
    assert tuned["max_linear_step"] == 0.15
    assert engine.shield.config.cbf_alpha == 0.42
    assert engine.policy.patch_cache.delta_threshold == 0.045
    assert engine.policy.teleop.substeps == 6


def test_arena_obstacles_control_estop_and_export(tmp_path: Path):
    client, engine = _make_client()

    # Add & remove keep-out zone dynamically
    add_obs = client.post(
        "/api/arena/obstacles",
        json={"name": "human_hand", "center": [0.12, -0.08, 0.20], "radius": 0.09},
    )
    assert add_obs.status_code == 200
    names = [o["name"] for o in add_obs.json()["obstacles"]]
    assert "human_hand" in names
    assert any(o.name == "human_hand" for o in engine.shield.config.obstacles)

    # Step closed-loop simulation and verify recorded steps + receipts
    for _ in range(5):
        tick = engine.step_tick()
        assert tick["receipt_digest"]
        assert tick["min_barrier_margin"] is not None

    # Engage hard E-STOP -> zeroes commands even under large raw proposal
    estop = client.post("/api/arena/control", json={"command": "estop"})
    assert estop.status_code == 200
    assert estop.json()["estop_active"] is True

    # Cannot resume until cleared
    denied = client.post("/api/arena/control", json={"command": "start"})
    assert denied.status_code == 409

    tick_while_estop = engine.step_tick()
    assert tick_while_estop["safe_action"] == [0.0] * 7
    assert tick_while_estop["estop_active"] is True

    cleared = client.post("/api/arena/control", json={"command": "clear_estop"})
    assert cleared.status_code == 200
    assert cleared.json()["estop_active"] is False

    # Delete custom obstacle
    del_obs = client.delete("/api/arena/obstacles/human_hand")
    assert del_obs.status_code == 200
    assert "human_hand" not in [o["name"] for o in del_obs.json()["obstacles"]]

    # Export .mcap + LeRobot v2.1 + swizzle_deploy_profile.json
    export = client.post(
        "/api/arena/export",
        json={"out_dir": str(tmp_path), "episode_label": "ci_verify"},
    )
    assert export.status_code == 200
    exp = export.json()
    assert exp["mcap_bytes"] > 100
    assert Path(exp["mcap_path"]).is_file()
    assert Path(exp["profile_path"]).is_file()
    assert exp["deploy_profile"]["tuned_params"]["cbf_alpha"] == 0.65


def test_real_robot_websocket_connection_and_live_tuning():
    client, _engine = _make_client()

    # Register physical robot metadata first
    reg = client.post(
        "/api/arena/robots/connect",
        json={
            "robot_id": "franka-lab-01",
            "name": "Franka FR3 Lab Cell",
            "adapter": "franka_fci",
            "endpoint": "tcp://192.168.1.50:7447",
            "dof": 7,
        },
    )
    assert reg.status_code == 200
    assert reg.json()["ws_ingest_path"] == "/ws/robot/franka-lab-01"

    # Connect external physical robot over WebSocket & test CBF shield projection
    with client.websocket_connect("/ws/robot/franka-lab-01") as robot_ws:
        # Send aggressive impulse moving directly toward pillar obstacle at (0, 0, 0.24)
        robot_ws.send_json(
            {
                "name": "Franka FR3 Lab Cell",
                "pose": [-0.11, 0.0, 0.24, 0.0, 0.0, 0.0],
                "raw_action": [0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "heartbeat_ok": True,
            }
        )
        reply = robot_ws.receive_json()
        assert reply["command_type"] == "safe_actuation"
        assert reply["cbf_intervened"] is True
        assert abs(reply["safe_action"][0]) < 0.95
        assert reply["receipt_digest"].startswith(("sha256:", "blake3:"))

    # Verify studio WS receives state sync cleanly
    with client.websocket_connect("/ws/studio") as studio_ws:
        init_msg = studio_ws.receive_json()
        assert init_msg["type"] == "state_sync"
        studio_ws.send_json({"type": "patch_params", "params": {"cbf_alpha": 0.35}})
        sync_msg = studio_ws.receive_json()
        assert sync_msg["snapshot"]["params"]["cbf_alpha"] == 0.35
