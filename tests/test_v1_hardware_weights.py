from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from swizzle_robotics.arena_backend import RealLifeArenaEngine, create_arena_app
from swizzle_robotics.core.native_accel import (
    fast_cbf_halfspace_clamp,
    fast_patch_delta_mask,
    probe_native_runtime,
)
from swizzle_robotics.core.recorder import MCAP_MAGIC, parse_mcap_messages
from swizzle_robotics.drivers import (
    RobotCalibrationProfile,
    SO100SerialDriver,
    STS3215Protocol,
    URRTDEDriver,
)
from swizzle_robotics.models import TorchEdgeVLAPolicy, load_real_vla_checkpoint


def test_sts3215_protocol_and_so100_driver():
    pkt = STS3215Protocol.encode_goal_position(servo_id=1, tick=2048, speed=800)
    assert pkt.startswith(b"\xff\xff")
    assert pkt[2] == 1
    assert pkt[4] == STS3215Protocol.INSTR_WRITE

    # Build valid position read-reply frame (pos = 1500 -> 0xDC, 0x05)
    params = [1500 & 0xFF, (1500 >> 8) & 0xFF]
    length = len(params) + 2
    chk = STS3215Protocol.checksum(1, length, 0x00, params)
    reply = b"\xff\xff" + bytes([1, length, 0x00, *params, chk])
    assert STS3215Protocol.decode_position_reply(reply) == 1500

    with pytest.raises(ValueError):
        STS3215Protocol.decode_position_reply(reply[:-1] + bytes([chk ^ 0xFF]))

    cal = RobotCalibrationProfile(dof=6, min_rad=[-1.0] * 6, max_rad=[1.0] * 6)
    cal.record_zero([0.1, -0.1, 0.0, 0.0, 0.0, 0.0])
    assert cal.clamp_joints([2.5, -2.5, 0.2, 0.0, 0.0, 0.0])[:3] == [1.0, -1.0, 0.2]
    ticks = cal.rad_to_sts_ticks([0.1, -0.1, 0.0, 0.0, 0.0, 0.0])
    assert all(2040 <= t <= 2055 for t in ticks)

    so = SO100SerialDriver("serial://COM99@1000000", dof=6)
    so.connect()
    tel = so.send_safe_action([0.3, 0.2, -0.1, 0.0, 0.0, 0.0, 1.0])
    assert tel.connected is True
    assert tel.packets_sent == 6
    assert len(so.last_packets) == 6
    so.emergency_stop()
    so.close()

    ur = URRTDEDriver("rtde://127.0.0.1:30004", dof=7)
    ur_tel = ur.send_safe_action([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    assert ur_tel.packets_sent == 1


def test_native_accel_and_hf_vla_checkpoint(tmp_path: Path):
    status = probe_native_runtime()
    assert status.backend in ("rust-pyo3", "rust-capi", "rust-daemon", "numpy-vectorized")

    prev = np.full(196, 0.5, dtype=np.float32)
    curr = prev.copy()
    curr[10] = 0.9
    mask, reused = fast_patch_delta_mask(curr, prev, threshold=0.025)
    assert reused == 195
    assert bool(mask[10]) is False

    safe_d, min_margin, hits = fast_cbf_halfspace_clamp(
        delta_xyz=np.array([0.8, 0.0, 0.0]),
        eef_xyz=np.array([-0.11, 0.0, 0.24]),
        centers_xyz=np.array([[0.0, 0.0, 0.24]]),
        radii=np.array([0.10]),
        alpha=0.5,
    )
    assert min_margin < 0.02
    assert safe_d[0] < 0.8
    assert "obs_0" in hits

    weights_path = tmp_path / "edge_flow.safetensors"
    src = TorchEdgeVLAPolicy()
    src.save_safetensors(weights_path)
    loaded, ckpt_status = load_real_vla_checkpoint(str(weights_path), quantize_int8=True)
    assert ckpt_status.loaded_real_weights is True
    assert ckpt_status.sparse_vit_hooked is True
    act, _, _, _ = loaded.step(np.zeros((224, 224, 3), dtype=np.uint8), "pick up cube")
    assert len(act) == 7


def test_arena_calibration_checkpoint_mcap_download_and_replay():
    engine = RealLifeArenaEngine(prefer_mujoco=False)
    app = create_arena_app(engine=engine, enable_background_loop=False)
    client = TestClient(app)

    for _ in range(4):
        tick = engine.step_tick()
        assert len(tick["patch_intensities"]) == 196
        assert len(tick["patch_reuse_mask"]) == 196
        assert len(tick["joint_positions"]) == 7

    cal_zero = client.post("/api/arena/calibrate", json={"action": "record_zero"})
    assert cal_zero.status_code == 200
    assert cal_zero.json()["calibration"]["calibrated_at"] > 0

    cal_payload = client.post(
        "/api/arena/calibrate",
        json={"action": "set_payload", "payload_mass_kg": 1.25},
    )
    assert cal_payload.status_code == 200
    assert cal_payload.json()["calibration"]["payload_mass_kg"] == 1.25

    cal_step = client.post("/api/arena/calibrate", json={"action": "step_test"})
    assert cal_step.status_code == 200
    assert cal_step.json()["packets_sent"] >= 1

    ckpt = client.post(
        "/api/arena/checkpoint",
        json={"repo_or_path": "swizzle/edge-flow-vla", "quantize_int8": True},
    )
    assert ckpt.status_code == 200
    assert ckpt.json()["quantized_int8"] is True

    rt = client.get("/api/arena/runtime")
    assert rt.status_code == 200
    assert "native_runtime" in rt.json()

    replay = client.get("/api/arena/episodes/replay")
    assert replay.status_code == 200
    assert len(replay.json()["frames"]) == 4

    dl = client.get("/api/arena/mcap/download")
    assert dl.status_code == 200
    assert dl.content.startswith(MCAP_MAGIC)
    assert dl.content.endswith(MCAP_MAGIC)
    parsed = parse_mcap_messages(dl.content)
    assert len(parsed) >= 16
