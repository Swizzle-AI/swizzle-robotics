"""
Swizzle Embodied Flight Recorder: Binary MCAP + LeRobot v2.1 + Cryptographic Receipts
====================================================================================
Provides unified AI-era robot black-box logging and ISO-10218-ready cryptographic
action auditability with zero third-party binary dependencies:

1. Binary MCAP v0 Writer (`McapBinaryWriter`):
   Writes valid `\x89MCAP0\r\n` binary logs compatible with Foxglove Studio and `mcap` CLI
   covering `/joint_states`, `/action/raw_vla`, `/action/cbf_safe`, and `/swizzle/receipt`.
2. Cryptographic Embodied Receipts (`PhysicalReceipt`):
   Commits `model_id`, `trace_id`, camera frame digest, proprioceptive pose, raw policy
   output, and CBF-shielded command into a deterministic tamper-evident digest.
3. Hugging Face LeRobot v2.1 Export (`EmbodiedFlightRecorder.export_lerobot`):
   Generates `meta/info.json`, `meta/episodes.jsonl`, `meta/stats.json`, and trajectory
   chunks with automatic `dagger_flagged` tagging whenever Swizzle-CBF intervenes.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .safety import SafetyTelemetry

MCAP_MAGIC = b"\x89MCAP0\r\n"

# Official MCAP opcodes
OP_HEADER = 0x01
OP_FOOTER = 0x02
OP_SCHEMA = 0x03
OP_CHANNEL = 0x04
OP_MESSAGE = 0x05
OP_DATA_END = 0x0F


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_bytes(data: bytes) -> str:
    """Computes BLAKE3 if available, else SHA-256 hex digest."""
    try:
        import blake3  # type: ignore

        return "blake3:" + blake3.blake3(data).hexdigest()
    except Exception:
        return "sha256:" + hashlib.sha256(data).hexdigest()


def hash_observation(frame: Any) -> str:
    """Computes deterministic content digest over an incoming camera/patch observation."""
    if isinstance(frame, np.ndarray):
        return _digest_bytes(np.ascontiguousarray(frame).tobytes())
    return _digest_bytes(_canonical_json(list(frame) if hasattr(frame, "__iter__") else str(frame)))


@dataclass
class PhysicalReceipt:
    """Tamper-evident audit record binding sensor input -> model -> CBF-safe action."""

    episode_id: str
    step_index: int
    timestamp_ns: int
    model_id: str
    trace_id: str
    observation_digest: str
    pose: List[float]
    raw_action: List[float]
    safe_action: List[float]
    cbf_intervened: bool
    min_barrier_margin: float
    receipt_digest: str = ""

    def compute_digest(self) -> str:
        payload = {
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "timestamp_ns": self.timestamp_ns,
            "model_id": self.model_id,
            "trace_id": self.trace_id,
            "observation_digest": self.observation_digest,
            "pose": [round(float(x), 6) for x in self.pose],
            "raw_action": [round(float(x), 6) for x in self.raw_action],
            "safe_action": [round(float(x), 6) for x in self.safe_action],
            "cbf_intervened": bool(self.cbf_intervened),
            "min_barrier_margin": round(float(self.min_barrier_margin), 6),
        }
        return _digest_bytes(_canonical_json(payload))

    def seal(self) -> "PhysicalReceipt":
        self.receipt_digest = self.compute_digest()
        return self

    def verify(self) -> bool:
        if not self.receipt_digest:
            return False
        return self.receipt_digest == self.compute_digest()


def _mcap_str(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<I", len(raw)) + raw


def _mcap_bytes(b: bytes) -> bytes:
    return struct.pack("<I", len(b)) + b


def _mcap_map(m: Dict[str, str]) -> bytes:
    body = bytearray()
    for k, v in m.items():
        body.extend(_mcap_str(str(k)))
        body.extend(_mcap_str(str(v)))
    return struct.pack("<I", len(body)) + bytes(body)


def _mcap_record(opcode: int, payload: bytes) -> bytes:
    return struct.pack("<BQ", opcode, len(payload)) + payload


class McapBinaryWriter:
    """Zero-dependency Foxglove-compatible MCAP v0 binary stream writer."""

    def __init__(self, profile: str = "swizzle-robotics", library: str = "swizzle-recorder-v0.3"):
        self.profile = profile
        self.library = library
        self._channels: Dict[str, int] = {}
        self._channel_records: List[bytes] = []
        self._schema_ids: Dict[str, int] = {}
        self._schema_records: List[bytes] = []
        self._message_records: List[bytes] = []
        self._seq: Dict[int, int] = {}
        self._data_crc = 0

    def register_json_channel(self, topic: str, schema_name: str, json_schema: Dict[str, Any]) -> int:
        if topic in self._channels:
            return self._channels[topic]

        if schema_name not in self._schema_ids:
            sid = len(self._schema_ids) + 1
            self._schema_ids[schema_name] = sid
            schema_bytes = _canonical_json(json_schema)
            payload = (
                struct.pack("<H", sid)
                + _mcap_str(schema_name)
                + _mcap_str("jsonschema")
                + _mcap_bytes(schema_bytes)
            )
            self._schema_records.append(_mcap_record(OP_SCHEMA, payload))
        else:
            sid = self._schema_ids[schema_name]

        cid = len(self._channels) + 1
        self._channels[topic] = cid
        self._seq[cid] = 0
        payload = (
            struct.pack("<HH", cid, sid)
            + _mcap_str(topic)
            + _mcap_str("json")
            + _mcap_map({})
        )
        self._channel_records.append(_mcap_record(OP_CHANNEL, payload))
        return cid

    def add_message(self, topic: str, timestamp_ns: int, data: Dict[str, Any]) -> None:
        if topic not in self._channels:
            self.register_json_channel(
                topic,
                schema_name=f"swizzle.{topic.strip('/').replace('/', '.')}",
                json_schema={"type": "object"},
            )
        cid = self._channels[topic]
        seq = self._seq[cid]
        self._seq[cid] = seq + 1
        body = _canonical_json(data)
        payload = struct.pack("<HIQQ", cid, seq, int(timestamp_ns), int(timestamp_ns)) + body
        rec = _mcap_record(OP_MESSAGE, payload)
        self._data_crc = zlib.crc32(rec, self._data_crc)
        self._message_records.append(rec)

    def to_bytes(self) -> bytes:
        buf = bytearray()
        buf.extend(MCAP_MAGIC)

        header_payload = _mcap_str(self.profile) + _mcap_str(self.library)
        buf.extend(_mcap_record(OP_HEADER, header_payload))

        for rec in self._schema_records:
            buf.extend(rec)
        for rec in self._channel_records:
            buf.extend(rec)
        for rec in self._message_records:
            buf.extend(rec)

        data_end_payload = struct.pack("<I", self._data_crc & 0xFFFFFFFF)
        buf.extend(_mcap_record(OP_DATA_END, data_end_payload))

        summary_start = len(buf)
        for rec in self._schema_records:
            buf.extend(rec)
        for rec in self._channel_records:
            buf.extend(rec)

        summary_offset_start = len(buf)
        footer_crc_body = struct.pack("<QQ", summary_start, summary_offset_start)
        summary_crc = zlib.crc32(footer_crc_body) & 0xFFFFFFFF
        footer_payload = struct.pack("<QQI", summary_start, summary_offset_start, summary_crc)
        buf.extend(_mcap_record(OP_FOOTER, footer_payload))
        buf.extend(MCAP_MAGIC)
        return bytes(buf)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.to_bytes())
        return out


@dataclass
class EpisodeStepRecord:
    step_index: int
    timestamp_ns: int
    prompt: str
    pose: List[float]
    raw_action: List[float]
    safe_action: List[float]
    cbf_intervened: bool
    min_barrier_margin: float
    receipt: PhysicalReceipt


class EmbodiedFlightRecorder:
    """
    Captures closed-loop robot episodes and exports both:
    - Foxglove-ready `.mcap` binary black-box logs
    - Hugging Face `LeRobot v2.1` dataset layout (`meta/` + `data/`) with DAgger tags
    """

    def __init__(self, episode_id: str = "ep_000000", model_id: str = "lerobot/smolvla-450m"):
        self.episode_id = episode_id
        self.model_id = model_id
        self.steps: List[EpisodeStepRecord] = []
        self.dagger_flagged = False
        self.intervention_count = 0

    def record_step(
        self,
        frame: Any,
        pose: Sequence[float],
        safety: SafetyTelemetry,
        prompt: str = "",
        trace_id: str = "local",
        timestamp_ns: Optional[int] = None,
    ) -> PhysicalReceipt:
        ts = int(timestamp_ns if timestamp_ns is not None else time.time_ns())
        idx = len(self.steps)
        pose_vec = [round(float(x), 5) for x in pose[:6]]
        while len(pose_vec) < 6:
            pose_vec.append(0.0)

        receipt = PhysicalReceipt(
            episode_id=self.episode_id,
            step_index=idx,
            timestamp_ns=ts,
            model_id=self.model_id,
            trace_id=trace_id,
            observation_digest=hash_observation(frame),
            pose=pose_vec,
            raw_action=list(safety.raw_action),
            safe_action=list(safety.safe_action),
            cbf_intervened=bool(safety.cbf_intervened),
            min_barrier_margin=float(safety.min_barrier_margin),
        ).seal()

        if safety.cbf_intervened:
            self.dagger_flagged = True
            self.intervention_count += 1

        self.steps.append(
            EpisodeStepRecord(
                step_index=idx,
                timestamp_ns=ts,
                prompt=prompt,
                pose=pose_vec,
                raw_action=list(safety.raw_action),
                safe_action=list(safety.safe_action),
                cbf_intervened=bool(safety.cbf_intervened),
                min_barrier_margin=float(safety.min_barrier_margin),
                receipt=receipt,
            )
        )
        return receipt

    def export_mcap(self, output_path: str | Path) -> Path:
        writer = McapBinaryWriter()
        for s in self.steps:
            writer.add_message("/joint_states", s.timestamp_ns, {"pose": s.pose, "step": s.step_index})
            writer.add_message("/action/raw_vla", s.timestamp_ns, {"action": s.raw_action})
            writer.add_message("/action/cbf_safe", s.timestamp_ns, {"action": s.safe_action})
            writer.add_message(
                "/cbf/safety",
                s.timestamp_ns,
                {"intervened": s.cbf_intervened, "min_margin": s.min_barrier_margin},
            )
            writer.add_message("/swizzle/receipt", s.timestamp_ns, asdict(s.receipt))
        return writer.save(output_path)

    def export_lerobot_v2(self, dataset_root: str | Path, episode_index: int = 0, fps: int = 50) -> Dict[str, Any]:
        """Writes Hugging Face LeRobot v2.1 compatible metadata (`meta/`) and trajectory arrays (`data/`)."""
        root = Path(dataset_root)
        meta_dir = root / "meta"
        chunk_dir = root / "data" / "chunk-000"
        meta_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir.mkdir(parents=True, exist_ok=True)

        ep_name = f"episode_{episode_index:06d}"
        rows = [
            {
                "episode_index": episode_index,
                "frame_index": s.step_index,
                "timestamp": round(s.step_index / float(max(1, fps)), 4),
                "task": s.prompt,
                "observation.state": s.pose,
                "action.raw_vla": s.raw_action,
                "action": s.safe_action,
                "cbf_intervened": s.cbf_intervened,
                "min_barrier_margin": s.min_barrier_margin,
                "receipt_digest": s.receipt.receipt_digest,
            }
            for s in self.steps
        ]

        jsonl_path = chunk_dir / f"{ep_name}.jsonl"
        jsonl_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        states_arr = np.array([s.pose for s in self.steps], dtype=np.float32) if self.steps else np.zeros((0, 6), dtype=np.float32)
        actions_arr = np.array([s.safe_action for s in self.steps], dtype=np.float32) if self.steps else np.zeros((0, 7), dtype=np.float32)
        np.save(chunk_dir / f"{ep_name}_states.npy", states_arr)
        np.save(chunk_dir / f"{ep_name}_actions.npy", actions_arr)

        info = {
            "codebase_version": "v2.1",
            "robot_type": "swizzle_7dof_manipulator",
            "total_episodes": episode_index + 1,
            "total_frames": len(self.steps),
            "fps": fps,
            "splits": {"train": f"0:{episode_index + 1}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.jsonl",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [6]},
                "action": {"dtype": "float32", "shape": [7]},
                "action.raw_vla": {"dtype": "float32", "shape": [7]},
            },
        }
        (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

        ep_meta = {
            "episode_index": episode_index,
            "episode_id": self.episode_id,
            "model_id": self.model_id,
            "length": len(self.steps),
            "dagger_flagged": self.dagger_flagged,
            "cbf_interventions": self.intervention_count,
            "tasks": sorted({s.prompt for s in self.steps if s.prompt}),
        }
        with (meta_dir / "episodes.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(ep_meta) + "\n")

        stats = {
            "observation.state": _array_stats(states_arr),
            "action": _array_stats(actions_arr),
        }
        (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

        return {
            "dataset_root": str(root),
            "episode_file": str(jsonl_path),
            "frames": len(self.steps),
            "dagger_flagged": self.dagger_flagged,
            "cbf_interventions": self.intervention_count,
        }


def _array_stats(arr: np.ndarray) -> Dict[str, List[float]]:
    if arr.shape[0] == 0:
        return {"mean": [], "std": [], "min": [], "max": []}
    return {
        "mean": [round(float(x), 5) for x in np.mean(arr, axis=0)],
        "std": [round(float(x), 5) for x in np.std(arr, axis=0)],
        "min": [round(float(x), 5) for x in np.min(arr, axis=0)],
        "max": [round(float(x), 5) for x in np.max(arr, axis=0)],
    }
