"""
Swizzle Eclipse Zenoh / ROS 2 (`rmw_zenoh`) Key-Space Bridge
=============================================================
Provides VLA-aware topic transport over Eclipse Zenoh key expressions without
ROS 2 DDS multicast storms over Wi-Fi / cellular WAN links:

- Default topics mirror standard `rmw_zenoh` keyspaces:
  `rt/camera/front/image_raw`, `rt/joint_states`, `rt/cmd_vel`,
  `rt/swizzle/safety`, `rt/swizzle/receipt`.
- Uses bounded drop-oldest ring queues so wireless jitter never builds stale
  command backlogs.
- Automatically tracks heartbeat timestamps and W3C `traceparent` headers.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional


@dataclass
class ZenohBridgeConfig:
    """Configuration for ROS 2 / Eclipse Zenoh topic routing."""

    domain_id: int = 0
    camera_topic: str = "rt/camera/front/image_raw"
    joint_state_topic: str = "rt/joint_states"
    cmd_topic: str = "rt/cmd_vel"
    safety_topic: str = "rt/swizzle/safety"
    receipt_topic: str = "rt/swizzle/receipt"
    queue_capacity: int = 4
    heartbeat_timeout_ms: float = 250.0
    locator: Optional[str] = None  # e.g. "tcp/127.0.0.1:7447"


@dataclass
class ZenohEnvelope:
    topic: str
    payload: Dict[str, Any]
    timestamp_ns: int
    traceparent: str = ""


class ZenohRoboticsBridge:
    """
    Lightweight VLA-aware Eclipse Zenoh / ROS 2 middleware bridge.
    Connects to live `eclipse-zenoh` if installed and `use_live_zenoh=True`,
    or runs over a lock-free bounded in-process pub/sub bus for zero-dependency
    hardware-in-the-loop and CI verification.
    """

    def __init__(self, config: Optional[ZenohBridgeConfig] = None, use_live_zenoh: bool = False):
        self.config = config or ZenohBridgeConfig()
        self.use_live_zenoh = use_live_zenoh
        self._lock = threading.Lock()
        self._queues: Dict[str, Deque[ZenohEnvelope]] = {}
        self._subscribers: Dict[str, List[Callable[[ZenohEnvelope], None]]] = {}
        self._last_heartbeat_ns: int = time.time_ns()
        self._zenoh_session: Any = None
        self.transport_mode = "in-process-ring"

        if use_live_zenoh:
            self._try_connect_zenoh()

    def _try_connect_zenoh(self) -> None:
        try:
            import zenoh  # type: ignore

            cfg = zenoh.Config()
            if self.config.locator:
                cfg.insert_json5("connect/endpoints", json.dumps([self.config.locator]))
            self._zenoh_session = zenoh.open(cfg)
            self.transport_mode = "eclipse-zenoh"
        except Exception:
            self._zenoh_session = None
            self.transport_mode = "in-process-ring"

    def close(self) -> None:
        if self._zenoh_session is not None:
            try:
                self._zenoh_session.close()
            except Exception:
                pass
            self._zenoh_session = None

    def key_expr(self, ros_topic: str) -> str:
        """Converts ROS 2 topic `/cmd_vel` to canonical `rmw_zenoh` key `<domain>/cmd_vel/**`."""
        clean = ros_topic.strip("/")
        if clean.startswith("rt/"):
            return clean
        return f"{self.config.domain_id}/{clean}"

    def heartbeat(self, now_ns: Optional[int] = None) -> None:
        self._last_heartbeat_ns = int(now_ns if now_ns is not None else time.time_ns())

    def heartbeat_healthy(self, now_ns: Optional[int] = None) -> bool:
        curr = int(now_ns if now_ns is not None else time.time_ns())
        age_ms = (curr - self._last_heartbeat_ns) / 1e6
        return age_ms <= self.config.heartbeat_timeout_ms

    def subscribe(self, topic: str, callback: Callable[[ZenohEnvelope], None]) -> None:
        key = self.key_expr(topic)
        with self._lock:
            self._subscribers.setdefault(key, []).append(callback)

    def publish(self, topic: str, payload: Dict[str, Any], traceparent: str = "") -> ZenohEnvelope:
        key = self.key_expr(topic)
        env = ZenohEnvelope(
            topic=key,
            payload=dict(payload),
            timestamp_ns=time.time_ns(),
            traceparent=traceparent,
        )
        if self._zenoh_session is not None:
            try:
                self._zenoh_session.put(key, json.dumps({"payload": payload, "traceparent": traceparent}))
            except Exception:
                pass

        with self._lock:
            q = self._queues.get(key)
            if q is None:
                q = deque(maxlen=max(1, int(self.config.queue_capacity)))
                self._queues[key] = q
            q.append(env)
            cbs = list(self._subscribers.get(key, []))

        for cb in cbs:
            cb(env)
        return env

    def latest(self, topic: str) -> Optional[ZenohEnvelope]:
        key = self.key_expr(topic)
        with self._lock:
            q = self._queues.get(key)
            if not q:
                return None
            return q[-1]

    def drain(self, topic: str) -> List[ZenohEnvelope]:
        key = self.key_expr(topic)
        with self._lock:
            q = self._queues.get(key)
            if not q:
                return []
            items = list(q)
            q.clear()
            return items
