from __future__ import annotations
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
import numpy as np

from .types import StudentHardwareProfile
from .hardware import HardwareDetector
from .models import get_policy, get_world_model, MODEL_REGISTRY
from .core.camera import MockCamera
from .core.patch_cache import extract_patch_intensities_from_frame
from .core.recorder import EmbodiedFlightRecorder, PhysicalReceipt
from .core.safety import ControlBarrierShield

_GLOBAL_SAFETY_SHIELD = ControlBarrierShield()
_GLOBAL_FLIGHT_RECORDER = EmbodiedFlightRecorder()

class RoboticsAPIHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP Handler for Swizzle Robotics VLA and World Model Serving."""

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"status": "ok"})

    def do_GET(self):
        if self.path in ("/v1/robotics/models", "/v1/robotics/catalog"):
            profile = HardwareDetector.detect()
            models = [
                {
                    "id": "nvidia/cosmos-predict-mini",
                    "label": "NVIDIA Cosmos Predict (World Foundation Model)",
                    "type": "world-model",
                    "vram_req_gb": 4.0,
                    "target_hz": 60.0,
                    "description": "Spatial-temporal continuous tokenizer and latent physical rollout simulator."
                },
                {
                    "id": "nvidia/gr00t-manipulator-v1",
                    "label": "NVIDIA Project GR00T (Humanoid & Dual-Arm)",
                    "type": "vla",
                    "vram_req_gb": 6.0,
                    "target_hz": 50.0,
                    "description": "Generalist robot foundation model with bimanual 7-DoF action chunking."
                },
                {
                    "id": "lerobot/smolvla-450m",
                    "label": "Hugging Face LeRobot SmolVLA (450M)",
                    "type": "vla",
                    "vram_req_gb": 1.5,
                    "target_hz": 80.5,
                    "description": "Ultra-lightweight edge VLA running locally on consumer laptops and Mac M-series."
                },
                {
                    "id": "openvla/openvla-7b-swizzle",
                    "label": "OpenVLA 7B (Swizzle-SpecAct Accelerated)",
                    "type": "vla",
                    "vram_req_gb": 6.5,
                    "target_hz": 31.2,
                    "description": "7B VLA foundation model with 2nd-order Taylor kinematic speculation."
                },
                {
                    "id": "physical-intelligence/pi0-fast",
                    "label": "Physical Intelligence Pi0 (Flow-Matching)",
                    "type": "vla",
                    "vram_req_gb": 4.5,
                    "target_hz": 45.0,
                    "description": "Fast ODE vector field integration for multi-task robotic manipulation."
                },
                {
                    "id": "meta/vjepa2-latent-mpc",
                    "label": "Meta V-JEPA 2 / DINO-WM (Latent CEM)",
                    "type": "world-model",
                    "vram_req_gb": 2.5,
                    "target_hz": 80.0,
                    "description": "Action-conditional joint embedding predictive architecture with sub-50ms CEM planning."
                }
            ]
            self._send_json(200, {
                "hardware": {
                    "device_type": profile.device_type,
                    "device_name": profile.device_name,
                    "vram_gb": profile.vram_gb,
                    "recommended_vla": profile.recommended_vla,
                    "recommended_world_model": profile.recommended_world_model,
                    "recommended_quantization": profile.recommended_quantization,
                    "needs_colab_tunnel": profile.needs_colab_tunnel_for_7b
                },
                "models": models
            })
        elif self.path in ("/health", "/v1/health"):
            self._send_json(200, {"status": "healthy", "service": "swizzle-robotics", "time": time.time()})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if self.path == "/v1/robotics/act":
            model_name = payload.get("model", "lerobot/smolvla-450m")
            prompt = payload.get("prompt", "Pick up target object")
            frame_data = payload.get("frame")
            if frame_data is None:
                # Use synthetic test frame
                cam = MockCamera()
                _, frame_data = cam.read()

            policy = get_policy(model_name)
            act, substeps, pc_decision, spec_act = policy.step(frame_data, prompt)
            pose = payload.get("pose", [0.0, 0.0, 0.25, 0.0, 0.0, 0.0])
            heartbeat_ok = bool(payload.get("heartbeat_ok", True))
            safety = _GLOBAL_SAFETY_SHIELD.project(act, pose, heartbeat_ok=heartbeat_ok)
            trace_id = self.headers.get("traceparent", "local")
            receipt = _GLOBAL_FLIGHT_RECORDER.record_step(
                frame_data, pose, safety, prompt=prompt, trace_id=trace_id
            )
            self._send_json(200, {
                "model": model_name,
                "action_7dof": safety.safe_action,
                "raw_action_7dof": safety.raw_action,
                "substeps_count": len(substeps),
                "patch_cache": {
                    "reused_patches": pc_decision.reused_patches,
                    "total_patches": pc_decision.total_patches,
                    "hit_ratio": pc_decision.patch_hit_ratio,
                    "prefill_speedup": pc_decision.prefill_speedup,
                    "swizzle_prefill_ms": pc_decision.swizzle_prefill_ms
                },
                "spec_act": {
                    "acceptance_rate": spec_act.acceptance_rate,
                    "speedup": spec_act.decoding_speedup
                },
                "cbf_safety": {
                    "intervened": safety.cbf_intervened,
                    "min_barrier_margin": safety.min_barrier_margin,
                    "clamped_axes": safety.clamped_axes,
                    "reasons": safety.intervention_reasons,
                    "solver_latency_ms": safety.solver_latency_ms,
                    "watchdog_active": safety.watchdog_active,
                },
                "receipt": {
                    "episode_id": receipt.episode_id,
                    "step_index": receipt.step_index,
                    "observation_digest": receipt.observation_digest,
                    "receipt_digest": receipt.receipt_digest,
                    "dagger_flagged": _GLOBAL_FLIGHT_RECORDER.dagger_flagged,
                },
            })

        elif self.path == "/v1/robotics/receipt/verify":
            try:
                rec = PhysicalReceipt(**payload)
                valid = rec.verify()
                self._send_json(200, {"valid": valid, "receipt_digest": rec.receipt_digest})
            except Exception as exc:
                self._send_json(400, {"valid": False, "error": str(exc)})

        elif self.path == "/v1/robotics/safety":
            raw_act = payload.get("action", [0.0] * 7)
            pose = payload.get("pose", [0.0, 0.0, 0.25, 0.0, 0.0, 0.0])
            heartbeat_ok = bool(payload.get("heartbeat_ok", True))
            safety = _GLOBAL_SAFETY_SHIELD.project(raw_act, pose, heartbeat_ok=heartbeat_ok)
            self._send_json(200, {
                "safe_action": safety.safe_action,
                "raw_action": safety.raw_action,
                "cbf_intervened": safety.cbf_intervened,
                "min_barrier_margin": safety.min_barrier_margin,
                "clamped_axes": safety.clamped_axes,
                "intervention_reasons": safety.intervention_reasons,
                "solver_latency_ms": safety.solver_latency_ms,
                "watchdog_active": safety.watchdog_active,
            })

        elif self.path == "/v1/world/plan":
            model_name = payload.get("model", "nvidia/cosmos-predict-mini")
            horizon = payload.get("horizon", 5)
            samples = payload.get("samples", 32)
            wm = get_world_model(model_name)

            cam = MockCamera()
            _, curr_frame = cam.read()
            cam.frame_idx = 10
            _, goal_frame = cam.read()

            best_act, full_traj, latency = wm.plan_cem(curr_frame, goal_frame, horizon=horizon, num_samples=samples)
            self._send_json(200, {
                "model": model_name,
                "horizon": horizon,
                "best_action": best_act,
                "planned_trajectory": full_traj,
                "planning_latency_ms": round(latency, 2),
                "samples_evaluated": samples
            })
        else:
            self._send_json(404, {"error": "Endpoint not found"})

def run_server(port: int = 8080, host: str = "0.0.0.0"):
    server = HTTPServer((host, port), RoboticsAPIHandler)
    print(f"🚀 Swizzle Robotics Server running at http://{host}:{port}")
    print(f"  - Model Catalog: http://{host}:{port}/v1/robotics/models")
    print(f"  - Action Endpoint: POST http://{host}:{port}/v1/robotics/act")
    print(f"  - World Model Planner: POST http://{host}:{port}/v1/world/plan")
    server.serve_forever()

if __name__ == "__main__":
    run_server()
