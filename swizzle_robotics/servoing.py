import time
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import numpy as np

@dataclass
class ServoingConfig:
    target_hz: float = 80.5
    grid_size: int = 14
    patch_size: int = 16
    smoothing_factor: float = 0.85

class VisualServoingLoop:
    """
    80.5 Hz sub-millisecond visual servoing control loop.
    Extracts 14x14 ViT patch luminance representations directly from frame buffers
    and computes closed-loop end-effector actuation deltas.
    """
    def __init__(self, config: Optional[ServoingConfig] = None):
        self.config = config or ServoingConfig()
        self.prev_delta = np.zeros(6, dtype=np.float32)
        self.frame_count = 0
        self.interval_sec = 1.0 / self.config.target_hz

    def extract_patch_luminance(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Extracts 14x14 luminance grid from incoming 224x224 frame."""
        if frame_bgr.ndim == 3:
            # Grayscale luminance: 0.299 R + 0.587 G + 0.114 B
            gray = 0.299 * frame_bgr[:, :, 2] + 0.587 * frame_bgr[:, :, 1] + 0.114 * frame_bgr[:, :, 0]
        else:
            gray = frame_bgr

        # Downsample to 14x14 grid
        h, w = gray.shape
        gh = h // self.config.grid_size
        gw = w // self.config.grid_size
        cropped = gray[:gh * self.config.grid_size, :gw * self.config.grid_size]
        grid = cropped.reshape(self.config.grid_size, gh, self.config.grid_size, gw).mean(axis=(1, 3))
        return grid.astype(np.float32) / 255.0

    def step(self, current_frame: np.ndarray, goal_features: np.ndarray) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        curr_feat = self.extract_patch_luminance(current_frame)
        error = goal_features - curr_feat

        # Proportional visual servoing error projection to 6-DoF end-effector delta [dx, dy, dz, droll, dpitch, dyaw]
        dx = float(np.mean(error[:, :7]) - np.mean(error[:, 7:]))
        dy = float(np.mean(error[:7, :]) - np.mean(error[7:, :]))
        dz = float(np.mean(error))
        d_rot = np.array([dy * 0.5, -dx * 0.5, 0.0], dtype=np.float32)

        raw_delta = np.array([dx, dy, dz, d_rot[0], d_rot[1], d_rot[2]], dtype=np.float32)
        # Exponential smoothing to prevent motor oscillation
        smoothed_delta = self.config.smoothing_factor * self.prev_delta + (1.0 - self.config.smoothing_factor) * raw_delta
        self.prev_delta = smoothed_delta
        self.frame_count += 1

        compute_latency_ms = (time.perf_counter() - t0) * 1000.0
        return smoothed_delta, compute_latency_ms
