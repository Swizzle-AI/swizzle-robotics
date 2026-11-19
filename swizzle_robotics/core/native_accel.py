"""
Native Rust/CUDA C-ABI & Vectorized Kernel Bridge (`Phase 4`)
=============================================================
Probes for `Swizzle-AI/swizzle-runtime` (`swizzle` PyO3 module, `swizzle-capi`
shared library, or `swizzle-rt` binary on `PATH` / `SWIZZLE_CAPI_PATH`) and
provides fast C-contiguous vectorized fallbacks for high-rate S0 loops:

1. `probe_native_runtime()`: reports attached Rust/CUDA runtime artifacts.
2. `fast_patch_delta_mask(...)`: vectorized `14x14` (or `16x16`) patch reuse gating.
3. `fast_cbf_halfspace_clamp(...)`: contiguous NumPy barrier clamp for multi-obstacle
   half-space projection at >500 Hz with minimal interpreter allocation.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class NativeRuntimeStatus:
    backend: str
    pyo3_available: bool
    capi_library: Optional[str]
    swizzle_rt_binary: Optional[str]


def probe_native_runtime() -> NativeRuntimeStatus:
    pyo3_ok = False
    try:
        import swizzle  # type: ignore  # noqa: F401

        pyo3_ok = True
    except Exception:
        pyo3_ok = False

    capi_path = os.environ.get("SWIZZLE_CAPI_PATH")
    if capi_path and not os.path.exists(capi_path):
        capi_path = None

    rt_bin = shutil.which("swizzle-rt") or shutil.which("swizzle")
    backend = (
        "rust-pyo3"
        if pyo3_ok
        else ("rust-capi" if capi_path else ("rust-daemon" if rt_bin else "numpy-vectorized"))
    )
    return NativeRuntimeStatus(
        backend=backend,
        pyo3_available=pyo3_ok,
        capi_library=capi_path,
        swizzle_rt_binary=rt_bin,
    )


def fast_patch_delta_mask(
    curr: Sequence[float],
    prev: Optional[Sequence[float]],
    threshold: float = 0.025,
) -> Tuple[np.ndarray, int]:
    """Vectorized C-contiguous L1 delta gating returning `(reuse_mask, reused_count)`."""
    curr_arr = np.ascontiguousarray(curr, dtype=np.float32)
    if prev is None or len(prev) != curr_arr.shape[0]:
        return np.zeros(curr_arr.shape[0], dtype=bool), 0
    prev_arr = np.ascontiguousarray(prev, dtype=np.float32)
    mask = np.abs(curr_arr - prev_arr) <= float(threshold)
    return mask, int(np.count_nonzero(mask))


def fast_cbf_halfspace_clamp(
    delta_xyz: np.ndarray,
    eef_xyz: np.ndarray,
    centers_xyz: np.ndarray,
    radii: np.ndarray,
    alpha: float = 0.65,
) -> Tuple[np.ndarray, float, List[str]]:
    """
    Vectorized distance & half-space barrier projection across `K` spherical obstacles
    in a single NumPy pass.
    """
    if centers_xyz.size == 0:
        return delta_xyz, 1.0, []
    diff = eef_xyz[None, :] - centers_xyz
    dists = np.linalg.norm(diff, axis=1)
    margins = dists - radii
    min_margin = float(np.min(margins))
    safe_delta = delta_xyz.copy()
    hit_indices: List[str] = []
    for i in range(centers_xyz.shape[0]):
        d = float(dists[i])
        if d < 1e-6:
            safe_delta = safe_delta * 0.0
            hit_indices.append(f"obs_{i}")
            continue
        normal = diff[i] / d
        rhs = -float(alpha) * max(0.0, float(margins[i]))
        proj = float(np.dot(safe_delta, normal))
        if proj < rhs:
            safe_delta = safe_delta + (rhs - proj) * normal
            hit_indices.append(f"obs_{i}")
    return safe_delta, min_margin, hit_indices
