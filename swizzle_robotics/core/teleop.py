from __future__ import annotations
import math
import numpy as np
from typing import List, Sequence, Tuple
from ..types import clamp_unit

class QuinticSplineInterpolator:
    """
    Quintic (5th order) polynomial spline generator.
    Guarantees continuous position, velocity, and acceleration (jerk-free)
    boundary conditions:
      p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
    """

    @staticmethod
    def interpolate(
        p0: Sequence[float],
        p1: Sequence[float],
        v0: Optional[Sequence[float]] = None,
        v1: Optional[Sequence[float]] = None,
        duration: float = 0.05,
        substeps: int = 4,
    ) -> List[List[float]]:
        dim = len(p0)
        v0_vec = list(v0) if v0 is not None else [0.0] * dim
        v1_vec = list(v1) if v1 is not None else [0.0] * dim

        T = max(1e-4, float(duration))
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        A = np.array([
            [T3, T4, T5],
            [3 * T2, 4 * T3, 5 * T4],
            [6 * T, 12 * T2, 20 * T3]
        ], dtype=np.float64)

        result_steps = []
        time_points = [(s + 1) / float(substeps) * T for s in range(substeps)]

        for d in range(dim):
            c0 = float(p0[d])
            c1 = float(v0_vec[d])
            c2 = 0.0  # Zero initial acceleration

            b = np.array([
                float(p1[d]) - c0 - c1 * T - c2 * T2,
                float(v1_vec[d]) - c1 - 2 * c2 * T,
                -2 * c2
            ], dtype=np.float64)

            try:
                c3, c4, c5 = np.linalg.solve(A, b)
            except Exception:
                c3, c4, c5 = 0.0, 0.0, 0.0

            dim_values = []
            for t in time_points:
                val = c0 + c1 * t + c2 * (t**2) + c3 * (t**3) + c4 * (t**4) + c5 * (t**5)
                dim_values.append(clamp_unit(float(val)))
            result_steps.append(dim_values)

        # Transpose to shape (substeps, dim)
        transposed = []
        for s in range(substeps):
            transposed.append([round(result_steps[d][s], 4) for d in range(dim)])
        return transposed

class TemporalActionEnsembler:
    """
    Swizzle-Teleop: Receding-Horizon Action Buffer & 50Hz/80Hz Temporal Ensembler.
    Smooths chunk boundaries using exponential decay weights w_i = exp(-decay_lambda * age_i)
    and generates jerk-free quintic spline substeps for robot motors.
    """

    def __init__(self, decay_lambda: float = 0.05, substeps: int = 4):
        self.decay_lambda = decay_lambda
        self.substeps = substeps
        self._chunks: List[Tuple[int, List[List[float]]]] = []

    def push_chunk(
        self, start_step: int, chunk: Sequence[Sequence[float]]
    ) -> Tuple[List[float], List[List[float]]]:
        clean_chunk = [[clamp_unit(v) for v in row[:7]] for row in chunk]
        self._chunks.append((start_step, clean_chunk))
        self._chunks = [
            (st, ch) for (st, ch) in self._chunks if st + len(ch) > start_step
        ]

        def _ensemble_at(step: int) -> List[float]:
            weighted = [0.0] * 7
            w_sum = 0.0
            for st, ch in self._chunks:
                offset = step - st
                if 0 <= offset < len(ch):
                    w = math.exp(-self.decay_lambda * offset)
                    for d in range(7):
                        weighted[d] += w * ch[offset][d]
                    w_sum += w
            if w_sum > 0:
                return [round(weighted[d] / w_sum, 4) for d in range(7)]
            return list(clean_chunk[0])

        curr_act = _ensemble_at(start_step)
        next_act = _ensemble_at(start_step + 1)

        # Generate quintic spline substeps
        substeps_act = QuinticSplineInterpolator.interpolate(
            p0=curr_act,
            p1=next_act,
            duration=0.02 * self.substeps,
            substeps=self.substeps
        )

        return curr_act, substeps_act

    @staticmethod
    def rtc_soft_mask_weights(
        execution_delay: int,
        overlap_horizon: int,
        chunk_horizon: int,
    ) -> List[float]:
        """
        Computes Real-Time Chunking (RTC) inpainting soft-mask weights w_i in [0.0, 1.0].
        - i < execution_delay: w_i = 1.0 (frozen committed prefix while GPU infers)
        - execution_delay <= i < execution_delay + overlap_horizon: exponential blend
        - i >= execution_delay + overlap_horizon: w_i = 0.0 (trust fresh chunk)
        """
        d = max(0, int(execution_delay))
        s = max(1, int(overlap_horizon))
        H = max(1, int(chunk_horizon))
        denom = math.expm1(float(s)) if s > 0 else 1.0
        weights: List[float] = []
        for i in range(H):
            if i < d:
                weights.append(1.0)
            elif i < d + s:
                c = float(d + s - i)
                w = math.expm1(c) / denom if denom > 0 else 0.0
                weights.append(round(max(0.0, min(1.0, w)), 6))
            else:
                weights.append(0.0)
        return weights

    def push_rtc_chunk(
        self,
        start_step: int,
        new_chunk: Sequence[Sequence[float]],
        execution_delay: int = 2,
        overlap_horizon: int = 4,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Applies Real-Time Chunking (RTC) inpainting to blend `new_chunk` against the
        currently executing previous chunk, guaranteeing smooth continuous motor transitions
        even under 50-150 ms inference latency.

        Returns `(inpainted_chunk, quintic_substeps_for_current_step)`.
        """
        clean_new = [[clamp_unit(v) for v in row[:7]] for row in new_chunk]
        if not clean_new:
            clean_new = [[0.0] * 7]
        H = len(clean_new)
        weights = self.rtc_soft_mask_weights(execution_delay, overlap_horizon, H)

        blended_chunk: List[List[float]] = []
        for i in range(H):
            step_idx = start_step + i
            # Evaluate weighted prediction from active previous chunks at step_idx
            prev_vec = None
            w_sum = 0.0
            acc = [0.0] * 7
            for st, ch in self._chunks:
                off = step_idx - st
                if 0 <= off < len(ch):
                    wt = math.exp(-self.decay_lambda * off)
                    for d_idx in range(7):
                        acc[d_idx] += wt * ch[off][d_idx]
                    w_sum += wt
            if w_sum > 0.0:
                prev_vec = [acc[d_idx] / w_sum for d_idx in range(7)]

            if prev_vec is None or weights[i] <= 0.0:
                blended_chunk.append([round(x, 4) for x in clean_new[i]])
            elif weights[i] >= 1.0:
                blended_chunk.append([round(clamp_unit(x), 4) for x in prev_vec])
            else:
                w = weights[i]
                row = [
                    round(clamp_unit(w * prev_vec[d_idx] + (1.0 - w) * clean_new[i][d_idx]), 4)
                    for d_idx in range(7)
                ]
                blended_chunk.append(row)

        _, substeps = self.push_chunk(start_step, blended_chunk)
        return blended_chunk, substeps
