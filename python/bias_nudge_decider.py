"""
Bias vs Nudge Decider — Python replication of BiasNudgeDecider.cpp

Distinguishes real nudge maneuvers from systematic bias (e.g., driving alongside
a row of parked cars). Uses a V3 two-question approach:

  Q1: Is there an EXCEPTIONAL object?
      - Single object in causal zone -> automatically exceptional (nothing to compare)
      - Multiple objects -> exceptional if exceptionality_ratio >= 1.5
        (ratio = trimmed_mean(others_raw_dist) / closest_adjusted_dist)
      - If Q1=NO -> BIAS (the deviation is from a uniform environment, not one obstacle)

  Q2: Is the exceptional object a valid nudge candidate?
      Both must pass:
      a) AHEAD OF EGO: distanceAheadOfEgo > -2m (allows objects slightly behind ego front)
      b) NEAR THE PEAK: the object is at a position where the ego was actually deviating.
         Two zones (either satisfies):
           Primary:  |dist_from_peak| <= 10m AND deviation_ratio >= 0.5
           Extended: 10m < signed_dist <= 20m AND deviation_ratio >= 0.8
         The deviation_ratio is the ego's lateral deviation at the object's s-position
         divided by the peak deviation: dev(obj_s) / dev(peak_s).
         This catches the nudge-abort-then-nudge pattern where the peak is behind
         but the real obstacle is ahead with the ego still deviating.

  Final: NUDGE = Q1 AND Q2a AND Q2b; else BIAS.

Exceptionality ratio computation:
  1. Collect raw distances of all other objects in causal zone (raw dist <= 4m)
  2. Prefer non-VRU objects for the pool (VRUs have artificially low adjusted dist)
  3. Trim min/max if >= 3 objects for robust mean
  4. Ratio = mean_of_pool / closest_adjusted_dist
"""

from __future__ import annotations

import bisect
from typing import Dict, List, Tuple

from .data_types import BiasNudgeOutput, Config


class BiasNudgeDecider:
    """Matches C++ bias_nudge::evaluate() and supporting functions."""

    def __init__(self, config: Config):
        self.config = config

    # ==============================================================================================
    # Public API — matches bias_nudge::evaluate()
    # ==============================================================================================

    def evaluate(
        self,
        candidates: List[Dict],
        peak_s_m: float,
        peak_magnitude_m: float,
        deviations: List[List[float]],
    ) -> BiasNudgeOutput:
        """Run the full bias vs nudge evaluation for one hypothesis.

        Args:
            candidates: Causal objects that passed all filters, each with
                        {"obj": PreprocessedObjectInfo, "adjusted_dist": float}.
            peak_s_m: Longitudinal position of the deviation peak (Frenet s).
            peak_magnitude_m: Absolute magnitude of the peak.
            deviations: Detrended [s, l] deviation series (for interpolation).

        Returns:
            BiasNudgeOutput with is_bias=True/False and all intermediate metrics.
        """
        output = BiasNudgeOutput()

        if not candidates:
            output.is_bias = True
            output.reason = "BIAS: No candidates"
            return output

        # Step 1: Find closest object and count zone memberships
        closest_obj = None
        closest_adj = float("inf")
        objects_in_causal = 0
        objects_within_critical = 0

        for c in candidates:
            obj = c["obj"]
            raw_dist = abs(obj.minLateralDistToBaseline_m)
            adj_dist = c["adjusted_dist"]
            if raw_dist <= self.config.causal_zone_m:
                objects_in_causal += 1
            if adj_dist <= self.config.critical_threshold_m:
                objects_within_critical += 1
            if adj_dist < closest_adj:
                closest_adj = adj_dist
                closest_obj = obj

        output.objects_in_causal_zone = objects_in_causal
        output.objects_within_critical = objects_within_critical

        if closest_obj is None:
            output.is_bias = True
            output.reason = "BIAS: No closest object found"
            return output

        output.closest_object_id = closest_obj.objectId
        output.closest_adjusted_dist = closest_adj
        output.closest_raw_dist = abs(closest_obj.minLateralDistToBaseline_m)
        output.closest_is_vru = closest_obj.isVRU

        # Step 2: Compute exceptionality ratio
        ratio, mean_others = self._compute_exceptionality_ratio(
            candidates, closest_obj.objectId, closest_adj
        )
        output.exceptionality_ratio = ratio
        output.mean_of_others_m = mean_others

        # Step 3 — Q1: Is there an exceptional object?
        is_exceptional = (
            objects_in_causal == 1 or ratio >= self.config.exceptionality_threshold
        )
        output.q1_is_exceptional = is_exceptional

        if not is_exceptional:
            output.is_bias = True
            output.reason = f"BIAS: No exceptional object ({objects_in_causal} objects, ratio={ratio:.2f})"
            return output

        # Step 4 — Q2a: Is the exceptional object ahead of ego?
        output.q2a_ahead_of_ego = (
            closest_obj.distanceAheadOfEgo_m > self.config.ahead_of_ego_threshold_m
        )

        # Step 5 — Q2b: Is the exceptional object near the deviation peak?
        # Check both current and future positions.
        signed_dist = closest_obj.centerSOnLaneCenter_m - peak_s_m
        dev_ratio = self._compute_deviation_ratio(
            closest_obj.centerSOnLaneCenter_m, peak_s_m, peak_magnitude_m, deviations
        )
        output.closest_signed_dist_to_peak_m = signed_dist
        output.closest_dist_from_peak_m = abs(signed_dist)
        output.closest_deviation_ratio = dev_ratio

        future_signed = closest_obj.futureCenterSOnLaneCenter_m - peak_s_m
        future_dev_ratio = self._compute_deviation_ratio(
            closest_obj.futureCenterSOnLaneCenter_m,
            peak_s_m,
            peak_magnitude_m,
            deviations,
        )
        output.closest_future_signed_dist_m = future_signed
        output.closest_future_deviation_ratio = future_dev_ratio

        # Case 1: Object crosses through peak area (current behind, future ahead or vice versa)
        current_behind = signed_dist < 0
        future_behind = future_signed < 0
        if current_behind != future_behind:
            near_peak = True
        else:
            # Case 2: Same side — check either current or future position
            near_peak = self._is_near_peak_at_position(
                signed_dist, dev_ratio
            ) or self._is_near_peak_at_position(future_signed, future_dev_ratio)
        output.q2b_near_peak = near_peak

        # Step 6: Final decision
        is_valid = output.q2a_ahead_of_ego and output.q2b_near_peak

        if is_valid:
            output.is_bias = False
            output.reason = (
                f"NUDGE: Exceptional (ratio={ratio:.2f}, dist={closest_adj:.2f}m, "
                f"peak={abs(signed_dist):.1f}m, dev={dev_ratio:.2f})"
            )
        else:
            output.is_bias = True
            fails = []
            if not output.q2a_ahead_of_ego:
                fails.append("not ahead")
            if not near_peak:
                fails.append(
                    f"not near peak (dist={abs(signed_dist):.1f}m, dev={dev_ratio:.2f})"
                )
            output.reason = f"BIAS: Exceptional fails check ({', '.join(fails)})"

        return output

    # ==============================================================================================
    # Private helpers
    # ==============================================================================================

    def interpolate_deviation_at_s(
        self, s_m: float, deviations: List[List[float]]
    ) -> float:
        """Linearly interpolate the lateral deviation at a given longitudinal position.

        Clamps to first/last value if s is outside the deviation series range.
        Uses binary search for O(log n) lookup.
        Matches C++ bias_nudge::interpolateDeviationAtS().
        """
        if not deviations:
            return 0.0
        if s_m <= deviations[0][0]:
            return deviations[0][1]
        if s_m >= deviations[-1][0]:
            return deviations[-1][1]

        # Binary search for the bracket containing s_m
        s_values = [d[0] for d in deviations]
        idx = bisect.bisect_left(s_values, s_m)
        if idx == 0:
            return deviations[0][1]
        if idx >= len(deviations):
            return deviations[-1][1]

        s0, l0 = deviations[idx - 1]
        s1, l1 = deviations[idx]
        ds = s1 - s0
        if ds < 1e-6:
            return l0
        t = (s_m - s0) / ds
        return l0 + t * (l1 - l0)

    def _compute_deviation_ratio(
        self,
        s_m: float,
        peak_s_m: float,
        peak_magnitude_m: float,
        deviations: List[List[float]],
    ) -> float:
        """Compute how much the ego was deviating at the object's position relative to the peak.

        deviation_ratio = dev(object_s) / dev(peak_s)
        A ratio of 1.0 means the ego was deviating as much at the object as at the peak.
        A ratio of 0.0 means the ego was back on the baseline at the object's position.

        Matches C++ computeDeviationRatio().
        """
        if not deviations or peak_magnitude_m <= 1e-6:
            return 0.0
        peak_dev = self.interpolate_deviation_at_s(peak_s_m, deviations)
        if abs(peak_dev) <= 1e-6:
            return 0.0
        obj_dev = self.interpolate_deviation_at_s(s_m, deviations)
        return obj_dev / peak_dev

    def _compute_exceptionality_ratio(
        self,
        candidates: List[Dict],
        closest_id: int,
        closest_adj_dist: float,
    ) -> Tuple[float, float]:
        """Compute how exceptional the closest object is compared to others.

        Collects raw distances of other objects in the causal zone (<=4m),
        preferring non-VRU objects. Trims min/max if >= 3 for robustness.
        Ratio = trimmed_mean(pool) / closest_adjusted_dist.

        High ratio (>= 1.5) means the closest object is much closer than average —
        it stands out from the background and likely caused the deviation.

        Matches C++ computeExceptionalityRatio().
        Returns: (ratio, mean_of_others).
        """
        non_vru_dists: List[float] = []
        all_dists: List[float] = []

        for c in candidates:
            obj = c["obj"]
            if obj.objectId == closest_id:
                continue
            raw_dist = abs(obj.minLateralDistToBaseline_m)
            if raw_dist <= self.config.causal_zone_m:
                all_dists.append(raw_dist)
                if not obj.isVRU:
                    non_vru_dists.append(raw_dist)

        # Prefer non-VRU pool (VRUs have buffers that distort the mean)
        pool = non_vru_dists if non_vru_dists else all_dists

        # Trim min/max for robust mean (removes outliers)
        if len(pool) >= 3:
            pool = sorted(pool)[1:-1]

        if not pool:
            return 0.0, 0.0

        mean_others = sum(pool) / len(pool)
        if closest_adj_dist > 1e-6:
            return mean_others / closest_adj_dist, mean_others
        return 100.0, mean_others  # Object on baseline = maximally exceptional

    def _is_near_peak_at_position(
        self, signed_dist: float, deviation_ratio: float
    ) -> bool:
        """Check if a position qualifies as "near the peak" for Q2b.

        Two zones:
          Primary:  |dist_from_peak| <= 10m AND deviation_ratio >= 0.5
                    (object is close to peak and ego was deviating at that position)
          Extended: 10m < signed_dist <= 20m AND deviation_ratio >= 0.8
                    (object is ahead of peak but ego was still strongly deviating)

        Matches C++ isNearPeakAtPosition().
        """
        dist_from_peak = abs(signed_dist)
        primary = (
            dist_from_peak <= self.config.primary_peak_dist_m
            and deviation_ratio >= self.config.primary_dev_ratio
        )
        extended = (
            signed_dist > self.config.extended_min_signed_dist_m
            and signed_dist <= self.config.extended_max_signed_dist_m
            and deviation_ratio >= self.config.extended_dev_ratio
        )
        return primary or extended
