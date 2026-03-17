"""
Nudge Object Analyzer — Python replication of NudgeObjectAnalyzer.cpp

For each trajectory hypothesis, finds the best matching obstacle that caused
the ego to deviate. Objects are filtered through a 4-stage causality pipeline,
then the closest surviving object (by adjusted distance) is selected.

Filtering stages:
  1. LATERAL CAUSALITY — Object must be on the opposite side of the trajectory
     from the nudge direction. If ego nudged LEFT, the object must be on the RIGHT.
     Also filters objects crossing the trajectory (future projection check).

  2. LONGITUDINAL CAUSALITY — Object bbox must overlap the hypothesis spatial window
     [start_s - 10m, end_s + 10m]. Objects far ahead or behind the maneuver are filtered.

  3. SAFETY CLEARANCE — Ego must have sufficient lateral clearance to physically pass
     the object: |minEdgeDist| >= egoHalfWidth + margin. The margin is speed-dependent
     for VRUs (0.3m at low speed, 1.0m at high speed) and fixed 0.3m for vehicles.

  4. DISTANCE CRITERIA — Object must be within detection range:
     ahead < 55m, not already passed, lateral dist < 3.0m from lane center.

Adjusted distance:
  After filtering, each surviving object gets an "adjusted distance" that accounts
  for object class: VRUs get 1.0m subtracted (they need more clearance to feel safe),
  large vehicles 0.6m, perpendicular vehicles 1.0m. Lower adjusted distance = closer
  = more critical. The object with the minimum adjusted distance is selected.

Note: Object tracking hysteresis (prevents flickering between objects across frames)
is omitted in this single-frame Python version.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .data_types import (
    compute_vru_safety_margin,
    Config,
    NudgeHypothesis,
    PreprocessedObjectInfo,
)


class NudgeObjectAnalyzer:
    """Matches C++ NudgeObjectAnalyzer class (single-frame, no hysteresis)."""

    def __init__(
        self, config: Config, vehicle_width_m: float, ego_speed_mps: float = 10.0
    ):
        self.config = config
        self.ego_half_width = vehicle_width_m * 0.5
        self.ego_speed_mps = ego_speed_mps

    # ==============================================================================================
    # Public API — matches analyzeSingleHypothesis() -> findBestMatchingObject()
    # ==============================================================================================

    def evaluate_objects_for_hypothesis(
        self,
        objects: List[PreprocessedObjectInfo],
        hypothesis: NudgeHypothesis,
    ) -> Tuple[Optional[int], float, List[Dict], Dict[str, Any]]:
        """Evaluate all objects for a single hypothesis.

        Runs each object through the 4-stage filter pipeline. Surviving objects
        become "causal objects" — candidates for the bias/nudge decision.

        Returns:
            best_object_id: ID of closest object (None if no match).
            best_adjusted_dist: Adjusted distance of closest object.
            causal_objects: List of dicts with "obj", "adjusted_dist", "in_lane_margin".
            debug: Filter counts and per-object analysis results.
        """
        hypothesis_left = hypothesis.direction == "LEFT"
        best_id = None
        best_score = float("inf")
        causal_objects: List[Dict] = []
        debug: Dict[str, Any] = {"filter_counts": {}, "analyzed": []}

        for obj in objects:
            obj_debug: Dict[str, Any] = {
                "objectId": obj.objectId,
                "class": obj.obstacleClass,
            }

            # Stage 1: Lateral causality
            filtered, reason = self._should_filter_lateral(obj, hypothesis_left)
            if filtered:
                obj_debug["filtered"] = reason
                debug["analyzed"].append(obj_debug)
                continue

            # Stage 2: Longitudinal causality
            filtered, reason = self._should_filter_longitudinal(obj, hypothesis)
            if filtered:
                obj_debug["filtered"] = reason
                debug["analyzed"].append(obj_debug)
                continue

            # Stage 3: Safety clearance
            filtered, reason = self._should_filter_safety(obj)
            if filtered:
                obj_debug["filtered"] = reason
                debug["analyzed"].append(obj_debug)
                continue

            # Stage 4: Distance criteria
            if not self._meets_distance_criteria(obj):
                obj_debug["filtered"] = "FAILED_DISTANCE_CRITERIA"
                debug["analyzed"].append(obj_debug)
                continue

            # Passed all filters — compute adjusted distance
            raw_dist = abs(obj.minLateralDistToBaseline_m)
            adj_dist = calculate_adjusted_distance(raw_dist, obj, self.config)

            causal_objects.append(
                {
                    "obj": obj,
                    "adjusted_dist": adj_dist,
                    "in_lane_margin": obj.inLaneRightMargin_m
                    if obj.isOnLeftOfTrajectory
                    else obj.inLaneLeftMargin_m,
                }
            )

            # Track best (closest) object
            if adj_dist < best_score:
                best_score = adj_dist
                best_id = obj.objectId

            obj_debug["adjusted_dist"] = round(adj_dist, 3)
            obj_debug["raw_dist"] = round(raw_dist, 3)
            obj_debug["kept"] = True
            debug["analyzed"].append(obj_debug)

        return best_id, best_score, causal_objects, debug

    # ==============================================================================================
    # Filtering stages — each matches a shouldFilter*() method in C++
    # ==============================================================================================

    def _should_filter_lateral(
        self,
        obj: PreprocessedObjectInfo,
        hypothesis_left: bool,
    ) -> Tuple[bool, str]:
        """Stage 1: Object must be on the opposite side of trajectory from nudge direction.

        If hypothesis direction is LEFT (ego swerved left), the causing object
        must be on the RIGHT of the trajectory. Also filters objects that are
        crossing the trajectory (current side != future side).

        Matches C++ shouldFilterLateralCausality().
        """
        # Object on same side as hypothesis = wrong side (not causing the nudge)
        if obj.isOnLeftOfTrajectory == hypothesis_left:
            return True, "WRONG_SIDE_OF_TRAJECTORY"
        # Crossing detection: if object is moving across trajectory, it's transient
        if obj.hasFutureProjection:
            current_on_left = obj.lateralDistToTrajectory_m > 0.0
            future_on_left = obj.futureDistToTrajectory_m > 0.0
            if current_on_left != future_on_left:
                return True, "CROSSING_TRAJECTORY"
        return False, ""

    def _should_filter_longitudinal(
        self,
        obj: PreprocessedObjectInfo,
        hypothesis: NudgeHypothesis,
    ) -> Tuple[bool, str]:
        """Stage 2: Object bbox must overlap the hypothesis spatial window +/- 10m buffer.

        Matches C++ shouldFilterLongitudinalCausality().
        """
        if hypothesis.end_s_m <= hypothesis.start_s_m + 1e-3:
            return False, ""  # Invalid window, skip check
        window_min = hypothesis.start_s_m - self.config.longitudinal_buffer_m
        window_max = hypothesis.end_s_m + self.config.longitudinal_buffer_m
        has_overlap = (
            obj.bboxLongitudinalMax_m >= window_min
            and obj.bboxLongitudinalMin_m <= window_max
        )
        if not has_overlap:
            return True, "LONGITUDINAL_CAUSALITY"
        return False, ""

    def _should_filter_safety(
        self,
        obj: PreprocessedObjectInfo,
    ) -> Tuple[bool, str]:
        """Stage 3: Ego must have physical clearance to pass the object.

        Required clearance = egoHalfWidth + margin.
        If both current AND future distances are below this, the object is too
        close for the ego to have passed — filter it out.

        Matches C++ shouldFilterSafetyCausality().
        """
        margin = (
            compute_vru_safety_margin(self.ego_speed_mps)
            if obj.isVRU
            else self.config.vehicle_safety_margin_m
        )
        min_clearance = self.ego_half_width + margin
        current_too_close = abs(obj.minLateralDistToTrajectory_m) < min_clearance
        future_too_close = (not obj.hasFutureProjection) or (
            abs(obj.futureMinLateralDistToTrajectory_m) < min_clearance
        )
        if current_too_close and future_too_close:
            return (
                True,
                f"SAFETY_CLEARANCE: dist={abs(obj.minLateralDistToTrajectory_m):.2f}m < {min_clearance:.2f}m",
            )
        return False, ""

    def _meets_distance_criteria(self, obj: PreprocessedObjectInfo) -> bool:
        """Stage 4: Basic distance range check (new object thresholds, no hysteresis).

        Matches C++ meetsDistanceCriteria() with isTracked=false.
        """
        if obj.distanceAheadOfEgo_m > self.config.max_longitudinal_dist_m:
            return False
        if obj.distancePassedByEgo_m > 0.0:  # MAX_PASSED_DIST_NEW_M = 0
            return False
        if abs(obj.minLateralDistToLaneCenter_m) > self.config.max_lateral_dist_m:
            return False
        return True


# ==================================================================================================
# Adjusted distance calculation — matches bias_nudge::calculateAdjustedDistance()
#
# The adjusted distance subtracts a class-specific buffer from the raw lateral distance.
# This accounts for the fact that different object types need different clearance:
#   - VRU (pedestrian, cyclist): -1.0m (drivers give VRUs extra room)
#   - Large vehicle (truck, bus): -0.6m (wider objects feel closer)
#   - Perpendicular vehicle (not parked, heading ~90 degrees): -1.0m
# The result is clamped to >= 0. Lower adjusted distance = more critical.
# ==================================================================================================


def calculate_adjusted_distance(
    raw_dist: float,
    obj: PreprocessedObjectInfo,
    config: Config,
) -> float:
    """Calculate adjusted distance. Matches C++ bias_nudge::calculateAdjustedDistance()."""
    buffer = 0.0
    if obj.isVRU:
        buffer = config.vru_buffer_m
    elif obj.isLargeVehicle:
        buffer = config.large_vehicle_buffer_m
    elif (
        obj.isVehicle
        and obj.obstacleState != "PARKED"
        and abs(obj.headingDiffWithLaneCenter) >= config.perpendicular_threshold_rad
        and abs(obj.headingDiffWithLaneCenter) <= config.perpendicular_upper_bound_rad
    ):
        buffer = config.perpendicular_buffer_m
    return max(0.0, raw_dist - buffer)
