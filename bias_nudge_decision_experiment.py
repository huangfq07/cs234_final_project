#!/usr/bin/env python3
"""
Bias vs Nudge Decision Experiment

Collects labeled test cases from bias_nudge_decision_data.json and verifies whether
the decision logic correctly classifies each case as nudge or bias based on trajectory
peakedness metrics and object distribution metrics.

Data sources per frame (from C++ debug info):
  - hypothesis_summary: DgpsHypothesisSummaryDebugInfo
    Contains merged hypotheses with peakedness metrics (PPR, PFI, CV, etc.)
  - bias_nudge_decision: NudgeObjectAnalysisBiasNudgeDecisionDebugInfo
    Contains object distribution metrics (exceptionality, critical object, criteria)

JSON schema per test case:
  {
    "case_id": {
      "name": "Short description",
      "ground_truth": "nudge" or "bias",
      "clip_id": "UUID",
      "frames": [
        {
          "hypothesis_summary": { ... },   // Raw from debug info
          "bias_nudge_decision": { ... }   // Raw from debug info
        }
      ],
      "notes": "Optional context"
    }
  }

Usage:
    python bias_nudge_decision_experiment.py all              # Run all test cases
    python bias_nudge_decision_experiment.py <case_id>        # Run specific test case
    python bias_nudge_decision_experiment.py summary          # Show summary statistics

TODO (after collecting enough data, improve decision logic first, then address these):
  - Rule B: Single object failing all criteria should default to BIAS, not NUDGE
    (see parked_cars_right_001 F1: 1 object, no criteria met, falls to default NUDGE)
  - VRU as critical object: VRU with near-zero adjusted distance should be treated as
    critical object (see vru_near_parked_cars_001 F1: VRU buffer zeros distance but
    criterion3_aligned_with_peak=False)
  - Rule 1 + VRU fix: Rule 1 (all objects critical + close -> BIAS) shouldn't trigger when
    VRU buffer is the cause of low min_adjusted_dist (see vru_near_parked_cars_001 F2)
  - Deviation-at-object metric: Instead of checking longitudinal_dist_from_peak, look up the
    actual detrended deviation d at the object's longitudinal position and compare to d_max.
    Ratio d/d_max (same sign required) is more robust than peak index alignment.
    (see out_of_lane_nudge_001 vs transient_oncoming_bias_001: objects near peak have
    d/d_max ~0.8-1.0, while transient oncoming vehicle in baseline region has d/d_max ~0)
  - Baseline disabled on curvy road: Adaptive baseline suddenly disabled when curvy road
    detected, causing deviation jump (see parked_cars_right_001 F3). Should freeze baseline
    instead of hard-disabling.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

# ==================================================================================================
# Data Loading
# ==================================================================================================


def load_test_cases() -> Dict[str, Dict[str, Any]]:
    """Load test cases from bias_nudge_decision_data.json."""
    data_file = Path(__file__).parent / "bias_nudge_decision_data.json"
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)
    with open(data_file, "r") as f:
        return json.load(f)


def extract_trajectory_metrics(hypothesis_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Extract trajectory peakedness metrics from hypothesis_summary debug info.

    Uses the first (dominant) merged hypothesis's peakedness_metrics.
    """
    merged = hypothesis_summary.get("merged_hypotheses", [])
    if not merged:
        return {
            "deviation_mean_m": 0.0,
            "deviation_std_m": 0.0,
            "deviation_cv": 0.0,
            "peak_prominence_ratio": 0.0,
            "plateau_flatness_index": 0.0,
            "baseline_point_count": 0,
        }
    # Use dominant (first) hypothesis
    pm = merged[0].get("peakedness_metrics", {})
    return {
        "deviation_mean_m": pm.get("deviation_mean_m", 0.0),
        "deviation_std_m": pm.get("deviation_std_m", 0.0),
        "deviation_cv": pm.get("deviation_cv", 0.0),
        "peak_prominence_ratio": pm.get("peak_prominence_ratio", 0.0),
        "plateau_flatness_index": pm.get("plateau_flatness_index", 0.0),
        "baseline_point_count": pm.get("baseline_point_count", 0),
    }


def extract_object_metrics(bias_nudge_decision: Dict[str, Any]) -> Dict[str, Any]:
    """Extract object distribution metrics from bias_nudge_decision debug info."""
    return {
        "total_causal_objects": bias_nudge_decision.get("total_causal_objects", 0),
        "objects_in_causal_zone": bias_nudge_decision.get("objects_in_causal_zone", 0),
        "objects_within_critical_threshold": bias_nudge_decision.get(
            "objects_within_critical_threshold", 0
        ),
        "min_adjusted_dist_to_baseline_m": bias_nudge_decision.get(
            "min_adjusted_dist_to_baseline_m", 999.0
        ),
        "mean_of_others_in_causal_zone_m": bias_nudge_decision.get(
            "mean_of_others_in_causal_zone_m", 999.0
        ),
        "exceptionality_ratio": bias_nudge_decision.get("exceptionality_ratio", 0.0),
        "closest_object_is_vru": bias_nudge_decision.get(
            "closest_object_is_vru", False
        ),
        "closest_object_dist_from_peak_m": bias_nudge_decision.get(
            "closest_object_dist_from_peak_m", 999.0
        ),
        "closest_object_signed_dist_to_peak_m": bias_nudge_decision.get(
            "closest_object_signed_dist_to_peak_m", 0.0
        ),
        "closest_object_future_signed_dist_to_peak_m": bias_nudge_decision.get(
            "closest_object_future_signed_dist_to_peak_m", 0.0
        ),
        "criterion1_within_threshold": bias_nudge_decision.get(
            "criterion1_within_threshold", False
        ),
        "criterion2_exceptional": bias_nudge_decision.get(
            "criterion2a_single_exceptional", False
        ),
        "criterion3_aligned_with_peak": bias_nudge_decision.get(
            "criterion3_aligned_with_peak", False
        ),
        "criterion4_ahead_of_ego": bias_nudge_decision.get(
            "criterion4_ahead_of_ego", False
        ),
        "has_critical_object": bias_nudge_decision.get("has_critical_object", False),
    }


# ==================================================================================================
# Decision Logic V1 (original, using C++ metrics directly)
# ==================================================================================================


def decide_bias_vs_nudge(
    trajectory_metrics: Dict[str, Any], object_metrics: Dict[str, Any]
) -> Tuple[str, float, str]:
    """
    V1 decision logic using C++ debug info metrics directly.

    Returns:
        Tuple of (decision, confidence, reason)
        - decision: "nudge" or "bias"
        - confidence: 0.0 to 1.0
        - reason: Human-readable explanation
    """

    # Extract trajectory metrics
    ppr = trajectory_metrics.get("peak_prominence_ratio", 0.0)
    pfi = trajectory_metrics.get("plateau_flatness_index", 0.0)
    cv = trajectory_metrics.get("deviation_cv", 0.0)

    # Extract object metrics
    has_critical = object_metrics.get("has_critical_object", False)
    objects_in_zone = object_metrics.get("objects_in_causal_zone", 0)
    exceptionality = object_metrics.get("exceptionality_ratio", 0.0)
    min_dist = object_metrics.get("min_adjusted_dist_to_baseline_m", 999.0)

    # ======================================================================================
    # Decision Logic (to be tuned based on test cases)
    # ======================================================================================

    # Rule 1: ALL objects within critical threshold -> Likely incorrect baseline (BIAS)
    objects_within_critical = object_metrics.get("objects_within_critical_threshold", 0)
    if (
        objects_in_zone >= 6
        and objects_within_critical >= objects_in_zone
        and min_dist < 0.5
    ):
        confidence = 0.9
        reason = f"All {objects_in_zone} objects critical and close (dist<0.5m) - likely incorrect baseline"
        return "bias", confidence, reason

    # Rule 2: Multiple objects in zone, low exceptionality -> Bias (convoy/parked cars)
    if objects_in_zone >= 2 and exceptionality < 1.5:
        confidence = 0.85
        reason = f"Multiple objects ({objects_in_zone}), none exceptional (ratio={exceptionality:.2f})"
        return "bias", confidence, reason

    # Rule 3: Strong object evidence -> Nudge
    if has_critical:
        confidence = 0.9
        reason = (
            f"Critical object found (dist={min_dist:.2f}m, ratio={exceptionality:.2f})"
        )
        return "nudge", confidence, reason

    # Rule 4: Strong trajectory peak with exceptional object nearby -> Nudge
    if ppr > 5.0 and objects_in_zone > 0 and min_dist < 3.0 and exceptionality >= 1.5:
        confidence = 0.8
        reason = f"Strong peak (PPR={ppr:.2f}) with exceptional object nearby (dist={min_dist:.2f}m, ratio={exceptionality:.2f})"
        return "nudge", confidence, reason

    # Rule 5: Flat plateau with no close objects -> Bias
    if ppr < 2.0 and pfi < 0.05 and (objects_in_zone == 0 or min_dist > 3.0):
        confidence = 0.7
        reason = f"Flat plateau (PPR={ppr:.2f}, PFI={pfi:.3f}), no close objects"
        return "bias", confidence, reason

    # Default: Conservative nudge classification with low confidence
    confidence = 0.3
    reason = f"Ambiguous case (PPR={ppr:.2f}, {objects_in_zone} objects, dist={min_dist:.2f}m)"
    return "nudge", confidence, reason


# ==================================================================================================
# Improved Object Metrics (V2) - Recomputed from raw causal objects
# ==================================================================================================

# Constants matching C++ (NudgeObjectAnalyzer.cpp lines 2079-2083)
MAX_CAUSAL_DISTANCE_M = 4.0
CRITICAL_THRESHOLD_M = 2.0
EXCEPTIONAL_RATIO = 2.0
PEAK_ALIGNMENT_THRESHOLD_M = 10.0
AHEAD_OF_EGO_THRESHOLD_M = -2.0

# VRU and vehicle buffers matching C++ (lines 2249-2256)
VRU_CLEARANCE_BUFFER_M = 1.0
LARGE_VEHICLE_BUFFER_M = 0.6
PERPENDICULAR_VEHICLE_BUFFER_M = 1.0


def calculate_adjusted_distance(raw_dist: float, obj: Dict[str, Any]) -> float:
    """Calculate adjusted distance matching C++ calculateAdjustedDistanceToBaseline()."""
    buffer = 0.0
    if obj.get("is_vru", False):
        buffer = VRU_CLEARANCE_BUFFER_M
    elif obj.get("is_large_vehicle", False):
        buffer = LARGE_VEHICLE_BUFFER_M
    # Note: perpendicular vehicle detection requires heading info not always in debug data
    return max(0.0, raw_dist - buffer)


def compute_improved_object_metrics(
    causal_objects: List[Dict[str, Any]],
    max_lateral_deviation_m: float,
) -> Dict[str, Any]:
    """Recompute object metrics from raw causal objects with improvements:

    V2 improvements over V1:
    1. VRU exclusion from mean pool - VRUs have artificially low distances due to buffer,
       which contaminates the mean and makes exceptionality borderline for VRU nudge cases.
    2. Outlier trimming - Remove closest and farthest non-VRU objects from mean pool
       to get a cleaner "typical background vehicle distance."
    3. Adjusted distance in exceptionality denominator - VRU buffer naturally increases
       exceptionality for VRUs since they "need less distance to be considered close."
    4. Single object alignment - When only 1 object exists, check if aligned with peak
       to identify valid nudge (fixes bus_on_lane_center F3 pattern).
    """
    defaults = {
        "total_causal_objects": 0,
        "objects_in_causal_zone": 0,
        "objects_within_critical_threshold": 0,
        "min_adjusted_dist_to_baseline_m": 999.0,
        "mean_of_others_cleaned_m": None,
        "exceptionality_ratio": 0.0,
        "closest_object_is_vru": False,
        "closest_object_dist_from_peak_m": 999.0,
        "criterion1_within_threshold": False,
        "criterion2_exceptional": False,
        "criterion3_aligned_with_peak": False,
        "criterion4_ahead_of_ego": False,
        "has_critical_object": False,
    }

    if not causal_objects:
        return defaults

    # Step 1: Compute raw and adjusted distances for each object
    enriched = []
    for obj in causal_objects:
        raw_dist = abs(obj.get("lateral_dist_to_baseline_m", 999.0))
        adjusted = obj.get("adjusted_dist_to_baseline_m", raw_dist)
        # Use provided adjusted distance, or recalculate if not available
        if adjusted is None or adjusted >= 900:
            adjusted = calculate_adjusted_distance(raw_dist, obj)
        enriched.append({**obj, "_raw_dist": raw_dist, "_adjusted_dist": adjusted})

    # Step 2: Count objects in zones and find closest by adjusted distance
    objects_in_zone = 0
    objects_within_critical = 0
    closest = None
    closest_adjusted = float("inf")

    for obj in enriched:
        if obj["_raw_dist"] <= MAX_CAUSAL_DISTANCE_M:
            objects_in_zone += 1
        if obj["_adjusted_dist"] <= CRITICAL_THRESHOLD_M:
            objects_within_critical += 1
        if obj["_adjusted_dist"] < closest_adjusted:
            closest_adjusted = obj["_adjusted_dist"]
            closest = obj

    if closest is None:
        return defaults

    dist_from_peak = abs(closest.get("longitudinal_dist_from_peak_m", 999.0))
    dist_ahead_of_ego = closest.get("distance_ahead_of_ego_m", 0.0)
    signed_dist_to_peak = closest.get(
        "closest_object_signed_dist_to_peak_m", -dist_from_peak
    )  # fallback: use negative abs
    future_signed_dist_to_peak = closest.get(
        "closest_object_future_signed_dist_to_peak_m", signed_dist_to_peak
    )  # fallback: same as current

    # Step 3: Build cleaned mean pool
    # Start with all objects in causal zone except the closest
    others_in_zone = [
        o
        for o in enriched
        if o.get("object_id") != closest.get("object_id")
        and o["_raw_dist"] <= MAX_CAUSAL_DISTANCE_M
    ]

    # V2 improvement: exclude VRUs from mean pool, then trim closest+farthest
    non_vru_others = [o for o in others_in_zone if not o.get("is_vru", False)]

    if len(non_vru_others) >= 3:
        # Trim closest and farthest by raw distance for robust mean
        sorted_pool = sorted(non_vru_others, key=lambda o: o["_raw_dist"])
        trimmed = sorted_pool[1:-1]
    elif non_vru_others:
        # Not enough for trimming, use all non-VRU others
        trimmed = non_vru_others
    elif others_in_zone:
        # No non-VRU others, fall back to all others (including VRUs)
        trimmed = others_in_zone
    else:
        trimmed = []

    # Compute cleaned mean
    if trimmed:
        mean_of_others = sum(o["_raw_dist"] for o in trimmed) / len(trimmed)
    else:
        mean_of_others = None

    # Step 4: Compute exceptionality using ADJUSTED distance in denominator
    # This naturally gives VRUs higher exceptionality due to the buffer
    if mean_of_others is not None and closest_adjusted > 1e-6:
        exceptionality = mean_of_others / closest_adjusted
    elif mean_of_others is not None:
        # Closest on baseline (adjusted ~0) → maximally exceptional
        exceptionality = 100.0
    else:
        # Single object → no exceptionality comparison possible
        exceptionality = 0.0

    # Step 5: Evaluate criteria
    c1_within_threshold = closest_adjusted <= CRITICAL_THRESHOLD_M
    c2_exceptional = exceptionality >= EXCEPTIONAL_RATIO or objects_within_critical >= 2
    c3_aligned_with_peak = dist_from_peak <= PEAK_ALIGNMENT_THRESHOLD_M
    # TODO: Add d/d_max as alternative: c3 = (dist_from_peak <= 10m) OR (d/d_max > threshold)
    # Requires C++ to add deviation_at_object_m to CausalObjectInfo
    c4_ahead_of_ego = dist_ahead_of_ego > AHEAD_OF_EGO_THRESHOLD_M

    has_critical = (
        c1_within_threshold
        and c2_exceptional
        and c3_aligned_with_peak
        and c4_ahead_of_ego
    )

    return {
        "total_causal_objects": len(causal_objects),
        "objects_in_causal_zone": objects_in_zone,
        "objects_within_critical_threshold": objects_within_critical,
        "min_adjusted_dist_to_baseline_m": closest_adjusted,
        "mean_of_others_cleaned_m": mean_of_others,
        "exceptionality_ratio": exceptionality,
        "closest_object_is_vru": closest.get("is_vru", False),
        "closest_object_dist_from_peak_m": dist_from_peak,
        "closest_object_signed_dist_to_peak_m": signed_dist_to_peak,
        "closest_object_future_signed_dist_to_peak_m": future_signed_dist_to_peak,
        "criterion1_within_threshold": c1_within_threshold,
        "criterion2_exceptional": c2_exceptional,
        "criterion3_aligned_with_peak": c3_aligned_with_peak,
        "criterion4_ahead_of_ego": c4_ahead_of_ego,
        "has_critical_object": has_critical,
        "closest_object_deviation_ratio": closest.get("deviation_at_object_ratio", 0.8),
    }


# ==================================================================================================
# Decision Logic V2 (improved metrics + single object rule)
# ==================================================================================================


def decide_bias_vs_nudge_v3(
    trajectory_metrics: Dict[str, Any], object_metrics: Dict[str, Any]
) -> Tuple[str, float, str]:
    """
    V3 decision logic — simplified two-question approach.
    Mirrors C++ shouldFilterAsRegularBias() in NudgeObjectAnalyzer.cpp.

    Question 1: Is there an exceptional object?
      - Multiple objects: exceptionality ratio >= 1.5 (one stands out)
      - Single object: automatically exceptional (nothing to compare)
      - No objects: BIAS

    Question 2: Is the exceptional object a valid nudge candidate?
      Both must pass:
      a) Ahead of ego (distance_ahead > -2m)
      b) Near the peak (either condition satisfies):
         Primary:  |dist_from_peak| <= 10m AND deviation_ratio >= 0.5
         Extended: 10m < signed_dist_to_peak <= 20m AND deviation_ratio >= 0.8
                   (handles nudge-abort-then-nudge pattern)

    If Q1=no → BIAS. If Q1=yes but Q2=no → BIAS. Both yes → NUDGE.
    """
    objects_in_zone = object_metrics.get("objects_in_causal_zone", 0)
    exceptionality = object_metrics.get("exceptionality_ratio", 0.0)
    min_dist = object_metrics.get("min_adjusted_dist_to_baseline_m", 999.0)
    dist_from_peak = object_metrics.get("closest_object_dist_from_peak_m", 999.0)
    ahead = object_metrics.get("criterion4_ahead_of_ego", False)
    deviation_ratio = object_metrics.get("closest_object_deviation_ratio", 0.0)
    # Fallback: compute from raw data if not directly available
    if deviation_ratio == 0.0:
        deviation_ratio = object_metrics.get("deviation_at_object_ratio", 0.0)

    # No objects at all
    if objects_in_zone == 0:
        return "bias", 0.9, "No causal objects"

    # Question 1: Is there an exceptional object?
    is_exceptional = False
    if objects_in_zone == 1:
        # Single object is automatically exceptional
        is_exceptional = True
    elif exceptionality >= 1.5:
        # Multiple objects but one stands out
        is_exceptional = True

    if not is_exceptional:
        confidence = 0.9
        reason = f"BIAS: Multiple objects ({objects_in_zone}), none exceptional (ratio={exceptionality:.2f})"
        return "bias", confidence, reason

    # Question 2: Is the exceptional object a valid nudge candidate?
    # 2a) Ahead of ego (distance_ahead > -2m, allows objects slightly behind ego front)
    cond_ahead = object_metrics.get("criterion4_ahead_of_ego", False)

    # 2b) Near the peak — two conditions (either satisfies):
    #   Primary: close to peak AND ego deviating (|dist| <= 10m AND ratio >= 0.5)
    #   Extended: object ahead of peak in nudge-abort-then-nudge pattern
    #             (10m < signed_dist <= 25m AND ratio >= 0.8)
    #             The abort creates a peak behind, but the real obstacle is ahead
    #             with the ego still strongly deviating at its location.
    signed_dist_to_peak = object_metrics.get(
        "closest_object_signed_dist_to_peak_m", 0.0
    )
    cond_near_peak_primary = dist_from_peak <= 10.0 and deviation_ratio >= 0.5
    cond_near_peak_extended = (
        10.0 < signed_dist_to_peak <= 20.0
    ) and deviation_ratio >= 0.8
    cond_near_peak = cond_near_peak_primary or cond_near_peak_extended

    is_valid_candidate = cond_ahead and cond_near_peak

    if is_valid_candidate:
        confidence = 0.9
        reason = (
            f"NUDGE: Exceptional object (ratio={exceptionality:.2f}, dist={min_dist:.2f}m, "
            f"peak_dist={dist_from_peak:.1f}m, dev_ratio={deviation_ratio:.2f})"
        )
        return "nudge", confidence, reason
    else:
        fails = []
        if not cond_ahead:
            fails.append("not ahead of ego")
        if not cond_near_peak:
            fails.append(
                f"not near peak (dist={dist_from_peak:.1f}m, dev_ratio={deviation_ratio:.2f})"
            )
        confidence = 0.7
        reason = f"BIAS: Exceptional object fails candidate check ({', '.join(fails)})"
        return "bias", confidence, reason


def is_object_far_from_peak_at_both_positions(
    current_signed_dist_m: float,
    future_signed_dist_m: float,
    threshold_m: float = 15.0,
) -> bool:
    """Check if object is far from peak at both current and future positions.

    Returns True only if both positions are on the SAME side of peak and beyond threshold.
    Mirrors C++ isObjectFarFromPeakAtBothPositions().
    """
    # If signs differ, object crosses the peak
    if (current_signed_dist_m > 0.0) != (future_signed_dist_m > 0.0):
        return False
    return (
        abs(current_signed_dist_m) > threshold_m
        and abs(future_signed_dist_m) > threshold_m
    )


def decide_bias_vs_nudge_v2(
    trajectory_metrics: Dict[str, Any], object_metrics: Dict[str, Any]
) -> Tuple[str, float, str]:
    """
    V2 decision logic matching C++ shouldFilterAsRegularBias() rule ordering.

    Rule ordering (first match wins):
    1. Multiple objects + low exceptionality → BIAS
    2. Critical object (all 4 criteria) → NUDGE
    3. Single object + aligned + ahead → NUDGE
    4. Far from peak at both positions → BIAS
    5. Exceptional nearby → NUDGE
    6. Default → NUDGE

    Returns:
        Tuple of (decision, confidence, reason)
    """
    # Extract object metrics
    has_critical = object_metrics.get("has_critical_object", False)
    objects_in_zone = object_metrics.get("objects_in_causal_zone", 0)
    exceptionality = object_metrics.get("exceptionality_ratio", 0.0)
    min_dist = object_metrics.get("min_adjusted_dist_to_baseline_m", 999.0)
    aligned = object_metrics.get("criterion3_aligned_with_peak", False)
    ahead = object_metrics.get("criterion4_ahead_of_ego", False)
    signed_dist_to_peak = object_metrics.get(
        "closest_object_signed_dist_to_peak_m", 0.0
    )
    future_signed_dist_to_peak = object_metrics.get(
        "closest_object_future_signed_dist_to_peak_m", 0.0
    )

    # Rule 1: Multiple objects in zone, none exceptional → BIAS (parked cars / convoy)
    if objects_in_zone >= 2 and exceptionality < 1.5:
        confidence = 0.9
        reason = f"Multiple objects ({objects_in_zone}), none exceptional (ratio={exceptionality:.2f})"
        return "bias", confidence, reason

    # Rule 2: Has critical object → NUDGE (all 4 criteria met)
    if has_critical:
        confidence = 0.95
        reason = f"Critical object (dist={min_dist:.2f}m, ratio={exceptionality:.2f})"
        return "nudge", confidence, reason

    # Rule 3: Single object aligned with peak and ahead → NUDGE
    if objects_in_zone == 1 and aligned and ahead:
        confidence = 0.8
        reason = f"Single object aligned with peak (dist={min_dist:.2f}m, peak_dist={object_metrics.get('closest_object_dist_from_peak_m', 999):.1f}m)"
        return "nudge", confidence, reason

    # Rule 4: Far from peak at both current and future position → BIAS
    # Must be checked BEFORE the exceptional-nearby rule, because an object can be
    # "exceptional" (VRU with large buffer) yet far from the peak (not causing the nudge).
    if is_object_far_from_peak_at_both_positions(
        signed_dist_to_peak, future_signed_dist_to_peak
    ):
        confidence = 0.7
        reason = f"Object far from peak (curr={signed_dist_to_peak:.1f}m, future={future_signed_dist_to_peak:.1f}m)"
        return "bias", confidence, reason

    # Rule 5: Exceptional object nearby but not full critical → NUDGE
    if exceptionality >= 2.0 and min_dist < 3.0:
        confidence = 0.7
        reason = f"Exceptional nearby object (ratio={exceptionality:.2f}, dist={min_dist:.2f}m)"
        return "nudge", confidence, reason

    # Default: NUDGE with low confidence (conservative)
    confidence = 0.3
    reason = f"Default nudge ({objects_in_zone} objects, ratio={exceptionality:.2f}, dist={min_dist:.2f}m)"
    return "nudge", confidence, reason


# ==================================================================================================
# Verification and Reporting
# ==================================================================================================


def verify_case(test_case: Dict[str, Any], version: int = 1) -> Dict[str, Any]:
    """Verify a single test case with multiple frames.

    Args:
        test_case: Test case data with ground_truth and frames.
        version: 1 for V1 (C++ metrics), 2 for V2 (improved metrics from raw objects).
    """
    ground_truth = test_case["ground_truth"]
    frames = test_case["frames"]

    frame_results = []
    for frame_idx, frame in enumerate(frames):
        hypothesis_summary = frame.get("hypothesis_summary", {})
        bias_nudge_decision = frame.get("bias_nudge_decision", {})
        trajectory_metrics = extract_trajectory_metrics(hypothesis_summary)

        if version == 3:
            # V3: simplified two-question approach
            causal_objects = bias_nudge_decision.get("causal_objects", [])
            merged = hypothesis_summary.get("merged_hypotheses", [])
            max_dev = merged[0].get("max_lateral_deviation_m", 0) if merged else 0
            object_metrics = compute_improved_object_metrics(causal_objects, max_dev)
            # Override with top-level fields from debug info (more reliable than causal object fallbacks);
            # default to 0.8 / 0.0 for old cases that don't have these fields
            object_metrics["closest_object_deviation_ratio"] = bias_nudge_decision.get(
                "closest_object_deviation_ratio", 0.8
            )
            if "closest_object_signed_dist_to_peak_m" in bias_nudge_decision:
                object_metrics[
                    "closest_object_signed_dist_to_peak_m"
                ] = bias_nudge_decision["closest_object_signed_dist_to_peak_m"]
            if "closest_object_future_signed_dist_to_peak_m" in bias_nudge_decision:
                object_metrics[
                    "closest_object_future_signed_dist_to_peak_m"
                ] = bias_nudge_decision["closest_object_future_signed_dist_to_peak_m"]
            decision, confidence, reason = decide_bias_vs_nudge_v3(
                trajectory_metrics, object_metrics
            )
        elif version == 2:
            # V2: recompute metrics from raw causal objects
            causal_objects = bias_nudge_decision.get("causal_objects", [])
            merged = hypothesis_summary.get("merged_hypotheses", [])
            max_dev = merged[0].get("max_lateral_deviation_m", 0) if merged else 0
            object_metrics = compute_improved_object_metrics(causal_objects, max_dev)
            decision, confidence, reason = decide_bias_vs_nudge_v2(
                trajectory_metrics, object_metrics
            )
        else:
            # V1: use C++ metrics directly
            object_metrics = extract_object_metrics(bias_nudge_decision)
            decision, confidence, reason = decide_bias_vs_nudge(
                trajectory_metrics, object_metrics
            )

        correct = decision == ground_truth

        frame_results.append(
            {
                "frame_idx": frame_idx,
                "decision": decision,
                "confidence": confidence,
                "reason": reason,
                "correct": correct,
                "trajectory_metrics": trajectory_metrics,
                "object_metrics": object_metrics,
            }
        )

    # Aggregate: majority voting
    decisions = [f["decision"] for f in frame_results]
    nudge_votes = decisions.count("nudge")
    bias_votes = decisions.count("bias")

    if nudge_votes > bias_votes:
        aggregate_decision = "nudge"
        aggregate_confidence = nudge_votes / len(decisions)
    else:
        aggregate_decision = "bias"
        aggregate_confidence = bias_votes / len(decisions)

    aggregate_correct = aggregate_decision == ground_truth
    all_frames_correct = all(f["correct"] for f in frame_results)

    return {
        "ground_truth": ground_truth,
        "frames": frame_results,
        "aggregate_decision": aggregate_decision,
        "aggregate_confidence": aggregate_confidence,
        "aggregate_correct": aggregate_correct,
        "all_frames_correct": all_frames_correct,
        "agreement": nudge_votes if aggregate_decision == "nudge" else bias_votes,
    }


def print_case_result(
    case_id: str,
    test_case: Dict[str, Any],
    result: Dict[str, Any],
    verbose: bool = True,
):
    """Print the results for a single test case."""

    gt = result["ground_truth"].upper()
    agg_decision = result["aggregate_decision"].upper()
    agg_status = "CORRECT" if result["aggregate_correct"] else "WRONG"
    all_frames_status = "ALL CORRECT" if result["all_frames_correct"] else "SOME WRONG"
    agreement = result["agreement"]
    total_frames = len(result["frames"])

    print(f"\n{'=' * 80}")
    print(f"Case: {case_id} - {test_case['name']}")
    print(f"{'=' * 80}")
    print(f"Ground Truth:      {gt}")
    print(
        f"Aggregate Decision: {agg_decision} ({agreement}/{total_frames} frames agree) {agg_status}"
    )
    print(f"Frame Agreement:   {all_frames_status}")

    print(f"\n{'-' * 80}")
    print("Frame-by-Frame Results:")
    print(f"{'-' * 80}")

    for frame_result in result["frames"]:
        frame_idx = frame_result["frame_idx"]
        decision = frame_result["decision"].upper()
        conf = frame_result["confidence"]
        status = "OK" if frame_result["correct"] else "FAIL"

        print(f"\n  Frame {frame_idx + 1}: [{status}] {decision} (conf: {conf:.2f})")
        print(f"    Reason: {frame_result['reason']}")

        if verbose:
            tm = frame_result["trajectory_metrics"]
            om = frame_result["object_metrics"]

            print(
                f"    Trajectory: PPR={tm.get('peak_prominence_ratio', 0):.2f}, "
                f"PFI={tm.get('plateau_flatness_index', 0):.3f}, "
                f"CV={tm.get('deviation_cv', 0):.2f}"
            )

            print(
                f"    Object:     {om.get('objects_in_causal_zone', 0)} objects, "
                f"dist={om.get('min_adjusted_dist_to_baseline_m', 999):.2f}m, "
                f"ratio={om.get('exceptionality_ratio', 0):.2f}, "
                f"critical={om.get('has_critical_object', False)}"
            )

    if verbose and result["frames"]:
        frame0 = result["frames"][0]
        tm = frame0["trajectory_metrics"]
        om = frame0["object_metrics"]

        print(f"\n{'-' * 80}")
        print("Detailed Metrics (Frame 1):")
        print(f"{'-' * 80}")

        print(f"\nTrajectory Metrics:")
        print(f"  Deviation Mean:          {tm.get('deviation_mean_m', 0):.3f} m")
        print(f"  Deviation Std:           {tm.get('deviation_std_m', 0):.3f} m")
        print(f"  Deviation CV:            {tm.get('deviation_cv', 0):.3f}")
        print(f"  Peak Prominence Ratio:   {tm.get('peak_prominence_ratio', 0):.3f}")
        print(f"  Plateau Flatness Index:  {tm.get('plateau_flatness_index', 0):.4f}")
        print(f"  Baseline Point Count:    {tm.get('baseline_point_count', 0)}")

        print(f"\nObject Metrics:")
        print(f"  Total Causal Objects:    {om.get('total_causal_objects', 0)}")
        print(f"  Objects in Causal Zone:  {om.get('objects_in_causal_zone', 0)}")
        print(
            f"  Objects Within Critical: {om.get('objects_within_critical_threshold', 0)}"
        )
        print(
            f"  Min Adjusted Distance:   {om.get('min_adjusted_dist_to_baseline_m', 999):.3f} m"
        )
        mean_others = om.get("mean_of_others_in_causal_zone_m")
        print(
            f"  Mean of Others:          {f'{mean_others:.3f} m' if mean_others is not None else 'N/A (single object)'}"
        )
        print(f"  Exceptionality Ratio:    {om.get('exceptionality_ratio', 0):.3f}")
        print(f"  Closest is VRU:          {om.get('closest_object_is_vru', False)}")
        print(f"  Has Critical Object:     {om.get('has_critical_object', False)}")

        print(f"\nCriteria Breakdown:")
        print(
            f"  Within Threshold:        {om.get('criterion1_within_threshold', False)}"
        )
        print(f"  Exceptional:             {om.get('criterion2_exceptional', False)}")
        print(
            f"  Aligned with Peak:       {om.get('criterion3_aligned_with_peak', False)}"
        )
        print(f"  Ahead of Ego:            {om.get('criterion4_ahead_of_ego', False)}")

    if "notes" in test_case:
        print(f"\nNotes: {test_case['notes']}")

    return result["aggregate_correct"]


def run_all_tests(test_cases: Dict[str, Dict], verbose: bool = False, version: int = 1):
    """Run all test cases and print summary."""

    if not test_cases:
        print(
            "No test cases available yet. Add cases to bias_nudge_decision_data.json."
        )
        return

    version_label = f"V{version}"
    results = {}
    for case_id in sorted(test_cases.keys()):
        test_case = test_cases[case_id]
        result = verify_case(test_case, version=version)
        passed = print_case_result(case_id, test_case, result, verbose)
        results[case_id] = {"passed": passed, "result": result, "test_case": test_case}

    # Summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY ({version_label})")
    print(f"{'=' * 80}")

    total = len(results)
    correct_count = sum(1 for r in results.values() if r["passed"])

    nudge_cases = [
        r for r in results.values() if r["test_case"]["ground_truth"] == "nudge"
    ]
    bias_cases = [
        r for r in results.values() if r["test_case"]["ground_truth"] == "bias"
    ]

    nudge_correct = sum(1 for r in nudge_cases if r["passed"])
    bias_correct = sum(1 for r in bias_cases if r["passed"])

    print(
        f"\nOverall Accuracy: {correct_count}/{total} ({100*correct_count/total:.1f}%)"
    )
    if nudge_cases:
        print(
            f"Nudge Cases:      {nudge_correct}/{len(nudge_cases)} ({100*nudge_correct/len(nudge_cases):.1f}%)"
        )
    if bias_cases:
        print(
            f"Bias Cases:       {bias_correct}/{len(bias_cases)} ({100*bias_correct/len(bias_cases):.1f}%)"
        )

    print(f"\nDetailed Results:")
    for case_id in sorted(results.keys()):
        r = results[case_id]
        status = "OK" if r["passed"] else "FAIL"
        gt = r["test_case"]["ground_truth"].upper()
        decision = r["result"]["aggregate_decision"].upper()
        conf = r["result"]["aggregate_confidence"]
        print(
            f"  [{status}] {case_id}: GT={gt:<5} Decision={decision:<5} (conf={conf:.2f}) | {r['test_case']['name']}"
        )

    print(f"{'=' * 80}\n")
    return results


def run_comparison(test_cases: Dict[str, Dict]):
    """Run both V1 and V2 and compare results side by side."""

    if not test_cases:
        print("No test cases available yet.")
        return

    print(f"\n{'=' * 90}")
    print("V1 vs V2 COMPARISON")
    print(f"{'=' * 90}")

    v1_correct = 0
    v2_correct = 0
    total = len(test_cases)
    changes = []

    print(f"\n{'Case':<35} {'GT':>5} {'V1':>8} {'V2':>8} {'Change':>10}")
    print("-" * 75)

    for case_id in sorted(test_cases.keys()):
        tc = test_cases[case_id]
        gt = tc["ground_truth"].upper()

        r1 = verify_case(tc, version=1)
        r2 = verify_case(tc, version=2)

        d1 = r1["aggregate_decision"].upper()
        d2 = r2["aggregate_decision"].upper()
        s1 = "OK" if r1["aggregate_correct"] else "FAIL"
        s2 = "OK" if r2["aggregate_correct"] else "FAIL"

        if r1["aggregate_correct"]:
            v1_correct += 1
        if r2["aggregate_correct"]:
            v2_correct += 1

        change = ""
        if s1 != s2:
            if s2 == "OK":
                change = "FIXED"
            else:
                change = "REGRESSED"
            changes.append((case_id, s1, s2, change))

        print(f"  {case_id:<33} {gt:>5} {d1:>5}({s1}) {d2:>5}({s2}) {change:>10}")

    print(f"\n{'=' * 90}")
    print(f"V1 Accuracy: {v1_correct}/{total} ({100*v1_correct/total:.1f}%)")
    print(f"V2 Accuracy: {v2_correct}/{total} ({100*v2_correct/total:.1f}%)")

    if changes:
        print(f"\nChanges:")
        for case_id, s1, s2, change in changes:
            print(f"  {change}: {case_id} ({s1} -> {s2})")
    else:
        print(f"\nNo changes between V1 and V2.")

    # Show per-frame detail for changed cases
    if changes:
        print(f"\n{'=' * 90}")
        print("DETAIL ON CHANGED CASES")
        print(f"{'=' * 90}")
        for case_id, s1, s2, change in changes:
            tc = test_cases[case_id]
            r1 = verify_case(tc, version=1)
            r2 = verify_case(tc, version=2)
            print(f"\n  {case_id} ({change}): GT={tc['ground_truth'].upper()}")
            for fi in range(len(r1["frames"])):
                f1 = r1["frames"][fi]
                f2 = r2["frames"][fi]
                print(f"    F{fi+1}: V1={f1['decision'].upper()} ({f1['reason']})")
                print(f"        V2={f2['decision'].upper()} ({f2['reason']})")
                # Show V2 improved metrics
                om2 = f2["object_metrics"]
                print(
                    f"        V2 metrics: ratio={om2.get('exceptionality_ratio', 0):.2f}, "
                    f"mean_cleaned={om2.get('mean_of_others_cleaned_m')}, "
                    f"dist={om2.get('min_adjusted_dist_to_baseline_m', 999):.2f}m"
                )

    print(f"{'=' * 90}\n")


def show_summary(test_cases: Dict[str, Dict]):
    """Show summary statistics of all test cases."""

    if not test_cases:
        print("No test cases available yet.")
        return

    print(f"\n{'=' * 80}")
    print("TEST CASES SUMMARY")
    print(f"{'=' * 80}")

    nudge_count = sum(1 for tc in test_cases.values() if tc["ground_truth"] == "nudge")
    bias_count = sum(1 for tc in test_cases.values() if tc["ground_truth"] == "bias")

    print(f"\nTotal Test Cases: {len(test_cases)}")
    print(f"  Nudge: {nudge_count}")
    print(f"  Bias:  {bias_count}")

    print(f"\nAvailable Test Cases:")
    for case_id in sorted(test_cases.keys()):
        tc = test_cases[case_id]
        gt = tc["ground_truth"].upper()
        n_frames = len(tc.get("frames", []))
        print(f"  [{gt:<5}] {case_id} ({n_frames} frames) - {tc['name']}")

    print(f"{'=' * 80}\n")


# ==================================================================================================
# Main Entry Point
# ==================================================================================================


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "  python bias_nudge_decision_experiment.py all [-v] [--v2]  # Run all test cases"
        )
        print(
            "  python bias_nudge_decision_experiment.py compare          # Compare V1 vs V2"
        )
        print(
            "  python bias_nudge_decision_experiment.py <case_id> [--v2] # Run specific test case"
        )
        print(
            "  python bias_nudge_decision_experiment.py summary          # Show summary statistics"
        )
        sys.exit(1)

    test_cases = load_test_cases()
    command = sys.argv[1]
    if "--v3" in sys.argv:
        version = 3
    elif "--v2" in sys.argv:
        version = 2
    else:
        version = 1

    if command == "all":
        verbose = "--verbose" in sys.argv or "-v" in sys.argv
        run_all_tests(test_cases, verbose=verbose, version=version)

    elif command == "compare":
        run_comparison(test_cases)

    elif command == "summary":
        show_summary(test_cases)

    else:
        case_id = command
        if case_id not in test_cases:
            print(f"Error: Test case '{case_id}' not found")
            print(f"Available: {', '.join(sorted(test_cases.keys()))}")
            sys.exit(1)

        test_case = test_cases[case_id]
        result = verify_case(test_case, version=version)
        print_case_result(case_id, test_case, result, verbose=True)


if __name__ == "__main__":
    main()
