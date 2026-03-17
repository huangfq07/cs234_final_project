"""
Shared data types for nudge detection pipeline.

This file is the Python equivalent of the C++ header files. It defines all
dataclasses (structs), configuration constants, and shared utility functions
used across the detection modules.

C++ header mapping:
  DgpsTrajectoryProcessor.hpp    -> DgpsPoint
  DgpsTrajectoryAnalyzer.hpp     -> DeviationPeak, PeakednessMetrics, TrajectoryFeatures, NudgeHypothesis
  PreprocessedObjectInfo.hpp     -> PreprocessedObjectInfo
  BiasNudgeDecider.hpp           -> BiasNudgeOutput
  NudgeClassifierOutput.hpp      -> NudgeEvent
  NudgePostprocessorUtils.hpp    -> Config, calculate_in_lane_threshold(), compute_vru_safety_margin()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ==================================================================================================
# Config — all tunable constants from the C++ pipeline
#
# Each group is annotated with the C++ source file where the default value is defined.
# When tuning parameters in Python, change them here; when porting back to C++, update
# the corresponding constant in the C++ file.
# ==================================================================================================


@dataclass
class Config:
    """Configuration parameters matching C++ defaults."""

    # --- DgpsTrajectoryAnalyzer.hpp::Config ---
    # Minimum |deviation| for a point to be considered a peak (absolute threshold).
    min_nudge_deviation_m: float = 0.4
    # After detrending by baseline, peaks below this are filtered out.
    min_peak_after_detrend_m: float = 0.3
    # If ego deviation at s=0 exceeds peak * this ratio, the peak is a return-to-center, not avoidance.
    ego_deviation_to_peak_ratio: float = 1.1
    # Start of deviation window = where |dev| drops below peak * this factor (scanning backward).
    threshold_factor_start: float = 0.3
    # End of deviation window = where |dev| drops below peak * this factor (scanning forward).
    threshold_factor_end: float = 0.5
    # Hypotheses with overlapping time windows or gap < this are merged into one.
    merge_time_s: float = 2.5
    # Minimum duration for a valid hypothesis (speed-adaptive: max(this, min_distance/speed)).
    min_hypothesis_duration_s: float = 0.5
    # Minimum maneuver distance — hypothesis valid if distance >= this even if duration is short.
    min_hypothesis_distance_m: float = 4.0
    # Cumulative bias must exceed this to assign LEFT/RIGHT direction (avoids noise).
    direction_threshold_m: float = 0.5

    # --- NudgePostprocessorUtils.hpp — straight trajectory filter ---
    # If trajectory within peak window has maxPerp < 0.2m AND rms < 0.1m, it's straight
    # (the peak is from road geometry, not an avoidance maneuver).
    straight_max_perp_m: float = 0.2
    straight_max_rms_m: float = 0.1
    # Number of trajectory points before/after the window to use as baseline for straight check.
    baseline_points_per_side: int = 5

    # --- NudgeObjectAnalyzer.cpp constants ---
    # Longitudinal buffer: objects must overlap [hyp.start - buffer, hyp.end + buffer].
    longitudinal_buffer_m: float = 10.0
    # Safety margin for non-VRU objects (VRU uses speed-dependent margin instead).
    vehicle_safety_margin_m: float = 0.3
    # Maximum distance ahead for new (untracked) objects to be considered.
    max_longitudinal_dist_m: float = 55.0
    # Maximum lateral distance from lane center for new objects.
    max_lateral_dist_m: float = 3.0

    # --- BiasNudgeDecider.cpp constants ---
    # Adjusted distance buffers: VRU get 1.0m subtracted (they need more clearance),
    # large vehicles 0.6m, perpendicular vehicles 1.0m.
    vru_buffer_m: float = 1.0
    large_vehicle_buffer_m: float = 0.6
    perpendicular_buffer_m: float = 1.0
    # Heading thresholds to detect perpendicular vehicles (radians).
    perpendicular_threshold_rad: float = 1.3
    perpendicular_upper_bound_rad: float = 1.84159265
    # Objects within this raw distance from baseline are in the "causal zone" (counted for ratios).
    causal_zone_m: float = 4.0
    # Objects with adjusted dist <= this are "within critical threshold".
    critical_threshold_m: float = 2.0
    # Exceptionality ratio threshold: if mean_of_others / closest_adj_dist >= this, object is exceptional.
    exceptionality_threshold: float = 1.5
    # Objects must be ahead of this threshold (allows 2m behind ego front).
    ahead_of_ego_threshold_m: float = -2.0
    # Q2b primary zone: |dist_from_peak| <= 10m AND deviation_ratio >= 0.5.
    primary_peak_dist_m: float = 10.0
    primary_dev_ratio: float = 0.5
    # Q2b extended zone: 10m < signed_dist <= 20m AND deviation_ratio >= 0.8.
    # Catches nudge-abort-then-nudge pattern where peak is behind but obstacle is ahead.
    extended_min_signed_dist_m: float = 10.0
    extended_max_signed_dist_m: float = 20.0
    extended_dev_ratio: float = 0.8

    # --- Peakedness metrics ---
    # Number of points before/after peak window used as baseline for Peak Prominence Ratio.
    peakedness_baseline_points: int = 5
    # Fraction of top deviation values used for Plateau Flatness Index.
    peakedness_top_fraction: float = 0.4


# ==================================================================================================
# DgpsTrajectoryProcessor.hpp
# ==================================================================================================


@dataclass
class DgpsPoint:
    """Single DGPS trajectory point.

    Matches C++ DgpsTrajectoryProcessor::DgpsPoint.
    Coordinates are in ego-local frame (typically ENU or map frame).
    """

    egoX: float  # Ego X position (meters)
    egoY: float  # Ego Y position (meters)
    velX: float = 0.0  # Ego velocity X (m/s)
    velY: float = 0.0  # Ego velocity Y (m/s)
    cameraTime: int = 0  # Timestamp (microseconds since epoch)


# ==================================================================================================
# DgpsTrajectoryAnalyzer.hpp
# ==================================================================================================


@dataclass
class DeviationPeak:
    """A local maximum in lateral deviation from lane center.

    Matches C++ DeviationPeak struct.
    Detected where |dev[i]| > |dev[i-1]| AND |dev[i]| > |dev[i+1]| AND |dev[i]| > threshold.
    """

    index: int  # Index in the deviation array
    time_s: float  # Time relative to trajectory start (seconds)
    magnitude_m: float  # Absolute magnitude of the peak (meters)
    signed_deviation_m: float  # Signed lateral deviation (positive=left, negative=right)
    direction: str  # "LEFT", "RIGHT", or "UNSPECIFIED"


@dataclass
class PeakednessMetrics:
    """Shape characteristics of a deviation window — used for bias vs nudge differentiation.

    Matches C++ PeakednessMetrics struct.
    A sharp peak (high PPR, low PFI) suggests a single obstacle nudge.
    A flat plateau (low PPR, high PFI) suggests systematic bias from multiple objects.
    """

    deviation_mean_m: float = 0.0  # Mean |deviation| in the window
    deviation_std_m: float = 0.0  # Std dev of |deviation| in the window
    deviation_cv: float = 0.0  # Coefficient of variation (std/mean)
    peak_prominence_ratio: float = 0.0  # (peak - baseline_mean) / baseline_std
    plateau_flatness_index: float = (
        0.0  # Variance of top 40% deviations (low = flat top)
    )
    baseline_point_count: int = 0  # Number of baseline points used for PPR


@dataclass
class TrajectoryFeatures:
    """Accumulated trajectory features for a hypothesis window.

    Matches C++ NudgeHypothesis::trajectoryFeatures.
    These are computed from the extended-field deviations within [start, end].
    """

    max_lateral_deviation_m: float = 0.0  # Peak |deviation| in window
    avg_lateral_deviation_m: float = 0.0  # Mean |deviation| in window
    deviation_duration_s: float = 0.0  # Duration of the maneuver (seconds)
    out_of_lane_duration_s: float = 0.0  # Time spent outside lane boundary
    lateral_jerk_m_s3: float = 0.0  # Maximum lateral jerk (m/s^3)
    returned_to_center: bool = False  # True if deviation < 0.1m at window end
    total_maneuver_distance_m: float = 0.0  # Longitudinal distance covered by window
    dominant_direction: str = "UNSPECIFIED"  # Overall direction from cumulative bias
    peakedness_metrics: PeakednessMetrics = field(default_factory=PeakednessMetrics)


@dataclass
class NudgeHypothesis:
    """A detected nudge hypothesis — a time/spatial window around a deviation peak.

    Matches C++ DgpsTrajectoryAnalysisOutput (aliased as NudgeHypothesis).
    Each hypothesis represents a potential nudge maneuver: ego deviated laterally
    (start -> peak -> return to center) within a specific window.
    """

    start_time_s: float = 0.0  # When deviation began (relative to trajectory start)
    peak_time_s: float = 0.0  # When deviation was maximum
    end_time_s: float = 0.0  # When ego returned to near-center
    start_s_m: float = 0.0  # Longitudinal station at start (Frenet s)
    peak_s_m: float = 0.0  # Longitudinal station at peak
    end_s_m: float = 0.0  # Longitudinal station at end
    direction: str = "UNSPECIFIED"  # "LEFT" or "RIGHT" (from cumulative bias)
    type: str = "UNSPECIFIED"  # "IN_LANE" or "OUT_OF_LANE" (from peak vs lane width)
    left_bias_sum_m: float = 0.0  # Cumulative leftward deviation in window
    right_bias_sum_m: float = 0.0  # Cumulative rightward deviation in window
    peak_index: int = 0  # Index of the peak in the deviation array
    adaptive_baseline_m: float = 0.0  # Baseline used for detrending
    ego_lane_width_m: float = 0.0  # Lane width at detection time
    trajectory_features: TrajectoryFeatures = field(default_factory=TrajectoryFeatures)


# ==================================================================================================
# NudgeObjectAnalyzer.hpp / PreprocessedObjectInfo.hpp
#
# All distances are relative to the LANE CENTERLINE (not the planning reference line).
# The C++ preprocessing converts from reference-line-relative to centerline-relative.
# ==================================================================================================


@dataclass
class PreprocessedObjectInfo:
    """Preprocessed object information.

    Matches C++ PreprocessedObjectInfo struct. All lateral distances use the
    minimum-edge convention (closest bbox edge, not center), and are relative
    to lane centerline.
    """

    objectId: int = 0
    obstacleClass: str = "UNKNOWN"  # VEHICLE, PEDESTRIAN, CYCLIST, etc.
    obstacleState: str = "UNKNOWN"  # PARKED, STOPPED, MOVING, REVERSE
    speed_mps: float = 0.0
    isVRU: bool = False  # Vulnerable Road User (pedestrian, cyclist, etc.)
    isVehicle: bool = False
    isLargeVehicle: bool = False  # Truck, bus, etc.
    isOncomingVehicle: bool = False
    isOnLeftOfTrajectory: bool = False  # Object center is left of DGPS trajectory
    isOnLeftOfBaseline: bool = False  # Object center is left of adaptive baseline

    # Lateral distances (meters, signed: positive=left, negative=right)
    minLateralDistToLaneCenter_m: float = 0.0  # Closest bbox edge to lane center
    lateralDistToTrajectory_m: float = 0.0  # Center to DGPS trajectory
    minLateralDistToTrajectory_m: float = 0.0  # Closest edge to DGPS trajectory
    minLateralDistToBaseline_m: float = 0.0  # Closest edge to adaptive baseline

    # Longitudinal distances (meters)
    distanceAheadOfEgo_m: float = 0.0  # Object rear to ego front (+ahead, -behind)
    distancePassedByEgo_m: float = 0.0  # Ego rear to object front (+passed, -not yet)
    centerSOnLaneCenter_m: float = 0.0  # Object center longitudinal station (Frenet s)
    bboxLongitudinalMin_m: float = 0.0  # Bbox rear edge station
    bboxLongitudinalMax_m: float = 0.0  # Bbox front edge station

    # Future projection (for slow oncoming / crossing detection)
    hasFutureProjection: bool = False
    futureDistToTrajectory_m: float = 0.0
    futureMinLateralDistToTrajectory_m: float = 0.0
    futureCenterSOnLaneCenter_m: float = 0.0

    # Heading relative to lane center (radians, used for perpendicular vehicle detection)
    headingDiffWithLaneCenter: float = 0.0

    # In-lane margins: how far the object's closest edge is from the lane boundary
    # Positive = inside lane, negative = protruding into lane
    inLaneLeftMargin_m: float = 0.0
    inLaneRightMargin_m: float = 0.0

    stateFlags: int = 0  # Signal lights (hazard, emergency)
    description: str = ""  # Optional pre-built description string


# ==================================================================================================
# BiasNudgeDecider.hpp
# ==================================================================================================


@dataclass
class BiasNudgeOutput:
    """Output of the bias vs nudge decision.

    Matches C++ bias_nudge::BiasNudgeOutput.
    Contains the binary BIAS/NUDGE decision plus all intermediate metrics
    for debugging (zone counts, exceptionality ratio, Q1/Q2 results).
    """

    is_bias: bool = True  # True = BIAS (systematic), False = NUDGE (real)
    reason: str = ""  # Human-readable decision explanation
    q1_is_exceptional: bool = False  # Q1: Is there an exceptional object?
    q2a_ahead_of_ego: bool = False  # Q2a: Is it ahead of ego?
    q2b_near_peak: bool = False  # Q2b: Is it near the deviation peak?
    objects_in_causal_zone: int = 0  # Objects with raw dist <= 4.0m
    objects_within_critical: int = 0  # Objects with adjusted dist <= 2.0m
    exceptionality_ratio: float = 0.0  # trimmed_mean(others) / closest_adjusted_dist
    mean_of_others_m: float = 0.0  # Trimmed mean of other objects in causal zone
    closest_object_id: int = 0
    closest_adjusted_dist: float = float(
        "inf"
    )  # After class-specific buffer subtraction
    closest_raw_dist: float = 0.0  # Before buffer subtraction
    closest_is_vru: bool = False
    closest_dist_from_peak_m: float = 0.0  # |signed_dist|
    closest_signed_dist_to_peak_m: float = (
        0.0  # Object center s - peak s (positive=ahead)
    )
    closest_deviation_ratio: float = 0.0  # dev(object_s) / dev(peak_s)
    closest_future_signed_dist_m: float = 0.0  # Future position signed dist to peak
    closest_future_deviation_ratio: float = 0.0  # Future position deviation ratio


# ==================================================================================================
# NudgeClassifierOutput.hpp
# ==================================================================================================


@dataclass
class NudgeEvent:
    """A classified nudge event with matched object and reasoning.

    Matches C++ NudgeClassifierOutput::NudgeEvent.
    Built from a hypothesis + its matched object + the bias/nudge decision.
    """

    critical_object_id: int = 0
    obstacle_class: str = "UNKNOWN"
    obstacle_state: str = "UNKNOWN"
    nudge_direction: str = "UNSPECIFIED"  # LEFT or RIGHT
    candidate_type: str = "UNSPECIFIED"  # IN_LANE or OUT_OF_LANE
    start_time_s: float = 0.0
    peak_time_s: float = 0.0
    end_time_s: float = 0.0
    max_lateral_deviation_m: float = 0.0
    adjusted_dist_m: float = float("inf")  # Primary sorting key (closest wins)
    distance_ahead_of_ego_m: float = 0.0
    object_speed_mps: float = 0.0
    object_was_stationary: bool = False
    object_was_oncoming: bool = False
    is_bilateral: bool = False  # Both left and right bias > 0.5m
    reasoning: str = (
        ""  # e.g. "out of lane nudge left to a parked vehicle on the right"
    )
    description: str = ""  # e.g. "a parked vehicle"


@dataclass
class DetectionResult:
    """Full pipeline output with debug info. Used by the nudge_detection.py orchestrator."""

    is_nudge_scenario: bool = False
    events: List[NudgeEvent] = field(default_factory=list)
    primary_event: Optional[NudgeEvent] = None
    reasoning: str = ""
    debug: Dict[str, Any] = field(default_factory=dict)


# ==================================================================================================
# NudgePostprocessorUtils.hpp — shared utility functions
# ==================================================================================================


def calculate_in_lane_threshold(lane_width_m: float, vehicle_width_m: float) -> float:
    """Calculate the in-lane threshold: deviation beyond this is out-of-lane.

    Matches nudge_utils::calculateInLaneThreshold().
    Formula: (lane_width / 2) - (vehicle_width / 2) + tolerance
    The 0.1m tolerance avoids false out-of-lane near the lane edge.
    """
    return (lane_width_m * 0.5) - (vehicle_width_m * 0.5) + 0.1


def compute_vru_safety_margin(ego_speed_mps: float) -> float:
    """Speed-dependent VRU safety margin for the safety clearance filter.

    Matches nudge_utils::computeVruSafetyMargin().
    At low speed (parking lot), drivers pass closer to VRUs.
    At high speed, more lateral margin is needed.
    """
    if ego_speed_mps < 5.0:
        return 0.3  # Parking-lot speed
    if ego_speed_mps < 10.0:
        return 0.6  # Urban speed
    return 1.0  # Higher speed
