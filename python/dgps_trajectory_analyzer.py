"""
DGPS Trajectory Analyzer — Python replication of DgpsTrajectoryAnalyzer.cpp

Analyzes the ego vehicle's DGPS trajectory to detect lateral deviation peaks
that may indicate nudge maneuvers (swerving around obstacles).

Algorithm overview:
  1. DETREND: Subtract the adaptive baseline from lateral deviations.
     The baseline represents the driver's habitual lateral position in wide lanes
     (e.g., 0.5m left of center). Detrending removes this bias so peaks represent
     true avoidance maneuvers, not positioning preference.

  2. PEAK DETECTION: Find local maxima in |lateral deviation|.
     A peak at index i must satisfy:
       |dev[i]| > |dev[i-1]| AND |dev[i]| > |dev[i+1]| AND |dev[i]| > 0.4m
     Peaks are filtered by:
       - Minimum magnitude after detrend (0.3m)
       - Ego-deviation ratio: if ego is already offset more than the peak, this is
         a return-to-center, not an avoidance maneuver

  3. HYPOTHESIS WINDOWS: For each surviving peak, find the start and end:
     - Start: scan backward from peak until |dev| < peak * 0.3  (30% threshold)
     - End: scan forward until |dev| < peak * 0.5  (50% threshold)
     - Uses near-field deviations (0-40m) for start, extended (0-60m) for end
     - Filters out straight trajectory segments (road geometry, not avoidance)
     - Validates by speed-adaptive duration OR distance >= 4m

  4. MERGING: Hypotheses that overlap in time or have gap < 2.5s are merged.
     The merged hypothesis takes the larger peak's characteristics.

  5. TYPE CLASSIFICATION: IN_LANE if peak < in-lane threshold, else OUT_OF_LANE.

Key coordinate system:
  Deviations are [s, l] pairs in Frenet coordinates:
    s = longitudinal distance along lane centerline (meters)
    l = lateral offset from lane centerline (positive=left, negative=right)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .data_types import (
    calculate_in_lane_threshold,
    Config,
    DeviationPeak,
    DgpsPoint,
    NudgeHypothesis,
    PeakednessMetrics,
    TrajectoryFeatures,
)


class DgpsTrajectoryAnalyzer:
    """Matches C++ DgpsTrajectoryAnalyzer class."""

    def __init__(self, config: Config):
        self.config = config

    # ==============================================================================================
    # Public API — matches DgpsTrajectoryAnalyzer::process()
    # ==============================================================================================

    def analyze(
        self,
        nf_deviations: List[List[float]],
        ext_deviations: List[List[float]],
        trajectory: List[DgpsPoint],
        adaptive_baseline_m: float,
        ego_lane_width_m: float,
        vehicle_width_m: float,
        use_adaptive_baseline: bool = True,
    ) -> Tuple[List[NudgeHypothesis], Dict[str, Any]]:
        """Run full trajectory analysis pipeline.

        Args:
            nf_deviations: Near-field [s, l] pairs (0-40m range, for peak detection).
            ext_deviations: Extended [s, l] pairs (0-60m range, for end detection).
            trajectory: DGPS points with positions, velocities, and timestamps.
            adaptive_baseline_m: Driver's habitual lateral offset to subtract.
            ego_lane_width_m: Current ego lane width.
            vehicle_width_m: Ego vehicle width.

        Returns:
            (hypotheses, debug_info) — list of validated/merged hypotheses + debug dict.
        """
        debug: Dict[str, Any] = {}

        # Step 1: Detrend — subtract baseline from lateral component only
        nf_devs = nf_deviations
        ext_devs = ext_deviations
        if use_adaptive_baseline and abs(adaptive_baseline_m) > 1e-6:
            nf_devs = self.detrend_deviations(nf_deviations, adaptive_baseline_m)
            ext_devs = self.detrend_deviations(ext_deviations, adaptive_baseline_m)

        debug["baseline_subtracted_m"] = (
            adaptive_baseline_m if use_adaptive_baseline else 0.0
        )
        debug["nf_point_count"] = len(nf_devs)
        debug["ext_point_count"] = len(ext_devs)

        # Step 2: Detect peaks in near-field deviations
        all_peaks = self.detect_local_maxima(nf_devs, trajectory)
        debug["peaks_before_filter"] = len(all_peaks)

        # Step 3: Filter peaks by magnitude and ego-deviation ratio
        filtered_peaks = self.filter_peaks(all_peaks, nf_devs, use_adaptive_baseline)
        debug["peaks_after_filter"] = len(filtered_peaks)
        debug["peaks"] = [
            {
                "index": p.index,
                "time_s": round(p.time_s, 3),
                "magnitude_m": round(p.magnitude_m, 3),
                "direction": p.direction,
                "survived_filter": p.index in {fp.index for fp in filtered_peaks},
            }
            for p in all_peaks
        ]

        if not filtered_peaks:
            return [], debug

        in_lane_threshold_m = calculate_in_lane_threshold(
            ego_lane_width_m, vehicle_width_m
        )

        # Step 4: Create hypothesis windows around each peak
        hypotheses = self._create_hypothesis_windows(
            filtered_peaks,
            nf_devs,
            ext_devs,
            trajectory,
            ego_lane_width_m,
            adaptive_baseline_m,
            in_lane_threshold_m,
        )
        debug["hypotheses_before_merge"] = len(hypotheses)

        if not hypotheses:
            return [], debug

        # Step 5: Merge overlapping or close hypotheses
        hypotheses = self._merge_close_hypotheses(hypotheses)
        debug["hypotheses_after_merge"] = len(hypotheses)

        # Step 6: Classify each hypothesis as IN_LANE or OUT_OF_LANE
        for h in hypotheses:
            h.type = self._classify_type(h, in_lane_threshold_m)

        debug["hypotheses"] = [
            {
                "start_time_s": round(h.start_time_s, 3),
                "peak_time_s": round(h.peak_time_s, 3),
                "end_time_s": round(h.end_time_s, 3),
                "start_s_m": round(h.start_s_m, 1),
                "peak_s_m": round(h.peak_s_m, 1),
                "end_s_m": round(h.end_s_m, 1),
                "direction": h.direction,
                "type": h.type,
                "max_deviation_m": round(
                    h.trajectory_features.max_lateral_deviation_m, 3
                ),
                "duration_s": round(h.trajectory_features.deviation_duration_s, 3),
                "distance_m": round(h.trajectory_features.total_maneuver_distance_m, 1),
            }
            for h in hypotheses
        ]

        return hypotheses, debug

    # ==============================================================================================
    # Stage 1: Peak Detection
    #
    # C++ references:
    #   detectLocalMaxima()      — DgpsTrajectoryAnalyzer.cpp:858
    #   detectAndFilterPeaks()   — DgpsTrajectoryAnalyzer.cpp:552
    # ==============================================================================================

    def detrend_deviations(
        self, deviations: List[List[float]], baseline: float
    ) -> List[List[float]]:
        """Subtract baseline from lateral component only (l' = l - baseline).

        The longitudinal component (s) is preserved unchanged.
        Matches C++ detrendDeviations().
        """
        return [[s, l - baseline] for s, l in deviations]

    def detect_local_maxima(
        self,
        deviations: List[List[float]],
        trajectory: List[DgpsPoint],
    ) -> List[DeviationPeak]:
        """Find local maxima in |lateral deviation|.

        A point is a peak if its absolute deviation is strictly greater than both
        neighbors AND exceeds the minimum threshold (0.4m).
        Matches C++ detectLocalMaxima().
        """
        peaks = []
        if len(deviations) < 3:
            return peaks

        t0 = trajectory[0].cameraTime if trajectory else 0

        for i in range(1, len(deviations) - 1):
            lat = deviations[i][1]
            abs_dev = abs(lat)
            abs_prev = abs(deviations[i - 1][1])
            abs_next = abs(deviations[i + 1][1])

            if (
                abs_dev > abs_prev
                and abs_dev > abs_next
                and abs_dev > self.config.min_nudge_deviation_m
            ):
                time_s = (trajectory[i].cameraTime - t0) / 1e6 if trajectory else 0.0
                direction = (
                    "LEFT" if lat > 0 else ("RIGHT" if lat < 0 else "UNSPECIFIED")
                )
                peaks.append(
                    DeviationPeak(
                        index=i,
                        time_s=time_s,
                        magnitude_m=abs_dev,
                        signed_deviation_m=lat,
                        direction=direction,
                    )
                )
        return peaks

    def filter_peaks(
        self,
        peaks: List[DeviationPeak],
        deviations: List[List[float]],
        use_adaptive_baseline: bool,
    ) -> List[DeviationPeak]:
        """Filter peaks by two criteria.

        1. Minimum magnitude after detrend (0.3m) — removes residual noise after baseline.
        2. Ego-deviation ratio — if the ego is already more offset than the peak
           (ego_dev > peak * 1.1), the peak is a return-to-center, not avoidance.

        Matches C++ detectAndFilterPeaks().
        """
        filtered = list(peaks)

        # Filter 1: magnitude after detrend
        if use_adaptive_baseline:
            filtered = [
                p
                for p in filtered
                if abs(p.magnitude_m) >= self.config.min_peak_after_detrend_m
            ]

        # Filter 2: ego deviation ratio (ego's current deviation at s=0)
        if deviations:
            ego_dev = abs(deviations[0][1])
            filtered = [
                p
                for p in filtered
                if ego_dev <= p.magnitude_m * self.config.ego_deviation_to_peak_ratio
            ]

        return filtered

    # ==============================================================================================
    # Stage 2: Hypothesis Window Creation
    #
    # C++ references:
    #   createHypothesisWindows() — DgpsTrajectoryAnalyzer.cpp:1170
    #   findDeviationStart()     — DgpsTrajectoryAnalyzer.cpp:1283
    #   findDeviationEnd()       — DgpsTrajectoryAnalyzer.cpp:1314
    # ==============================================================================================

    def find_deviation_start(
        self,
        peak_index: int,
        peak_magnitude: float,
        deviations: List[List[float]],
    ) -> int:
        """Find where the deviation began by scanning backward from the peak.

        Scans backward until |dev| drops below peak * threshold_factor_start (0.3).
        Returns the index where the maneuver started.
        Matches C++ findDeviationStart().
        """
        if peak_index == 0:
            return 0
        threshold = peak_magnitude * self.config.threshold_factor_start
        for i in range(peak_index, 0, -1):
            if abs(deviations[i][1]) < threshold:
                return i
        return 0

    def find_deviation_end(
        self,
        peak_index: int,
        peak_magnitude: float,
        deviations: List[List[float]],
    ) -> Optional[int]:
        """Find where ego returned to center by scanning forward from the peak.

        Scans forward until |dev| drops below peak * threshold_factor_end (0.5).
        Returns None if no return found within the extended field — the peak is
        rejected (incomplete maneuver or monotonic drift).
        Matches C++ findDeviationEnd().
        """
        if peak_index >= len(deviations) - 1:
            return None
        threshold = peak_magnitude * self.config.threshold_factor_end
        for i in range(peak_index + 1, len(deviations)):
            if abs(deviations[i][1]) < threshold:
                return i
        return None

    def is_straight_trajectory_window(
        self,
        trajectory: List[DgpsPoint],
        start_idx: int,
        end_idx: int,
    ) -> bool:
        """Check if the ego trajectory within the peak window is geometrically straight.

        If the physical trajectory is straight (the car didn't actually swerve),
        the lateral deviation peak is caused by road geometry (curve, lane shift),
        not by an obstacle avoidance maneuver.

        Method:
          1. Collect 5 trajectory points before and after the window as "baseline"
          2. Fit a direction line through the baseline points (centroid + direction vector)
          3. Measure perpendicular distance of each window point to this line
          4. If maxPerp < 0.2m AND rms < 0.1m, the trajectory is straight

        Matches C++ nudge_utils::isStraightTrajectoryWindow().
        """
        if not trajectory or start_idx >= end_idx or end_idx >= len(trajectory):
            return False
        if end_idx - start_idx + 1 < 2:
            return False

        n = self.config.baseline_points_per_side

        # Collect baseline points before and after window
        baseline_pts = []
        for i in range(max(0, start_idx - n), start_idx):
            baseline_pts.append((trajectory[i].egoX, trajectory[i].egoY))
        for i in range(end_idx + 1, min(end_idx + n + 1, len(trajectory))):
            baseline_pts.append((trajectory[i].egoX, trajectory[i].egoY))

        if len(baseline_pts) < 2:
            return False

        # Check minimum window distance (too short = unreliable)
        total_dist = 0.0
        for i in range(start_idx + 1, end_idx + 1):
            dx = trajectory[i].egoX - trajectory[i - 1].egoX
            dy = trajectory[i].egoY - trajectory[i - 1].egoY
            total_dist += math.sqrt(dx * dx + dy * dy)
        if total_dist < 3.0:
            return False

        # Fit baseline direction (centroid + first-to-last vector)
        cx = sum(p[0] for p in baseline_pts) / len(baseline_pts)
        cy = sum(p[1] for p in baseline_pts) / len(baseline_pts)
        dx = baseline_pts[-1][0] - baseline_pts[0][0]
        dy = baseline_pts[-1][1] - baseline_pts[0][1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return False
        dir_x, dir_y = dx / length, dy / length

        # Measure perpendicular distances of window points to baseline line
        max_dist = 0.0
        sum_sq = 0.0
        count = 0
        for i in range(start_idx, end_idx + 1):
            px = trajectory[i].egoX - cx
            py = trajectory[i].egoY - cy
            perp = abs(
                px * dir_y - py * dir_x
            )  # Cross product gives perpendicular distance
            max_dist = max(max_dist, perp)
            sum_sq += perp * perp
            count += 1

        rms_dist = math.sqrt(sum_sq / count) if count > 0 else 0.0
        return (
            max_dist < self.config.straight_max_perp_m
            and rms_dist < self.config.straight_max_rms_m
        )

    def compute_peakedness_metrics(
        self,
        start_idx: int,
        peak_idx: int,
        end_idx: int,
        deviations: List[List[float]],
    ) -> PeakednessMetrics:
        """Compute shape metrics to distinguish sharp peaks (nudge) from flat plateaus (bias).

        Three metrics:
          - PPR (Peak Prominence Ratio): How much the peak stands out from the baseline.
            Computed as (peak_value - baseline_mean) / baseline_std.
            High PPR = sharp peak = likely nudge. Low PPR = flat = likely bias.
          - PFI (Plateau Flatness Index): Variance of the top 40% of deviation values.
            Low PFI = flat top = bias. High PFI = peaked = nudge.
          - CV (Coefficient of Variation): std/mean of deviations in window.

        Matches C++ calculatePeakednessMetrics().
        """
        metrics = PeakednessMetrics()
        if end_idx <= start_idx or peak_idx < start_idx or peak_idx > end_idx:
            return metrics
        window_size = end_idx - start_idx + 1
        if window_size < 3:
            return metrics

        # Deviation statistics within the window
        abs_devs = [abs(deviations[i][1]) for i in range(start_idx, end_idx + 1)]
        mean_dev = sum(abs_devs) / len(abs_devs)
        variance = sum((d - mean_dev) ** 2 for d in abs_devs) / len(abs_devs)
        std_dev = math.sqrt(max(0, variance))

        metrics.deviation_mean_m = mean_dev
        metrics.deviation_std_m = std_dev
        if mean_dev > 1e-6:
            metrics.deviation_cv = std_dev / mean_dev

        # PPR: collect baseline points (5 before start, 5 after end)
        n = self.config.peakedness_baseline_points
        baseline_vals = []
        for i in range(max(0, start_idx - n), start_idx):
            baseline_vals.append(abs(deviations[i][1]))
        for i in range(end_idx + 1, min(end_idx + n + 1, len(deviations))):
            baseline_vals.append(abs(deviations[i][1]))
        metrics.baseline_point_count = len(baseline_vals)

        if baseline_vals:
            bl_mean = sum(baseline_vals) / len(baseline_vals)
            bl_var = sum((v - bl_mean) ** 2 for v in baseline_vals) / len(baseline_vals)
            bl_std = math.sqrt(bl_var)
            peak_value = abs(deviations[peak_idx][1])
            prominence = peak_value - bl_mean
            metrics.peak_prominence_ratio = (
                prominence / bl_std if bl_std > 1e-6 else prominence * 10.0
            )

        # PFI: variance of top 40% deviation values
        sorted_devs = sorted(abs_devs, reverse=True)
        top_count = max(1, int(len(sorted_devs) * self.config.peakedness_top_fraction))
        top_vals = sorted_devs[:top_count]
        top_mean = sum(top_vals) / len(top_vals)
        metrics.plateau_flatness_index = sum(
            (v - top_mean) ** 2 for v in top_vals
        ) / len(top_vals)

        return metrics

    # ==============================================================================================
    # Private: Window creation, merging, classification
    # ==============================================================================================

    def _create_hypothesis_windows(
        self,
        peaks,
        nf_devs,
        ext_devs,
        trajectory,
        ego_lane_width_m,
        adaptive_baseline_m,
        in_lane_threshold_m,
    ) -> List[NudgeHypothesis]:
        """For each peak, create a hypothesis window and validate it.

        Matches C++ createHypothesisWindows().
        """
        hypotheses = []
        t0 = trajectory[0].cameraTime if trajectory else 0
        cfg = self.config

        for peak in peaks:
            # Find start in near-field, end in extended field
            start_idx = self.find_deviation_start(peak.index, peak.magnitude_m, nf_devs)
            end_idx = self.find_deviation_end(peak.index, peak.magnitude_m, ext_devs)
            if end_idx is None or end_idx <= start_idx:
                continue

            # Reject if trajectory is physically straight during the window
            nf_limit = len(nf_devs) - 1 if nf_devs else 0
            straight_end = min(end_idx, nf_limit)
            if straight_end > start_idx and self.is_straight_trajectory_window(
                trajectory, start_idx, straight_end
            ):
                continue

            # Accumulate window metrics: directional bias, max deviation, distance, jerk
            left_sum = (
                right_sum
            ) = sum_abs = max_abs = total_distance = out_of_lane_dur = max_jerk = 0.0
            prev_lat_vel = None
            count = 0

            for i in range(start_idx, min(end_idx + 1, len(ext_devs))):
                dev = ext_devs[i][1]
                abs_dev = abs(dev)
                sum_abs += abs_dev
                max_abs = max(max_abs, abs_dev)
                count += 1
                if dev > 0:
                    left_sum += dev
                elif dev < 0:
                    right_sum += abs_dev

                if i > start_idx:
                    seg_dist = ext_devs[i][0] - ext_devs[i - 1][0]
                    total_distance += max(0, seg_dist)
                    # Out-of-lane duration: time spent beyond lane boundary
                    if abs_dev > in_lane_threshold_m and i < len(trajectory):
                        speed = math.sqrt(
                            trajectory[i].velX ** 2 + trajectory[i].velY ** 2
                        )
                        if seg_dist > 1e-6:
                            out_of_lane_dur += seg_dist / max(1.0, speed)
                    # Lateral jerk: max |d(lat_vel)/dt|
                    if i < len(trajectory) and i > 0:
                        dt = (
                            trajectory[i].cameraTime - trajectory[i - 1].cameraTime
                        ) / 1e6
                        if dt > 1e-6:
                            lat_vel = (dev - ext_devs[i - 1][1]) / dt
                            if prev_lat_vel is not None:
                                max_jerk = max(
                                    max_jerk, abs((lat_vel - prev_lat_vel) / dt)
                                )
                            prev_lat_vel = lat_vel

            # Determine direction from cumulative bias
            direction = "UNSPECIFIED"
            if left_sum > right_sum and left_sum > cfg.direction_threshold_m:
                direction = "LEFT"
            elif right_sum > left_sum and right_sum > cfg.direction_threshold_m:
                direction = "RIGHT"

            returned = abs(ext_devs[end_idx][1]) < 0.1

            # Compute timing (relative to trajectory start)
            start_time = (
                (trajectory[start_idx].cameraTime - t0) / 1e6 if trajectory else 0.0
            )
            end_time = (
                (trajectory[min(end_idx, len(trajectory) - 1)].cameraTime - t0) / 1e6
                if trajectory
                else 0.0
            )
            duration = end_time - start_time
            if duration <= 0 and total_distance > 0:
                duration = (
                    total_distance / 12.0
                )  # Fallback: assume 12 m/s typical urban speed

            # Speed-adaptive duration validation:
            # requiredDuration = max(minDuration, minDistance / avgSpeed)
            # Hypothesis is valid if: duration >= required OR distance >= minDistance
            avg_speed = total_distance / duration if duration > 1e-6 else 12.0
            required_dur = max(
                cfg.min_hypothesis_duration_s,
                cfg.min_hypothesis_distance_m / avg_speed
                if avg_speed > 1e-3
                else cfg.min_hypothesis_duration_s,
            )

            if not (
                duration >= required_dur
                or total_distance >= cfg.min_hypothesis_distance_m
            ):
                continue

            def get_station(devs, idx):
                if not devs:
                    return 0.0
                return devs[min(idx, len(devs) - 1)][0]

            hyp = NudgeHypothesis(
                start_time_s=start_time,
                peak_time_s=peak.time_s,
                end_time_s=end_time if end_time > start_time else start_time + duration,
                start_s_m=get_station(nf_devs, start_idx),
                peak_s_m=get_station(nf_devs, peak.index),
                end_s_m=get_station(ext_devs, end_idx),
                direction=direction,
                left_bias_sum_m=left_sum,
                right_bias_sum_m=right_sum,
                peak_index=peak.index,
                adaptive_baseline_m=adaptive_baseline_m,
                ego_lane_width_m=ego_lane_width_m,
                trajectory_features=TrajectoryFeatures(
                    max_lateral_deviation_m=max_abs,
                    avg_lateral_deviation_m=sum_abs / count if count > 0 else 0.0,
                    deviation_duration_s=duration,
                    out_of_lane_duration_s=out_of_lane_dur,
                    lateral_jerk_m_s3=max_jerk,
                    returned_to_center=returned,
                    total_maneuver_distance_m=total_distance,
                    dominant_direction=direction,
                    peakedness_metrics=self.compute_peakedness_metrics(
                        start_idx, peak.index, end_idx, ext_devs
                    ),
                ),
            )
            hypotheses.append(hyp)

        return hypotheses

    def _merge_close_hypotheses(
        self, hypotheses: List[NudgeHypothesis]
    ) -> List[NudgeHypothesis]:
        """Merge overlapping or temporally close hypotheses.

        Two hypotheses are merged if:
          - Their time windows overlap, OR
          - The gap between them is less than merge_time_s (2.5s)

        When merging, the combined hypothesis takes the wider time/spatial window
        and the characteristics (direction, peakedness) of the larger peak.

        Matches C++ mergeCloseHypotheses().
        """
        if len(hypotheses) <= 1:
            return list(hypotheses)

        merged = []
        used = [False] * len(hypotheses)

        for i in range(len(hypotheses)):
            if used[i]:
                continue
            # Copy current hypothesis to avoid mutating the original
            current = NudgeHypothesis(
                **{
                    f.name: getattr(hypotheses[i], f.name)
                    for f in hypotheses[i].__dataclass_fields__.values()
                }
            )
            used[i] = True

            for j in range(i + 1, len(hypotheses)):
                if used[j]:
                    continue
                other = hypotheses[j]
                overlaps = (
                    current.start_time_s <= other.end_time_s
                    and other.start_time_s <= current.end_time_s
                )
                gap = min(
                    abs(other.start_time_s - current.end_time_s),
                    abs(current.start_time_s - other.end_time_s),
                )

                if overlaps or gap < self.config.merge_time_s:
                    # Expand window to encompass both
                    current.start_time_s = min(current.start_time_s, other.start_time_s)
                    current.end_time_s = max(current.end_time_s, other.end_time_s)
                    current.start_s_m = min(current.start_s_m, other.start_s_m)
                    current.end_s_m = max(current.end_s_m, other.end_s_m)
                    # Take the larger peak's characteristics
                    if (
                        other.trajectory_features.max_lateral_deviation_m
                        > current.trajectory_features.max_lateral_deviation_m
                    ):
                        current.peak_time_s = other.peak_time_s
                        current.peak_s_m = other.peak_s_m
                        current.direction = other.direction
                        current.trajectory_features.max_lateral_deviation_m = (
                            other.trajectory_features.max_lateral_deviation_m
                        )
                        current.peak_index = other.peak_index
                        current.trajectory_features.peakedness_metrics = (
                            other.trajectory_features.peakedness_metrics
                        )
                    used[j] = True

            merged.append(current)
        return merged

    def _classify_type(
        self, hypothesis: NudgeHypothesis, in_lane_threshold_m: float
    ) -> str:
        """Classify as IN_LANE or OUT_OF_LANE based on peak magnitude vs lane width.

        Matches C++ classifyHypothesisType().
        """
        if hypothesis.trajectory_features.max_lateral_deviation_m > in_lane_threshold_m:
            return "OUT_OF_LANE"
        return "IN_LANE"
