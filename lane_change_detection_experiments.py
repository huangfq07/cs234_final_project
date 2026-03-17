#!/usr/bin/env python3
"""
Lane change detection experiments — faithful reproduction of C++ algorithm.

Mirrors NudgeGatingDecider.cpp:detectLaneChangeViaDeviation() and helpers.
Run over labeled test cases to measure precision/recall and iterate on logic.

Usage:
    python lane_change_detection_experiments.py              # Run all cases
    python lane_change_detection_experiments.py case_001     # Run one case
    python lane_change_detection_experiments.py --verbose    # Show phase-by-phase trace
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys

# =============================================================================
# Config — mirrors NudgeGatingDecider::Config defaults
# =============================================================================

EGO_WIDTH_M = 1.95  # Constant for all cases
MAX_EGO_LANE_LENGTH_M = 100.0  # Upper bound — deviations beyond 100m are unreliable


@dataclass
class LaneChangeConfig:
    beyond_boundary_m: float = 0.8  # laneChangeBeyondBoundary_m (reduced from 1.0)
    stable_zone_stddev_threshold_m: float = (
        0.5  # deviationLCStableZoneStdDevThreshold_m
    )
    min_stable_distance_m: float = (
        12.0  # deviationLCMinStableDistance_m (reduced from 15.0)
    )
    window_size: int = 13  # deviationLCWindowSize (must cover min_stable_distance)
    start_check_points: int = 3  # deviationLCStartCheckPoints (reduced from 5)
    max_transition_rate: float = (
        20.0  # max transition_distance / lateral_displacement (m/m)
    )


# =============================================================================
# Result structs
# =============================================================================


@dataclass
class StableZoneResult:
    found: bool = False
    mean_m: float = 0.0
    stddev_m: float = 0.0
    start_x_m: float = 0.0
    end_x_m: float = 0.0
    distance_m: float = 0.0


@dataclass
class DetectionResult:
    detected: bool = False
    returned_to_ego_lane: bool = False
    direction: int = 0  # -1=left, +1=right
    onset_distance_m: float = float("inf")
    stable_zone_mean_m: float = 0.0
    stable_zone_stddev_m: float = 0.0
    stable_zone_distance_m: float = 0.0
    stable_zone_end_x_m: float = 0.0
    # Diagnostics (not in C++, for debugging)
    phase_reached: str = ""
    magnitude_threshold_m: float = 0.0
    num_qualifying_points: int = 0


# =============================================================================
# Phase 0: Verify trajectory starts in-lane
# =============================================================================


def verify_trajectory_starts_in_lane(
    deviations: list[list[float]],
    in_lane_threshold_m: float,
    check_points: int,
) -> bool:
    """First N points must have |l| < in_lane_threshold."""
    count = min(check_points, len(deviations))
    for i in range(count):
        if abs(deviations[i][1]) >= in_lane_threshold_m:
            return False
    return True


# =============================================================================
# Phase 1: Collect qualifying points (|l| > magnitude threshold)
# =============================================================================


def collect_qualifying_points(
    deviations: list[list[float]],
    magnitude_threshold_m: float,
) -> list[list[float]]:
    return [pt for pt in deviations if abs(pt[1]) > magnitude_threshold_m]


# =============================================================================
# Phase 2: Sliding window stable zone search
# =============================================================================


def has_consistent_sign(
    qualifying_points: list[list[float]],
    start_idx: int,
    end_idx: int,
    mean: float,
) -> bool:
    mean_positive = mean > 0.0
    for i in range(start_idx, end_idx):
        if (qualifying_points[i][1] > 0.0) != mean_positive:
            return False
    return True


def find_stable_zone(
    deviations: list[list[float]],
    magnitude_threshold_m: float,
    window_size: int,
    stddev_threshold_m: float,
) -> StableZoneResult:
    """Slide fixed-size window along full deviation series. Mirrors C++ findStableZone().

    Only consecutive points above magnitude_threshold contribute to the window.
    When a point drops below the threshold, the window resets.

    Algorithm:
      1. Point below threshold -> reset window
      2. Point above threshold -> add to window, pop oldest if full
      3. Window full -> check stddev
      4. Stddev passes -> check sign consistency -> done
         (distance is guaranteed by window_size — no separate distance check needed)
    """
    result = StableZoneResult()
    n = len(deviations)

    window_start = 0
    window_count = 0
    s = 0.0
    s_sq = 0.0

    for i in range(n):
        if abs(deviations[i][1]) <= magnitude_threshold_m:
            # Below threshold: reset window
            window_count = 0
            s = 0.0
            s_sq = 0.0
            continue

        # Above threshold: add to window
        val = deviations[i][1]
        s += val
        s_sq += val * val
        window_count += 1

        if window_count == 1:
            window_start = i

        # Pop oldest if window exceeds size
        if window_count > window_size:
            out_val = deviations[window_start][1]
            s -= out_val
            s_sq -= out_val * out_val
            window_count -= 1
            window_start += 1

        if window_count < window_size:
            continue

        # Window full — check stddev
        count = float(window_count)
        mean = s / count
        var = max(0.0, s_sq / count - mean * mean)
        stddev = math.sqrt(var)

        if stddev >= stddev_threshold_m:
            continue

        # Stddev passes — check sign consistency (distance guaranteed by window_size)
        zone_mean = mean

        if has_consistent_sign(deviations, window_start, i + 1, zone_mean):
            dist = deviations[i][0] - deviations[window_start][0]
            result.found = True
            result.mean_m = zone_mean
            result.stddev_m = stddev
            result.start_x_m = deviations[window_start][0]
            result.end_x_m = deviations[i][0]
            result.distance_m = dist
            return result

    return result


# =============================================================================
# Phase 3: Return check after stable zone
# =============================================================================


def check_deviation_returns_after_stable_zone(
    deviations: list[list[float]],
    in_lane_threshold_m: float,
    stable_zone_end_x_m: float,
) -> bool:
    """Check if deviation returns to in-lane after stable zone end."""
    MIN_RETURN_POINTS = 3
    tail_count = 0
    tail_in_lane_count = 0

    for pt in deviations:
        if pt[0] > stable_zone_end_x_m:
            tail_count += 1
            if abs(pt[1]) < in_lane_threshold_m:
                tail_in_lane_count += 1

    # If at least MIN_RETURN_POINTS tail points are within the lane boundary, ego returned.
    # Previously required majority (>50%), but "double hump" patterns have the ego returning
    # briefly then departing again — still counts as a return (not a permanent lane change).
    return (tail_count >= MIN_RETURN_POINTS) and (
        tail_in_lane_count >= MIN_RETURN_POINTS
    )


# =============================================================================
# Onset distance: first point where |l| > threshold
# =============================================================================


def find_deviation_onset_distance(
    deviations: list[list[float]],
    threshold_m: float,
) -> float:
    for pt in deviations:
        if abs(pt[1]) > threshold_m:
            return pt[0]
    return float("inf")


# =============================================================================
# Main detection function — mirrors detectLaneChangeViaDeviation()
# =============================================================================


def detect_lane_change(
    deviations: list[list[float]],
    half_lane_width_m: float,
    in_lane_threshold_m: float,
    config: LaneChangeConfig,
    verbose: bool = False,
) -> DetectionResult:
    result = DetectionResult()

    if len(deviations) < 3:
        result.phase_reached = "too_few_points"
        return result

    # Phase 0: Verify trajectory starts in-lane
    if not verify_trajectory_starts_in_lane(
        deviations, in_lane_threshold_m, config.start_check_points
    ):
        result.phase_reached = "phase0_not_in_lane"
        if verbose:
            first_n = deviations[: config.start_check_points]
            print(
                f"  Phase 0 FAIL: first {config.start_check_points} points not all in-lane (threshold={in_lane_threshold_m:.3f}m)"
            )
            for i, pt in enumerate(first_n):
                flag = " <-- FAIL" if abs(pt[1]) >= in_lane_threshold_m else ""
                print(f"    [{i}] s={pt[0]:.2f} l={pt[1]:.3f}{flag}")
        return result

    if verbose:
        print(f"  Phase 0 PASS: first {config.start_check_points} points in-lane")

    # Magnitude threshold
    magnitude_threshold = half_lane_width_m + config.beyond_boundary_m
    result.magnitude_threshold_m = magnitude_threshold

    qualifying = [pt for pt in deviations if abs(pt[1]) > magnitude_threshold]
    result.num_qualifying_points = len(qualifying)

    if verbose:
        print(
            f"  Phase 1+2: magnitude_threshold={magnitude_threshold:.3f}m, qualifying_points={len(qualifying)}/{len(deviations)}"
        )

    # Phase 1+2: Slide window along full deviation series. Only consecutive points
    # above magnitudeThreshold contribute. Window resets when a point drops below.
    stable_zone = find_stable_zone(
        deviations,
        magnitude_threshold,
        config.window_size,
        config.stable_zone_stddev_threshold_m,
    )

    if verbose:
        if stable_zone.found:
            print(f"  Phase 2 PASS: stable zone found")
            print(
                f"    range: s=[{stable_zone.start_x_m:.2f}, {stable_zone.end_x_m:.2f}] ({stable_zone.distance_m:.1f}m)"
            )
            print(
                f"    mean={stable_zone.mean_m:.3f}m, stddev={stable_zone.stddev_m:.3f}m"
            )
        else:
            # Diagnostic: show why no stable zone was found
            print(
                f"  Phase 2 FAIL: no stable zone found (window={config.window_size}, stddev_thresh={config.stable_zone_stddev_threshold_m}m, min_dist={config.min_stable_distance_m}m)"
            )
            # Show stddev at each window position
            if len(qualifying) >= config.window_size:
                print(f"    Window stddevs (first 5 positions):")
                for start in range(min(5, len(qualifying) - config.window_size + 1)):
                    vals = [
                        qualifying[i][1]
                        for i in range(start, start + config.window_size)
                    ]
                    m = sum(vals) / len(vals)
                    v = max(0.0, sum(x * x for x in vals) / len(vals) - m * m)
                    sd = math.sqrt(v)
                    s_range = f"s=[{qualifying[start][0]:.1f}, {qualifying[start + config.window_size - 1][0]:.1f}]"
                    print(
                        f"      pos {start}: {s_range} stddev={sd:.3f}m {'<-- PASS' if sd < config.stable_zone_stddev_threshold_m else ''}"
                    )

    if not stable_zone.found:
        result.phase_reached = "phase2_no_stable_zone"
        return result

    # Phase 2a: Transition rate check — reject gradual drift patterns.
    # A real lane change transitions quickly; gradual curvature drift has a high
    # transition_distance / lateral_displacement ratio.
    onset_s = find_deviation_onset_distance(deviations, in_lane_threshold_m)
    displacement_m = abs(stable_zone.mean_m) - in_lane_threshold_m
    if displacement_m > 0.0 and onset_s != float("inf"):
        transition_distance = stable_zone.start_x_m - onset_s
        transition_rate = transition_distance / displacement_m
        if verbose:
            print(
                f"  Phase 2a: onset={onset_s:.1f}m, stable_start={stable_zone.start_x_m:.1f}m, "
                f"transition={transition_distance:.1f}m, displacement={displacement_m:.2f}m, "
                f"rate={transition_rate:.1f} m/m (max={config.max_transition_rate})"
            )
        if transition_rate > config.max_transition_rate:
            result.phase_reached = "phase2a_gradual_drift"
            if verbose:
                print(
                    f"  Phase 2a FAIL: gradual drift (rate {transition_rate:.1f} > {config.max_transition_rate})"
                )
            return result
        if verbose:
            print(f"  Phase 2a PASS")
    elif verbose:
        print(f"  Phase 2a SKIP: displacement={displacement_m:.2f}m, onset={onset_s}")

    # Phase 2b: Mid-trajectory return check — reject if ego returned within the lane
    # AFTER first crossing the lane boundary but BEFORE the stable zone. This catches
    # "double hump" trajectories where the first excursion is an OOLN and the second
    # is not a lane change.
    mid_returned = False
    crossed_lane_boundary = False
    for pt in deviations:
        if pt[0] >= stable_zone.start_x_m:
            break
        if not crossed_lane_boundary and abs(pt[1]) > half_lane_width_m:
            crossed_lane_boundary = True
        elif crossed_lane_boundary and abs(pt[1]) < half_lane_width_m:
            mid_returned = True
            break

    if verbose:
        if mid_returned:
            print(
                f"  Phase 2b FAIL: ego crossed lane boundary then returned within lane before stable zone (s={stable_zone.start_x_m:.1f})"
            )
        elif crossed_lane_boundary:
            print(
                f"  Phase 2b PASS: ego crossed lane boundary and stayed out until stable zone"
            )
        else:
            print(
                f"  Phase 2b PASS: ego never crossed lane boundary before stable zone"
            )

    if mid_returned:
        result.returned_to_ego_lane = True
        result.phase_reached = "phase2b_mid_return"
        return result

    # Phase 3: Return check — did ego return within the lane boundary?
    # Use halfLaneWidth (not inLaneThreshold) because the question is "is ego back in the lane",
    # not "is ego back at lane center".
    returned = check_deviation_returns_after_stable_zone(
        deviations, half_lane_width_m, stable_zone.end_x_m
    )

    if verbose:
        print(f"  Phase 3: returned_to_ego_lane={returned}")

    if returned:
        result.returned_to_ego_lane = True
        result.phase_reached = "phase3_returned"
        return result

    # Phase 4: Detection confirmed
    result.detected = True
    result.direction = -1 if stable_zone.mean_m > 0.0 else 1
    result.stable_zone_mean_m = stable_zone.mean_m
    result.stable_zone_stddev_m = stable_zone.stddev_m
    result.stable_zone_distance_m = stable_zone.distance_m
    result.stable_zone_end_x_m = stable_zone.end_x_m
    result.onset_distance_m = find_deviation_onset_distance(
        deviations, in_lane_threshold_m
    )
    result.phase_reached = "phase4_detected"

    if verbose:
        dir_str = "LEFT" if result.direction == -1 else "RIGHT"
        print(f"  Phase 4: DETECTED {dir_str}, onset={result.onset_distance_m:.2f}m")

    return result


# =============================================================================
# Evaluation harness
# =============================================================================


def evaluate_case(
    case_id: str,
    case_data: dict,
    config: LaneChangeConfig,
    verbose: bool = False,
) -> dict:
    """Run detection on a single case and return result summary."""
    deviations = [
        pt for pt in case_data["deviations"] if pt[0] <= MAX_EGO_LANE_LENGTH_M
    ]
    ego_lane_width_m = case_data["ego_lane_width_m"]
    half_lane_width_m = ego_lane_width_m * 0.5
    in_lane_threshold_m = max(0.5, half_lane_width_m - (EGO_WIDTH_M * 0.5))
    ground_truth = case_data["ground_truth"]

    if verbose:
        print(f"\n{'='*70}")
        print(f"Case {case_id}: {case_data['name']}")
        print(f"  ground_truth={ground_truth}")
        print(
            f"  ego_lane_width={ego_lane_width_m:.3f}m, in_lane_threshold={in_lane_threshold_m:.3f}m"
        )
        print(
            f"  half_lane_width={half_lane_width_m:.3f}m, num_deviations={len(deviations)}"
        )
        if deviations:
            print(
                f"  deviation range: s=[{deviations[0][0]:.1f}, {deviations[-1][0]:.1f}], l=[{min(d[1] for d in deviations):.2f}, {max(d[1] for d in deviations):.2f}]"
            )
        print()

    result = detect_lane_change(
        deviations, half_lane_width_m, in_lane_threshold_m, config, verbose
    )

    predicted = "lane_change" if result.detected else "no_lane_change"
    correct = predicted == ground_truth

    return {
        "case_id": case_id,
        "name": case_data["name"],
        "ground_truth": ground_truth,
        "predicted": predicted,
        "correct": correct,
        "phase_reached": result.phase_reached,
        "result": result,
    }


def print_summary(results: list[dict]) -> None:
    """Print precision/recall summary table."""
    tp = sum(
        1
        for r in results
        if r["ground_truth"] == "lane_change" and r["predicted"] == "lane_change"
    )
    fn = sum(
        1
        for r in results
        if r["ground_truth"] == "lane_change" and r["predicted"] == "no_lane_change"
    )
    fp = sum(
        1
        for r in results
        if r["ground_truth"] == "no_lane_change" and r["predicted"] == "lane_change"
    )
    tn = sum(
        1
        for r in results
        if r["ground_truth"] == "no_lane_change" and r["predicted"] == "no_lane_change"
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Case':<12} {'GT':<16} {'Predicted':<16} {'Phase':<24} {'Result'}")
    print(f"{'-'*12} {'-'*16} {'-'*16} {'-'*24} {'-'*6}")
    for r in results:
        mark = "OK" if r["correct"] else "FAIL"
        print(
            f"{r['case_id']:<12} {r['ground_truth']:<16} {r['predicted']:<16} {r['phase_reached']:<24} {mark}"
        )

    print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.0f}%)")
    print(f"TP={tp} FN={fn} FP={fp} TN={tn}")
    print(f"Precision: {precision:.2f}  Recall: {recall:.2f}")


def main():
    args = sys.argv[1:]
    verbose = "--verbose" in args or "-v" in args
    case_filter = [a for a in args if not a.startswith("-")]

    data_path = Path(__file__).parent / "lane_change_detection_data.json"
    with open(data_path) as f:
        all_cases = json.load(f)

    config = LaneChangeConfig()

    if case_filter:
        cases = {k: v for k, v in all_cases.items() if k in case_filter}
        if not cases:
            print(f"No matching cases found. Available: {list(all_cases.keys())}")
            sys.exit(1)
    else:
        cases = all_cases

    results = []
    for case_id, case_data in sorted(cases.items()):
        r = evaluate_case(case_id, case_data, config, verbose)
        results.append(r)

    print_summary(results)


if __name__ == "__main__":
    main()
