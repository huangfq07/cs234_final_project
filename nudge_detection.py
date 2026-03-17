#!/usr/bin/env python3
"""
Nudge Detection — Pipeline Orchestrator + CLI

Python replication of the C++ nudge detection pipeline for rapid prototyping.
Allows testing algorithm changes in Python before porting to C++.

The pipeline detects nudge maneuvers from DGPS trajectory data:
  1. DgpsTrajectoryAnalyzer — detect lateral deviation peaks, build hypothesis windows
  2. NudgeObjectAnalyzer    — match nearby objects to each hypothesis
  3. BiasNudgeDecider       — distinguish real nudges from systematic bias (e.g. parked car rows)
  4. NudgeClassifier        — build final events with reasoning strings

Scope: Core detection only. Excludes gating (ODD, lane change), state machine,
       object tracking hysteresis, and multi-frame smoothing.

Usage:
    python nudge_detection.py input.json               # Run detection
    python nudge_detection.py input.json --verbose      # Verbose debug output
    python nudge_detection.py input.json --json         # Output as JSON

Architecture (mirrors C++ file structure):
    python/data_types.py                <- Shared structs + utils (like the .hpp headers)
    python/dgps_trajectory_analyzer.py  <- DgpsTrajectoryAnalyzer.cpp
    python/nudge_object_analyzer.py     <- NudgeObjectAnalyzer.cpp
    python/bias_nudge_decider.py        <- BiasNudgeDecider.cpp
    python/nudge_classifier.py          <- NudgeClassifier.cpp
    nudge_detection.py                  <- Pipeline orchestrator + CLI (this file)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

from python.bias_nudge_decider import BiasNudgeDecider
from python.data_types import (
    Config,
    DetectionResult,
    DgpsPoint,
    PreprocessedObjectInfo,
)
from python.dgps_trajectory_analyzer import DgpsTrajectoryAnalyzer
from python.nudge_classifier import NudgeClassifier
from python.nudge_object_analyzer import NudgeObjectAnalyzer

# ==================================================================================================
# Pipeline Orchestrator
# ==================================================================================================


def run_detection(input_data: Dict[str, Any]) -> DetectionResult:
    """Run the full nudge detection pipeline.

    This is the Python equivalent of the C++ NudgePostprocessor::process() call chain:
      NudgeDetectionContext  (provides deviations + baseline — pre-computed in input JSON)
      -> DgpsTrajectoryAnalyzer.process()   (peaks -> hypotheses)
      -> NudgeObjectAnalyzer.process()      (match objects per hypothesis)
         -> bias_nudge::evaluate()          (bias vs nudge decision per hypothesis)
      -> NudgeClassifier.process()          (build events, select primary)

    Args:
        input_data: Parsed JSON matching nudge_detection_example_input.json schema.

    Returns:
        DetectionResult with is_nudge_scenario, events, reasoning, and per-step debug info.
    """
    result = DetectionResult()
    config = Config()

    # --- Parse input ---
    trajectory = [DgpsPoint(**pt) for pt in input_data.get("dgpsTrajectory", [])]
    ego_lane_width_m = input_data.get("egoLane", {}).get("width_m", 3.5)
    vehicle_width_m = input_data.get("vehicleWidth_m", 1.8)
    nf_deviations = input_data.get(
        "nearFieldDeviations", input_data.get("deviations", [])
    )
    ext_deviations = input_data.get("extendedDeviations", nf_deviations)
    adaptive_baseline_m = input_data.get("adaptiveBaseline_m", 0.0)
    ego_speed_mps = input_data.get("egoSpeed_mps", 10.0)
    objects = [PreprocessedObjectInfo(**obj) for obj in input_data.get("objects", [])]

    # --- Stage 1: DgpsTrajectoryAnalyzer ---
    # Detects lateral deviation peaks in the DGPS trajectory and creates
    # nudge hypotheses (time/spatial windows around each peak).
    analyzer = DgpsTrajectoryAnalyzer(config)
    hypotheses, traj_debug = analyzer.analyze(
        nf_deviations,
        ext_deviations,
        trajectory,
        adaptive_baseline_m,
        ego_lane_width_m,
        vehicle_width_m,
    )
    result.debug["trajectory_analysis"] = traj_debug

    if not hypotheses:
        result.reasoning = "No valid hypotheses detected"
        return result

    # --- Stage 2 + 3: NudgeObjectAnalyzer + BiasNudgeDecider (per hypothesis) ---
    # For each hypothesis, find the best matching object (closest by adjusted distance),
    # then decide if the detection is a real nudge or just systematic bias.
    obj_analyzer = NudgeObjectAnalyzer(config, vehicle_width_m, ego_speed_mps)
    bias_decider = BiasNudgeDecider(config)

    # Detrend deviations for bias/nudge decision (same baseline as trajectory analyzer).
    # The BiasNudgeDecider needs detrended deviations to compute deviation ratios
    # at object positions relative to the peak.
    ext_detrended = ext_deviations
    if abs(adaptive_baseline_m) > 1e-6:
        ext_detrended = analyzer.detrend_deviations(ext_deviations, adaptive_baseline_m)

    object_results = []
    bias_results = []
    obj_debug_list = []
    bias_debug_list = []

    for hyp in hypotheses:
        # Object analysis: filter by causality (lateral, longitudinal, safety),
        # then select the closest object by adjusted distance.
        (
            best_id,
            best_dist,
            causal_objs,
            obj_debug,
        ) = obj_analyzer.evaluate_objects_for_hypothesis(objects, hyp)
        object_results.append((best_id, best_dist, causal_objs, obj_debug))
        obj_debug_list.append(obj_debug)

        # Bias vs nudge: given all causal objects, decide if the closest one
        # is an exceptional outlier (nudge) or part of a uniform row (bias).
        bias_output = bias_decider.evaluate(
            causal_objs,
            hyp.peak_s_m,
            hyp.trajectory_features.max_lateral_deviation_m,
            ext_detrended,
        )
        bias_results.append(bias_output)
        bias_debug_list.append(
            {
                "is_bias": bias_output.is_bias,
                "reason": bias_output.reason,
                "q1_exceptional": bias_output.q1_is_exceptional,
                "q2a_ahead": bias_output.q2a_ahead_of_ego,
                "q2b_near_peak": bias_output.q2b_near_peak,
                "exceptionality_ratio": round(bias_output.exceptionality_ratio, 3),
                "closest_object_id": bias_output.closest_object_id,
                "closest_adjusted_dist": round(bias_output.closest_adjusted_dist, 3),
                "closest_deviation_ratio": round(
                    bias_output.closest_deviation_ratio, 3
                ),
                "objects_in_causal_zone": bias_output.objects_in_causal_zone,
            }
        )

    result.debug["object_analysis"] = obj_debug_list
    result.debug["bias_nudge_decisions"] = bias_debug_list

    # --- Stage 4: NudgeClassifier ---
    # Combines hypotheses + object matches + bias/nudge decisions into final events.
    # Selects primary event by closest adjusted distance.
    classifier = NudgeClassifier()
    events = classifier.build_events(hypotheses, object_results, bias_results, objects)
    result.events = events

    primary = classifier.select_primary(events)
    result.primary_event = primary

    if primary:
        result.is_nudge_scenario = True
        result.reasoning = primary.reasoning
    else:
        all_bias = all(b.is_bias for b in bias_results)
        result.reasoning = (
            "All hypotheses classified as BIAS"
            if all_bias
            else "No object matched to any hypothesis"
        )

    return result


# ==================================================================================================
# CLI Output Formatting
# ==================================================================================================


def print_result(result: DetectionResult, verbose: bool = False):
    """Print detection result in human-readable format."""
    print("=" * 70)
    print("NUDGE DETECTION RESULT")
    print("=" * 70)
    print(f"  Is Nudge Scenario: {result.is_nudge_scenario}")
    print(f"  Reasoning:         {result.reasoning}")
    print(f"  Events:            {len(result.events)}")

    if result.primary_event:
        e = result.primary_event
        print(f"\n  Primary Event:")
        print(f"    Object ID:       {e.critical_object_id}")
        print(f"    Class:           {e.obstacle_class}")
        print(f"    Direction:       {e.nudge_direction}")
        print(f"    Type:            {e.candidate_type}")
        print(f"    Adjusted Dist:   {e.adjusted_dist_m:.3f}m")
        print(f"    Description:     {e.description}")
        print(f"    Reasoning:       {e.reasoning}")

    if verbose and result.debug:
        print(f"\n{'=' * 70}")
        print("DEBUG INFO")
        print(f"{'=' * 70}")

        ta = result.debug.get("trajectory_analysis", {})
        print(f"\n  Trajectory Analysis:")
        print(f"    Baseline subtracted: {ta.get('baseline_subtracted_m', 0):.3f}m")
        print(f"    NF points:           {ta.get('nf_point_count', 0)}")
        print(f"    Ext points:          {ta.get('ext_point_count', 0)}")
        print(f"    Peaks (before):      {ta.get('peaks_before_filter', 0)}")
        print(f"    Peaks (after):       {ta.get('peaks_after_filter', 0)}")
        print(f"    Hypotheses (raw):    {ta.get('hypotheses_before_merge', 0)}")
        print(f"    Hypotheses (merged): {ta.get('hypotheses_after_merge', 0)}")

        for p in ta.get("peaks", []):
            status = "KEPT" if p.get("survived_filter") else "FILTERED"
            print(
                f"    Peak[{p['index']}]: {p['magnitude_m']:.3f}m {p['direction']} @ {p['time_s']:.3f}s [{status}]"
            )

        for i, h in enumerate(ta.get("hypotheses", [])):
            print(
                f"    Hyp[{i}]: {h['direction']} {h['type']} s=[{h['start_s_m']:.1f}, {h['end_s_m']:.1f}]m "
                f"peak={h['max_deviation_m']:.3f}m dur={h['duration_s']:.3f}s"
            )

        for i, bd in enumerate(result.debug.get("bias_nudge_decisions", [])):
            print(f"\n  Bias/Nudge Decision [{i}]:")
            print(f"    Decision:        {'BIAS' if bd['is_bias'] else 'NUDGE'}")
            print(f"    Reason:          {bd['reason']}")
            print(f"    Q1 Exceptional:  {bd['q1_exceptional']}")
            print(f"    Q2a Ahead:       {bd['q2a_ahead']}")
            print(f"    Q2b Near Peak:   {bd['q2b_near_peak']}")
            print(f"    Exceptionality:  {bd['exceptionality_ratio']:.3f}")
            if bd["closest_object_id"]:
                print(f"    Closest Object:  {bd['closest_object_id']}")
                print(f"    Closest Adj Dist: {bd['closest_adjusted_dist']:.3f}m")
                print(f"    Closest Dev Ratio: {bd['closest_deviation_ratio']:.3f}")

    print(f"\n{'=' * 70}")


def print_json(result: DetectionResult):
    """Print detection result as JSON."""
    output = {
        "isNudgeScenario": result.is_nudge_scenario,
        "reasoning": result.reasoning,
        "events": [
            {
                "criticalObjectId": e.critical_object_id,
                "obstacleClass": e.obstacle_class,
                "nudgeDirection": e.nudge_direction,
                "candidateType": e.candidate_type,
                "adjustedDist_m": round(e.adjusted_dist_m, 3),
                "reasoning": e.reasoning,
            }
            for e in result.events
        ],
        "debug": result.debug,
    }
    print(json.dumps(output, indent=2, default=str))


# ==================================================================================================
# CLI Entry Point
# ==================================================================================================


def run_all_test_cases(input_data: Dict[str, Any], verbose: bool = False) -> bool:
    """Run all test cases and print pass/fail summary.

    Args:
        input_data: Parsed JSON with "test_cases" key.
        verbose: If True, print detailed debug output per case.

    Returns:
        True if all test cases passed.
    """
    test_cases = input_data["test_cases"]
    passed = 0
    failed = 0
    results = []

    for case_id, case_data in test_cases.items():
        expected_result = case_data.get("expected_result", "")
        expected_direction = case_data.get("expected_direction")
        name = case_data.get("name", case_id)

        result = run_detection(case_data)

        # Determine actual result
        if result.is_nudge_scenario:
            actual_result = "nudge"
            actual_direction = (
                result.primary_event.nudge_direction if result.primary_event else None
            )
        elif result.reasoning and "BIAS" in result.reasoning.upper():
            actual_result = "bias"
            actual_direction = None
        else:
            actual_result = "no_detection"
            actual_direction = None

        # Compare
        result_match = actual_result == expected_result
        direction_match = (
            expected_direction is None or actual_direction == expected_direction
        )
        case_passed = result_match and direction_match

        if case_passed:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        results.append(
            (
                case_id,
                status,
                expected_result,
                expected_direction,
                actual_result,
                actual_direction,
                name,
            )
        )

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"CASE: {case_id} — {name}")
            print_result(result, verbose=True)

    # Print summary
    print(f"\n{'=' * 70}")
    print("TEST RESULTS")
    print(f"{'=' * 70}")

    for case_id, status, exp_res, exp_dir, act_res, act_dir, name in results:
        dir_info = ""
        if exp_dir or act_dir:
            dir_info = f" dir: expected={exp_dir}, got={act_dir}"
        if status == "PASS":
            print(f"  [PASS] {case_id}: {act_res}" + (f" {act_dir}" if act_dir else ""))
        else:
            print(
                f"  [FAIL] {case_id}: expected {exp_res}"
                + (f" {exp_dir}" if exp_dir else "")
                + f", got {act_res}"
                + (f" {act_dir}" if act_dir else "")
                + f"{dir_info}"
            )

    total = passed + failed
    print(f"\n  {passed}/{total} passed")
    if failed > 0:
        print(f"  {failed}/{total} FAILED")
    print(f"{'=' * 70}")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Nudge Detection — Python Replication")
    parser.add_argument("input_file", help="JSON input file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose debug output"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    with open(input_path) as f:
        input_data = json.load(f)

    # Multi-case mode: if JSON has "test_cases" key, run all and print summary
    if "test_cases" in input_data:
        all_passed = run_all_test_cases(input_data, verbose=args.verbose)
        sys.exit(0 if all_passed else 1)

    # Single-case mode: original behavior
    result = run_detection(input_data)

    if args.json:
        print_json(result)
    else:
        print_result(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
