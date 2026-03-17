#!/usr/bin/env python3
"""
Adaptive Baseline Estimation - Algorithm Prototyping & Experiments

Purpose:
    Rapid prototyping and testing of adaptive baseline algorithms without C++ compilation.
    Test cases extracted from real debug logs with ground truth annotations.

Files:
    - adaptive_baseline_experiments.py (this file): Experimentation framework and algorithms
    - adaptive_baseline_data.json: Test case data (10 cases with ground truth)

Commands:
    python adaptive_baseline_experiments.py analyze [case_id]       # Show per-frame statistics
    python adaptive_baseline_experiments.py run [case_id]           # Run algorithms and compare
    python adaptive_baseline_experiments.py evaluate [case_id]      # Evaluate peak detectability
    python adaptive_baseline_experiments.py visualize <case_id>     # Visualize specific case
    python adaptive_baseline_experiments.py visualize_all           # Save all visualizations

Workflow:
    1. Collect test cases from debug logs -> adaptive_baseline_data.json
    2. Run 'analyze' to understand expected baselines per frame
    3. Run 'run' to compare current vs improved algorithms
    4. Run 'evaluate' to check peak detectability (the metric that matters)
    5. Iterate on algorithm design based on results
    6. Port final algorithm to C++ (NudgeDetectionContext.cpp)

Algorithm Selection Summary (Feb 2026):
    Six algorithms were tested (V1-V6). V6 was selected for C++ porting because:

    | Algorithm          | Window Selection      | Smoothing              | Key Property              |
    |--------------------|----------------------|------------------------|---------------------------|
    | V1-Current (C++)   | Global min-window    | EMA β=0.9 + L3 reject | Hard rejection can fail   |
    | V2-AdaptAlpha      | NF15 median          | Adaptive α EMA         | Too sensitive to NF noise |
    | V3-RunMed          | NF15 median          | Running median + EMA   | Fast convergence          |
    | V4-MinWinAdapt     | Global min-window    | Adaptive α EMA         | Global window contaminated|
    | V5-NFMinWin        | NF min-window        | Adaptive α EMA         | Good but slow early       |
    | V6-NFMinRunMed ✓   | NF min-window        | RunMed early + EMA     | Best overall              |

    V6 Results (evaluate command):
    - Zero TP regressions vs V1 across all 10 test cases
    - Eliminates FP on return_to_center_001 (V1 incorrectly detects)
    - Faster convergence: bias_then_nudge_001 baseline at F3 = -0.017m (V1: -0.279m at F6)

    Key Design Decisions:
    1. NF-only (15m): Avoids far-field contamination from actual nudge peaks
    2. Min-window (not plain median): Resists wide contamination within NF
    3. Running median early (5 frames): Instant convergence without EMA lag
    4. Adaptive alpha (0.20/0.12/0.05): No hard rejection, always converges
    5. No decay: Real bias should persist, not drift to zero
    6. No Layer 3: Hard rejection caused pathological cases; adaptive alpha is sufficient
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================
# Data Loading
# ============================================================


def load_cases(json_path: Optional[Path] = None) -> Dict:
    """Load test cases from JSON file as raw dicts"""
    if json_path is None:
        json_path = Path(__file__).parent / "adaptive_baseline_data.json"
    with open(json_path) as f:
        return json.load(f)


# ============================================================
# Frame Analysis Utilities
# ============================================================


def near_field_median(deviations: List[List[float]], max_s_m: float = 15.0) -> float:
    """Median of near-field deviations (s <= max_s_m)"""
    near = [d[1] for d in deviations if d[0] <= max_s_m]
    return float(np.median(near)) if near else 0.0


def min_sliding_window_median(
    deviations: List[List[float]], window_size: int = 5
) -> float:
    """Current C++ Layer 1: minimum absolute median over sliding windows"""
    l_vals = [d[1] for d in deviations]
    if len(l_vals) < window_size:
        return float(np.median(l_vals))

    min_abs_med = float("inf")
    result = 0.0
    for i in range(len(l_vals) - window_size + 1):
        med = float(np.median(l_vals[i : i + window_size]))
        if abs(med) < abs(min_abs_med):
            min_abs_med = abs(med)
            result = med
    return result


def nf_min_sliding_window_median(
    deviations: List[List[float]], max_s_m: float = 15.0, window_size: int = 5
) -> float:
    """Near-field min sliding window: min absolute median within s <= max_s_m.

    This is the V6 Layer 1 window selection method. It combines:
    - Near-field locality (only s <= max_s_m): avoids far-field contamination
    - Min-window robustness: resists wide contamination even within NF

    Matches C++ computeCurrentWindowBaseline() in NudgeDetectionContext.cpp.
    """
    nf_vals = [d[1] for d in deviations if d[0] <= max_s_m]
    if len(nf_vals) < window_size:
        return float(np.median(nf_vals)) if nf_vals else 0.0

    min_abs_med = float("inf")
    result = 0.0
    for i in range(len(nf_vals) - window_size + 1):
        med = float(np.median(nf_vals[i : i + window_size]))
        if abs(med) < abs(min_abs_med):
            min_abs_med = abs(med)
            result = med
    return result


def most_stable_window(
    deviations: List[List[float]], window_size: int = 5
) -> Tuple[float, float, int]:
    """Find most stable (lowest std) window. Returns (median, std, start_index)"""
    l_vals = [d[1] for d in deviations]
    if len(l_vals) < window_size:
        return float(np.median(l_vals)), float(np.std(l_vals)), 0

    min_std = float("inf")
    result_med = 0.0
    result_idx = 0
    for i in range(len(l_vals) - window_size + 1):
        w = l_vals[i : i + window_size]
        std = float(np.std(w))
        if std < min_std:
            min_std = std
            result_med = float(np.median(w))
            result_idx = i
    return result_med, min_std, result_idx


def compute_frame_stats(deviations: List[List[float]]) -> Dict:
    """Compute comprehensive statistics for a frame"""
    s_vals = np.array([d[0] for d in deviations])
    l_vals = np.array([d[1] for d in deviations])

    nf_mask = s_vals <= 15.0
    nf_l = l_vals[nf_mask]
    nf20_mask = s_vals <= 20.0
    nf20_l = l_vals[nf20_mask]

    stable_med, stable_std, stable_idx = most_stable_window(deviations)

    return {
        "n_points": len(deviations),
        "s_range": (float(s_vals[0]), float(s_vals[-1])),
        "full_median": float(np.median(l_vals)),
        "full_range": (float(np.min(l_vals)), float(np.max(l_vals))),
        "nf15_median": float(np.median(nf_l)) if len(nf_l) > 0 else None,
        "nf15_std": float(np.std(nf_l)) if len(nf_l) > 0 else None,
        "nf15_range": (float(np.min(nf_l)), float(np.max(nf_l)))
        if len(nf_l) > 0
        else None,
        "nf20_median": float(np.median(nf20_l)) if len(nf20_l) > 0 else None,
        "min_window": min_sliding_window_median(deviations),
        "stable_window": stable_med,
        "stable_std": stable_std,
        "stable_idx": stable_idx,
    }


# ============================================================
# Peak Detection & Freeze Logic
# (Simplified from DgpsTrajectoryAnalyzer.cpp)
# ============================================================


def detect_peaks_detrended(
    deviations: List[List[float]],
    baseline: float,
    min_magnitude: float = 0.4,
) -> List[Dict]:
    """
    Detect local maxima in detrended deviations.
    Matches C++ detectLocalMaxima() logic (DgpsTrajectoryAnalyzer.cpp:1022-1074).
    """
    peaks = []
    for i in range(1, len(deviations) - 1):
        l_det = deviations[i][1] - baseline
        l_prev = deviations[i - 1][1] - baseline
        l_next = deviations[i + 1][1] - baseline

        abs_curr = abs(l_det)
        abs_prev = abs(l_prev)
        abs_next = abs(l_next)

        if abs_curr > abs_prev and abs_curr > abs_next and abs_curr > min_magnitude:
            peaks.append(
                {
                    "index": i,
                    "s_m": deviations[i][0],
                    "magnitude": abs_curr,
                    "signed": l_det,
                }
            )
    return peaks


def find_peak_end(
    deviations: List[List[float]],
    peak_index: int,
    peak_magnitude: float,
    baseline: float,
    end_factor: float = 0.5,
) -> Optional[int]:
    """
    Scan forward from peak for return below threshold.
    Matches C++ findDeviationEnd() (DgpsTrajectoryAnalyzer.cpp:1448-1486).
    Returns end index or None if no valid end found.
    """
    threshold = peak_magnitude * end_factor
    for i in range(peak_index + 1, len(deviations)):
        if abs(deviations[i][1] - baseline) < threshold:
            return i
    return None


def find_peak_start(
    deviations: List[List[float]],
    peak_index: int,
    peak_magnitude: float,
    baseline: float,
    start_factor: float = 0.3,
) -> int:
    """
    Scan backward from peak to find where deviation starts rising.
    Matches C++ findDeviationStart() (DgpsTrajectoryAnalyzer.cpp:1417-1446).
    """
    threshold = peak_magnitude * start_factor
    for i in range(peak_index, 0, -1):
        if abs(deviations[i][1] - baseline) < threshold:
            return i
    return 0


def detect_freeze_trigger(
    deviations: List[List[float]],
    baseline: float,
    near_field_m: float = 30.0,
    min_peak: float = 0.4,
) -> Tuple[bool, Optional[Dict]]:
    """
    Check if any valid peak (with start+end) exists in near-field.
    If so, this frame triggers freeze for baseline estimation.

    This simulates the C++ freeze logic: when a nudge is actively being tracked,
    we freeze the baseline to prevent the nudge deviation from contaminating
    the baseline estimate.

    A valid peak requires:
    1. Local maximum > min_peak (after detrending)
    2. Located within near_field_m of ego
    3. Has a valid end (returns below 50% of peak magnitude)
    """
    peaks = detect_peaks_detrended(deviations, baseline, min_peak)

    for peak in peaks:
        if peak["s_m"] > near_field_m:
            continue
        end_idx = find_peak_end(deviations, peak["index"], peak["magnitude"], baseline)
        if end_idx is not None:
            start_idx = find_peak_start(
                deviations, peak["index"], peak["magnitude"], baseline
            )
            if end_idx > start_idx:
                return True, peak
    return False, None


# ============================================================
# Baseline Estimation Algorithms
# ============================================================


class BaselineState:
    """Mutable state for baseline estimator"""

    def __init__(self):
        self.initialized = False
        self.frame_count = 0
        self.ema_baseline = 0.0
        self.sustained_direction = 0  # +1 above, -1 below, 0 neutral
        self.sustained_frames = 0

    def reset(self):
        self.__init__()


class CurrentAlgorithm:
    """
    Approximate reproduction of current C++ implementation.

    Layer 1: Min absolute median sliding window (5 points)
    Layer 2: EMA with alpha=0.1
    Layer 3: Anomaly rejection (hard reject if deviation > threshold and frame > 3)
    Layer 4: Dynamic bounds ± max(0.3, laneWidth * 0.4) + 0.5% decay
    """

    NAME = "V1-Current"

    def __init__(self):
        self.state = BaselineState()

    def reset(self):
        self.state.reset()

    def estimate(self, deviations: List[List[float]], lane_width: float) -> Dict:
        # Dynamic bounds
        max_offset = max(0.3, lane_width * 0.4)

        # Layer 1: Min sliding window
        current_window = min_sliding_window_median(deviations)

        # First frame: initialize
        if not self.state.initialized:
            self.state.ema_baseline = float(
                np.clip(current_window, -max_offset, max_offset)
            )
            self.state.frame_count = 1
            self.state.initialized = True
            return {
                "window": current_window,
                "ema": self.state.ema_baseline,
                "final": self.state.ema_baseline,
                "rejected": False,
                "alpha_used": 1.0,
            }

        # Layer 3: Anomaly detection
        deviation_from_history = abs(current_window - self.state.ema_baseline)
        anomaly_threshold = float(np.clip(max_offset * 0.5, 0.4, 0.8))

        rejected = (
            deviation_from_history > anomaly_threshold and self.state.frame_count > 3
        )

        if rejected:
            estimate = self.state.ema_baseline
        else:
            estimate = current_window

        # Layer 2: EMA
        alpha = 0.1
        new_ema = (1 - alpha) * self.state.ema_baseline + alpha * estimate

        # Layer 4: Bounds + decay
        new_ema *= 0.995  # 0.5% decay
        final = float(np.clip(new_ema, -max_offset, max_offset))

        self.state.ema_baseline = final
        self.state.frame_count += 1

        return {
            "window": current_window,
            "ema": new_ema,
            "final": final,
            "rejected": rejected,
            "alpha_used": alpha if not rejected else 0.0,
        }


class ImprovedAlgorithm:
    """
    V2: NF15 median + adaptive alpha (no sustained shift).

    Changes from V1:
    1. Layer 1: Near-field median (first 15m) - better current-position estimate
    2. Layer 3: Adaptive alpha instead of hard rejection - always converges
    3. No sustained shift detection - too dangerous without freeze guarantee
    4. No decay - fights against real bias in wide lanes
    """

    NAME = "V2-AdaptAlpha"

    def __init__(self, near_field_m: float = 15.0):
        self.state = BaselineState()
        self.near_field_m = near_field_m

    def reset(self):
        self.state.reset()

    def estimate(self, deviations: List[List[float]], lane_width: float) -> Dict:
        max_offset = max(0.3, lane_width * 0.4)
        current_window = near_field_median(deviations, max_s_m=self.near_field_m)

        if not self.state.initialized:
            self.state.ema_baseline = float(
                np.clip(current_window, -max_offset, max_offset)
            )
            self.state.frame_count = 1
            self.state.initialized = True
            return {
                "window": current_window,
                "ema": self.state.ema_baseline,
                "final": self.state.ema_baseline,
                "rejected": False,
                "alpha_used": 1.0,
                "sustained_frames": 0,
            }

        abs_deviation = abs(current_window - self.state.ema_baseline)

        # Adaptive alpha (no hard rejection, no sustained shift)
        if abs_deviation < 0.2:
            alpha = 0.20  # Close: fast
        elif abs_deviation < 0.5:
            alpha = 0.12  # Moderate
        else:
            alpha = 0.05  # Large: slow but converging

        new_ema = (1 - alpha) * self.state.ema_baseline + alpha * current_window
        final = float(np.clip(new_ema, -max_offset, max_offset))

        self.state.ema_baseline = final
        self.state.frame_count += 1

        return {
            "window": current_window,
            "ema": new_ema,
            "final": final,
            "rejected": False,
            "alpha_used": alpha,
            "sustained_frames": 0,
        }


class MinWindowAdaptAlphaAlgorithm:
    """
    V4: Min sliding window (like V1) + adaptive alpha (like V2) + no decay.

    Combines V1's conservative window selection with V2's soft convergence.
    - Layer 1: Min absolute median sliding window (same as V1)
    - Layer 2: Adaptive alpha EMA (same as V2) - no hard rejection
    - No decay (same as V2/V3)
    """

    NAME = "V4-MinWinAdapt"

    def __init__(self):
        self.state = BaselineState()

    def reset(self):
        self.state.reset()

    def estimate(self, deviations: List[List[float]], lane_width: float) -> Dict:
        max_offset = max(0.3, lane_width * 0.4)
        current_window = min_sliding_window_median(deviations)

        if not self.state.initialized:
            self.state.ema_baseline = float(
                np.clip(current_window, -max_offset, max_offset)
            )
            self.state.frame_count = 1
            self.state.initialized = True
            return {
                "window": current_window,
                "ema": self.state.ema_baseline,
                "final": self.state.ema_baseline,
                "rejected": False,
                "alpha_used": 1.0,
                "sustained_frames": 0,
            }

        abs_deviation = abs(current_window - self.state.ema_baseline)

        # Adaptive alpha (same as V2) - no hard rejection
        if abs_deviation < 0.2:
            alpha = 0.20
        elif abs_deviation < 0.5:
            alpha = 0.12
        else:
            alpha = 0.05

        new_ema = (1 - alpha) * self.state.ema_baseline + alpha * current_window
        # No decay (unlike V1)
        final = float(np.clip(new_ema, -max_offset, max_offset))

        self.state.ema_baseline = final
        self.state.frame_count += 1

        return {
            "window": current_window,
            "ema": new_ema,
            "final": final,
            "rejected": False,
            "alpha_used": alpha,
            "sustained_frames": 0,
        }


class NfMinWindowAdaptAlphaAlgorithm:
    """
    V5: Near-field min sliding window + adaptive alpha + no decay.

    Combines locality (near-field only) with contamination resistance (min-window).
    - Layer 1: Min absolute median sliding window within first 15m only
    - Layer 2: Adaptive alpha EMA (no hard rejection)
    - No decay
    """

    NAME = "V5-NFMinWin"

    def __init__(self, near_field_m: float = 15.0):
        self.state = BaselineState()
        self.near_field_m = near_field_m

    def reset(self):
        self.state.reset()

    def estimate(self, deviations: List[List[float]], lane_width: float) -> Dict:
        max_offset = max(0.3, lane_width * 0.4)
        current_window = nf_min_sliding_window_median(
            deviations, max_s_m=self.near_field_m
        )

        if not self.state.initialized:
            self.state.ema_baseline = float(
                np.clip(current_window, -max_offset, max_offset)
            )
            self.state.frame_count = 1
            self.state.initialized = True
            return {
                "window": current_window,
                "ema": self.state.ema_baseline,
                "final": self.state.ema_baseline,
                "rejected": False,
                "alpha_used": 1.0,
                "sustained_frames": 0,
            }

        abs_deviation = abs(current_window - self.state.ema_baseline)

        if abs_deviation < 0.2:
            alpha = 0.20
        elif abs_deviation < 0.5:
            alpha = 0.12
        else:
            alpha = 0.05

        new_ema = (1 - alpha) * self.state.ema_baseline + alpha * current_window
        final = float(np.clip(new_ema, -max_offset, max_offset))

        self.state.ema_baseline = final
        self.state.frame_count += 1

        return {
            "window": current_window,
            "ema": new_ema,
            "final": final,
            "rejected": False,
            "alpha_used": alpha,
            "sustained_frames": 0,
        }


class RunningMedianAlgorithm:
    """
    V3: Running median for early convergence, then adaptive alpha EMA.
    Uses NF15 median as window selection.
    """

    NAME = "V3-RunMed"

    def __init__(self, near_field_m: float = 15.0, early_frames: int = 5):
        self.state = BaselineState()
        self.near_field_m = near_field_m
        self.early_frames = early_frames
        self.window_history: List[float] = []

    def reset(self):
        self.state.reset()
        self.window_history = []

    def estimate(self, deviations: List[List[float]], lane_width: float) -> Dict:
        max_offset = max(0.3, lane_width * 0.4)
        current_window = near_field_median(deviations, max_s_m=self.near_field_m)

        self.window_history.append(current_window)
        self.state.frame_count += 1

        if self.state.frame_count <= self.early_frames:
            baseline = float(np.median(self.window_history))
            alpha = 1.0
        else:
            if not self.state.initialized:
                baseline = float(np.median(self.window_history))
                alpha = 1.0
            else:
                abs_deviation = abs(current_window - self.state.ema_baseline)
                if abs_deviation < 0.2:
                    alpha = 0.20
                elif abs_deviation < 0.5:
                    alpha = 0.12
                else:
                    alpha = 0.05
                baseline = (
                    1 - alpha
                ) * self.state.ema_baseline + alpha * current_window

        final = float(np.clip(baseline, -max_offset, max_offset))

        self.state.ema_baseline = final
        self.state.initialized = True

        return {
            "window": current_window,
            "ema": baseline,
            "final": final,
            "rejected": False,
            "alpha_used": alpha,
            "sustained_frames": 0,
        }


class NfMinWindowRunningMedianAlgorithm:
    """
    V6: NF min sliding window + running median (early) + adaptive alpha (steady).
    >>> SELECTED FOR C++ PORTING <<<

    Combines:
    - V5's contamination-resistant window (NF min sliding window within first 15m)
    - V3's fast early convergence (running median for first 5 frames)
    - Adaptive alpha EMA for steady-state (0.20/0.12/0.05 based on deviation)

    Why V6 wins over all others:
    - Zero TP regressions vs V1 (current C++) across all 10 test cases
    - Eliminates FP on return_to_center_001 (V1 incorrectly detects)
    - Faster convergence: bias_then_nudge_001 at F3 = -0.017m (V1: -0.279m at F6)
    - No hard rejection: adaptive alpha always converges (avoids Layer 3 pathological cases)
    - No decay: real bias persists correctly (V1's decay fights against wide-lane positioning)

    C++ implementation: NudgeDetectionContext::estimateAdaptiveBaseline()
    """

    NAME = "V6-NFMinRunMed"

    def __init__(self, near_field_m: float = 15.0, early_frames: int = 5):
        self.state = BaselineState()
        self.near_field_m = near_field_m
        self.early_frames = early_frames
        self.window_history: List[float] = []

    def reset(self):
        self.state.reset()
        self.window_history = []

    def estimate(self, deviations: List[List[float]], lane_width: float) -> Dict:
        max_offset = max(0.3, lane_width * 0.4)
        current_window = nf_min_sliding_window_median(
            deviations, max_s_m=self.near_field_m
        )

        self.window_history.append(current_window)
        self.state.frame_count += 1

        if self.state.frame_count <= self.early_frames:
            # Early frames: running median of all NF min-window estimates
            baseline = float(np.median(self.window_history))
            alpha = 1.0
        else:
            if not self.state.initialized:
                baseline = float(np.median(self.window_history))
                alpha = 1.0
            else:
                abs_deviation = abs(current_window - self.state.ema_baseline)
                if abs_deviation < 0.2:
                    alpha = 0.20
                elif abs_deviation < 0.5:
                    alpha = 0.12
                else:
                    alpha = 0.05
                baseline = (
                    1 - alpha
                ) * self.state.ema_baseline + alpha * current_window

        final = float(np.clip(baseline, -max_offset, max_offset))

        self.state.ema_baseline = final
        self.state.initialized = True

        return {
            "window": current_window,
            "ema": baseline,
            "final": final,
            "rejected": False,
            "alpha_used": alpha,
            "sustained_frames": 0,
        }


# ============================================================
# Commands
# ============================================================


def cmd_analyze(cases: Dict, case_id: Optional[str] = None):
    """Analyze deviation data to understand expected baselines per frame"""
    target = {case_id: cases[case_id]} if case_id else cases
    for cid, cdata in target.items():
        _analyze_case(cid, cdata)


def _analyze_case(case_id: str, case_data: Dict):
    """Print detailed analysis for one case"""
    print(f"\n{'='*110}")
    print(f"Case: {case_id}")
    print(f"Type: {case_data.get('scenario_type', '?')}")
    print(f"Desc: {case_data.get('description', '')}")

    expected = case_data.get("expected_baseline_m")
    if expected is not None:
        print(f"Expected baseline: {expected:.3f}m")
    else:
        print("Expected baseline: VARIES (see frame annotations)")

    annotations = case_data.get("frame_annotations", {})

    header = (
        f"{'Frame':<6} {'LaneW':<7} {'NF15med':<9} {'NF15std':<9} "
        f"{'NF15 Range':<18} {'NF20med':<9} {'MinWin':<9} "
        f"{'StableW':<9} {'C++Fin':<9} {'C++Rej':<7}"
    )
    print(f"\n{header}")
    print(f"{'-'*len(header)}")

    for frame in case_data.get("frames", []):
        fid = frame["frame_id"]
        lw = frame["lane_width_m"]
        stats = compute_frame_stats(frame["deviations"])

        nf15 = stats["nf15_median"]
        nf15_std = stats["nf15_std"]
        nf15_range = stats["nf15_range"]
        nf20 = stats["nf20_median"]
        min_win = stats["min_window"]
        stable = stats["stable_window"]
        cpp_fin = frame.get("cpp_final", float("nan"))
        cpp_rej = frame.get("cpp_layer3_rejected", False)

        nf_str = f"[{nf15_range[0]:.2f},{nf15_range[1]:.2f}]" if nf15_range else "N/A"
        rej_str = "REJ" if cpp_rej else ""

        nf15_s = f"{nf15:.3f}" if nf15 is not None else "N/A"
        nf15_std_s = f"{nf15_std:.3f}" if nf15_std is not None else "N/A"
        nf20_s = f"{nf20:.3f}" if nf20 is not None else "N/A"

        print(
            f"F{fid:<5} {lw:<7.2f} {nf15_s:<9} {nf15_std_s:<9} "
            f"{nf_str:<18} {nf20_s:<9} {min_win:<9.3f} "
            f"{stable:<9.3f} {cpp_fin:<9.3f} {rej_str:<7}"
        )

        ann_key = f"frame_{fid}"
        if ann_key in annotations:
            ann = annotations[ann_key]
            print(
                f"       -> Expected: {ann.get('expected_baseline', '?')}m"
                f" | {ann.get('note', '')}"
            )


def _run_algo_on_case(
    algo, frames: List[Dict], expected, annotations: Dict, freeze_enabled: bool
):
    """Run a single algorithm on a case with optional freeze logic.

    Freeze behavior (matching C++ NudgeDetectionContext lifecycle):
    - After each frame, run peak detection on detrended deviations
    - If valid peak found in near-field → freeze baseline for NEXT frame
    - Unfreeze when no peak detected (simulates tracking end)
    """
    algo.reset()
    frozen = False
    current_baseline = 0.0
    rows = []

    for frame in frames:
        fid = frame["frame_id"]
        devs = frame["deviations"]
        lw = frame["lane_width_m"]

        if frozen and freeze_enabled:
            # Frozen: don't update baseline, return current value
            result = {
                "window": 0.0,
                "ema": current_baseline,
                "final": current_baseline,
                "rejected": False,
                "alpha_used": 0.0,
                "sustained_frames": 0,
            }
        else:
            result = algo.estimate(devs, lw)
            current_baseline = result["final"]

        # Check for peaks to determine freeze state for NEXT frame
        if freeze_enabled:
            has_peak, peak_info = detect_freeze_trigger(devs, current_baseline)
            if has_peak:
                frozen = True
            else:
                frozen = False  # Unfreeze when no peak (tracking ended)

        # Determine expected baseline for this frame
        ann_key = f"frame_{fid}"
        if (
            ann_key in annotations
            and annotations[ann_key].get("expected_baseline") is not None
        ):
            frame_expected = annotations[ann_key]["expected_baseline"]
        else:
            frame_expected = expected

        if frame_expected is not None:
            error = result["final"] - frame_expected
            err_str = f"{error:+.3f}"
        else:
            err_str = "N/A"

        frz_str = "FRZ" if (frozen and freeze_enabled) else ""
        rows.append((fid, lw, result, err_str, frz_str))

    return rows


def cmd_evaluate(cases: Dict, case_id: Optional[str] = None):
    """Evaluate what matters: can we detect nudge peaks after detrending?

    For each algorithm and each frame, compute:
    - Max detrended peak magnitude (is it > 0.4m threshold?)
    - Whether a valid peak (with start+end) is detected
    This tells us if baseline estimation hurts or helps TP nudge detection.
    """
    algorithms = [
        CurrentAlgorithm(),
        NfMinWindowAdaptAlphaAlgorithm(),
        NfMinWindowRunningMedianAlgorithm(),
    ]
    target = {case_id: cases[case_id]} if case_id else cases

    # Summary table
    summary_rows = []

    for cid, cdata in target.items():
        frames = cdata.get("frames", [])
        scenario = cdata.get("scenario_type", "?")
        expected = cdata.get("expected_baseline_m")

        if not frames:
            continue

        print(f"\n{'='*100}")
        print(f"Case: {cid}  |  Type: {scenario}  |  Expected baseline: {expected}")
        print(f"{'='*100}")

        # For each algorithm, run with freeze and check peak detection per frame
        algo_summaries = {}
        for algo in algorithms:
            results = _run_algo_with_freeze(algo, frames)

            print(f"\n  {algo.NAME}:")
            header = f"    {'Frame':<6} {'Baseline':<10} {'MaxPeak':<10} {'PeakS':<8} {'Valid':<7} {'Detected':<10}"
            print(header)
            print(f"    {'-'*len(header)}")

            best_peak = 0.0
            best_peak_frame = 0
            n_detected = 0
            first_detected_frame = None

            for frame, result in zip(frames, results):
                fid = frame["frame_id"]
                devs = frame["deviations"]
                bl = result["baseline"]

                # Find max detrended peak
                peaks = detect_peaks_detrended(devs, bl, min_magnitude=0.0)
                if peaks:
                    max_peak = max(peaks, key=lambda p: p["magnitude"])
                    mag = max_peak["magnitude"]
                    peak_s = max_peak["s_m"]

                    # Check if valid (has start + end)
                    end_idx = find_peak_end(devs, max_peak["index"], mag, bl)
                    has_valid = end_idx is not None
                    detected = mag > 0.4 and has_valid

                    det_str = "YES" if detected else ("weak" if mag > 0.3 else "no")
                    valid_str = "yes" if has_valid else "no"

                    print(
                        f"    F{fid:<5} {bl:<10.3f} {mag:<10.3f} {peak_s:<8.1f} "
                        f"{valid_str:<7} {det_str:<10}"
                    )

                    if mag > best_peak:
                        best_peak = mag
                        best_peak_frame = fid
                    if detected:
                        n_detected += 1
                        if first_detected_frame is None:
                            first_detected_frame = fid
                else:
                    print(
                        f"    F{fid:<5} {bl:<10.3f} {'--':<10} {'--':<8} {'--':<7} {'no':<10}"
                    )

            algo_summaries[algo.NAME] = {
                "best_peak": best_peak,
                "best_peak_frame": best_peak_frame,
                "n_detected": n_detected,
                "first_detected": first_detected_frame,
                "n_frames": len(frames),
            }

        # Print comparison summary for this case
        print(f"\n  SUMMARY for {cid}:")
        print(
            f"    {'Algorithm':<22} {'BestPeak':<10} {'@Frame':<8} {'Detected':<12} {'FirstAt':<8}"
        )
        print(f"    {'-'*60}")
        for algo in algorithms:
            s = algo_summaries[algo.NAME]
            first = f"F{s['first_detected']}" if s["first_detected"] else "--"
            print(
                f"    {algo.NAME:<22} {s['best_peak']:<10.3f} F{s['best_peak_frame']:<7} "
                f"{s['n_detected']}/{s['n_frames']:<10} {first:<8}"
            )
        summary_rows.append((cid, scenario, algo_summaries))

    # Final cross-case summary
    print(f"\n\n{'='*100}")
    print("CROSS-CASE SUMMARY: Peak Detection After Detrending")
    print(f"{'='*100}")
    algo_names = [a.NAME for a in algorithms]
    header = f"{'Case':<28} {'Type':<12} " + "  ".join(f"{n:<20}" for n in algo_names)
    print(header)
    print("-" * len(header))

    for cid, scenario, algo_sums in summary_rows:
        parts = [f"{cid:<28} {scenario:<12}"]
        for name in algo_names:
            s = algo_sums[name]
            first = f"F{s['first_detected']}" if s["first_detected"] else "--"
            parts.append(
                f"{s['best_peak']:.2f}m {s['n_detected']}/{s['n_frames']} @{first:<6}"
            )
        print("  ".join(parts))


def cmd_run(cases: Dict, case_id: Optional[str] = None):
    """Run algorithms on cases and compare (with freeze logic)"""
    algorithms = [
        CurrentAlgorithm(),
        ImprovedAlgorithm(),
        RunningMedianAlgorithm(),
        MinWindowAdaptAlphaAlgorithm(),
        NfMinWindowAdaptAlphaAlgorithm(),
        NfMinWindowRunningMedianAlgorithm(),
    ]
    target = {case_id: cases[case_id]} if case_id else cases

    for cid, cdata in target.items():
        expected = cdata.get("expected_baseline_m")
        annotations = cdata.get("frame_annotations", {})
        frames = cdata.get("frames", [])

        print(f"\n{'='*120}")
        print(f"Case: {cid}")
        print(f"Expected: {expected if expected is not None else 'VARIES'}")
        print(f"{'='*120}")

        for algo in algorithms:
            header = (
                f"  {'Frame':<6} {'LaneW':<7} {'Window':<9} {'Alpha':<7} "
                f"{'EMA':<9} {'Final':<9} {'Error':<9} {'Freeze':<7}"
            )
            print(f"\n  {algo.NAME} (with freeze):")
            print(f"  {header}")
            print(f"  {'-'*len(header)}")

            rows = _run_algo_on_case(algo, frames, expected, annotations, True)
            for fid, lw, result, err_str, frz_str in rows:
                print(
                    f"  F{fid:<5} {lw:<7.2f} "
                    f"{result['window']:<9.3f} {result['alpha_used']:<7.3f} "
                    f"{result['ema']:<9.3f} {result['final']:<9.3f} "
                    f"{err_str:<9} {frz_str:<7}"
                )


def _run_algo_with_freeze(algo, frames: List[Dict]) -> List[Dict]:
    """Run algorithm with freeze logic, return per-frame results with baseline."""
    algo.reset()
    frozen = False
    current_baseline = 0.0
    results = []

    for frame in frames:
        devs = frame["deviations"]
        lw = frame["lane_width_m"]

        if frozen:
            result = {
                "window": 0.0,
                "ema": current_baseline,
                "final": current_baseline,
                "rejected": False,
                "alpha_used": 0.0,
                "sustained_frames": 0,
                "frozen": True,
            }
        else:
            result = algo.estimate(devs, lw)
            result["frozen"] = False
            current_baseline = result["final"]

        # Check freeze trigger for next frame
        has_peak, _ = detect_freeze_trigger(devs, current_baseline)
        frozen = has_peak

        result["frame_id"] = frame["frame_id"]
        result["baseline"] = current_baseline
        results.append(result)

    return results


def cmd_visualize(cases: Dict, case_id: str, save_dir: Optional[Path] = None):
    """
    Visualize before/after detrend for each algorithm.

    Layout: 2 rows x 3 columns
      Row 0: Before detrend (raw deviations + baseline line)
      Row 1: After detrend (deviations - baseline, centered around 0)
      Columns: V1-Current, V2-AdaptAlpha, V3-RunMed
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    if case_id not in cases:
        print(f"Case '{case_id}' not found. Available: {list(cases.keys())}")
        return

    cdata = cases[case_id]
    frames = cdata.get("frames", [])
    expected = cdata.get("expected_baseline_m")

    if not frames:
        print("No frames in case")
        return

    # Run all algorithms with freeze
    algos = [
        CurrentAlgorithm(),
        ImprovedAlgorithm(),
        RunningMedianAlgorithm(),
        MinWindowAdaptAlphaAlgorithm(),
        NfMinWindowAdaptAlphaAlgorithm(),
        NfMinWindowRunningMedianAlgorithm(),
    ]
    algo_colors = [
        "#e74c3c",
        "#3498db",
        "#9b59b6",
        "#e67e22",
        "#27ae60",
        "#2c3e50",
    ]  # red, blue, purple, orange, green, dark
    algo_results = {}
    for algo in algos:
        algo_results[algo.NAME] = _run_algo_with_freeze(algo, frames)

    n_frames = len(frames)
    n_algos = len(algos)
    frame_cmap = plt.cm.viridis

    fig, axes = plt.subplots(2, n_algos, figsize=(7 * n_algos, 10))
    if n_algos == 1:
        axes = axes.reshape(2, 1)

    for col, algo in enumerate(algos):
        results = algo_results[algo.NAME]
        algo_color = algo_colors[col]

        # --- Row 0: Before detrend ---
        ax_raw = axes[0, col]
        for i, frame in enumerate(frames):
            color = frame_cmap(i / max(n_frames - 1, 1))
            s = [d[0] for d in frame["deviations"]]
            l_val = [d[1] for d in frame["deviations"]]
            label = f"F{frame['frame_id']}"
            ax_raw.plot(s, l_val, color=color, alpha=0.6, linewidth=1.2, label=label)

        # Draw per-frame baseline as horizontal segments
        for i, (frame, result) in enumerate(zip(frames, results)):
            bl = result["baseline"]
            s_vals = [d[0] for d in frame["deviations"]]
            s_min, s_max = s_vals[0], s_vals[-1]
            style = "--" if not result.get("frozen", False) else ":"
            lw = 2.0 if not result.get("frozen", False) else 1.0
            frz_tag = " FRZ" if result.get("frozen", False) else ""
            ax_raw.hlines(
                bl,
                s_min,
                s_max,
                colors=algo_color,
                linestyles=style,
                linewidth=lw,
                alpha=0.8,
            )
            # Annotate baseline value at left edge
            ax_raw.text(
                s_min + 1,
                bl + 0.03,
                f"F{frame['frame_id']}:{bl:.2f}{frz_tag}",
                fontsize=6,
                color=algo_color,
                alpha=0.9,
            )

        if expected is not None:
            ax_raw.axhline(
                expected,
                color="green",
                linestyle="-",
                linewidth=2.5,
                alpha=0.7,
                label=f"Expected={expected:.2f}m",
            )

        ax_raw.set_title(f"{algo.NAME}\nBefore Detrend", fontsize=11, fontweight="bold")
        ax_raw.set_xlabel("s (m)")
        if col == 0:
            ax_raw.set_ylabel("Lateral deviation (m)")
        ax_raw.legend(fontsize=6, loc="upper right", ncol=2)
        ax_raw.grid(True, alpha=0.3)

        # --- Row 1: After detrend ---
        ax_det = axes[1, col]
        for i, (frame, result) in enumerate(zip(frames, results)):
            color = frame_cmap(i / max(n_frames - 1, 1))
            bl = result["baseline"]
            s = [d[0] for d in frame["deviations"]]
            l_det = [d[1] - bl for d in frame["deviations"]]
            frz_tag = " (FRZ)" if result.get("frozen", False) else ""
            ax_det.plot(
                s,
                l_det,
                color=color,
                alpha=0.6,
                linewidth=1.2,
                label=f"F{frame['frame_id']}{frz_tag}",
            )

        # Zero line (ideal detrended baseline)
        ax_det.axhline(0, color="green", linestyle="-", linewidth=2.5, alpha=0.7)
        # Peak detection threshold
        ax_det.axhline(
            0.4,
            color="gray",
            linestyle=":",
            linewidth=1,
            alpha=0.5,
            label="Peak thr 0.4m",
        )
        ax_det.axhline(-0.4, color="gray", linestyle=":", linewidth=1, alpha=0.5)

        ax_det.set_title(f"{algo.NAME}\nAfter Detrend", fontsize=11, fontweight="bold")
        ax_det.set_xlabel("s (m)")
        if col == 0:
            ax_det.set_ylabel("Detrended deviation (m)")
        ax_det.legend(fontsize=6, loc="upper right", ncol=2)
        ax_det.grid(True, alpha=0.3)

    # Compute final errors for suptitle
    err_parts = []
    for algo in algos:
        results = algo_results[algo.NAME]
        last_bl = results[-1]["baseline"]
        if expected is not None:
            err = last_bl - expected
            err_parts.append(f"{algo.NAME}: {last_bl:.3f}m (err={err:+.3f})")
        else:
            err_parts.append(f"{algo.NAME}: {last_bl:.3f}m")

    fig.suptitle(
        f"{case_id}  |  " + "  |  ".join(err_parts),
        fontsize=10,
        y=1.01,
    )

    plt.tight_layout()

    out_dir = save_dir or Path(__file__).parent
    save_path = out_dir / f"{case_id}_baseline.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    cases = load_cases()

    if not cases:
        print("No test cases found in adaptive_baseline_data.json")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Adaptive Baseline Estimation Experiments")
        print("=" * 50)
        print("Commands:")
        print("  analyze [case_id]       Show per-frame deviation statistics")
        print("  run [case_id]           Run algorithms and compare")
        print("  visualize <case_id>     Visualize specific case")
        print("  visualize_all           Save all visualizations")
        print(f"\nAvailable cases ({len(cases)}): {list(cases.keys())}")
        sys.exit(0)

    command = sys.argv[1]
    case_arg = sys.argv[2] if len(sys.argv) > 2 else None

    if command == "analyze":
        cmd_analyze(cases, case_arg)
    elif command == "evaluate":
        cmd_evaluate(cases, case_arg)
    elif command == "run":
        cmd_run(cases, case_arg)
    elif command == "visualize":
        if case_arg:
            cmd_visualize(cases, case_arg)
        else:
            print("Usage: visualize <case_id>")
    elif command == "visualize_all":
        for cid in cases:
            cmd_visualize(cases, cid)
    elif command in cases:
        # Backward compat: treat as case_id for 'run'
        cmd_run(cases, command)
    else:
        print(f"Unknown command or case: {command}")
        print(f"Available: analyze, run, visualize, visualize_all")
        print(f"Cases: {list(cases.keys())}")
