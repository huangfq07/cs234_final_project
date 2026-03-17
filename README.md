# Nudge Detection Pipeline

Python replication of a C++ nudge detection system for rapid prototyping and algorithm development. Detects when an ego vehicle executes a **nudge maneuver** — a lateral deviation around a nearby obstacle — using DGPS trajectory data and preprocessed object perception.

---

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Stage Details](#stage-details)
   - [Stage 1: DgpsTrajectoryAnalyzer](#stage-1-dgpstrajectoryanalyzer)
   - [Stage 2: NudgeObjectAnalyzer](#stage-2-nudgeobjectanalyzer)
   - [Stage 3: BiasNudgeDecider](#stage-3-biasnudgedecider)
   - [Stage 4: NudgeClassifier](#stage-4-nudgeclassifier)
5. [Data Types](#data-types)
6. [Configuration Parameters](#configuration-parameters)
7. [Coordinate System](#coordinate-system)
8. [Input JSON Schema](#input-json-schema)
9. [Output Schema](#output-schema)
10. [Usage](#usage)

---

## Overview

A **nudge** is a deliberate lateral deviation of the ego vehicle to avoid a nearby obstacle (e.g., a parked car, a cyclist) without a full lane change. The pipeline answers:

> *Did the ego just nudge? If so, around what object, in which direction, and how far out of lane?*

The system works entirely offline on a snapshot of DGPS trajectory + perception data. It mirrors the production C++ implementation function-by-function, making it easy to port algorithm changes back.

**Scope:** Core detection only. Excluded: ODD/lane-change gating, state machine, object tracking hysteresis, multi-frame smoothing.

---

## File Structure

```
nudge_detection.py                  ← Pipeline orchestrator + CLI entry point
nudge_detection_example_input.json  ← Multi-scenario test suite (JSON)

python/
  __init__.py                       ← Package marker
  data_types.py                     ← All shared dataclasses, Config, utility functions
  dgps_trajectory_analyzer.py       ← Stage 1: trajectory peak detection
  nudge_object_analyzer.py          ← Stage 2: object filtering + selection
  bias_nudge_decider.py             ← Stage 3: bias vs nudge decision
  nudge_classifier.py               ← Stage 4: event building + reasoning
```

### C++ File Mapping

| Python file | C++ source |
|---|---|
| `python/data_types.py` | `DgpsTrajectoryProcessor.hpp`, `DgpsTrajectoryAnalyzer.hpp`, `PreprocessedObjectInfo.hpp`, `BiasNudgeDecider.hpp`, `NudgeClassifierOutput.hpp`, `NudgePostprocessorUtils.hpp` |
| `python/dgps_trajectory_analyzer.py` | `DgpsTrajectoryAnalyzer.cpp` |
| `python/nudge_object_analyzer.py` | `NudgeObjectAnalyzer.cpp` |
| `python/bias_nudge_decider.py` | `BiasNudgeDecider.cpp` |
| `python/nudge_classifier.py` | `NudgeClassifier.cpp` |
| `nudge_detection.py` | `NudgePostprocessor.cpp` (orchestration) |

---

## Pipeline Architecture

```
JSON Input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  run_detection()  [nudge_detection.py]                  │
│                                                         │
│  ① DgpsTrajectoryAnalyzer.analyze()                    │
│     │  nf_deviations, ext_deviations, trajectory,      │
│     │  adaptive_baseline_m, lane_width, vehicle_width   │
│     ▼                                                   │
│     [hypotheses]  ← list of NudgeHypothesis             │
│                                                         │
│  ② (per hypothesis)                                     │
│     NudgeObjectAnalyzer.evaluate_objects_for_hypothesis()│
│     │  objects, hypothesis                              │
│     ▼                                                   │
│     [causal_objects, best_id, best_dist]                │
│                                                         │
│  ③ (per hypothesis)                                     │
│     BiasNudgeDecider.evaluate()                         │
│     │  causal_objects, peak_s_m, peak_magnitude,        │
│     │  detrended_deviations                             │
│     ▼                                                   │
│     BiasNudgeOutput  (is_bias: True / False)            │
│                                                         │
│  ④ NudgeClassifier.build_events()                      │
│     NudgeClassifier.select_primary()                    │
│     ▼                                                   │
│     DetectionResult                                     │
└─────────────────────────────────────────────────────────┘
    │
    ▼
DetectionResult {is_nudge_scenario, events, primary_event,
                 reasoning, debug}
```

---

## Stage Details

### Stage 1: DgpsTrajectoryAnalyzer

**File:** `python/dgps_trajectory_analyzer.py`
**Input:** Raw deviation series, DGPS trajectory, lane geometry
**Output:** List of `NudgeHypothesis` objects

Detects lateral deviation peaks in the DGPS trajectory and creates a time/spatial window (hypothesis) around each one.

#### Sub-steps

```
1. DETREND
   l'[i] = l[i] - adaptive_baseline_m
   Removes habitual driver positioning offset (e.g., habitually driving 0.5m left).

2. PEAK DETECTION  (detect_local_maxima)
   A point i is a peak if:
     |dev[i]| > |dev[i-1]|  AND  |dev[i]| > |dev[i+1]|  AND  |dev[i]| > 0.4m

3. PEAK FILTERING  (filter_peaks)
   Filter A — minimum magnitude after detrend:  |peak| >= 0.3m
   Filter B — ego-deviation ratio:  ego_dev <= peak * 1.1
              (if ego is already more offset than the peak, it's returning to center)

4. HYPOTHESIS WINDOW  (_create_hypothesis_windows, per peak)
   start: scan backward until |dev| < peak * 0.3  (near-field array)
   end:   scan forward  until |dev| < peak * 0.5  (extended-field array)
   Reject if trajectory in window is geometrically straight (road geometry, not avoidance)
   Validate: duration >= required_duration  OR  distance >= 4.0m

5. MERGE  (_merge_close_hypotheses)
   Overlapping or gap < 2.5s → merge into one (larger peak wins characteristics)

6. CLASSIFY TYPE
   peak > in_lane_threshold → OUT_OF_LANE
   else                     → IN_LANE
   in_lane_threshold = (lane_width/2) - (vehicle_width/2) + 0.1m
```

#### Trajectory Straightness Check (`is_straight_trajectory_window`)

Fits a direction line through 5 points before + 5 after the window. Measures perpendicular distance of window points to this line. If `max_perp < 0.2m` AND `rms < 0.1m` → peak is road geometry, not avoidance.

#### Peakedness Metrics (`compute_peakedness_metrics`)

Three shape metrics distinguish a sharp nudge peak from a flat bias plateau:

| Metric | Meaning | Nudge | Bias |
|---|---|---|---|
| PPR (Peak Prominence Ratio) | `(peak - baseline_mean) / baseline_std` | High | Low |
| PFI (Plateau Flatness Index) | Variance of top 40% deviation values | High | Low |
| CV (Coefficient of Variation) | `std / mean` in window | High | Low |

---

### Stage 2: NudgeObjectAnalyzer

**File:** `python/nudge_object_analyzer.py`
**Input:** All perception objects, one `NudgeHypothesis`
**Output:** `causal_objects` list, best matching object ID and adjusted distance

Filters all objects through a 4-stage causality pipeline, then selects the closest survivor.

#### Filter Stages (applied in order)

```
Stage 1 — LATERAL CAUSALITY  (_should_filter_lateral)
  Object must be on the OPPOSITE side of the trajectory from the nudge direction.
  If ego nudged LEFT → causal object must be on the RIGHT.
  Also filters objects currently crossing the trajectory (current side ≠ future side).

Stage 2 — LONGITUDINAL CAUSALITY  (_should_filter_longitudinal)
  Object bbox must overlap [hyp.start_s - 10m, hyp.end_s + 10m].
  Objects spatially unrelated to the maneuver are removed.

Stage 3 — SAFETY CLEARANCE  (_should_filter_safety)
  Required clearance = ego_half_width + margin
    VRU margin:  speed < 5 m/s → 0.3m | speed < 10 m/s → 0.6m | else → 1.0m
    Vehicle margin: 0.3m (fixed)
  If BOTH current AND future distances are below clearance → filter out.
  (Ego physically couldn't have passed: not the causal object)

Stage 4 — DISTANCE CRITERIA  (_meets_distance_criteria)
  distanceAheadOfEgo <= 55m
  distancePassedByEgo <= 0m  (not already fully behind ego)
  |minLateralDistToLaneCenter| <= 3.0m
```

#### Adjusted Distance

After passing all filters, each object gets an adjusted distance that accounts for perceived safety need:

```
VRU (pedestrian/cyclist):      raw_dist - 1.0m
Large vehicle (truck/bus):     raw_dist - 0.6m
Perpendicular vehicle          raw_dist - 1.0m
  (non-parked, heading ~90°)
All others:                    raw_dist
Clamped to >= 0.
```

The object with the **lowest adjusted distance** is selected as the best match.

---

### Stage 3: BiasNudgeDecider

**File:** `python/bias_nudge_decider.py`
**Input:** Causal objects from Stage 2, deviation series, peak info
**Output:** `BiasNudgeOutput` with `is_bias: True/False`

Distinguishes a **real nudge** (single exceptional obstacle) from **systematic bias** (e.g., driving alongside a long row of parked cars, where the deviation is from the row, not one car).

#### Decision Logic (V3 two-question approach)

```
Q1: Is there an EXCEPTIONAL object?
  ┌── 1 object in causal zone (raw dist <= 4m)?
  │       → automatically exceptional (nothing to compare to)
  │
  └── Multiple objects in causal zone?
          exceptionality_ratio = trimmed_mean(others_raw_dist) / closest_adjusted_dist
          exceptional if ratio >= 1.5

  Q1 = NO  →  BIAS  (uniform environment, not one obstacle)
  Q1 = YES →  proceed to Q2

Q2: Is the exceptional object a valid nudge candidate?
  Both sub-questions must pass:

  Q2a — AHEAD OF EGO:
    distanceAheadOfEgo > -2.0m  (allows slightly behind ego front)

  Q2b — NEAR THE PEAK:
    Primary zone:  |dist_from_peak| <= 10m  AND  deviation_ratio >= 0.5
    Extended zone: 10m < signed_dist <= 20m  AND  deviation_ratio >= 0.8
    (Extended zone catches "nudge-abort-then-nudge" where peak is behind but
     the real obstacle is still ahead with ego still deviating)

    deviation_ratio = dev(object_s) / dev(peak_s)
    (how much ego was deviating at the object's position, relative to the peak)

    Checks BOTH current and future projected positions; crosses through peak area also pass.

NUDGE = Q1 AND Q2a AND Q2b
BIAS  = anything else
```

#### Exceptionality Ratio Details

1. Collect raw distances of all *other* objects in causal zone (raw dist <= 4m)
2. Prefer non-VRU pool (VRUs have artificially low adjusted distance from buffers)
3. Trim min + max if pool has >= 3 objects (robust trimmed mean)
4. `ratio = mean_of_pool / closest_adjusted_dist`
5. If closest object is *on the baseline* (dist → 0): ratio = 100 (maximally exceptional)

---

### Stage 4: NudgeClassifier

**File:** `python/nudge_classifier.py`
**Input:** Hypotheses, object results, bias results, object list
**Output:** List of `NudgeEvent`, primary event, reasoning string

Final assembly stage.

#### build_events

For each hypothesis:
- Skip if `bias_result.is_bias == True`
- Skip if no object matched (`best_id` is None or 0)
- Create a `NudgeEvent` from hypothesis trajectory features + matched object info
- Compute `is_bilateral`: both left and right bias sums > 0.5m (rare, e.g., island)
- Generate reasoning string

#### select_primary

```python
primary = min(events, key=lambda e: e.adjusted_dist_m)
```
Purely proximity-based. No weighting.

#### Reasoning Format

```
"[in lane | out of lane] nudge [left | right] to [object description] on the [right | left]"
```
Examples:
- `"out of lane nudge left to a parked vehicle on the right"`
- `"in lane nudge right to a cyclist on the left"`

The side ("on the right/left") is always opposite to the nudge direction — that's where the causal object is.

---

## Data Types

All defined in `python/data_types.py`.

### Input types

| Class | Description |
|---|---|
| `DgpsPoint` | Single DGPS trajectory point: `(egoX, egoY, velX, velY, cameraTime)` |
| `PreprocessedObjectInfo` | Perception object with lateral/longitudinal distances, class flags, heading, bbox stations |

### Internal pipeline types

| Class | Description | Created by |
|---|---|---|
| `DeviationPeak` | Local maximum in `\|lateral deviation\|` | `DgpsTrajectoryAnalyzer` |
| `PeakednessMetrics` | Shape metrics (PPR, PFI, CV) of a deviation window | `DgpsTrajectoryAnalyzer` |
| `TrajectoryFeatures` | Window statistics: max/avg deviation, duration, jerk, direction | `DgpsTrajectoryAnalyzer` |
| `NudgeHypothesis` | Full time/spatial window around a peak, with features | `DgpsTrajectoryAnalyzer` |
| `BiasNudgeOutput` | Q1/Q2 results, exceptionality ratio, closest object info | `BiasNudgeDecider` |

### Output types

| Class | Description |
|---|---|
| `NudgeEvent` | Classified event: object, direction, type, distances, reasoning |
| `DetectionResult` | Top-level: `is_nudge_scenario`, `events`, `primary_event`, `reasoning`, `debug` |

### Config

`Config` dataclass in `data_types.py` holds **all tunable constants** organized by origin C++ file. Key groups:

| Group | Key parameters |
|---|---|
| DgpsTrajectoryAnalyzer | `min_nudge_deviation_m=0.4`, `merge_time_s=2.5`, `threshold_factor_start=0.3`, `threshold_factor_end=0.5` |
| Straight trajectory filter | `straight_max_perp_m=0.2`, `straight_max_rms_m=0.1` |
| NudgeObjectAnalyzer | `longitudinal_buffer_m=10.0`, `max_longitudinal_dist_m=55.0`, `max_lateral_dist_m=3.0` |
| BiasNudgeDecider | `causal_zone_m=4.0`, `exceptionality_threshold=1.5`, `primary_peak_dist_m=10.0`, `primary_dev_ratio=0.5` |

---

## Configuration Parameters

Full list in `python/data_types.py:Config`. Parameters to tune most often:

| Parameter | Default | Effect |
|---|---|---|
| `min_nudge_deviation_m` | 0.4 m | Minimum peak to detect. Lower → more sensitivity, more false positives. |
| `min_peak_after_detrend_m` | 0.3 m | Minimum peak after baseline subtraction. |
| `ego_deviation_to_peak_ratio` | 1.1 | Filters return-to-center peaks. Lower → stricter. |
| `merge_time_s` | 2.5 s | Hypotheses closer than this are merged. |
| `min_hypothesis_duration_s` | 0.5 s | Minimum maneuver time. |
| `min_hypothesis_distance_m` | 4.0 m | Minimum maneuver distance. |
| `exceptionality_threshold` | 1.5 | Ratio to qualify as exceptional. Higher → fewer nudge detections. |
| `causal_zone_m` | 4.0 m | Raw lateral distance defining the "causal zone" for ratio computation. |
| `critical_threshold_m` | 2.0 m | Adjusted distance for "within critical threshold" count. |
| `primary_peak_dist_m` | 10.0 m | Primary zone longitudinal span for Q2b. |
| `primary_dev_ratio` | 0.5 | Minimum deviation ratio for Q2b primary zone. |

---

## Coordinate System

```
Lane centerline → s axis (longitudinal)
Perpendicular  → l axis (lateral)
  l > 0 : LEFT of lane center
  l < 0 : RIGHT of lane center

DGPS deviation: [s_m, l_m]
  s_m = Frenet longitudinal station along lane centerline
  l_m = lateral offset from lane centerline

Object distances:
  All lateral distances use the CLOSEST BBOX EDGE, not object center.
  Distances are relative to the LANE CENTERLINE (not the planning reference line).
  The C++ preprocessing converts reference-line-relative → centerline-relative before input.

Nudge direction:
  direction=LEFT  → ego swerved LEFT → causal object is on the RIGHT
  direction=RIGHT → ego swerved RIGHT → causal object is on the LEFT
```

---

## Input JSON Schema

Defined by `nudge_detection_example_input.json`. Fields accepted by `run_detection()`:

```jsonc
{
  "dgpsTrajectory": [
    { "egoX": float, "egoY": float,
      "velX": float, "velY": float,
      "cameraTime": int }   // microseconds since epoch
  ],
  "egoLane": { "width_m": float },   // default 3.5
  "vehicleWidth_m": float,           // default 1.8
  "egoSpeed_mps": float,             // default 10.0 (used for VRU safety margin)
  "nearFieldDeviations":  [[s_m, l_m], ...],  // 0-40m range, for peak detection
  "extendedDeviations":   [[s_m, l_m], ...],  // 0-60m range, for end detection
  "adaptiveBaseline_m": float,       // driver offset bias to subtract; 0.0 = none
  "objects": [
    {
      "objectId": int,
      "obstacleClass": str,          // VEHICLE, PEDESTRIAN, CYCLIST, ...
      "obstacleState": str,          // PARKED, STOPPED, MOVING, REVERSE
      "speed_mps": float,
      "isVRU": bool,
      "isVehicle": bool,
      "isLargeVehicle": bool,
      "isOncomingVehicle": bool,
      "isOnLeftOfTrajectory": bool,
      "isOnLeftOfBaseline": bool,
      "minLateralDistToLaneCenter_m": float,
      "lateralDistToTrajectory_m": float,
      "minLateralDistToTrajectory_m": float,
      "minLateralDistToBaseline_m": float,
      "distanceAheadOfEgo_m": float,
      "distancePassedByEgo_m": float,
      "centerSOnLaneCenter_m": float,
      "bboxLongitudinalMin_m": float,
      "bboxLongitudinalMax_m": float,
      "hasFutureProjection": bool,
      "futureDistToTrajectory_m": float,
      "futureMinLateralDistToTrajectory_m": float,
      "futureCenterSOnLaneCenter_m": float,
      "headingDiffWithLaneCenter": float,  // radians
      "inLaneLeftMargin_m": float,
      "inLaneRightMargin_m": float,
      "description": str             // optional; overrides auto-generated description
    }
  ]
}
```

### Multi-case test file format

Wrap multiple scenarios under a `"test_cases"` key:

```jsonc
{
  "test_cases": {
    "case_name": {
      "name": "Human-readable name",
      "expected_result": "nudge | bias | no_detection",
      "expected_direction": "LEFT | RIGHT | null",
      // ... same fields as single-case above
    }
  }
}
```

---

## Output Schema

`DetectionResult` fields:

| Field | Type | Description |
|---|---|---|
| `is_nudge_scenario` | bool | True if at least one nudge event was detected |
| `events` | list[NudgeEvent] | All classified nudge events |
| `primary_event` | NudgeEvent or None | Event with the minimum adjusted distance |
| `reasoning` | str | Human-readable summary (from primary event, or failure reason) |
| `debug` | dict | Per-stage debug info (see below) |

`debug` keys:

| Key | Contents |
|---|---|
| `trajectory_analysis` | Peak counts, hypothesis counts, per-peak filter results, per-hypothesis window details |
| `object_analysis` | Per-hypothesis: per-object filter reasons, kept objects with distances |
| `bias_nudge_decisions` | Per-hypothesis: Q1/Q2 results, exceptionality ratio, closest object metrics |

---

## Usage

### Single scenario

```bash
python nudge_detection.py input.json
python nudge_detection.py input.json --verbose   # Show per-stage debug info
python nudge_detection.py input.json --json      # Machine-readable JSON output
```

### Verify with the bundled test suite

Run all 5 built-in scenarios against `nudge_detection_example_input.json`:

```bash
python3 nudge_detection.py nudge_detection_example_input.json
```

Expected output:

```
======================================================================
TEST RESULTS
======================================================================
  [PASS] single_parked_vehicle_nudge: nudge LEFT
  [PASS] parked_car_row_bias: bias
  [PASS] vru_pedestrian_nudge: nudge LEFT
  [PASS] wide_lane_baseline_nudge: nudge RIGHT
  [PASS] no_peak_flat_trajectory: no_detection

  5/5 passed
======================================================================
```

Exit code is `0` if all cases pass, `1` if any fail.

To inspect the per-stage debug info for each case:

```bash
python3 nudge_detection.py nudge_detection_example_input.json --verbose
```

### Programmatic use

```python
from nudge_detection import run_detection

with open("input.json") as f:
    input_data = json.load(f)

result = run_detection(input_data)

print(result.is_nudge_scenario)     # bool
print(result.primary_event)         # NudgeEvent or None
print(result.reasoning)             # str
```
