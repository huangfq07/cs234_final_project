"""
Nudge Classifier — Python replication of NudgeClassifier.cpp

Final stage of the pipeline: combines trajectory hypotheses, object matches,
and bias/nudge decisions into classified NudgeEvent objects.

Pipeline:
  1. BUILD EVENTS: For each hypothesis, if the bias/nudge decision was NUDGE and
     an object was matched, create a NudgeEvent with all relevant fields from both
     the hypothesis (trajectory features) and the object (class, state, distances).

  2. SELECT PRIMARY: Choose the event with the minimum adjusted distance (closest
     object to the ego's adaptive baseline). This is the most critical event.

  3. GENERATE REASONING: Create a human-readable string describing the detection,
     e.g., "out of lane nudge left to a parked vehicle on the right".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .data_types import (
    BiasNudgeOutput,
    NudgeEvent,
    NudgeHypothesis,
    PreprocessedObjectInfo,
)


class NudgeClassifier:
    """Matches C++ NudgeClassifier class."""

    # ==============================================================================================
    # Public API — matches NudgeClassifier::process()
    # ==============================================================================================

    def build_events(
        self,
        hypotheses: List[NudgeHypothesis],
        object_results: List[Tuple[Optional[int], float, List[Dict], Dict]],
        bias_results: List[BiasNudgeOutput],
        objects: List[PreprocessedObjectInfo],
    ) -> List[NudgeEvent]:
        """Build NudgeEvents from hypotheses + matched objects.

        Skips hypotheses where:
          - The bias/nudge decision was BIAS (not a real nudge)
          - No object was matched (best_id is None or 0)

        Matches C++ buildPathOneEvents() -> buildEventFromHypothesisAndObjectAnalysis().
        """
        events = []
        obj_map = {o.objectId: o for o in objects}

        for i, hyp in enumerate(hypotheses):
            if i >= len(object_results) or i >= len(bias_results):
                continue

            best_id, best_dist, _, _ = object_results[i]
            bias = bias_results[i]

            # Skip hypotheses classified as bias or with no matched object
            if bias.is_bias or best_id is None or best_id == 0:
                continue

            obj = obj_map.get(best_id)
            if obj is None:
                continue

            # Bilateral: ego deviated in both directions (unusual, e.g., island obstacle)
            is_bilateral = (
                abs(hyp.left_bias_sum_m) > 0.5 and abs(hyp.right_bias_sum_m) > 0.5
            )

            event = NudgeEvent(
                critical_object_id=best_id,
                obstacle_class=obj.obstacleClass,
                obstacle_state=obj.obstacleState,
                nudge_direction=hyp.direction,
                candidate_type=hyp.type,
                start_time_s=hyp.start_time_s,
                peak_time_s=hyp.peak_time_s,
                end_time_s=hyp.end_time_s,
                max_lateral_deviation_m=hyp.trajectory_features.max_lateral_deviation_m,
                adjusted_dist_m=best_dist,
                distance_ahead_of_ego_m=obj.distanceAheadOfEgo_m,
                object_speed_mps=obj.speed_mps,
                object_was_stationary=obj.obstacleState in ("PARKED", "STOPPED"),
                object_was_oncoming=obj.isOncomingVehicle,
                is_bilateral=is_bilateral,
                description=_build_object_description(obj),
            )
            event.reasoning = _generate_reasoning(event)
            events.append(event)

        return events

    def select_primary(self, events: List[NudgeEvent]) -> Optional[NudgeEvent]:
        """Select the primary (most critical) event by minimum adjusted distance.

        No weighting or threshold — purely proximity-based selection.
        Matches C++ selectPrimaryEvent().
        """
        if not events:
            return None
        return min(events, key=lambda e: e.adjusted_dist_m)


# ==================================================================================================
# Reasoning generation
#
# C++ references:
#   buildObjectDescription() — NudgeClassifier.cpp (nested hierarchy)
#   generateReasoning()      — NudgeClassifier.cpp:573
#
# Simplified from C++ which has a complex hierarchy for VRU spatial context
# (e.g., "a person opening the left door of a parked vehicle").
# ==================================================================================================


def _build_object_description(obj: PreprocessedObjectInfo) -> str:
    """Build a short human-readable description of the object.

    Examples: "a parked vehicle", "a pedestrian", "a stopped truck"
    """
    if obj.description:
        return obj.description

    state = obj.obstacleState.lower() if obj.obstacleState != "UNKNOWN" else ""

    if obj.isVRU:
        cls = "pedestrian" if "PEDESTRIAN" in obj.obstacleClass else "cyclist"
    elif obj.isLargeVehicle:
        cls = "truck"
    elif obj.isVehicle:
        cls = "vehicle"
    else:
        cls = obj.obstacleClass.lower() if obj.obstacleClass != "UNKNOWN" else "object"

    parts = []
    if state and state != "unknown":
        parts.append(state)
    parts.append(cls)
    return "a " + " ".join(parts)


def _generate_reasoning(event: NudgeEvent) -> str:
    """Generate a one-line reasoning string for the event.

    Format: "[type] nudge [direction] to [description] [side]"
    Example: "out of lane nudge left to a parked vehicle on the right"

    The side ("on the right") is opposite to the nudge direction — the object
    is on the right, so ego nudged left to avoid it.

    Truncated to 255 chars to match C++ buffer limit.
    Matches C++ generateReasoning().
    """
    type_str = event.candidate_type.lower().replace("_", " ")
    direction = event.nudge_direction.lower()
    desc = event.description or f"a {event.obstacle_class.lower()}"
    side = "on the right" if direction == "left" else "on the left"
    return f"{type_str} nudge {direction} to {desc} {side}"[:255]
