// @ts-nocheck
/**
 * Simplified Nudge SL Graph — Path 1 peak detection + objects
 *
 * X axis: s_m (meters along projection curve)
 * Y axis: l_m (meters lateral offset, scaled by Y_PLOT_SCALE)
 *
 * Yellow curve   = baseline-subtracted deviation series
 * Gray curve     = raw lane-center deviations
 * Orange spheres = detected peaks (with tooltips)
 * Green/Magenta  = peak start/end markers
 * Red dashed     = ±peak threshold
 * Rectangles     = objects (green=critical, orange=causal, gray=active/filtered)
 */

import { Input, Message } from "./types";
import { Messages, Time } from "ros";
import { buildRosMarker, IRosMarker, MarkerTypes, Marker } from "./markers";

export const inputs = ["/BpPostprocessDebugInfo"];
export const output = "/BpPostprocessDebugInfo/NudgeDeviationSLGraph";

type InputEvent = Input<"/BpPostprocessDebugInfo">;

const LIFE_SPAN_NANO_SECONDS = 5.0 * 66666666;
const FRAME_IDS: string[] = ["car", ""];

// Plot scales (meters -> visualization units)
const X_PLOT_SCALE = 1.0;
const Y_PLOT_SCALE = 5.0;

// Grid config
const GRID_S_RESOLUTION_M = 5.0;
const GRID_L_RESOLUTION_M = 0.5;
const X_PLOT_LIMIT_M = 100.0;
const Y_PLOT_LIMIT_M = 3.0;

const GRID_NUM_S_STEPS = Math.floor(X_PLOT_LIMIT_M / GRID_S_RESOLUTION_M);
const GRID_NUM_L_STEPS = Math.floor((2 * Y_PLOT_LIMIT_M) / GRID_L_RESOLUTION_M);

// ========== Colors ==========
const GRID_COLOR = { r: 0.6, g: 0.6, b: 0.6, a: 0.25 };
const AXIS_COLOR = { r: 0.6, g: 0.6, b: 0.6, a: 1.0 };
const ZERO_LINE_COLOR = { r: 0.7, g: 0.7, b: 0.7, a: 0.9 };

// Path 1
const FULL_SERIES_COLOR = { r: 1.0, g: 1.0, b: 0.0, a: 1.0 }; // yellow
const RAW_SERIES_COLOR = { r: 0.8, g: 0.8, b: 0.8, a: 0.75 }; // gray
const THRESHOLD_COLOR = { r: 1.0, g: 0.3, b: 0.3, a: 0.9 }; // red
const PEAK_COLOR = { r: 1.0, g: 0.5, b: 0.0, a: 1.0 }; // orange
const START_COLOR = { r: 0.2, g: 1.0, b: 0.2, a: 1.0 }; // green
const END_COLOR = { r: 1.0, g: 0.2, b: 1.0, a: 1.0 }; // magenta

// Objects
const OBJ_OUTLINE_COLOR = { r: 0.7, g: 0.7, b: 0.7, a: 0.6 };
const OBJ_ACTIVE_COLOR = { r: 0.7, g: 0.7, b: 0.7, a: 0.2 };
const OBJ_CAUSAL_COLOR = { r: 1.0, g: 0.6, b: 0.0, a: 0.35 };
const OBJ_CRITICAL_COLOR = { r: 0.2, g: 1.0, b: 0.2, a: 0.5 };
const OBJ_LABEL_COLOR = { r: 0.9, g: 0.9, b: 0.9, a: 0.9 };

// ========== Scales ==========
const GRID_SCALE = { x: 0.15, y: 0.15, z: 0.15 };
const AXIS_SCALE = { x: 0.35, y: 0.35, z: 0.35 };
const CURVE_SCALE = { x: 0.25, y: 0.25, z: 0.25 };
const PEAK_SCALE = { x: 2.5, y: 2.5, z: 2.5 };
const START_END_SCALE = { x: 2.0, y: 2.0, z: 2.0 };

function asNumber(v: any, fallback = 0): number {
  return typeof v === "number" && isFinite(v) ? v : fallback;
}

function getNudgePostprocessDebugInfo(root: any): any | undefined {
  return root?.nudge_postprocess_debug_info ?? root?.nudgePostprocessDebugInfo;
}

// ---------------------------------------------------------------------------
// Grid, Axes, Tick Labels
// ---------------------------------------------------------------------------

function drawGrid(
  time: Time,
  markerArray: Messages.visualization_msgs__Marker[],
) {
  const points: any[] = [];
  for (let i = 0; i <= GRID_NUM_S_STEPS; i++) {
    const x = i * GRID_S_RESOLUTION_M * X_PLOT_SCALE;
    points.push({ x, y: -Y_PLOT_LIMIT_M * Y_PLOT_SCALE, z: 0.0 });
    points.push({ x, y: Y_PLOT_LIMIT_M * Y_PLOT_SCALE, z: 0.0 });
  }
  for (let j = 0; j <= GRID_NUM_L_STEPS; j++) {
    const y = (-Y_PLOT_LIMIT_M + j * GRID_L_RESOLUTION_M) * Y_PLOT_SCALE;
    points.push({ x: 0.0, y, z: 0.0 });
    points.push({ x: X_PLOT_LIMIT_M * X_PLOT_SCALE, y, z: 0.0 });
  }
  for (const frameId of FRAME_IDS) {
    markerArray.push(
      buildRosMarker({
        type: MarkerTypes.LINE_LIST,
        id: markerArray.length,
        ns: `nudge_sl_grid_${frameId || "none"}`,
        points,
        lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
        header: { frame_id: frameId, seq: 0, stamp: time },
        scale: GRID_SCALE,
        color: GRID_COLOR,
      }),
    );
  }
}

function drawAxis(
  time: Time,
  markerArray: Messages.visualization_msgs__Marker[],
) {
  const points = [
    { x: 0.0, y: -Y_PLOT_LIMIT_M * Y_PLOT_SCALE, z: 0.0 },
    { x: 0.0, y: Y_PLOT_LIMIT_M * Y_PLOT_SCALE, z: 0.0 },
    { x: 0.0, y: 0.0, z: 0.0 },
    { x: X_PLOT_LIMIT_M * X_PLOT_SCALE, y: 0.0, z: 0.0 },
  ];
  for (const frameId of FRAME_IDS) {
    markerArray.push(
      buildRosMarker({
        type: MarkerTypes.LINE_LIST,
        id: markerArray.length,
        ns: `nudge_sl_axis_${frameId || "none"}`,
        points,
        lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
        header: { frame_id: frameId, seq: 0, stamp: time },
        scale: AXIS_SCALE,
        color: AXIS_COLOR,
      }),
    );
  }
}

function drawTickLabels(
  time: Time,
  markerArray: Messages.visualization_msgs__Marker[],
) {
  const xTickEvery_m = 10.0;
  for (let s = 0.0; s <= X_PLOT_LIMIT_M + 1e-3; s += xTickEvery_m) {
    for (const frameId of FRAME_IDS) {
      markerArray.push(
        buildRosMarker({
          type: MarkerTypes.TEXT,
          id: markerArray.length,
          ns: `nudge_sl_xticks_${frameId || "none"}`,
          pose: {
            position: {
              x: s * X_PLOT_SCALE,
              y: -Y_PLOT_LIMIT_M * Y_PLOT_SCALE - 1.2,
              z: 0.0,
            },
            orientation: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
          },
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          text: `${s.toFixed(0)}m`,
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: { x: 0.9, y: 0.9, z: 0.9 },
          color: { r: 0.8, g: 0.8, b: 0.8, a: 0.9 },
        }),
      );
    }
  }

  const yTickEvery_m = 1.0;
  for (let l = -Y_PLOT_LIMIT_M; l <= Y_PLOT_LIMIT_M + 1e-3; l += yTickEvery_m) {
    for (const frameId of FRAME_IDS) {
      markerArray.push(
        buildRosMarker({
          type: MarkerTypes.TEXT,
          id: markerArray.length,
          ns: `nudge_sl_yticks_${frameId || "none"}`,
          pose: {
            position: { x: -2.0, y: l * Y_PLOT_SCALE, z: 0.0 },
            orientation: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
          },
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          text: `${l.toFixed(0)}m`,
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: { x: 0.9, y: 0.9, z: 0.9 },
          color: { r: 0.8, g: 0.8, b: 0.8, a: 0.9 },
        }),
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Threshold lines
// ---------------------------------------------------------------------------

function drawHorizontalLine(
  time: Time,
  markerArray: Messages.visualization_msgs__Marker[],
  y_m: number,
  color: any,
  ns: string,
  labelText?: string,
) {
  const pts = [
    { x: 0.0, y: y_m * Y_PLOT_SCALE, z: 0.0 },
    { x: X_PLOT_LIMIT_M * X_PLOT_SCALE, y: y_m * Y_PLOT_SCALE, z: 0.0 },
  ];
  for (const frameId of FRAME_IDS) {
    markerArray.push(
      buildRosMarker({
        type: MarkerTypes.LINE_STRIP,
        id: markerArray.length,
        ns: `${ns}_${frameId || "none"}`,
        points: pts,
        lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
        header: { frame_id: frameId, seq: 0, stamp: time },
        scale: { x: 0.2, y: 0.2, z: 0.2 },
        color,
      }),
    );
    if (labelText) {
      markerArray.push(
        buildRosMarker({
          type: MarkerTypes.TEXT,
          id: markerArray.length,
          ns: `${ns}_label_${frameId || "none"}`,
          pose: {
            position: {
              x: X_PLOT_LIMIT_M * X_PLOT_SCALE + 1.0,
              y: y_m * Y_PLOT_SCALE,
              z: 0.0,
            },
            orientation: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
          },
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          text: labelText,
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: { x: 0.85, y: 0.85, z: 0.85 },
          color,
        }),
      );
    }
  }
}

function drawThresholdLines(
  time: Time,
  markerArray: Messages.visualization_msgs__Marker[],
  threshold_m: number,
) {
  drawHorizontalLine(time, markerArray, 0.0, ZERO_LINE_COLOR, "nudge_sl_zero");
  drawHorizontalLine(
    time, markerArray, threshold_m, THRESHOLD_COLOR,
    "nudge_sl_thr_pos", `T=${threshold_m.toFixed(2)}m`,
  );
  drawHorizontalLine(
    time, markerArray, -threshold_m, THRESHOLD_COLOR,
    "nudge_sl_thr_neg",
  );
}

// ---------------------------------------------------------------------------
// Path 1: Deviation curves + peak detection
// ---------------------------------------------------------------------------

function drawPath1Curve(
  time: Time,
  markerArray: Messages.visualization_msgs__Marker[],
  dgpsAnalysis: any,
) {
  const series = dgpsAnalysis?.deviation_series;
  const rawSeries = dgpsAnalysis?.lane_center_deviation_series;
  const peakDet = dgpsAnalysis?.peak_detection;
  const samples: any[] = Array.isArray(series?.samples) ? series.samples : [];
  const rawSamples: any[] = Array.isArray(rawSeries?.samples)
    ? rawSeries.samples
    : [];

  const s0 = samples.length > 0 ? asNumber(samples[0]?.s_m, 0) : 0;
  const rawS0 = rawSamples.length > 0 ? asNumber(rawSamples[0]?.s_m, 0) : 0;

  const full = samples.map((s: any) => ({
    x: (asNumber(s?.s_m, 0) - s0) * X_PLOT_SCALE,
    y: asNumber(s?.l_m, 0) * Y_PLOT_SCALE,
    z: 0.0,
  }));

  const rawFull = rawSamples.map((s: any) => ({
    x: (asNumber(s?.s_m, 0) - rawS0) * X_PLOT_SCALE,
    y: asNumber(s?.l_m, 0) * Y_PLOT_SCALE,
    z: 0.0,
  }));

  for (const frameId of FRAME_IDS) {
    // Raw lane-center series (gray)
    if (rawFull.length > 1) {
      markerArray.push(
        buildRosMarker({
          type: MarkerTypes.LINE_STRIP,
          id: markerArray.length,
          ns: `nudge_sl_curve_raw_${frameId || "none"}`,
          points: rawFull,
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: CURVE_SCALE,
          color: RAW_SERIES_COLOR,
        }),
      );
    }

    // Baseline-subtracted series (yellow)
    if (full.length > 1) {
      markerArray.push(
        buildRosMarker({
          type: MarkerTypes.LINE_STRIP,
          id: markerArray.length,
          ns: `nudge_sl_p1_curve_${frameId || "none"}`,
          points: full,
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: CURVE_SCALE,
          color: FULL_SERIES_COLOR,
        }),
      );
    }
  }

  // Peak detection markers
  const peaks: any[] = Array.isArray(peakDet?.peaks) ? peakDet.peaks : [];

  for (const p of peaks) {
    const idx = asNumber(p?.index, -1);
    if (idx < 0 || idx >= full.length) continue;
    const pt = full[idx];
    const l_m = pt.y / Y_PLOT_SCALE;

    const startIdx = asNumber(p?.start_index, -1);
    const endIdx = asNumber(p?.end_index, -1);
    const endFound = Boolean(p?.end_found ?? false);

    const tooltip =
      `Peak idx=${idx}  dev=${asNumber(p?.signed_deviation_m).toFixed(3)}m  ` +
      `mag=${asNumber(p?.magnitude_m).toFixed(3)}m\n` +
      `start=${startIdx}  end=${endIdx}  ` +
      `window=${Boolean(p?.created_hypothesis_window)}  valid=${Boolean(p?.created_valid_hypothesis)}`;

    for (const frameId of FRAME_IDS) {
      // Peak sphere (orange, larger)
      markerArray.push(
        buildRosMarker({
          type: MarkerTypes.SPHERE_LIST,
          id: markerArray.length,
          ns: `nudge_sl_peak_${frameId || "none"}`,
          points: [pt],
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: PEAK_SCALE,
          color: PEAK_COLOR,
        }),
      );

      // Tooltip (hover)
      markerArray.push(
        buildRosMarker({
          type: MarkerTypes.LINE_LIST,
          id: markerArray.length,
          ns: `nudge_sl_peak_tip_${frameId || "none"}`,
          points: [pt, { x: pt.x + 0.001, y: pt.y + 0.001, z: 0.0 }],
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: { x: 0.4, y: 0.4, z: 0.4 },
          color: PEAK_COLOR,
          tooltipText: tooltip,
        }),
      );

      // Peak value label
      markerArray.push(
        buildRosMarker({
          type: MarkerTypes.TEXT,
          id: markerArray.length,
          ns: `nudge_sl_peak_val_${frameId || "none"}`,
          pose: {
            position: { x: pt.x, y: pt.y + 1.8, z: 0.0 },
            orientation: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
          },
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          text: `PEAK ${l_m.toFixed(2)}m`,
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: { x: 1.5, y: 1.5, z: 1.5 },
          color: PEAK_COLOR,
        }),
      );

      // Start marker (green)
      if (startIdx >= 0 && startIdx < full.length) {
        const spt = full[startIdx]!;
        markerArray.push(
          buildRosMarker({
            type: MarkerTypes.SPHERE_LIST,
            id: markerArray.length,
            ns: `nudge_sl_start_${frameId || "none"}`,
            points: [spt],
            lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
            header: { frame_id: frameId, seq: 0, stamp: time },
            scale: START_END_SCALE,
            color: START_COLOR,
          }),
        );
        markerArray.push(
          buildRosMarker({
            type: MarkerTypes.TEXT,
            id: markerArray.length,
            ns: `nudge_sl_start_lbl_${frameId || "none"}`,
            pose: {
              position: { x: spt.x, y: spt.y - 1.5, z: 0.0 },
              orientation: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
            },
            lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
            text: `START`,
            header: { frame_id: frameId, seq: 0, stamp: time },
            scale: { x: 1.3, y: 1.3, z: 1.3 },
            color: START_COLOR,
          }),
        );
      }

      // End marker (magenta if found, gray if not)
      if (endIdx >= 0 && endIdx < full.length) {
        const ept = full[endIdx]!;
        const endColor = endFound
          ? END_COLOR
          : { r: 0.7, g: 0.7, b: 0.7, a: 0.9 };
        markerArray.push(
          buildRosMarker({
            type: MarkerTypes.SPHERE_LIST,
            id: markerArray.length,
            ns: `nudge_sl_end_${frameId || "none"}`,
            points: [ept],
            lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
            header: { frame_id: frameId, seq: 0, stamp: time },
            scale: START_END_SCALE,
            color: endColor,
          }),
        );
        markerArray.push(
          buildRosMarker({
            type: MarkerTypes.TEXT,
            id: markerArray.length,
            ns: `nudge_sl_end_lbl_${frameId || "none"}`,
            pose: {
              position: { x: ept.x, y: ept.y - 1.5, z: 0.0 },
              orientation: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
            },
            lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
            text: endFound ? `END` : `END*`,
            header: { frame_id: frameId, seq: 0, stamp: time },
            scale: { x: 1.3, y: 1.3, z: 1.3 },
            color: endColor,
          }),
        );
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Objects: rectangles on the SL graph
// ---------------------------------------------------------------------------

function drawObjectRect(
  time: Time,
  markerArray: Messages.visualization_msgs__Marker[],
  sMin: number,
  sMax: number,
  lLeft: number,
  lRight: number,
  fillColor: any | null,
  outlineColor: any,
  ns: string,
) {
  const x1 = sMin * X_PLOT_SCALE;
  const x2 = sMax * X_PLOT_SCALE;
  const y1 = lRight * Y_PLOT_SCALE;
  const y2 = lLeft * Y_PLOT_SCALE;

  const rectPts = [
    { x: x1, y: y1, z: 0.0 },
    { x: x2, y: y1, z: 0.0 },
    { x: x2, y: y2, z: 0.0 },
    { x: x1, y: y2, z: 0.0 },
    { x: x1, y: y1, z: 0.0 },
  ];

  for (const frameId of FRAME_IDS) {
    markerArray.push(
      buildRosMarker({
        type: MarkerTypes.LINE_STRIP,
        id: markerArray.length,
        ns: `${ns}_outline_${frameId || "none"}`,
        points: rectPts,
        lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
        header: { frame_id: frameId, seq: 0, stamp: time },
        scale: { x: 0.15, y: 0.15, z: 0.15 },
        color: outlineColor,
      }),
    );

    if (fillColor) {
      const cx = (x1 + x2) * 0.5;
      const cy = (y1 + y2) * 0.5;
      markerArray.push(
        buildRosMarker({
          type: MarkerTypes.CUBE,
          id: markerArray.length,
          ns: `${ns}_fill_${frameId || "none"}`,
          pose: {
            position: { x: cx, y: cy, z: -0.05 },
            orientation: { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
          },
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: {
            x: Math.max(0.1, Math.abs(x2 - x1)),
            y: Math.max(0.1, Math.abs(y2 - y1)),
            z: 0.02,
          },
          color: fillColor,
        }),
      );
    }
  }
}

function drawObjects(
  time: Time,
  markerArray: Messages.visualization_msgs__Marker[],
  nudgeDbg: any,
  s0: number,
) {
  const preproc = nudgeDbg?.preprocessing;
  const objAnalysis = nudgeDbg?.object_analysis ?? nudgeDbg?.objectAnalysis;
  const classifier =
    nudgeDbg?.classifier_debug_info ?? nudgeDbg?.classifierDebugInfo;
  const oolnDecider = nudgeDbg?.ooln_decider ?? nudgeDbg?.oolnDecider;
  const detCtx = nudgeDbg?.detection_context ?? nudgeDbg?.detectionContext;

  const preprocObjects: any[] =
    preproc?.preprocessed_objects ?? preproc?.preprocessedObjects ?? [];
  const causalObjects: any[] =
    objAnalysis?.causal_objects ?? objAnalysis?.causalObjects ?? [];
  const halfLaneWidth =
    asNumber(detCtx?.ego_lane_width_m ?? detCtx?.egoLaneWidthM, 0) * 0.5;

  const causalBbox = new Map<string, any>();
  for (const co of causalObjects) {
    causalBbox.set(String(co.object_id ?? co.objectId), co);
  }

  const p1CriticalId = String(
    objAnalysis?.summary?.tracked_object_id ??
      objAnalysis?.summary?.trackedObjectId ??
      classifier?.summary?.primary_event_object_id ??
      classifier?.summary?.primaryEventObjectId ??
      "0",
  );

  const oolnCriticalIds = new Set<string>();
  for (const ev of (oolnDecider?.events ?? [])) {
    const oid = String(ev?.critical_object_id ?? ev?.criticalObjectId ?? "0");
    if (oid !== "0") oolnCriticalIds.add(oid);
  }

  const oolnCandidateIds = new Set<string>();
  for (const c of (oolnDecider?.phase1?.candidates ?? [])) {
    if (!(c?.rejection_reason ?? c?.rejectionReason)) {
      oolnCandidateIds.add(String(c?.object_id ?? c?.objectId));
    }
  }

  for (let i = 0; i < preprocObjects.length; i++) {
    const obj = preprocObjects[i];
    const objId = String(obj?.object_id ?? obj?.objectId ?? "?");
    const dists = obj?.distances ?? {};

    let sMin: number, sMax: number, lLeft: number, lRight: number;

    const causal = causalBbox.get(objId);
    if (causal) {
      sMin =
        asNumber(causal.bbox_longitudinal_min_m ?? causal.bboxLongitudinalMinM, 0) - s0;
      sMax =
        asNumber(causal.bbox_longitudinal_max_m ?? causal.bboxLongitudinalMaxM, 0) - s0;
      lLeft = asNumber(causal.bbox_left_edge_m ?? causal.bboxLeftEdgeM, 0);
      lRight = asNumber(causal.bbox_right_edge_m ?? causal.bboxRightEdgeM, 0);
    } else if (halfLaneWidth > 0) {
      sMin = asNumber(dists.distance_ahead_of_ego_m ?? dists.distanceAheadOfEgoM, 0);
      sMax = Math.abs(
        asNumber(dists.distance_passed_by_ego_m ?? dists.distancePassedByEgoM, 0),
      );
      if (sMin > sMax) {
        const tmp = sMin;
        sMin = sMax;
        sMax = tmp;
      }
      lLeft =
        halfLaneWidth -
        asNumber(dists.in_lane_left_margin_m ?? dists.inLaneLeftMarginM, halfLaneWidth);
      lRight = -(
        halfLaneWidth -
        asNumber(dists.in_lane_right_margin_m ?? dists.inLaneRightMarginM, halfLaneWidth)
      );
    } else {
      continue;
    }

    const isCausal = causalBbox.has(objId) || oolnCandidateIds.has(objId);
    const isCritical =
      (p1CriticalId === objId && p1CriticalId !== "0") ||
      oolnCriticalIds.has(objId);
    const filtered = obj?.is_base_filtered ?? obj?.isBaseFiltered ?? false;

    // Skip filtered objects entirely
    if (filtered) continue;

    let fillColor: any | null = OBJ_ACTIVE_COLOR;
    let outlineColor = OBJ_OUTLINE_COLOR;
    if (isCritical) {
      fillColor = OBJ_CRITICAL_COLOR;
      outlineColor = OBJ_CRITICAL_COLOR;
    } else if (isCausal) {
      fillColor = OBJ_CAUSAL_COLOR;
      outlineColor = OBJ_CAUSAL_COLOR;
    }

    drawObjectRect(
      time, markerArray, sMin, sMax, lLeft, lRight,
      fillColor, outlineColor, `nudge_sl_obj_${objId}`,
    );

    // Object ID label
    const cx = (sMin + sMax) * 0.5 * X_PLOT_SCALE;
    const cy = (lLeft + lRight) * 0.5 * Y_PLOT_SCALE;
    const statusTag = isCritical
      ? " [CRIT]"
      : isCausal
        ? " [CAUSAL]"
        : "";

    for (const frameId of FRAME_IDS) {
      markerArray.push(
        buildRosMarker({
          type: MarkerTypes.TEXT,
          id: markerArray.length,
          ns: `nudge_sl_obj_label_${frameId || "none"}`,
          pose: {
            position: { x: cx, y: cy + 0.6, z: 0.0 },
            orientation: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
          },
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          text: `${objId}${statusTag}`,
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: { x: 1.2, y: 1.2, z: 1.2 },
          color: isCritical
            ? OBJ_CRITICAL_COLOR
            : isCausal
              ? OBJ_CAUSAL_COLOR
              : OBJ_LABEL_COLOR,
        }),
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

export default function script(
  event: InputEvent,
): Messages.visualization_msgs__MarkerArray {
  const markers: IRosMarker[] = [];

  const deleteAllMarker = new Marker({ action: 3 });
  deleteAllMarker.header.stamp = {
    sec: event.receiveTime.sec,
    nsec: event.receiveTime.nsec - 1,
  };

  const root = event.message as any;
  const nudgeDbg = getNudgePostprocessDebugInfo(root);
  const dgpsAnalysis =
    nudgeDbg?.dgps_trajectory_analysis ?? nudgeDbg?.dgpsTrajectoryAnalysis;

  drawAxis(event.receiveTime, markers);
  drawGrid(event.receiveTime, markers);
  drawTickLabels(event.receiveTime, markers);

  const threshold_m = asNumber(
    dgpsAnalysis?.peak_detection?.min_peak_threshold_m,
    0.4,
  );
  drawThresholdLines(event.receiveTime, markers, threshold_m);

  if (dgpsAnalysis) {
    drawPath1Curve(event.receiveTime, markers, dgpsAnalysis);
  }

  if (nudgeDbg) {
    const gating = nudgeDbg?.gating;
    const rawDevs = gating?.raw_deviations ?? gating?.rawDeviations ?? [];
    const objS0 =
      rawDevs.length > 0 ? asNumber(rawDevs[0]?.s_m ?? rawDevs[0]?.sM, 0) : 0;
    drawObjects(event.receiveTime, markers, nudgeDbg, objS0);
  }

  return { markers: [deleteAllMarker, ...markers] as any };
}
