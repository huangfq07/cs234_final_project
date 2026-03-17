// @ts-nocheck
/**
 * Nudge DGPS Deviation SL Graph (Path 1 + Path 2 visualization)
 *
 * Renders an "SL graph":
 * - X axis: s_m  (meters, along projection curve)
 * - Y axis: l_m  (meters, lateral offset)
 *
 * Path 1 (DGPS Trajectory Analysis — baseline-subtracted):
 *   Yellow curve    = full deviation series (baseline-subtracted)
 *   Orange spheres  = detected peaks (with tooltips)
 *   Green/Magenta   = peak start/end markers
 *   Red dashed      = ±min_peak_threshold_m
 *
 * Path 2 (OOLN Decider — raw deviations from lane centerline):
 *   Cyan curve      = raw deviations (up to path2_cutoff_m)
 *   Teal sphere     = OOLN peak (with tooltip)
 *   Magenta dashed  = ±in_lane_threshold_m (OOLN peak validation)
 *
 * Shared:
 *   Gray curve      = raw lane-center deviations (if available)
 *   Vertical lines  = cutoff boundaries (P1 near/ext, P2)
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
// Shared
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

// Path 2
const PATH2_CURVE_COLOR = { r: 0.0, g: 0.85, b: 1.0, a: 0.85 }; // bright cyan
const PATH2_PEAK_COLOR = { r: 0.0, g: 0.9, b: 0.9, a: 1.0 }; // teal
const IN_LANE_THR_COLOR = { r: 0.75, g: 0.25, b: 0.85, a: 0.7 }; // purple
const CUTOFF_P1_COLOR = { r: 1.0, g: 1.0, b: 0.0, a: 0.35 }; // yellow
const CUTOFF_P2_COLOR = { r: 0.0, g: 0.85, b: 1.0, a: 0.35 }; // cyan
const BASELINE_COLOR = { r: 0.6, g: 0.6, b: 0.6, a: 0.6 }; // gray

// Objects
const OBJ_OUTLINE_COLOR = { r: 0.7, g: 0.7, b: 0.7, a: 0.6 };   // gray outline (base-filtered)
const OBJ_ACTIVE_COLOR = { r: 0.7, g: 0.7, b: 0.7, a: 0.2 };    // gray fill (preprocessed, not filtered)
const OBJ_CAUSAL_COLOR = { r: 1.0, g: 0.6, b: 0.0, a: 0.35 };   // orange fill (in bias-nudge decision)
const OBJ_CRITICAL_COLOR = { r: 0.2, g: 1.0, b: 0.2, a: 0.5 };  // green fill (final nudge candidate)
const OBJ_LABEL_COLOR = { r: 0.9, g: 0.9, b: 0.9, a: 0.9 };     // white label

// Lane / junction
const LANE_EDGE_COLOR = { r: 0.3, g: 0.5, b: 1.0, a: 0.5 };     // blue
const JUNCTION_COLOR = { r: 1.0, g: 0.4, b: 0.4, a: 0.6 };      // red-ish
const STATUS_TEXT_COLOR = { r: 1.0, g: 1.0, b: 1.0, a: 0.95 };   // white

// ========== Scales ==========
const GRID_SCALE = { x: 0.15, y: 0.15, z: 0.15 };
const AXIS_SCALE = { x: 0.35, y: 0.35, z: 0.35 };
const CURVE_SCALE = { x: 0.25, y: 0.25, z: 0.25 };
const PATH2_CURVE_SCALE = { x: 0.3, y: 0.3, z: 0.3 };
const PEAK_SCALE = { x: 1.1, y: 1.1, z: 1.1 };
const PATH2_PEAK_SCALE = { x: 1.3, y: 1.3, z: 1.3 };
const START_END_SCALE = { x: 1.0, y: 1.0, z: 1.0 };

function asNumber(v: any, fallback = 0): number {
  return typeof v === "number" && isFinite(v) ? v : fallback;
}

function getNudgePostprocessDebugInfo(root: any): any | undefined {
  return root?.nudge_postprocess_debug_info ?? root?.nudgePostprocessDebugInfo;
}

// ---------------------------------------------------------------------------
// Grid, Axes, Labels
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

function drawAxisText(
  time: Time,
  markerArray: Messages.visualization_msgs__Marker[],
) {
  const legend =
    `SL Graph (Nudge) — Yellow=P1(subtracted) Cyan=P2(raw)\n` +
    `X: s_m (m) * ${X_PLOT_SCALE}  Y: l_m (m) * ${Y_PLOT_SCALE}`;

  for (const frameId of FRAME_IDS) {
    markerArray.push(
      buildRosMarker({
        type: MarkerTypes.TEXT,
        id: markerArray.length,
        ns: `nudge_sl_axis_text_${frameId || "none"}`,
        pose: {
          position: { x: 1.0, y: Y_PLOT_LIMIT_M * Y_PLOT_SCALE + 1.0, z: 0.0 },
          orientation: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
        },
        lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
        text: legend,
        header: { frame_id: frameId, seq: 0, stamp: time },
        scale: { x: 1.2, y: 1.2, z: 1.2 },
        color: { r: 0.9, g: 0.9, b: 0.9, a: 0.95 },
      }),
    );
  }

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
// Threshold / reference lines
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

function drawVerticalLine(
  time: Time,
  markerArray: Messages.visualization_msgs__Marker[],
  x_m: number,
  color: any,
  ns: string,
  labelText?: string,
) {
  const pts = [
    { x: x_m * X_PLOT_SCALE, y: -Y_PLOT_LIMIT_M * Y_PLOT_SCALE, z: 0.0 },
    { x: x_m * X_PLOT_SCALE, y: Y_PLOT_LIMIT_M * Y_PLOT_SCALE, z: 0.0 },
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
        scale: { x: 0.15, y: 0.15, z: 0.15 },
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
              x: x_m * X_PLOT_SCALE,
              y: Y_PLOT_LIMIT_M * Y_PLOT_SCALE + 0.6,
              z: 0.0,
            },
            orientation: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
          },
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          text: labelText,
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: { x: 0.7, y: 0.7, z: 0.7 },
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
  // y = 0
  drawHorizontalLine(time, markerArray, 0.0, ZERO_LINE_COLOR, "nudge_sl_zero");
  // ±threshold (Path 1 peak threshold)
  drawHorizontalLine(
    time, markerArray, threshold_m, THRESHOLD_COLOR,
    "nudge_sl_p1_thr_pos", `P1 T=${threshold_m.toFixed(2)}m`,
  );
  drawHorizontalLine(
    time, markerArray, -threshold_m, THRESHOLD_COLOR,
    "nudge_sl_p1_thr_neg", `P1 -T=${threshold_m.toFixed(2)}m`,
  );
}

// ---------------------------------------------------------------------------
// Path 1: Deviation curve + peaks (baseline-subtracted)
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
    // Raw lane-center series (gray, if available)
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

  // Path 1 peaks
  const peaks: any[] = Array.isArray(peakDet?.peaks) ? peakDet.peaks : [];
  const peakPoints: any[] = [];
  for (const p of peaks) {
    const idx = asNumber(p?.index, -1);
    if (idx >= 0 && idx < full.length) {
      peakPoints.push(full[idx]);
    }
  }
  for (const frameId of FRAME_IDS) {
    if (peakPoints.length > 0) {
      markerArray.push(
        buildRosMarker({
          type: MarkerTypes.SPHERE_LIST,
          id: markerArray.length,
          ns: `nudge_sl_p1_peaks_${frameId || "none"}`,
          points: peakPoints,
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: PEAK_SCALE,
          color: PEAK_COLOR,
        }),
      );
    }
  }

  // Peak tooltips, value labels, start/end markers
  for (const p of peaks) {
    const idx = asNumber(p?.index, -1);
    if (idx < 0 || idx >= full.length) continue;
    const pt = full[idx];
    const l_m = pt.y / Y_PLOT_SCALE;
    const tooltip = `
P1 peak idx: ${String(idx)}
time_s: ${String(asNumber(p?.time_s, 0).toFixed(3))}
signed_dev_m: ${String(asNumber(p?.signed_deviation_m, 0).toFixed(3))}
abs_mag_m: ${String(asNumber(p?.magnitude_m, 0).toFixed(3))}
direction: ${String(p?.direction)}
start_idx: ${String(p?.start_index ?? "")}
end_idx: ${String(p?.end_index ?? "")}
created_window: ${String(Boolean(p?.created_hypothesis_window))}
created_valid: ${String(Boolean(p?.created_valid_hypothesis))}
`;
    for (const frameId of FRAME_IDS) {
      markerArray.push(
        buildRosMarker({
          type: MarkerTypes.LINE_LIST,
          id: markerArray.length,
          ns: `nudge_sl_p1_peak_tooltips_${frameId || "none"}`,
          points: [pt, { x: pt.x + 0.001, y: pt.y + 0.001, z: 0.0 }],
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: { x: 0.4, y: 0.4, z: 0.4 },
          color: PEAK_COLOR,
          tooltipText: tooltip,
        }),
      );

      markerArray.push(
        buildRosMarker({
          type: MarkerTypes.TEXT,
          id: markerArray.length,
          ns: `nudge_sl_p1_peak_values_${frameId || "none"}`,
          pose: {
            position: { x: pt.x + 0.8, y: pt.y + 0.8, z: 0.0 },
            orientation: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
          },
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          text: `P1 l=${l_m.toFixed(2)}m`,
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: { x: 0.85, y: 0.85, z: 0.85 },
          color: PEAK_COLOR,
        }),
      );

      // Start/End markers
      const startIdx = asNumber(p?.start_index, -1);
      const endIdx = asNumber(p?.end_index, -1);
      const endFound = Boolean(p?.end_found ?? false);

      if (startIdx >= 0 && startIdx < full.length) {
        const spt = full[startIdx]!;
        const sl_m = spt.y / Y_PLOT_SCALE;
        markerArray.push(
          buildRosMarker({
            type: MarkerTypes.SPHERE_LIST,
            id: markerArray.length,
            ns: `nudge_sl_p1_start_${frameId || "none"}`,
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
            ns: `nudge_sl_p1_start_text_${frameId || "none"}`,
            pose: {
              position: { x: spt.x + 0.8, y: spt.y - 0.8, z: 0.0 },
              orientation: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
            },
            lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
            text: `start l=${sl_m.toFixed(2)}m`,
            header: { frame_id: frameId, seq: 0, stamp: time },
            scale: { x: 0.8, y: 0.8, z: 0.8 },
            color: START_COLOR,
          }),
        );
      }

      if (endIdx >= 0 && endIdx < full.length) {
        const ept = full[endIdx]!;
        const el_m = ept.y / Y_PLOT_SCALE;
        const endColor = endFound
          ? END_COLOR
          : { r: 0.7, g: 0.7, b: 0.7, a: 0.9 };
        const endLabelPrefix = endFound ? "end" : "end*";
        markerArray.push(
          buildRosMarker({
            type: MarkerTypes.SPHERE_LIST,
            id: markerArray.length,
            ns: `nudge_sl_p1_end_${frameId || "none"}`,
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
            ns: `nudge_sl_p1_end_text_${frameId || "none"}`,
            pose: {
              position: { x: ept.x + 0.8, y: ept.y - 0.8, z: 0.0 },
              orientation: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
            },
            lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
            text: `${endLabelPrefix} l=${el_m.toFixed(2)}m`,
            header: { frame_id: frameId, seq: 0, stamp: time },
            scale: { x: 0.8, y: 0.8, z: 0.8 },
            color: endColor,
          }),
        );
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Path 2: Raw deviation curve + OOLN peak
// ---------------------------------------------------------------------------

function drawPath2Curve(
  time: Time,
  markerArray: Messages.visualization_msgs__Marker[],
  nudgeDbg: any,
) {
  const gating = nudgeDbg?.gating;
  const rawDeviations: any[] =
    gating?.raw_deviations ?? gating?.rawDeviations ?? [];
  const cutoffConfig = gating?.cutoff_config ?? gating?.cutoffConfig ?? {};
  const path2Cutoff_m = asNumber(
    cutoffConfig?.path2_cutoff_m ?? cutoffConfig?.path2CutoffM, 0,
  );

  if (rawDeviations.length < 2) return;

  const s0 = asNumber(rawDeviations[0]?.s_m ?? rawDeviations[0]?.sM, 0);

  // Filter to path2_cutoff_m using cumulative_distance_m (or use all if no cutoff)
  const path2Points: any[] = [];
  for (const d of rawDeviations) {
    const cumDist = asNumber(
      d?.cumulative_distance_m ?? d?.cumulativeDistanceM, 0,
    );
    if (path2Cutoff_m > 0 && cumDist > path2Cutoff_m) break;
    path2Points.push({
      x: (asNumber(d?.s_m ?? d?.sM, 0) - s0) * X_PLOT_SCALE,
      y: asNumber(d?.l_m ?? d?.lM, 0) * Y_PLOT_SCALE,
      z: 0.0,
    });
  }

  if (path2Points.length < 2) return;

  for (const frameId of FRAME_IDS) {
    markerArray.push(
      buildRosMarker({
        type: MarkerTypes.LINE_STRIP,
        id: markerArray.length,
        ns: `nudge_sl_p2_curve_${frameId || "none"}`,
        points: path2Points,
        lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
        header: { frame_id: frameId, seq: 0, stamp: time },
        scale: PATH2_CURVE_SCALE,
        color: PATH2_CURVE_COLOR,
      }),
    );
  }

  // OOLN peak
  const oolnDecider = nudgeDbg?.ooln_decider ?? nudgeDbg?.oolnDecider;
  const hasValidPeak =
    oolnDecider?.has_valid_peak ?? oolnDecider?.hasValidPeak ?? false;
  const peakS_m = asNumber(
    oolnDecider?.peak_longitudinal_m ?? oolnDecider?.peakLongitudinalM, 0,
  );
  const peakSignedDev_m = asNumber(
    oolnDecider?.peak_signed_deviation_m ?? oolnDecider?.peakSignedDeviationM, 0,
  );
  const peakMag_m = asNumber(
    oolnDecider?.peak_magnitude_m ?? oolnDecider?.peakMagnitudeM, 0,
  );

  if (hasValidPeak && peakS_m > 0) {
    const peakPt = {
      x: (peakS_m - s0) * X_PLOT_SCALE,
      y: peakSignedDev_m * Y_PLOT_SCALE,
      z: 0.0,
    };

    const oolnNotes =
      oolnDecider?.processing_notes ?? oolnDecider?.processingNotes ?? "";
    const p1Eval = asNumber(
      oolnDecider?.phase1?.candidates_evaluated ??
      oolnDecider?.phase1?.candidatesEvaluated, 0,
    );
    const p1Pass = asNumber(
      oolnDecider?.phase1?.passing_count ?? oolnDecider?.phase1?.passingCount, 0,
    );

    const tooltip = `
P2 OOLN peak
s_m: ${peakS_m.toFixed(2)}
signed_dev_m: ${peakSignedDev_m.toFixed(3)}
magnitude_m: ${peakMag_m.toFixed(3)}
phase1: ${p1Eval} eval, ${p1Pass} pass
notes: ${oolnNotes}
`;
    for (const frameId of FRAME_IDS) {
      markerArray.push(
        buildRosMarker({
          type: MarkerTypes.SPHERE_LIST,
          id: markerArray.length,
          ns: `nudge_sl_p2_peak_${frameId || "none"}`,
          points: [peakPt],
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: PATH2_PEAK_SCALE,
          color: PATH2_PEAK_COLOR,
        }),
      );

      // Tooltip
      markerArray.push(
        buildRosMarker({
          type: MarkerTypes.LINE_LIST,
          id: markerArray.length,
          ns: `nudge_sl_p2_peak_tooltip_${frameId || "none"}`,
          points: [peakPt, { x: peakPt.x + 0.001, y: peakPt.y + 0.001, z: 0.0 }],
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: { x: 0.4, y: 0.4, z: 0.4 },
          color: PATH2_PEAK_COLOR,
          tooltipText: tooltip,
        }),
      );

      // Label
      markerArray.push(
        buildRosMarker({
          type: MarkerTypes.TEXT,
          id: markerArray.length,
          ns: `nudge_sl_p2_peak_label_${frameId || "none"}`,
          pose: {
            position: { x: peakPt.x - 0.8, y: peakPt.y + 0.8, z: 0.0 },
            orientation: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
          },
          lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
          text: `P2 l=${peakSignedDev_m.toFixed(2)}m`,
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: { x: 0.85, y: 0.85, z: 0.85 },
          color: PATH2_PEAK_COLOR,
        }),
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Cutoff boundary lines + baseline + in-lane threshold
// ---------------------------------------------------------------------------

function drawReferenceLines(
  time: Time,
  markerArray: Messages.visualization_msgs__Marker[],
  nudgeDbg: any,
) {
  const gating = nudgeDbg?.gating;
  const cutoffConfig = gating?.cutoff_config ?? gating?.cutoffConfig ?? {};
  const nearCutoff_m = asNumber(
    cutoffConfig?.near_field_cutoff_m ?? cutoffConfig?.nearFieldCutoffM, 0,
  );
  const extCutoff_m = asNumber(
    cutoffConfig?.extended_field_cutoff_m ?? cutoffConfig?.extendedFieldCutoffM, 0,
  );
  const path2Cutoff_m = asNumber(
    cutoffConfig?.path2_cutoff_m ?? cutoffConfig?.path2CutoffM, 0,
  );

  // Vertical cutoff boundaries
  if (nearCutoff_m > 0) {
    drawVerticalLine(time, markerArray, nearCutoff_m, CUTOFF_P1_COLOR,
      "nudge_sl_cutoff_p1_near", `P1 near ${nearCutoff_m.toFixed(0)}m`);
  }
  if (extCutoff_m > 0) {
    drawVerticalLine(time, markerArray, extCutoff_m, CUTOFF_P1_COLOR,
      "nudge_sl_cutoff_p1_ext", `P1 ext ${extCutoff_m.toFixed(0)}m`);
  }
  if (path2Cutoff_m > 0) {
    drawVerticalLine(time, markerArray, path2Cutoff_m, CUTOFF_P2_COLOR,
      "nudge_sl_cutoff_p2", `P2 ${path2Cutoff_m.toFixed(0)}m`);
  }

  // Half lane width (lane edge boundaries)
  const detCtx = nudgeDbg?.detection_context ?? nudgeDbg?.detectionContext;
  const egoLaneWidth_m = asNumber(
    detCtx?.ego_lane_width_m ?? detCtx?.egoLaneWidthM, 0,
  );
  if (egoLaneWidth_m > 0) {
    const hlw = egoLaneWidth_m * 0.5;
    drawHorizontalLine(time, markerArray, hlw, LANE_EDGE_COLOR,
      "nudge_sl_lane_edge_l", `Lane +${hlw.toFixed(2)}m`);
    drawHorizontalLine(time, markerArray, -hlw, LANE_EDGE_COLOR,
      "nudge_sl_lane_edge_r", `Lane -${hlw.toFixed(2)}m`);
  }

  // Junction entry / exit vertical lines
  const inputCtx = nudgeDbg?.input_context ?? nudgeDbg?.inputContext;
  const jctEntry_m = asNumber(
    inputCtx?.ego_distance_to_next_junction_entry ??
    inputCtx?.egoDistanceToNextJunctionEntry, NaN,
  );
  const jctExit_m = asNumber(
    inputCtx?.ego_distance_to_next_junction_exit ??
    inputCtx?.egoDistanceToNextJunctionExit, NaN,
  );
  if (isFinite(jctEntry_m) && jctEntry_m > 0) {
    drawVerticalLine(time, markerArray, jctEntry_m, JUNCTION_COLOR,
      "nudge_sl_jct_entry", `Jct Entry ${jctEntry_m.toFixed(0)}m`);
  }
  if (isFinite(jctExit_m) && jctExit_m > 0) {
    drawVerticalLine(time, markerArray, jctExit_m, JUNCTION_COLOR,
      "nudge_sl_jct_exit", `Jct Exit ${jctExit_m.toFixed(0)}m`);
  }

  // Adaptive baseline (horizontal gray line at baseline_m on raw scale)
  const baselineInfo = detCtx?.baseline;
  const finalBaseline_m = asNumber(
    baselineInfo?.final_baseline_m ?? baselineInfo?.finalBaselineM, 0,
  );
  const blEnabled =
    baselineInfo?.enable_adaptive_baseline ?? baselineInfo?.enableAdaptiveBaseline ?? false;

  if (blEnabled && Math.abs(finalBaseline_m) > 0.001) {
    drawHorizontalLine(time, markerArray, finalBaseline_m, BASELINE_COLOR,
      "nudge_sl_baseline", `BL=${finalBaseline_m.toFixed(3)}m`);
  }

  // In-lane threshold (used by OOLN on raw deviations)
  // Try dgps_trajectory_analysis first; if 0 (e.g. junction suppression),
  // estimate from ego_lane_width using the C++ formula:
  //   (laneWidth/2) - (vehicleWidth/2) + 0.2m
  const dgpsAnalysis =
    nudgeDbg?.dgps_trajectory_analysis ?? nudgeDbg?.dgpsTrajectoryAnalysis;
  let inLaneThr_m = asNumber(
    dgpsAnalysis?.in_lane_threshold_m ?? dgpsAnalysis?.inLaneThresholdM, 0,
  );
  if (inLaneThr_m <= 0) {
    const egoLaneWidth_m = asNumber(
      detCtx?.ego_lane_width_m ?? detCtx?.egoLaneWidthM, 0,
    );
    const DEFAULT_VEHICLE_WIDTH_M = 1.9;
    const OUT_OF_LANE_TOLERANCE_M = 0.2;
    if (egoLaneWidth_m > 0) {
      inLaneThr_m = egoLaneWidth_m * 0.5 - DEFAULT_VEHICLE_WIDTH_M * 0.5
                    + OUT_OF_LANE_TOLERANCE_M;
    }
  }
  // OOLN uses min(1.0, inLaneThreshold) to avoid overly strict bar in wide lanes
  const MIN_OOLN_PEAK_THRESHOLD_M = 1.0;
  const oolnPeakThr_m = inLaneThr_m > 0
    ? Math.min(MIN_OOLN_PEAK_THRESHOLD_M, inLaneThr_m) : 0;
  if (oolnPeakThr_m > 0) {
    drawHorizontalLine(time, markerArray, oolnPeakThr_m, IN_LANE_THR_COLOR,
      "nudge_sl_inlane_thr_pos", `P2 Thr=${oolnPeakThr_m.toFixed(2)}m`);
    drawHorizontalLine(time, markerArray, -oolnPeakThr_m, IN_LANE_THR_COLOR,
      "nudge_sl_inlane_thr_neg");
  }
}

// ---------------------------------------------------------------------------
// Status overlay: key state info in the top-left corner
// ---------------------------------------------------------------------------

function drawStatusOverlay(
  time: Time,
  markerArray: Messages.visualization_msgs__Marker[],
  nudgeDbg: any,
) {
  const output = nudgeDbg?.output;
  const inputCtx = nudgeDbg?.input_context ?? nudgeDbg?.inputContext;
  const detCtx = nudgeDbg?.detection_context ?? nudgeDbg?.detectionContext;
  const gating = nudgeDbg?.gating;
  const suppression = gating?.suppression;
  const oolnDecider = nudgeDbg?.ooln_decider ?? nudgeDbg?.oolnDecider;
  const baselineInfo = detCtx?.baseline;
  const outOfLaneState = nudgeDbg?.out_of_lane_state ?? nudgeDbg?.outOfLaneState;

  const lines: string[] = [];

  // Outcome
  const reasoning = output?.reasoning ?? "";
  if (reasoning) {
    lines.push(reasoning);
  }

  // Suppression
  const isSuppressed = suppression?.is_suppressed ?? suppression?.isSuppressed ?? false;
  const suppressionText =
    suppression?.suppression_reason_text ?? suppression?.suppressionReasonText ?? "";
  if (isSuppressed && suppressionText) {
    lines.push(`SUPPRESSED: ${suppressionText}`);
  }

  // State machine
  const smPhase = asNumber(outOfLaneState?.phase, 0);
  const smPhaseNames = ["Idle", "Prepare", "Execute", "Hysteresis"];
  const smConfirm = asNumber(outOfLaneState?.confirm_frames ?? outOfLaneState?.confirmFrames, 0);
  const smHyst = asNumber(outOfLaneState?.hysteresis_frames ?? outOfLaneState?.hysteresisFrames, 0);
  if (smPhase > 0) {
    lines.push(`SM: ${smPhaseNames[smPhase] ?? smPhase} (confirm=${smConfirm} hyst=${smHyst})`);
  }

  // Baseline
  const blFinal = asNumber(baselineInfo?.final_baseline_m ?? baselineInfo?.finalBaselineM, 0);
  const blEnabled =
    baselineInfo?.enable_adaptive_baseline ?? baselineInfo?.enableAdaptiveBaseline ?? false;
  const blNotes = baselineInfo?.processing_notes ?? baselineInfo?.processingNotes ?? "";
  if (blEnabled || blNotes) {
    lines.push(`Baseline: ${blFinal.toFixed(3)}m ${blNotes}`);
  }

  // Lane width
  const laneW = asNumber(detCtx?.ego_lane_width_m ?? detCtx?.egoLaneWidthM, 0);
  if (laneW > 0) {
    lines.push(`Lane: ${laneW.toFixed(2)}m`);
  }

  // OOLN summary
  const oolnNotes =
    oolnDecider?.processing_notes ?? oolnDecider?.processingNotes ?? "";
  if (oolnNotes) {
    lines.push(`OOLN: ${oolnNotes}`);
  }

  // Junction
  const inJunction = inputCtx?.ego_in_junction ?? inputCtx?.egoInJunction ?? false;
  const jctEntry = inputCtx?.ego_distance_to_next_junction_entry ??
    inputCtx?.egoDistanceToNextJunctionEntry;
  if (inJunction) {
    lines.push(`IN JUNCTION (entry=${typeof jctEntry === "number" ? jctEntry.toFixed(1) : "?"}m)`);
  }

  if (lines.length === 0) return;

  const text = lines.join("\n");

  for (const frameId of FRAME_IDS) {
    markerArray.push(
      buildRosMarker({
        type: MarkerTypes.TEXT,
        id: markerArray.length,
        ns: `nudge_sl_status_${frameId || "none"}`,
        pose: {
          position: {
            x: 40.0,
            y: Y_PLOT_LIMIT_M * Y_PLOT_SCALE - 1.0,
            z: 0.1,
          },
          orientation: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
        },
        lifetime: { sec: 0, nsec: LIFE_SPAN_NANO_SECONDS },
        text,
        header: { frame_id: frameId, seq: 0, stamp: time },
        scale: { x: 1.4, y: 1.4, z: 1.4 },
        color: STATUS_TEXT_COLOR,
      }),
    );
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
  idBase: number,
) {
  const x1 = sMin * X_PLOT_SCALE;
  const x2 = sMax * X_PLOT_SCALE;
  const y1 = lRight * Y_PLOT_SCALE; // right edge (more negative)
  const y2 = lLeft * Y_PLOT_SCALE;  // left edge (more positive)

  const rectPts = [
    { x: x1, y: y1, z: 0.0 },
    { x: x2, y: y1, z: 0.0 },
    { x: x2, y: y2, z: 0.0 },
    { x: x1, y: y2, z: 0.0 },
    { x: x1, y: y1, z: 0.0 }, // close
  ];

  for (const frameId of FRAME_IDS) {
    // Outline
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

    // Filled rectangle (CUBE)
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
  const preproc = nudgeDbg?.preprocessing ?? nudgeDbg?.preprocessing;
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

  // Build lookup: causal object ID -> bbox
  const causalBbox = new Map<string, any>();
  for (const co of causalObjects) {
    causalBbox.set(String(co.object_id ?? co.objectId), co);
  }

  // Critical object ID (Path 1 tracked or classifier primary)
  const p1CriticalId = String(
    objAnalysis?.summary?.tracked_object_id ??
    objAnalysis?.summary?.trackedObjectId ??
    classifier?.summary?.primary_event_object_id ??
    classifier?.summary?.primaryEventObjectId ?? "0",
  );

  // OOLN critical object IDs (from events if any)
  const oolnCriticalIds = new Set<string>();
  const oolnEvents: any[] = oolnDecider?.events ?? [];
  for (const ev of oolnEvents) {
    const oid = String(ev?.critical_object_id ?? ev?.criticalObjectId ?? "0");
    if (oid !== "0") oolnCriticalIds.add(oid);
  }

  // OOLN phase1 candidate IDs (that passed)
  const oolnCandidateIds = new Set<string>();
  const oolnCandidates: any[] =
    oolnDecider?.phase1?.candidates ?? [];
  for (const c of oolnCandidates) {
    const reason = c?.rejection_reason ?? c?.rejectionReason ?? "";
    if (!reason) {
      oolnCandidateIds.add(String(c?.object_id ?? c?.objectId));
    }
  }

  for (let i = 0; i < preprocObjects.length; i++) {
    const obj = preprocObjects[i];
    const objId = String(obj?.object_id ?? obj?.objectId ?? "?");
    const dists = obj?.distances ?? {};

    // Determine bbox edges
    let sMin: number, sMax: number, lLeft: number, lRight: number;

    const causal = causalBbox.get(objId);
    if (causal) {
      // Use exact SL bbox from causal objects
      sMin = asNumber(
        causal.bbox_longitudinal_min_m ?? causal.bboxLongitudinalMinM, 0) - s0;
      sMax = asNumber(
        causal.bbox_longitudinal_max_m ?? causal.bboxLongitudinalMaxM, 0) - s0;
      lLeft = asNumber(causal.bbox_left_edge_m ?? causal.bboxLeftEdgeM, 0);
      lRight = asNumber(causal.bbox_right_edge_m ?? causal.bboxRightEdgeM, 0);
    } else if (halfLaneWidth > 0) {
      // Approximate bbox from preprocessed distances
      sMin = asNumber(
        dists.distance_ahead_of_ego_m ?? dists.distanceAheadOfEgoM, 0);
      sMax = Math.abs(asNumber(
        dists.distance_passed_by_ego_m ?? dists.distancePassedByEgoM, 0));
      if (sMin > sMax) { const tmp = sMin; sMin = sMax; sMax = tmp; }
      lLeft = halfLaneWidth - asNumber(
        dists.in_lane_left_margin_m ?? dists.inLaneLeftMarginM, halfLaneWidth);
      lRight = -(halfLaneWidth - asNumber(
        dists.in_lane_right_margin_m ?? dists.inLaneRightMarginM, halfLaneWidth));
    } else {
      continue; // Cannot determine bbox
    }

    // Determine fill color based on status
    const isCausal = causalBbox.has(objId) || oolnCandidateIds.has(objId);
    const isCritical =
      (p1CriticalId === objId && p1CriticalId !== "0") ||
      oolnCriticalIds.has(objId);

    const filtered = obj?.is_base_filtered ?? obj?.isBaseFiltered ?? false;

    let fillColor: any | null = null;
    let outlineColor = OBJ_OUTLINE_COLOR;
    if (isCritical) {
      fillColor = OBJ_CRITICAL_COLOR;
      outlineColor = OBJ_CRITICAL_COLOR;
    } else if (isCausal) {
      fillColor = OBJ_CAUSAL_COLOR;
      outlineColor = OBJ_CAUSAL_COLOR;
    } else if (!filtered) {
      // Active preprocessed object (not filtered, not causal): gray fill
      fillColor = OBJ_ACTIVE_COLOR;
    }
    // else: base-filtered → outline only (no fill)

    drawObjectRect(
      time, markerArray, sMin, sMax, lLeft, lRight,
      fillColor, outlineColor, `nudge_sl_obj_${objId}`, i,
    );

    // Object ID label
    const cx = ((sMin + sMax) * 0.5) * X_PLOT_SCALE;
    const cy = ((lLeft + lRight) * 0.5) * Y_PLOT_SCALE;
    const statusTag = isCritical ? " [CRITICAL]" : isCausal ? " [CAUSAL]" : filtered ? " [filtered]" : "";
    const props = obj?.properties ?? {};
    const speed = asNumber(props?.speed_mps ?? props?.speedMps, 0);
    const oncoming = props?.is_oncoming_vehicle ?? props?.isOncomingVehicle ?? false;

    const labelText = `${objId}${statusTag}\n${speed.toFixed(1)}m/s${oncoming ? " onc" : ""}`;

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
          text: labelText,
          header: { frame_id: frameId, seq: 0, stamp: time },
          scale: { x: 1.2, y: 1.2, z: 1.2 },
          color: isCritical ? OBJ_CRITICAL_COLOR : isCausal ? OBJ_CAUSAL_COLOR : OBJ_LABEL_COLOR,
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

  const deleteAllMarker = new Marker({ action: 3 }); // DELETE_ALL
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
  drawAxisText(event.receiveTime, markers);

  const threshold_m = asNumber(
    dgpsAnalysis?.peak_detection?.min_peak_threshold_m,
    0.4,
  );
  drawThresholdLines(event.receiveTime, markers, threshold_m);

  // Path 1: baseline-subtracted deviations + peaks (yellow/orange)
  if (dgpsAnalysis) {
    drawPath1Curve(event.receiveTime, markers, dgpsAnalysis);
  }

  // Path 2: raw deviations + OOLN peak (cyan/teal)
  if (nudgeDbg) {
    drawPath2Curve(event.receiveTime, markers, nudgeDbg);
    drawReferenceLines(event.receiveTime, markers, nudgeDbg);
  }

  // Objects: rectangles on the SL graph
  if (nudgeDbg) {
    const gating = nudgeDbg?.gating;
    const rawDevs = gating?.raw_deviations ?? gating?.rawDeviations ?? [];
    const objS0 = rawDevs.length > 0 ? asNumber(rawDevs[0]?.s_m ?? rawDevs[0]?.sM, 0) : 0;
    drawObjects(event.receiveTime, markers, nudgeDbg, objS0);
  }

  // Status overlay: key state info
  if (nudgeDbg) {
    drawStatusOverlay(event.receiveTime, markers, nudgeDbg);
  }

  return { markers: [deleteAllMarker, ...markers] as any };
}
