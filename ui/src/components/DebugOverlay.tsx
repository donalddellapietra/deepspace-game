import { useEffect, useRef, useState } from "react";
import { useDebugOverlay } from "../hooks/useGameState";
import type { DebugOverlayState } from "../types";
import "./DebugOverlay.css";

function pad(label: string, value: string, width = 44): string {
  const gap = Math.max(1, width - label.length - value.length);
  return label + " ".repeat(gap) + value;
}

function fmtCell(v: number): string {
  if (!Number.isFinite(v)) return "—";
  const abs = Math.abs(v);
  if (abs === 0) return "0";
  if (abs >= 1e-3 && abs < 1e4) return v.toFixed(6);
  return v.toExponential(3);
}

function fmtDist(v: number): string {
  if (!Number.isFinite(v)) return "—";
  const abs = Math.abs(v);
  if (abs >= 1e-3 && abs < 1e4) return v.toFixed(5);
  return v.toExponential(3);
}

/// Format the debug-overlay state for the on-screen display.
/// Uses unicode for nicer formatting; not safe for terminal paste.
function formatDebug(s: DebugOverlayState): string {
  const [rx, ry, rz] = s.cameraRootXyz;
  const [lx, ly, lz] = s.cameraLocal;
  const [ox, oy, oz] = s.cameraOffset;
  const [dwx, dwy, dwz] = s.worldDelta;
  const [dox, doy, doz] = s.offsetDelta;
  const ts = new Date().toISOString();
  const lines = [
    `# debug overlay  ${ts}`,
    pad("fps", s.fps.toFixed(1)),
    pad("frame time", s.frameTimeMs.toFixed(2) + " ms"),
    "",
    "── zoom ──",
    pad("zoom level", String(s.zoomLevel)),
    pad("tree depth", String(s.treeDepth)),
    pad("edit depth", String(s.editDepth)),
    pad("visual depth", String(s.visualDepth)),
    pad("anchor depth", String(s.cameraAnchorDepth)),
    pad("anchor cell (root)", fmtCell(s.anchorCellSizeRoot)),
    "",
    "── camera ──",
    pad("root x", fmtDist(rx)),
    pad("root y", fmtDist(ry)),
    pad("root z", fmtDist(rz)),
    pad("local x", lx.toFixed(6)),
    pad("local y", ly.toFixed(6)),
    pad("local z", lz.toFixed(6)),
    pad("offset x", ox.toFixed(6)),
    pad("offset y", oy.toFixed(6)),
    pad("offset z", oz.toFixed(6)),
    pad("fov", s.fov.toFixed(3)),
    "",
    "── per-frame delta ──",
    pad("Δ world", `${fmtDist(dwx)}, ${fmtDist(dwy)}, ${fmtDist(dwz)}`),
    pad("Δ offset", `${dox.toFixed(6)}, ${doy.toFixed(6)}, ${doz.toFixed(6)}`),
    "",
    "── frame ──",
    pad("active kind", s.activeFrameKind),
    "render   [" + s.renderPathCsv + "]",
    "intended [" + (s.intendedRenderPathCsv || "") + "]",
    "anchor   [" + s.anchorSlotsCsv + "]",
    "stop     " + (s.renderStopReason || "ok"),
    "diag     " + (s.pathDiag || "-"),
    "",
    "── rotation ──",
    pad("TB on anchor path", s.tbOnAnchorPath ? "yes" : "no"),
    pad("cumulative yaw", s.anchorCumulativeYawDeg.toFixed(3) + "°"),
  ];
  if (s.lastTbCrossing) {
    const c = s.lastTbCrossing;
    const [bwx, bwy, bwz] = c.beforeWorld;
    const [box, boy, boz] = c.beforeOffset;
    const [awx, awy, awz] = c.afterWorld;
    const [aox, aoy, aoz] = c.afterOffset;
    lines.push(
      "",
      "── last TB boundary crossing ──",
      pad("before tb_on_anchor", c.beforeTbOnAnchor ? "yes" : "no"),
      pad("before world", `${fmtDist(bwx)}, ${fmtDist(bwy)}, ${fmtDist(bwz)}`),
      pad("before offset", `${box.toFixed(6)}, ${boy.toFixed(6)}, ${boz.toFixed(6)}`),
      "before anchor [" + c.beforeAnchor + "]",
      pad("after tb_on_anchor", c.afterTbOnAnchor ? "yes" : "no"),
      pad("after world", `${fmtDist(awx)}, ${fmtDist(awy)}, ${fmtDist(awz)}`),
      pad("after offset", `${aox.toFixed(6)}, ${aoy.toFixed(6)}, ${aoz.toFixed(6)}`),
      "after  anchor [" + c.afterAnchor + "]",
      pad("after yaw", c.afterYawDeg.toFixed(3) + "°"),
      pad("Δ world (cross)", `${fmtDist(awx - bwx)}, ${fmtDist(awy - bwy)}, ${fmtDist(awz - bwz)}`),
    );
  }
  lines.push("", pad("nodes", String(s.nodeCount)));
  return lines.join("\n");
}

/// Pure ASCII format for clipboard copy. Terminals choke on the
/// unicode block-drawing chars and trailing whitespace from pad();
/// this strips all that to a single-space label/value format.
function formatDebugAscii(s: DebugOverlayState): string {
  const [rx, ry, rz] = s.cameraRootXyz;
  const [lx, ly, lz] = s.cameraLocal;
  const [ox, oy, oz] = s.cameraOffset;
  const [dwx, dwy, dwz] = s.worldDelta;
  const [dox, doy, doz] = s.offsetDelta;
  const ts = new Date().toISOString();
  const kv = (k: string, v: string | number) => `${k}: ${v}`;
  const out: string[] = [
    `# debug ${ts}`,
    kv("fps", s.fps.toFixed(1)),
    kv("frame_ms", s.frameTimeMs.toFixed(2)),
    "",
    "[zoom]",
    kv("zoom_level", s.zoomLevel),
    kv("tree_depth", s.treeDepth),
    kv("edit_depth", s.editDepth),
    kv("visual_depth", s.visualDepth),
    kv("anchor_depth", s.cameraAnchorDepth),
    kv("anchor_cell_root", fmtCell(s.anchorCellSizeRoot)),
    "",
    "[camera]",
    kv("root", `${fmtDist(rx)}, ${fmtDist(ry)}, ${fmtDist(rz)}`),
    kv("local", `${lx.toFixed(6)}, ${ly.toFixed(6)}, ${lz.toFixed(6)}`),
    kv("offset", `${ox.toFixed(6)}, ${oy.toFixed(6)}, ${oz.toFixed(6)}`),
    kv("fov", s.fov.toFixed(3)),
    "",
    "[delta]",
    kv("d_world", `${fmtDist(dwx)}, ${fmtDist(dwy)}, ${fmtDist(dwz)}`),
    kv("d_offset", `${dox.toFixed(6)}, ${doy.toFixed(6)}, ${doz.toFixed(6)}`),
    "",
    "[frame]",
    kv("active_kind", s.activeFrameKind),
    kv("render", `[${s.renderPathCsv}]`),
    kv("intended", `[${s.intendedRenderPathCsv || ""}]`),
    kv("anchor", `[${s.anchorSlotsCsv}]`),
    kv("stop", s.renderStopReason || "ok"),
    kv("diag", s.pathDiag || "-"),
    "",
    "[rotation]",
    kv("tb_on_anchor", s.tbOnAnchorPath ? "yes" : "no"),
    kv("cumulative_yaw_deg", s.anchorCumulativeYawDeg.toFixed(3)),
  ];
  if (s.lastTbCrossing) {
    const c = s.lastTbCrossing;
    const [bwx, bwy, bwz] = c.beforeWorld;
    const [box, boy, boz] = c.beforeOffset;
    const [awx, awy, awz] = c.afterWorld;
    const [aox, aoy, aoz] = c.afterOffset;
    out.push(
      "",
      "[last_tb_crossing]",
      kv("before_tb_on_anchor", c.beforeTbOnAnchor ? "yes" : "no"),
      kv("before_world", `${fmtDist(bwx)}, ${fmtDist(bwy)}, ${fmtDist(bwz)}`),
      kv("before_offset", `${box.toFixed(6)}, ${boy.toFixed(6)}, ${boz.toFixed(6)}`),
      kv("before_anchor", `[${c.beforeAnchor}]`),
      kv("after_tb_on_anchor", c.afterTbOnAnchor ? "yes" : "no"),
      kv("after_world", `${fmtDist(awx)}, ${fmtDist(awy)}, ${fmtDist(awz)}`),
      kv("after_offset", `${aox.toFixed(6)}, ${aoy.toFixed(6)}, ${aoz.toFixed(6)}`),
      kv("after_anchor", `[${c.afterAnchor}]`),
      kv("after_yaw_deg", c.afterYawDeg.toFixed(3)),
      kv("d_world_cross", `${fmtDist(awx - bwx)}, ${fmtDist(awy - bwy)}, ${fmtDist(awz - bwz)}`),
    );
  }
  out.push("", kv("nodes", s.nodeCount));
  return out.join("\n");
}

async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fall through
  }
  // Fallback for environments without the modern clipboard API.
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}

export function DebugOverlay() {
  const s = useDebugOverlay();
  const [flash, setFlash] = useState<"copied" | "failed" | null>(null);
  const lastCopySeq = useRef<number>(0);

  // Watch the Rust-side `copySeq` counter. It increments each time
  // the user presses `[` while the overlay is visible (handled in
  // `src/app/input_handlers.rs`). The actual clipboard write has
  // to happen in JS (clipboard API is web-only) so we receive the
  // signal here, format the state, and copy.
  useEffect(() => {
    // Initialize the ref to whatever we first see, so we don't
    // copy on first render just because seq jumped from 0 → N
    // after a hot reload.
    if (lastCopySeq.current === 0 && s.copySeq !== 0) {
      lastCopySeq.current = s.copySeq;
      return;
    }
    if (s.copySeq === lastCopySeq.current) return;
    lastCopySeq.current = s.copySeq;
    copyToClipboard(formatDebugAscii(s)).then((ok) => {
      setFlash(ok ? "copied" : "failed");
      window.setTimeout(() => setFlash(null), 1500);
    });
  }, [s]);

  if (!s.visible) return null;

  const raw = "DEBUG  ]=toggle  [=copy\n" + formatDebug(s);
  const parts = raw.split("\n").map((line, i) => {
    const isPath = line.startsWith("render") || line.startsWith("intended") || line.startsWith("anchor");
    if (isPath) {
      return <span key={i} className="path-line">{line}{"\n"}</span>;
    }
    return <span key={i}>{line}{"\n"}</span>;
  });

  return (
    <div className="debug-overlay">
      {flash === "copied" && <div>✓ copied to clipboard</div>}
      {flash === "failed" && <div>✗ copy failed (clipboard blocked)</div>}
      {flash !== null && <div>{" "}</div>}
      {parts}
    </div>
  );
}
