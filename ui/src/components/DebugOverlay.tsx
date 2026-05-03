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

/// Format the debug-overlay state as a multi-line string suitable
/// for pasting into a bug report. Same content the on-screen
/// overlay shows. Available even when the overlay is hidden.
function formatDebug(s: DebugOverlayState): string {
  const [rx, ry, rz] = s.cameraRootXyz;
  const [lx, ly, lz] = s.cameraLocal;
  const ts = new Date().toISOString();
  return [
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
    pad("fov", s.fov.toFixed(3)),
    "",
    "── frame ──",
    pad("active kind", s.activeFrameKind),
    "render   [" + s.renderPathCsv + "]",
    "intended [" + (s.intendedRenderPathCsv || "") + "]",
    "anchor   [" + s.anchorSlotsCsv + "]",
    "stop     " + (s.renderStopReason || "ok"),
    "",
    "── rotation ──",
    pad("TB on anchor path", s.tbOnAnchorPath ? "yes" : "no"),
    pad("cumulative yaw", s.anchorCumulativeYawDeg.toFixed(3) + "°"),
    "",
    pad("nodes", String(s.nodeCount)),
  ].join("\n");
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
    copyToClipboard(formatDebug(s)).then((ok) => {
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
