import { useEffect, useState } from "react";
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
/// for pasting into a bug report. Same content the on-screen overlay
/// shows (sans the "DEBUG" header). Available even when the overlay
/// is hidden.
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
    pad("render path", "[" + s.renderPathCsv + "]"),
    pad("anchor path", "[" + s.anchorSlotsCsv + "]"),
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
    // fall through to legacy path
  }
  // Fallback for environments that don't expose the modern clipboard
  // API (older webviews, insecure contexts).
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

  // `Ctrl/Cmd+Shift+C` copies the current debug-overlay state to the
  // clipboard regardless of whether the overlay is visible. The
  // shortcut is namespaced behind Shift so it doesn't collide with
  // browser/OS native copy. See the formatDebug() output above for
  // exactly what gets copied.
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const isMod = e.ctrlKey || e.metaKey;
      const isCopy =
        isMod && e.shiftKey && (e.key === "C" || e.key === "c");
      if (!isCopy) return;
      e.preventDefault();
      e.stopPropagation();
      copyToClipboard(formatDebug(s)).then((ok) => {
        setFlash(ok ? "copied" : "failed");
        window.setTimeout(() => setFlash(null), 1500);
      });
    }
    window.addEventListener("keydown", handler, { capture: true });
    return () =>
      window.removeEventListener("keydown", handler, { capture: true });
  }, [s]);

  // Even when the overlay isn't visible, render a small "Copied!" /
  // "Copy failed" toast so the user has feedback that the shortcut
  // fired. Position it identically to the overlay so the corner of
  // the screen flashes.
  if (!s.visible && flash === null) return null;

  if (!s.visible && flash !== null) {
    return (
      <div className="debug-overlay">
        {flash === "copied" ? "✓ debug copied" : "✗ copy failed"}
      </div>
    );
  }

  const lines = [
    "DEBUG  [ ]=toggle  ⌘⇧C / ^⇧C=copy",
    formatDebug(s),
  ];

  return (
    <div className="debug-overlay">
      {flash === "copied" && <div>✓ copied to clipboard</div>}
      {flash === "failed" && <div>✗ copy failed (clipboard blocked)</div>}
      {flash !== null && <div>{" "}</div>}
      {lines.join("\n")}
    </div>
  );
}
