import { useDebugOverlay } from "../hooks/useGameState";
import "./DebugOverlay.css";

// Two widths: `pad` is the standard short-value row (numbers, depths);
// `padWide` is the left-aligned-label-then-value for strings like the
// anchor-slot CSV that can get long.
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

export function DebugOverlay() {
  const s = useDebugOverlay();
  if (!s.visible) return null;

  const [rx, ry, rz] = s.cameraRootXyz;
  const [lx, ly, lz] = s.cameraLocal;

  // Relative position to the outer shell in anchor-cell units —
  // the "you are N anchor cells above the surface" stat. Only
  // meaningful when a body is on the anchor path.
  const cellsAboveOuter = Number.isFinite(s.sphereDistOuter)
    ? s.sphereDistOuter / (s.anchorCellSizeRoot * 3) // body-local → root → cells
    : Number.NaN;

  const insphere = s.sphereState.length > 0;

  const lines = [
    "DEBUG [F6=sphere dbg  ]=toggle]",
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
    "── sphere ──",
    pad("state", insphere ? s.sphereState : "(none)"),
    pad("radii ir,or (body)", insphere
      ? `${s.sphereRadii[0].toFixed(3)}, ${s.sphereRadii[1].toFixed(3)}`
      : "—"),
    pad("dist to center (body)", fmtDist(s.sphereDistCenter)),
    pad("dist to outer (body)", fmtDist(s.sphereDistOuter)),
    pad("dist to inner (body)", fmtDist(s.sphereDistInner)),
    pad("cells above outer", Number.isFinite(cellsAboveOuter)
      ? cellsAboveOuter.toFixed(2)
      : "—"),
    pad("debug mode", String(s.sphereDebugMode)),
    "",
    pad("nodes", String(s.nodeCount)),
  ];

  return <div className="debug-overlay">{lines.join("\n")}</div>;
}
