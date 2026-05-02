import { useDebugOverlay } from "../hooks/useGameState";
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

export function DebugOverlay() {
  const s = useDebugOverlay();
  if (!s.visible) return null;

  const [rx, ry, rz] = s.cameraRootXyz;
  const [lx, ly, lz] = s.cameraLocal;

  const lines = [
    "DEBUG [ ]=toggle",
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
  ];

  return <div className="debug-overlay">{lines.join("\n")}</div>;
}
